'''NumPy raw-head decoding for ONNX/TensorRT runtimes without PyTorch.'''

import numpy as np

from .config import ANCHORS_416, MASKS


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(value, -80, 80)))


def decode_numpy_heads(heads: list[np.ndarray], threshold: float = .25,
                       nms_iou: float = .45, max_detections: int = 300) -> list[np.ndarray]:
    '''Return per-image [N,6] x1,y1,x2,y2,score,class in input coordinates.'''
    if len(heads) != 2 or not 0 <= threshold <= 1 or not 0 <= nms_iou <= 1:
        raise ValueError('invalid heads or thresholds')
    batch = heads[0].shape[0]
    candidates = [[] for _ in range(batch)]
    anchors = np.asarray(ANCHORS_416, dtype=np.float32) / 416
    for scale, prediction in enumerate(heads):
        if prediction.ndim != 5:
            raise ValueError('expected [batch,3,5+classes,grid,grid]')
        n, count, fields, height, width = prediction.shape
        if n != batch or count != 3 or fields < 6 or height != width:
            raise ValueError('invalid raw-head shape')
        grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        cx = (_sigmoid(prediction[:, :, 0]) + grid_x) / width
        cy = (_sigmoid(prediction[:, :, 1]) + grid_y) / height
        selected = anchors[list(MASKS[scale])]
        bw = np.exp(np.clip(prediction[:, :, 2], -10, 10)) * selected[None, :, 0, None, None]
        bh = np.exp(np.clip(prediction[:, :, 3], -10, 10)) * selected[None, :, 1, None, None]
        class_probs = _sigmoid(prediction[:, :, 5:])
        class_id = np.argmax(class_probs, axis=2)
        class_score = np.max(class_probs, axis=2)
        scores = _sigmoid(prediction[:, :, 4]) * class_score
        boxes = np.stack((cx - bw / 2, cy - bh / 2,
                          cx + bw / 2, cy + bh / 2), axis=-1)
        for image_id in range(batch):
            mask = scores[image_id] >= threshold
            if np.any(mask):
                candidates[image_id].append(np.concatenate((
                    np.clip(boxes[image_id][mask], 0, 1),
                    scores[image_id][mask][:, None],
                    class_id[image_id][mask][:, None].astype(np.float32),
                ), axis=1).astype(np.float32))
    results = []
    for parts in candidates:
        if not parts:
            results.append(np.empty((0, 6), dtype=np.float32))
            continue
        merged = np.concatenate(parts)
        order = np.argsort(-merged[:, 4])
        kept = []
        while len(order) and len(kept) < max_detections:
            index = order[0]
            kept.append(index)
            remaining = order[1:]
            if not len(remaining):
                break
            first = merged[index, :4]
            others = merged[remaining, :4]
            xy1 = np.maximum(first[:2], others[:, :2])
            xy2 = np.minimum(first[2:], others[:, 2:])
            intersection = np.prod(np.maximum(xy2 - xy1, 0), axis=1)
            area_a = np.prod(np.maximum(first[2:] - first[:2], 0))
            area_b = np.prod(np.maximum(others[:, 2:] - others[:, :2], 0), axis=1)
            overlap = intersection / np.maximum(area_a + area_b - intersection, 1e-9)
            same_class = merged[index, 5] == merged[remaining, 5]
            order = remaining[~((overlap > nms_iou) & same_class)]
        results.append(merged[kept])
    return results
