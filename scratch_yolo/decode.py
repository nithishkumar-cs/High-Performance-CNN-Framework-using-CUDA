'''Decode raw two-scale logits and run class-aware NMS in PyTorch.'''

import torch

try:
    from torchvision.ops import batched_nms
except (ImportError, RuntimeError):
    batched_nms = None

from .config import ANCHORS_416, MASKS


def box_iou(one: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(one[:2], others[:, :2])
    bottom_right = torch.minimum(one[2:], others[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(1)
    first_area = (one[2:] - one[:2]).clamp_min(0).prod()
    other_area = (others[:, 2:] - others[:, :2]).clamp_min(0).prod(1)
    return intersection / (first_area + other_area - intersection).clamp_min(1e-9)


@torch.no_grad()
def decode_heads(heads: tuple[torch.Tensor, torch.Tensor], threshold: float = 0.25,
                 nms_iou: float = 0.45, max_detections: int = 300,
                 fast_nms: bool = True) -> list[torch.Tensor]:
    '''Return one [N, 6] tensor per image: x1,y1,x2,y2,score,class.

    Coordinates are normalized to the square, letterboxed model input.
    '''
    if (len(heads) != 2 or not 0 <= threshold <= 1 or not 0 <= nms_iou <= 1
            or max_detections < 1):
        raise ValueError('invalid heads or thresholds')
    batch = heads[0].shape[0]
    candidates = [[] for _ in range(batch)]
    anchors = torch.tensor(ANCHORS_416, dtype=heads[0].dtype, device=heads[0].device) / 416
    for scale, prediction in enumerate(heads):
        n, count, fields, height, width = prediction.shape
        if n != batch or count != 3 or fields < 6 or height != width:
            raise ValueError('invalid raw-head shape')
        y, x = torch.meshgrid(torch.arange(height, device=prediction.device),
                              torch.arange(width, device=prediction.device), indexing='ij')
        center_x = (prediction[:, :, 0].sigmoid() + x) / width
        center_y = (prediction[:, :, 1].sigmoid() + y) / height
        selected_anchors = anchors[list(MASKS[scale])]
        box_w = prediction[:, :, 2].clamp(-10, 10).exp() * selected_anchors[:, 0, None, None]
        box_h = prediction[:, :, 3].clamp(-10, 10).exp() * selected_anchors[:, 1, None, None]
        class_score, class_id = prediction[:, :, 5:].sigmoid().max(dim=2)
        confidence = prediction[:, :, 4].sigmoid() * class_score
        boxes = torch.stack((center_x - box_w / 2, center_y - box_h / 2,
                             center_x + box_w / 2, center_y + box_h / 2), dim=-1)
        for image_id in range(batch):
            mask = confidence[image_id] >= threshold
            if mask.any():
                candidates[image_id].append(torch.cat((
                    boxes[image_id][mask].clamp(0, 1),
                    confidence[image_id][mask].unsqueeze(1),
                    class_id[image_id][mask].unsqueeze(1).to(prediction.dtype),
                ), dim=1))
    results = []
    for image_id, parts in enumerate(candidates):
        if not parts:
            results.append(heads[0].new_empty((0, 6)))
            continue
        merged = torch.cat(parts)
        if fast_nms and batched_nms is not None:
            kept = batched_nms(merged[:, :4], merged[:, 4], merged[:, 5].long(), nms_iou)
            results.append(merged[kept[:max_detections]])
            continue
        order = merged[:, 4].argsort(descending=True)
        kept = []
        while order.numel() and len(kept) < max_detections:
            index = order[0]
            kept.append(index)
            if order.numel() == 1:
                break
            remaining = order[1:]
            overlap = box_iou(merged[index, :4], merged[remaining, :4])
            same_class = merged[index, 5] == merged[remaining, 5]
            order = remaining[~((overlap > nms_iou) & same_class)]
        results.append(merged[torch.stack(kept)])
    return results
