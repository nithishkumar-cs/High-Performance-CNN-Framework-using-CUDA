'''Pascal-style class matching with 101-point interpolated AP at IoU 0.5.'''

import numpy as np
import torch

from .decode import box_iou


def ap50(predictions: list[torch.Tensor], ground_truth: list[torch.Tensor],
         classes: int) -> dict:
    '''Predictions are [xyxy, score, class]; truth is [class, cx,cy,w,h].

    All coordinates must use the same normalized, letterboxed image space.
    Each GT can be matched once. Classes without GT are omitted from mAP.
    '''
    if len(predictions) != len(ground_truth):
        raise ValueError('prediction and ground-truth image counts differ')
    result = {}
    class_aps = []
    for class_id in range(classes):
        truth = {}
        candidates = []
        for image_id, (found, labels) in enumerate(zip(predictions, ground_truth)):
            matching = labels[labels[:, 0] == class_id]
            center = matching[:, 1:3]
            half = matching[:, 3:5] / 2
            truth[image_id] = torch.cat((center - half, center + half), dim=1)
            for row in found[found[:, 5] == class_id]:
                candidates.append((float(row[4]), image_id, row[:4]))
        count = sum(len(boxes) for boxes in truth.values())
        if not count:
            result[class_id] = None
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        used = {image_id: torch.zeros(len(boxes), dtype=torch.bool)
                for image_id, boxes in truth.items()}
        tp = np.zeros(len(candidates), dtype=np.float64)
        fp = np.zeros(len(candidates), dtype=np.float64)
        for index, (_, image_id, box) in enumerate(candidates):
            boxes = truth[image_id]
            if not len(boxes):
                fp[index] = 1
                continue
            overlaps = box_iou(box, boxes)
            best = int(overlaps.argmax())
            if overlaps[best] >= .5 and not used[image_id][best]:
                tp[index] = 1
                used[image_id][best] = True
            else:
                fp[index] = 1
        recall = np.cumsum(tp) / count
        precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1e-12)
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        positions = np.searchsorted(recall, np.linspace(0, 1, 101), side='left')
        points = np.where(positions < len(envelope),
                          envelope[np.minimum(positions, max(len(envelope) - 1, 0))]
                          if len(envelope) else 0, 0)
        value = float(points.mean())
        result[class_id] = value
        class_aps.append(value)
    return {'map50': float(np.mean(class_aps)) if class_aps else None,
            'per_class_ap50': result, 'images': len(predictions)}
