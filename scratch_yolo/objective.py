'''Anchor assignment and a compact trainable two-scale detection objective.'''

import torch
from torch import nn
from torch.nn import functional as F

from .config import ANCHORS_416, MASKS


class DetectionLoss(nn.Module):
    '''Targets are [batch_index, class, cx, cy, width, height] in [0, 1].

    This is an independent, simplified YOLO objective, not Darknet's exact
    training loss. One anchor receives each object; collisions are unresolved.
    '''

    def __init__(self, classes: int = 20):
        super().__init__()
        self.classes = classes
        self.register_buffer('anchors', torch.tensor(ANCHORS_416, dtype=torch.float32) / 416)

    def forward(self, heads: tuple[torch.Tensor, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        if len(heads) != 2:
            raise ValueError('expected two detection heads')
        device = heads[0].device
        targets = targets.to(device=device, dtype=torch.float32)
        if targets.ndim != 2 or targets.shape[1] != 6:
            raise ValueError('targets must have six columns')
        if targets.numel() and (torch.any(targets[:, 2:] < 0) or torch.any(targets[:, 2:] > 1)):
            raise ValueError('normalized box coordinates must be in [0, 1]')
        anchors = self.anchors.to(device)
        assignments = []
        for row in targets:
            batch_id, class_id = int(row[0]), int(row[1])
            if batch_id < 0 or batch_id >= heads[0].shape[0] or class_id < 0 or class_id >= self.classes:
                raise ValueError('invalid batch or class index')
            wh = row[4:6].clamp_min(1e-6)
            overlap = torch.minimum(anchors, wh).prod(1)
            union = anchors.prod(1) + wh.prod() - overlap
            anchor_id = int(torch.argmax(overlap / union))
            assignments.append((row, batch_id, class_id, anchor_id))

        total = heads[0].new_zeros(())
        for scale, prediction in enumerate(heads):
            batch, count, fields, height, width = prediction.shape
            if count != 3 or fields != self.classes + 5 or height != width:
                raise ValueError('invalid detection-head shape')
            obj = torch.zeros_like(prediction[:, :, 4])
            xy = torch.zeros_like(prediction[:, :, :2])
            wh = torch.zeros_like(prediction[:, :, 2:4])
            cls = torch.zeros_like(prediction[:, :, 5:])
            selected = torch.zeros_like(obj, dtype=torch.bool)
            for row, batch_id, class_id, anchor_id in assignments:
                if anchor_id not in MASKS[scale]:
                    continue
                local_anchor = MASKS[scale].index(anchor_id)
                gx = min(int(row[2] * width), width - 1)
                gy = min(int(row[3] * height), height - 1)
                selected[batch_id, local_anchor, gy, gx] = True
                obj[batch_id, local_anchor, gy, gx] = 1
                xy[batch_id, local_anchor, 0, gy, gx] = row[2] * width - gx
                xy[batch_id, local_anchor, 1, gy, gx] = row[3] * height - gy
                wh[batch_id, local_anchor, :, gy, gx] = torch.log(
                    row[4:6].clamp_min(1e-6) / anchors[anchor_id]
                )
                cls[batch_id, local_anchor, class_id, gy, gx] = 1
            # Normalize foreground and background separately so sparse
            # positives are not diluted by thousands of empty cells.
            obj_logits = prediction[:, :, 4]
            obj_loss = .5 * F.binary_cross_entropy_with_logits(
                obj_logits[~selected], obj[~selected]
            ) if (~selected).any() else obj_logits.new_zeros(())
            if selected.any():
                obj_loss = obj_loss + F.binary_cross_entropy_with_logits(
                    obj_logits[selected], obj[selected]
                )
                pred_xy = prediction[:, :, :2].permute(0, 1, 3, 4, 2)[selected]
                true_xy = xy.permute(0, 1, 3, 4, 2)[selected]
                pred_wh = prediction[:, :, 2:4].permute(0, 1, 3, 4, 2)[selected]
                true_wh = wh.permute(0, 1, 3, 4, 2)[selected]
                pred_cls = prediction[:, :, 5:].permute(0, 1, 3, 4, 2)[selected]
                true_cls = cls.permute(0, 1, 3, 4, 2)[selected]
                total = total + 5 * F.binary_cross_entropy_with_logits(pred_xy, true_xy)
                total = total + 5 * F.smooth_l1_loss(pred_wh, true_wh)
                total = total + F.binary_cross_entropy_with_logits(pred_cls, true_cls)
            total = total + obj_loss
        return total
