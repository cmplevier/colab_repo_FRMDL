"""YOLOv3-style loss.

Components:
- CIoU regression loss on matched anchors
- BCE objectness loss using IoU-as-target on positives
- BCE classification loss on positives

This is a working baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchors import build_anchors, match_targets


def bbox_ciou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Both tensors (..., 4) in (cx, cy, w, h)."""
    # to xyxy
    px1 = pred[..., 0] - pred[..., 2] / 2
    py1 = pred[..., 1] - pred[..., 3] / 2
    px2 = pred[..., 0] + pred[..., 2] / 2
    py2 = pred[..., 1] + pred[..., 3] / 2
    tx1 = target[..., 0] - target[..., 2] / 2
    ty1 = target[..., 1] - target[..., 3] / 2
    tx2 = target[..., 0] + target[..., 2] / 2
    ty2 = target[..., 1] + target[..., 3] / 2

    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area_p = (px2 - px1) * (py2 - py1)
    area_t = (tx2 - tx1) * (ty2 - ty1)
    union = area_p + area_t - inter + eps
    iou = inter / union

    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw * cw + ch * ch + eps
    rho2 = ((target[..., 0] - pred[..., 0]) ** 2 + (target[..., 1] - pred[..., 1]) ** 2)

    import math
    v = (4 / math.pi ** 2) * (torch.atan(target[..., 2] / (target[..., 3] + eps))
                              - torch.atan(pred[..., 2] / (pred[..., 3] + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - rho2 / c2 - alpha * v


class ComputeLoss:
    """Callable: returns scalar loss and a dict of components."""

    def __init__(self, model, lambda_box: float = 0.05, lambda_obj: float = 1.0,
                 lambda_cls: float = 0.5):
        self.model = model
        self.nc = model.nc
        self.na = model.na
        self.strides = model.strides
        # Sensible default anchors (COCO yolov3-tiny style, just two scales)
        self.anchors = build_anchors(self.strides)  # tuple of (na, 2) per scale, in pixels
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def __call__(self, preds, targets):
        """preds: tuple of grids, each (B, A*(5+nc), Hi, Wi)
        targets: (M, 6) -> (batch_idx, cls, cx, cy, w, h) in [0,1]
        """
        device = preds[0].device
        loss_box = torch.zeros((), device=device)
        loss_obj = torch.zeros((), device=device)
        loss_cls = torch.zeros((), device=device)

        for i, p in enumerate(preds):
            B, _, Hi, Wi = p.shape
            p = p.view(B, self.na, 5 + self.nc, Hi, Wi).permute(0, 1, 3, 4, 2).contiguous()
            # decode shape-only for matching; actual decode happens after matching
            stride = self.strides[i]
            anchors = self.anchors[i].to(device)  # (na, 2) in pixels

            tcls, tbox, indices, anchor_w = match_targets(
                targets, anchors, stride, Hi, Wi, self.na
            )
            b, a, gj, gi = indices  # all (N,)
            n = b.numel()

            obj_target = torch.zeros_like(p[..., 0])  # (B, na, H, W)

            if n:
                ps = p[b, a, gj, gi]  # (N, 5+nc)
                # decode bbox
                pxy = ps[:, 0:2].sigmoid()
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchor_w  # (N, 2)
                pbox = torch.cat([pxy, pwh], dim=-1)
                ciou = bbox_ciou(pbox, tbox)
                loss_box = loss_box + (1.0 - ciou).mean()

                obj_target[b, a, gj, gi] = ciou.detach().clamp(0).type(obj_target.dtype)

                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], 0.0)
                    t[range(n), tcls] = 1.0
                    loss_cls = loss_cls + self.bce(ps[:, 5:], t)

            loss_obj = loss_obj + self.bce(p[..., 4], obj_target)

        loss = self.lambda_box * loss_box + self.lambda_obj * loss_obj + self.lambda_cls * loss_cls
        return loss, {
            "box": loss_box.detach(),
            "obj": loss_obj.detach(),
            "cls": loss_cls.detach(),
            "total": loss.detach(),
        }