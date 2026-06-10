"""Post-processing: decode raw grids to boxes and run NMS."""
from __future__ import annotations

import torch
from torchvision.ops import batched_nms

from .anchors import build_anchors


def decode_predictions(
    preds: tuple[torch.Tensor, ...],
    strides: tuple[int, ...],
    nc: int,
    na: int,
    conf_thresh: float = 0.001,
    anchors=None,
):
    """Decode raw grids -> list of (N, 6) per image: (x1, y1, x2, y2, score, cls).

    anchors: optional list of per-scale anchor lists [[w,h],...] in pixels.
             Must match what was passed to ComputeLoss during training.
             Falls back to build_anchors(strides) default if None.
    """
    device = preds[0].device
    if anchors is not None:
        anchors_all = tuple(
            torch.tensor(a, dtype=torch.float32) for a in anchors
        )
    else:
        anchors_all = build_anchors(strides)
    B = preds[0].size(0)

    outs = []
    for i, p in enumerate(preds):
        Hi, Wi = p.shape[-2:]
        stride = strides[i]
        anchors = anchors_all[i].to(device) / stride   # in grid units

        p = p.view(B, na, 5 + nc, Hi, Wi).permute(0, 1, 3, 4, 2).contiguous()
        # grid offsets
        yv, xv = torch.meshgrid(torch.arange(Hi, device=device),
                                torch.arange(Wi, device=device), indexing="ij")
        grid = torch.stack([xv, yv], dim=-1).view(1, 1, Hi, Wi, 2).float()

        xy = (p[..., 0:2].sigmoid() + grid) * stride
        wh = (p[..., 2:4].sigmoid() * 2) ** 2 * anchors.view(1, na, 1, 1, 2) * stride
        obj = p[..., 4:5].sigmoid()
        cls = p[..., 5:].sigmoid()

        scores = (obj * cls).view(B, -1, nc)  # (B, A*Hi*Wi, nc)
        xy = xy.view(B, -1, 2)
        wh = wh.view(B, -1, 2)
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], dim=-1)  # xyxy
        outs.append((boxes, scores))

    boxes = torch.cat([o[0] for o in outs], dim=1)   # (B, N, 4)
    scores = torch.cat([o[1] for o in outs], dim=1)  # (B, N, nc)

    results = []
    for i in range(B):
        s_max, cls_idx = scores[i].max(dim=1)
        keep = s_max > conf_thresh
        b = boxes[i][keep]
        s = s_max[keep]
        c = cls_idx[keep]
        if b.numel() == 0:
            results.append(torch.zeros((0, 6), device=device))
            continue
        nms_keep = batched_nms(b, s, c, iou_threshold=0.65)[:300]
        results.append(torch.cat([b[nms_keep], s[nms_keep, None], c[nms_keep, None].float()], dim=1))
    return results