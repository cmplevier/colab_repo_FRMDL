"""Anchor utilities for the 2-head YOLOv3-tiny-style detector."""
from __future__ import annotations

import torch


# YOLOv3-tiny default anchors (px @ 416 input). Rescale to img size if needed,
# but with letterbox to a fixed square the absolute pixel values are fine.
_DEFAULT_ANCHORS_BY_STRIDE = {
    16: [(10, 14), (23, 27), (37, 58)],     # smaller objects -> finer grid
    32: [(81, 82), (135, 169), (344, 319)],
}


def build_anchors(strides: tuple[int, ...]):
    """Return tuple of tensors (na, 2) in pixels, one per output scale."""
    return tuple(
        torch.tensor(_DEFAULT_ANCHORS_BY_STRIDE[s], dtype=torch.float32)
        for s in strides
    )


def match_targets(
    targets: torch.Tensor,    # (M, 6): (batch_idx, cls, cx, cy, w, h) in [0,1]
    anchors: torch.Tensor,    # (na, 2) in pixels
    stride: int,
    gh: int, gw: int,
    na: int,
    anchor_thresh: float = 4.0,
):
    """Match each ground-truth box to an anchor on this scale.

    Returns (tcls, tbox, indices, anchor_w):
        tcls   : (N,) long, class index
        tbox   : (N, 4) in grid units (cx, cy, w, h)
        indices: tuple of (b, a, gj, gi) each (N,) long
        anchor_w: (N, 2) anchor (w, h) in grid units, used to decode pred w/h
    """
    device = anchors.device
    if targets.numel() == 0:
        z = torch.zeros((0,), dtype=torch.long, device=device)
        return (z, torch.zeros((0, 4), device=device),
                (z, z, z, z), torch.zeros((0, 2), device=device))

    img_size_px = max(gh, gw) * stride  # square assumption from letterbox
    # to grid coords
    t = targets.to(device).clone()
    gxy = t[:, 2:4] * torch.tensor([gw, gh], device=device)
    gwh = t[:, 4:6] * torch.tensor([gw, gh], device=device)

    anchors_grid = anchors / stride          # (na, 2) in grid units

    # ratio-based matching: anchor whose w/h ratio to target is closest to 1
    # for each (target, anchor) compute max(t/a, a/t) and keep those <= thresh
    M = t.size(0)
    ratio = gwh[:, None, :] / anchors_grid[None, :, :]   # (M, na, 2)
    mratio = torch.max(ratio, 1.0 / ratio).max(dim=2).values  # (M, na)
    mask = mratio < anchor_thresh                              # (M, na)

    if not mask.any():
        # fall back: assign best anchor per target
        best = mratio.argmin(dim=1)
        mask = torch.zeros_like(mratio, dtype=torch.bool)
        mask[torch.arange(M), best] = True

    # Expand to (M*na,) and select positives
    t_idx, a_idx = mask.nonzero(as_tuple=True)
    n = t_idx.numel()
    b = t[t_idx, 0].long()
    tcls = t[t_idx, 1].long()
    txy = gxy[t_idx]                 # (n, 2)
    twh = gwh[t_idx]                 # (n, 2)
    gi = txy[:, 0].long().clamp(0, gw - 1)
    gj = txy[:, 1].long().clamp(0, gh - 1)
    # localise target xy to cell offset
    txy_off = txy - txy.floor()

    tbox = torch.cat([txy_off, twh], dim=1)
    anchor_w = anchors_grid[a_idx]

    return tcls, tbox, (b, a_idx, gj, gi), anchor_w