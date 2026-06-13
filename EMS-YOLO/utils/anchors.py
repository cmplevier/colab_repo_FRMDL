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
    # Multi-positive: assign center cell + up to 2 nearest neighbor cells (YOLOv5 g=0.5 pattern).
    # Subtracting off[i] before floor() shifts the cell index to the neighbor:
    #   [0,0]   center   txy_off ∈ [0,1)
    #   [.5,0]  left     floor(gxy-.5)=gi-1 when frac_x<.5  → txy_off_x=frac+1 ∈(1,1.5)
    #   [0,.5]  top      same logic, y axis
    #   [-.5,0] right    floor(gxy+.5)=gi+1 when frac_x>.5  → txy_off_x=frac-1 ∈(-0.5,0)
    #   [0,-.5] bottom
    g = 0.5
    off = torch.tensor([[0., 0.], [.5, 0.], [0., .5], [-.5, 0.], [0., -.5]], device=device)

    frac = txy % 1
    use = torch.stack([
        torch.ones(n, dtype=torch.bool, device=device),           # center always
        (frac[:, 0] < g) & (txy[:, 0] >= 1),                     # left
        (frac[:, 1] < g) & (txy[:, 1] >= 1),                     # top
        (frac[:, 0] > 1 - g) & (txy[:, 0] < gw - 1),            # right
        (frac[:, 1] > 1 - g) & (txy[:, 1] < gh - 1),            # bottom
    ])  # (5, n)
    flat = use.reshape(-1)

    e_txy  = txy.unsqueeze(0).expand(5, -1, -1).reshape(-1, 2)[flat]
    e_twh  = twh.unsqueeze(0).expand(5, -1, -1).reshape(-1, 2)[flat]
    e_b    = b.unsqueeze(0).expand(5, -1).reshape(-1)[flat]
    e_tcls = tcls.unsqueeze(0).expand(5, -1).reshape(-1)[flat]
    e_a    = a_idx.unsqueeze(0).expand(5, -1).reshape(-1)[flat]
    e_off  = off.unsqueeze(1).expand(-1, n, -1).reshape(-1, 2)[flat]

    gij = (e_txy - e_off).long()
    gi  = gij[:, 0].clamp(0, gw - 1)
    gj  = gij[:, 1].clamp(0, gh - 1)
    txy_off = e_txy - gij.float()   # cell-relative; ∈ (-0.5, 1.5) for neighbor cells

    tbox    = torch.cat([txy_off, e_twh], dim=1)
    anchor_w = anchors_grid[e_a]

    return e_tcls, tbox, (e_b, e_a, gj, gi), anchor_w