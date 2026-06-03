"""Minimal transforms. Add Mosaic by wrapping the dataset; this file just
exposes the primitives the dataset needs.
"""
from __future__ import annotations

import numpy as np


def letterbox(img: np.ndarray, labels: np.ndarray, new_shape: int = 640,
              color: tuple[int, int, int] = (114, 114, 114)):
    """Resize and pad image to (new_shape, new_shape), updating YOLO labels."""
    h0, w0 = img.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    if (w0, h0) != new_unpad:
        # bilinear resize without cv2: use PIL
        from PIL import Image
        img = np.array(Image.fromarray(img).resize(new_unpad, Image.BILINEAR))

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=color[0])

    if labels.size:
        # labels are (cls, cx, cy, w, h) normalised to original image
        labels = labels.copy()
        # convert to absolute on new canvas
        labels[:, 1] = labels[:, 1] * (w0 * r) + left
        labels[:, 2] = labels[:, 2] * (h0 * r) + top
        labels[:, 3] = labels[:, 3] * (w0 * r)
        labels[:, 4] = labels[:, 4] * (h0 * r)
        # back to normalised on letterboxed canvas
        labels[:, 1] /= new_shape
        labels[:, 2] /= new_shape
        labels[:, 3] /= new_shape
        labels[:, 4] /= new_shape

    return img, labels, r, (left, top)


def random_hflip(img: np.ndarray, labels: np.ndarray, p: float = 0.5):
    if np.random.rand() < p:
        img = img[:, ::-1, :].copy()
        if labels.size:
            labels[:, 1] = 1.0 - labels[:, 1]
    return img, labels


# ---- Mosaic hook -----------------------------------------------------------
# To use Mosaic from ultralytics:
#   from ultralytics.data.augment import Mosaic
# then wrap CocoYOLODataset.__getitem__ to call Mosaic on a batch of 4 indices
# before letterbox. The paper uses Mosaic (Bochkovskiy et al., YOLOv4) — without
# it, expect a few mAP points lower.