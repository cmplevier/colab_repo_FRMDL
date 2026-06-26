"""Image transforms for YOLO training."""
from __future__ import annotations

import math
import random

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


def random_perspective(img: np.ndarray, labels: np.ndarray,
                       degrees: float = 0.0, translate: float = 0.1,
                       scale: float = 0.5, shear: float = 0.0,
                       perspective: float = 0.0) -> tuple:
    """Random affine/perspective warp applied after mosaic assembly.

    img    : (H, W, 3) uint8
    labels : (N, 5) float32 — (cls, cx, cy, w, h) normalised to img dims
    Returns (img, labels) same format and size.
    Requires opencv-python-headless.
    """
    import cv2

    h, w = img.shape[:2]

    C = np.eye(3); C[0, 2] = -w / 2;  C[1, 2] = -h / 2

    P = np.eye(3)
    if perspective:
        P[2, 0] = random.uniform(-perspective, perspective)
        P[2, 1] = random.uniform(-perspective, perspective)

    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0.0, 0.0), scale=s)

    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * w
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * h

    M = T @ S @ R @ P @ C

    if perspective:
        img = cv2.warpPerspective(img, M, dsize=(w, h), borderValue=(114, 114, 114))
    else:
        img = cv2.warpAffine(img, M[:2], dsize=(w, h), borderValue=(114, 114, 114))

    n = len(labels)
    if n == 0:
        return img, labels

    # Abs xyxy corners (4 per box) → transform → new axis-aligned boxes
    bx1 = (labels[:, 1] - labels[:, 3] / 2) * w
    by1 = (labels[:, 2] - labels[:, 4] / 2) * h
    bx2 = (labels[:, 1] + labels[:, 3] / 2) * w
    by2 = (labels[:, 2] + labels[:, 4] / 2) * h

    pts = np.ones((n * 4, 3))
    pts[:, :2] = (np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]])
                  .transpose(2, 0, 1).reshape(-1, 2))
    pts = pts @ M.T
    if perspective:
        pts = (pts[:, :2] / pts[:, 2:3]).reshape(n, 8)
    else:
        pts = pts[:, :2].reshape(n, 8)

    x1n = pts[:, [0, 2, 4, 6]].min(1).clip(0, w)
    y1n = pts[:, [1, 3, 5, 7]].min(1).clip(0, h)
    x2n = pts[:, [0, 2, 4, 6]].max(1).clip(0, w)
    y2n = pts[:, [1, 3, 5, 7]].max(1).clip(0, h)

    valid = (x2n - x1n > 2) & (y2n - y1n > 2)
    if not valid.any():
        return img, np.zeros((0, 5), dtype=np.float32)

    out = np.zeros((valid.sum(), 5), dtype=np.float32)
    out[:, 0] = labels[valid, 0]
    out[:, 1] = (x1n[valid] + x2n[valid]) / 2 / w
    out[:, 2] = (y1n[valid] + y2n[valid]) / 2 / h
    out[:, 3] = (x2n[valid] - x1n[valid]) / w
    out[:, 4] = (y2n[valid] - y1n[valid]) / h
    return img, out