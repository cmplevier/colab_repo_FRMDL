"""COCO 2017 dataset for YOLO-style training."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from transforms import letterbox, random_hflip, random_perspective


def _build_id_map(cat_ids: list[int]) -> tuple[dict, dict]:
    cat_ids = sorted(cat_ids)
    id_to_idx = {c: i for i, c in enumerate(cat_ids)}
    idx_to_id = {i: c for c, i in id_to_idx.items()}
    return id_to_idx, idx_to_id


class CocoYOLODataset(Dataset):
    def __init__(self, root: str, split: str = "train2017", img_size: int = 640,
                 augment: bool = True):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = 1.0 if augment else 0.0

        ann_file = self.root / "annotations" / f"instances_{split}.json"
        with open(ann_file) as f:
            data = json.load(f)

        self.imgs = {im["id"]: im for im in data["images"]}
        self.cat_ids = [c["id"] for c in data["categories"]]
        self.id_to_idx, self.idx_to_id = _build_id_map(self.cat_ids)

        self.anns_by_img: dict[int, list[dict[str, Any]]] = {i: [] for i in self.imgs}
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            if ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0:
                continue
            self.anns_by_img[ann["image_id"]].append(ann)

        self.img_ids = list(self.imgs.keys())

    def __len__(self) -> int:
        return len(self.img_ids)

    def _load_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Load image and YOLO-normalised labels without any resizing."""
        meta = self.imgs[self.img_ids[idx]]
        img_path = self.root / "images" / self.split / meta["file_name"]
        from PIL import Image
        img = np.array(Image.open(img_path).convert("RGB"))
        H0, W0 = img.shape[:2]
        labels = []
        for ann in self.anns_by_img[self.img_ids[idx]]:
            x, y, w, h = ann["bbox"]
            labels.append([
                self.id_to_idx[ann["category_id"]],
                (x + w / 2) / W0, (y + h / 2) / H0,
                w / W0, h / H0,
            ])
        return img, np.array(labels, dtype=np.float32).reshape(-1, 5)

    def _load_mosaic(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Combine 4 images into a mosaic of size (img_size, img_size)."""
        s = self.img_size
        xc = int(random.uniform(s * 0.5, s * 1.5))
        yc = int(random.uniform(s * 0.5, s * 1.5))
        indices = [index] + random.choices(range(len(self)), k=3)

        canvas = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
        all_labels: list[np.ndarray] = []

        from PIL import Image as PImage
        for i, idx in enumerate(indices):
            img, labels = self._load_raw(idx)
            h, w = img.shape[:2]

            # Scale so longest side fits in s
            r = min(s / h, s / w)
            nw, nh = int(w * r), int(h * r)
            img_r = np.array(PImage.fromarray(img).resize((nw, nh), PImage.BILINEAR))

            # Quadrant boundaries on canvas and corresponding source slice
            if i == 0:      # top-left: image anchored at bottom-right corner (xc, yc)
                x1c, y1c = max(xc - nw, 0), max(yc - nh, 0)
                x2c, y2c = xc, yc
                x1i, y1i = nw - (x2c - x1c), nh - (y2c - y1c)
            elif i == 1:    # top-right
                x1c, y1c = xc, max(yc - nh, 0)
                x2c, y2c = min(xc + nw, 2 * s), yc
                x1i, y1i = 0, nh - (y2c - y1c)
            elif i == 2:    # bottom-left
                x1c, y1c = max(xc - nw, 0), yc
                x2c, y2c = xc, min(yc + nh, 2 * s)
                x1i, y1i = nw - (x2c - x1c), 0
            else:           # bottom-right
                x1c, y1c = xc, yc
                x2c, y2c = min(xc + nw, 2 * s), min(yc + nh, 2 * s)
                x1i, y1i = 0, 0

            x2i = x1i + (x2c - x1c)
            y2i = y1i + (y2c - y1c)
            canvas[y1c:y2c, x1c:x2c] = img_r[y1i:y2i, x1i:x2i]

            if labels.size:
                cx_a = labels[:, 1] * nw
                cy_a = labels[:, 2] * nh
                bw_a = labels[:, 3] * nw
                bh_a = labels[:, 4] * nh
                bx1 = (cx_a - bw_a / 2).clip(x1i, x2i)
                by1 = (cy_a - bh_a / 2).clip(y1i, y2i)
                bx2 = (cx_a + bw_a / 2).clip(x1i, x2i)
                by2 = (cy_a + bh_a / 2).clip(y1i, y2i)
                valid = (bx2 > bx1) & (by2 > by1)
                if valid.any():
                    all_labels.append(np.stack([
                        labels[valid, 0],
                        bx1[valid] - x1i + x1c,
                        by1[valid] - y1i + y1c,
                        bx2[valid] - x1i + x1c,
                        by2[valid] - y1i + y1c,
                    ], axis=1))

        # Crop canvas to (s, s) centred on (xc, yc)
        x_off = max(0, min(xc - s // 2, s))
        y_off = max(0, min(yc - s // 2, s))
        img_out = canvas[y_off:y_off + s, x_off:x_off + s]
        if img_out.shape[:2] != (s, s):
            ph, pw = s - img_out.shape[0], s - img_out.shape[1]
            img_out = np.pad(img_out, ((0, ph), (0, pw), (0, 0)), constant_values=114)

        if not all_labels:
            return img_out, np.zeros((0, 5), dtype=np.float32)

        labs = np.concatenate(all_labels, 0)
        labs[:, 1] -= x_off;  labs[:, 3] -= x_off
        labs[:, 2] -= y_off;  labs[:, 4] -= y_off
        labs[:, [1, 3]] = labs[:, [1, 3]].clip(0, s)
        labs[:, [2, 4]] = labs[:, [2, 4]].clip(0, s)
        valid = ((labs[:, 3] - labs[:, 1]) > 2) & ((labs[:, 4] - labs[:, 2]) > 2)
        labs = labs[valid]
        if not len(labs):
            return img_out, np.zeros((0, 5), dtype=np.float32)

        out = np.zeros((len(labs), 5), dtype=np.float32)
        out[:, 0] = labs[:, 0]
        out[:, 1] = (labs[:, 1] + labs[:, 3]) / 2 / s
        out[:, 2] = (labs[:, 2] + labs[:, 4]) / 2 / s
        out[:, 3] = (labs[:, 3] - labs[:, 1]) / s
        out[:, 4] = (labs[:, 4] - labs[:, 2]) / s
        return img_out, out

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]

        if self.augment and random.random() < self.mosaic_prob:
            img, labels = self._load_mosaic(idx)
            img, labels = random_perspective(img, labels,
                                             degrees=0.0, translate=0.1,
                                             scale=0.5, shear=0.0)
            img, labels = random_hflip(img, labels, p=0.5)
            H0, W0 = self.img_size, self.img_size
            ratio, pad = 1.0, (0, 0)
        else:
            img, labels = self._load_raw(idx)
            H0, W0 = img.shape[:2]
            img, labels, ratio, pad = letterbox(img, labels, new_shape=self.img_size)
            if self.augment:
                img, labels = random_hflip(img, labels, p=0.5)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        labels_t = torch.from_numpy(labels)
        return img_t, labels_t, img_id, (H0, W0), ratio, pad


def yolo_collate(batch):
    """Batch into:
        imgs:   (B, 3, H, W)
        labels: (M, 6) where col 0 is batch index, cols 1..5 are (cls, cx, cy, w, h)
        meta:   list of per-image (img_id, (H0, W0), ratio, pad) for eval
    """
    imgs, labels, ids, sizes, ratios, pads = zip(*batch)
    imgs = torch.stack(imgs, 0)
    out_labels = []
    for i, lb in enumerate(labels):
        if lb.numel() == 0:
            continue
        bi = torch.full((lb.size(0), 1), float(i))
        out_labels.append(torch.cat([bi, lb], dim=1))
    out_labels = torch.cat(out_labels, 0) if out_labels else torch.zeros((0, 6))
    meta = list(zip(ids, sizes, ratios, pads))
    return imgs, out_labels, meta
