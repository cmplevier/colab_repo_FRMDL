"""COCO 2017 dataset for YOLO-style training.

Returns: image tensor (3, H, W), labels tensor (N, 5) in YOLO format
[class_id, cx, cy, w, h] normalised to [0, 1], and a list of original image
sizes / ids needed by the COCO evaluator.

This is deliberately minimal — it does NOT include Mosaic augmentation. A hook
is provided in `transforms.py` so you can plug ultralytics' Mosaic in without
touching the training loop.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from transforms import letterbox, random_hflip


# COCO category id is not contiguous (1..90 with gaps). We map to 0..79.
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

        ann_file = self.root / "annotations" / f"instances_{split}.json"
        with open(ann_file) as f:
            data = json.load(f)

        self.imgs = {im["id"]: im for im in data["images"]}
        self.cat_ids = [c["id"] for c in data["categories"]]
        self.id_to_idx, self.idx_to_id = _build_id_map(self.cat_ids)

        # index annotations by image id
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

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        meta = self.imgs[img_id]
        img_path = self.root / "images" / self.split / meta["file_name"]

        # Use PIL to keep deps light; could swap to cv2.
        from PIL import Image
        img = np.array(Image.open(img_path).convert("RGB"))  # (H, W, 3) uint8
        H0, W0 = img.shape[:2]

        # Build YOLO labels from COCO anns
        labels = []
        for ann in self.anns_by_img[img_id]:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / W0
            cy = (y + h / 2) / H0
            ww = w / W0
            hh = h / H0
            cls = self.id_to_idx[ann["category_id"]]
            labels.append([cls, cx, cy, ww, hh])
        labels = np.array(labels, dtype=np.float32).reshape(-1, 5)

        # Letterbox to square img_size
        img, labels, ratio, pad = letterbox(img, labels, new_shape=self.img_size)

        if self.augment:
            img, labels = random_hflip(img, labels, p=0.5)

        # to tensor, normalize to [0,1]
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