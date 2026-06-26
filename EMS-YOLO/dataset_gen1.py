"""Gen1 dataset loader for EMS-YOLO.

Reads pre-processed .npy + .txt pairs produced by EventCamera_Gen1/preprocess_gen1.py.

Each .npy  : (T, H, W, 3) uint8  — T event frames per sample
Each .txt  : YOLO-format lines   — class cx cy w h, normalised to [0,1] vs original sensor size

Returns frames as (T, 3, H, W) float32 in [0, 1].
Evaluation uses a COCO-format GT dict built in-memory from the .txt files, with all
coordinates expressed at img_size x img_size so no rescaling is needed at eval time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Gen1 class mapping: stored index -> COCO category id (1-indexed)
GEN1_IDX_TO_ID = {0: 1, 1: 2}
GEN1_CATEGORIES = [{"id": 1, "name": "car"}, {"id": 2, "name": "pedestrian"}]


def _hflip_temporal(frames: np.ndarray, labels: np.ndarray) -> tuple:
    """Flip all T frames and their labels horizontally."""
    frames = frames[:, :, ::-1, :].copy()          # (T, H, W, 3)
    if labels.size:
        labels = labels.copy()
        labels[:, 1] = 1.0 - labels[:, 1]          # cx -> 1 - cx
    return frames, labels


def _load_labels(txt_path: Path) -> np.ndarray:
    if not txt_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    raw = txt_path.read_text().strip()
    if not raw:
        return np.zeros((0, 5), dtype=np.float32)
    rows = [list(map(float, line.split())) for line in raw.splitlines()]
    return np.array(rows, dtype=np.float32)


class Gen1Dataset(Dataset):
    """
    Args:
        root     : directory containing paired 0000001.npy / 0000001.txt files
        T        : number of temporal bins (must match the value used during preprocessing)
        img_size : spatial resolution of the stored frames (also used for GT JSON)
        augment  : whether to apply random horizontal flip
    """

    def __init__(self, root: str, T: int = 5, img_size: int = 320, augment: bool = True):
        self.root = Path(root)
        self.T = T
        self.img_size = img_size
        self.augment = augment

        self.samples = sorted(self.root.glob("*.npy"))
        if not self.samples:
            raise FileNotFoundError(f"No .npy files found in {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        npy_path = self.samples[idx]
        labels = _load_labels(npy_path.with_suffix(".txt"))

        frames = np.load(str(npy_path))             # (T, H, W, 3) uint8

        if self.augment and np.random.rand() < 0.5:
            frames, labels = _hflip_temporal(frames, labels)

        # (T, H, W, 3) -> (T, 3, H, W) float32 in [0, 1]
        frames_t = torch.from_numpy(
            frames.transpose(0, 3, 1, 2).copy()
        ).float() / 255.0

        labels_t = torch.from_numpy(labels)         # (N, 5)

        # meta is consumed by predictions_to_coco_json in the eval loop.
        # Predictions are decoded at img_size x img_size and GT JSON is also
        # built at img_size x img_size, so ratio=1 and no padding correction needed.
        sample_id = int(npy_path.stem)
        meta = (sample_id, (self.img_size, self.img_size), 1.0, (0, 0))

        return frames_t, labels_t, meta

    # ------------------------------------------------------------------
    # COCO GT helpers
    # ------------------------------------------------------------------

    def build_coco_gt(self) -> dict:
        """Return a COCO-format ground-truth dict for pycocotools evaluation.

        Coordinates are expressed in img_size x img_size pixels so they match
        the model output without any rescaling step.
        """
        images = []
        annotations = []
        ann_id = 1

        for npy_path in self.samples:
            sample_id = int(npy_path.stem)
            images.append({
                "id": sample_id,
                "width": self.img_size,
                "height": self.img_size,
                "file_name": npy_path.name,
            })

            for cls_f, cx, cy, w, h in _load_labels(npy_path.with_suffix(".txt")):
                x1 = (cx - w / 2) * self.img_size
                y1 = (cy - h / 2) * self.img_size
                bw = w * self.img_size
                bh = h * self.img_size
                annotations.append({
                    "id": ann_id,
                    "image_id": sample_id,
                    "category_id": GEN1_IDX_TO_ID[int(cls_f)],
                    "bbox": [float(x1), float(y1), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                })
                ann_id += 1

        return {"images": images, "annotations": annotations, "categories": GEN1_CATEGORIES}

    @property
    def idx_to_id(self) -> dict:
        return dict(GEN1_IDX_TO_ID)


def gen1_collate(batch):
    """Collate Gen1 samples into a batch.

    Returns:
        frames : (B, T, 3, H, W) float32
        labels : (M, 6) float32  [batch_idx, cls, cx, cy, w, h]
        meta   : list of (sample_id, (H, W), ratio, (pad_x, pad_y))
    """
    frames_list, labels_list, meta_list = zip(*batch)

    frames = torch.stack(frames_list, dim=0)        # (B, T, 3, H, W)

    out_labels = []
    for i, lb in enumerate(labels_list):
        if lb.numel() == 0:
            continue
        bi = torch.full((lb.size(0), 1), float(i))
        out_labels.append(torch.cat([bi, lb], dim=1))
    labels = torch.cat(out_labels, 0) if out_labels else torch.zeros((0, 6))

    return frames, labels, list(meta_list)
