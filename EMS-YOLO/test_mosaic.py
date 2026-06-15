"""Unit tests for mosaic augmentation — no COCO data or cv2 required."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Minimal stub so dataset.py can import without a real COCO directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

# Stub out random_perspective so cv2 import doesn't fail
import transforms as _transforms
_ORIG_RP = getattr(_transforms, "random_perspective", None)

def _noop_rp(img, labels, **kwargs):
    return img, labels

_transforms.random_perspective = _noop_rp

# Now we can import dataset safely
from dataset import CocoYOLODataset, yolo_collate  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: build a minimal CocoYOLODataset without touching disk
# ---------------------------------------------------------------------------
def _make_dataset(img_size=416):
    """Construct dataset object with faked annotation data."""
    ds = object.__new__(CocoYOLODataset)
    ds.root = Path("/fake")
    ds.split = "train2017"
    ds.img_size = img_size
    ds.augment = True
    ds.mosaic_prob = 1.0
    # 10 fake images
    n = 10
    ds.img_ids = list(range(n))
    ds.imgs = {i: {"id": i, "file_name": f"{i:012d}.jpg"} for i in range(n)}
    cat_ids = [1, 2, 3]
    ds.id_to_idx = {c: i for i, c in enumerate(cat_ids)}
    ds.idx_to_id = {i: c for c, i in ds.id_to_idx.items()}
    ds.anns_by_img = {i: [] for i in range(n)}
    return ds


def _make_raw(h=480, w=640, n_boxes=3):
    """Return a fake (img, labels) pair."""
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    if n_boxes == 0:
        return img, np.zeros((0, 5), dtype=np.float32)
    labels = np.zeros((n_boxes, 5), dtype=np.float32)
    labels[:, 0] = np.arange(n_boxes) % 3
    labels[:, 1] = np.linspace(0.25, 0.75, n_boxes)
    labels[:, 2] = np.linspace(0.25, 0.75, n_boxes)
    labels[:, 3] = 0.1
    labels[:, 4] = 0.1
    return img, labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMosaicAssembly(unittest.TestCase):

    def setUp(self):
        self.ds = _make_dataset(img_size=416)
        # Patch _load_raw to return synthetic data
        self.ds._load_raw = lambda idx: _make_raw()

    def test_output_shape(self):
        img, labels = self.ds._load_mosaic(0)
        self.assertEqual(img.shape, (416, 416, 3), "mosaic output must be (s, s, 3)")

    def test_output_dtype(self):
        img, labels = self.ds._load_mosaic(0)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(labels.dtype, np.float32)

    def test_labels_shape(self):
        img, labels = self.ds._load_mosaic(0)
        self.assertEqual(labels.ndim, 2)
        self.assertEqual(labels.shape[1], 5)

    def test_labels_normalised(self):
        img, labels = self.ds._load_mosaic(0)
        if labels.size:
            self.assertTrue((labels[:, 1:] >= 0).all(), "coords must be >= 0")
            self.assertTrue((labels[:, 1:] <= 1).all(), "coords must be <= 1")

    def test_wh_positive(self):
        img, labels = self.ds._load_mosaic(0)
        if labels.size:
            self.assertTrue((labels[:, 3] > 0).all(), "box widths must be > 0")
            self.assertTrue((labels[:, 4] > 0).all(), "box heights must be > 0")

    def test_empty_labels(self):
        """Mosaic should work when source images have no annotations."""
        self.ds._load_raw = lambda idx: (_make_raw()[0], np.zeros((0, 5), dtype=np.float32))
        img, labels = self.ds._load_mosaic(0)
        self.assertEqual(img.shape, (416, 416, 3))
        self.assertEqual(labels.shape, (0, 5))

    def test_determinism_off(self):
        """Two mosaic calls on the same index should differ (random)."""
        np.random.seed(0)
        img1, _ = self.ds._load_mosaic(0)
        np.random.seed(42)
        img2, _ = self.ds._load_mosaic(0)
        self.assertFalse(np.array_equal(img1, img2), "mosaic should be random")

    def test_canvas_not_all_gray(self):
        img, _ = self.ds._load_mosaic(0)
        self.assertFalse((img == 114).all(), "canvas should be at least partially filled")

    def test_many_calls_stable(self):
        """No exceptions across many random draws."""
        for i in range(50):
            img, labels = self.ds._load_mosaic(i % len(self.ds))
            self.assertEqual(img.shape, (416, 416, 3))


class TestMosaicLabelMath(unittest.TestCase):
    """Verify the box coordinate transform through mosaic assembly."""

    def test_box_stays_in_image(self):
        ds = _make_dataset(416)
        # single box at exact centre of a square image → should survive mosaic
        def _centred(idx):
            img = np.zeros((416, 416, 3), dtype=np.uint8)
            labels = np.array([[0, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
            return img, labels

        ds._load_raw = _centred
        for _ in range(20):
            img, labels = ds._load_mosaic(0)
            if labels.size:
                cx = labels[:, 1]
                cy = labels[:, 2]
                w2 = labels[:, 3] / 2
                h2 = labels[:, 4] / 2
                self.assertTrue((cx - w2 >= 0).all())
                self.assertTrue((cy - h2 >= 0).all())
                self.assertTrue((cx + w2 <= 1).all())
                self.assertTrue((cy + h2 <= 1).all())


class TestGetItemRouting(unittest.TestCase):

    def setUp(self):
        self.ds = _make_dataset(416)
        self.ds._load_raw = lambda idx: _make_raw()
        self.ds._load_mosaic = lambda idx: _make_raw(416, 416, 2)  # already s×s

    def test_mosaic_path_tensor_shape(self):
        import torch
        self.ds.mosaic_prob = 1.0
        img_t, labels_t, img_id, hw, ratio, pad = self.ds[0]
        self.assertEqual(img_t.shape, (3, 416, 416))
        self.assertEqual(img_t.dtype, torch.float32)
        self.assertTrue((img_t >= 0).all() and (img_t <= 1).all())

    def test_letterbox_path_tensor_shape(self):
        import torch
        self.ds.mosaic_prob = 0.0
        img_t, labels_t, img_id, hw, ratio, pad = self.ds[0]
        self.assertEqual(img_t.shape, (3, 416, 416))

    def test_mosaic_meta_placeholder(self):
        self.ds.mosaic_prob = 1.0
        _, _, _, hw, ratio, pad = self.ds[0]
        self.assertEqual(hw, (416, 416))
        self.assertEqual(ratio, 1.0)
        self.assertEqual(pad, (0, 0))

    def test_letterbox_meta_valid(self):
        self.ds.mosaic_prob = 0.0
        _, _, _, hw, ratio, pad = self.ds[0]
        h0, w0 = hw
        self.assertGreater(h0, 0)
        self.assertGreater(w0, 0)
        self.assertGreater(ratio, 0)


class TestYoloCollate(unittest.TestCase):

    def test_collate_shape(self):
        import torch
        ds = _make_dataset(416)
        ds._load_raw = lambda idx: _make_raw()
        ds._load_mosaic = lambda idx: _make_raw(416, 416, 3)
        ds.mosaic_prob = 1.0
        batch = [ds[i] for i in range(4)]
        imgs, labels, meta = yolo_collate(batch)
        self.assertEqual(imgs.shape, (4, 3, 416, 416))
        self.assertEqual(labels.shape[1], 6)     # (batch_idx, cls, cx, cy, w, h)
        self.assertEqual(len(meta), 4)

    def test_collate_batch_index(self):
        import torch
        ds = _make_dataset(416)
        ds._load_raw = lambda idx: _make_raw()
        ds._load_mosaic = lambda idx: _make_raw(416, 416, 2)
        ds.mosaic_prob = 1.0
        batch = [ds[i] for i in range(3)]
        _, labels, _ = yolo_collate(batch)
        batch_indices = labels[:, 0].long()
        self.assertTrue((batch_indices >= 0).all())
        self.assertTrue((batch_indices < 3).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
