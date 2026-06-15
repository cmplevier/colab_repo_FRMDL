"""Integration tests using real COCO val2017 data at C:\\coco.

cv2 is not available locally so random_perspective is patched to a no-op.
Everything else (JSON parsing, image reads, letterbox, mosaic assembly,
label maths, collate) runs end-to-end against real data.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

COCO_ROOT = "C:/coco"
SPLIT = "val2017"

def _noop_rp(img, labels, **kwargs):
    return img, labels


@patch("dataset.random_perspective", _noop_rp)
def _make_ds(augment=True, img_size=416):
    from dataset import CocoYOLODataset
    return CocoYOLODataset(COCO_ROOT, split=SPLIT, img_size=img_size, augment=augment)


class TestCocoInit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with patch("dataset.random_perspective", _noop_rp):
            from dataset import CocoYOLODataset
            cls.ds = CocoYOLODataset(COCO_ROOT, split=SPLIT, img_size=416, augment=False)

    def test_length(self):
        self.assertEqual(len(self.ds), 5000)

    def test_img_ids_populated(self):
        self.assertEqual(len(self.ds.img_ids), 5000)

    def test_anns_non_empty(self):
        # at least some images have annotations
        has_ann = sum(1 for v in self.ds.anns_by_img.values() if v)
        self.assertGreater(has_ann, 4000)

    def test_id_to_idx_range(self):
        self.assertEqual(min(self.ds.id_to_idx.values()), 0)
        self.assertEqual(max(self.ds.id_to_idx.values()), len(self.ds.id_to_idx) - 1)

    def test_80_classes(self):
        self.assertEqual(len(self.ds.id_to_idx), 80)


class TestLoadRaw(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with patch("dataset.random_perspective", _noop_rp):
            from dataset import CocoYOLODataset
            cls.ds = CocoYOLODataset(COCO_ROOT, split=SPLIT, img_size=416, augment=False)

    def _check_sample(self, idx):
        img, labels = self.ds._load_raw(idx)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(labels.ndim, 2)
        self.assertEqual(labels.shape[1], 5)
        if labels.size:
            self.assertTrue((labels[:, 1:] > 0).all(), "coords should be positive")
            # COCO raw annotations occasionally have boxes that marginally exceed
            # image bounds due to float rounding — allow 1% slack here.
            self.assertTrue((labels[:, 1:] <= 1.01).all(), "coords should be <= 1.01")
            self.assertTrue((labels[:, 3] > 0).all())
            self.assertTrue((labels[:, 4] > 0).all())
            cls_ids = labels[:, 0].astype(int)
            self.assertTrue((cls_ids >= 0).all())
            self.assertTrue((cls_ids < 80).all())

    def test_first_image(self):
        self._check_sample(0)

    def test_tenth_image(self):
        self._check_sample(10)

    def test_last_image(self):
        self._check_sample(len(self.ds) - 1)

    def test_several_random(self):
        rng = np.random.default_rng(0)
        for idx in rng.integers(0, len(self.ds), size=20):
            self._check_sample(int(idx))


class TestLetterboxPath(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with patch("dataset.random_perspective", _noop_rp):
            from dataset import CocoYOLODataset
            cls.ds = CocoYOLODataset(COCO_ROOT, split=SPLIT, img_size=416, augment=False)

    def _check_item(self, idx):
        with patch("dataset.random_perspective", _noop_rp):
            img_t, labels_t, img_id, hw, ratio, pad = self.ds[idx]

        self.assertEqual(img_t.shape, (3, 416, 416))
        self.assertEqual(img_t.dtype, torch.float32)
        self.assertGreaterEqual(img_t.min().item(), 0.0)
        self.assertLessEqual(img_t.max().item(), 1.0)

        h0, w0 = hw
        self.assertGreater(h0, 0)
        self.assertGreater(w0, 0)
        self.assertGreater(ratio, 0.0)

        if labels_t.numel():
            self.assertEqual(labels_t.shape[1], 5)
            self.assertTrue((labels_t[:, 1:] >= 0).all())
            self.assertTrue((labels_t[:, 1:] <= 1).all())

    def test_first(self):
        self._check_item(0)

    def test_several(self):
        for i in [0, 1, 5, 42, 99, 499, 999, 4999]:
            self._check_item(i)


class TestMosaicPath(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with patch("dataset.random_perspective", _noop_rp):
            from dataset import CocoYOLODataset
            cls.ds = CocoYOLODataset(COCO_ROOT, split=SPLIT, img_size=416, augment=True)
            cls.ds.mosaic_prob = 1.0

    def _check_mosaic_item(self, idx):
        with patch("dataset.random_perspective", _noop_rp):
            img_t, labels_t, img_id, hw, ratio, pad = self.ds[idx]

        self.assertEqual(img_t.shape, (3, 416, 416))
        self.assertEqual(img_t.dtype, torch.float32)
        self.assertGreaterEqual(img_t.min().item(), 0.0)
        self.assertLessEqual(img_t.max().item(), 1.0)

        self.assertEqual(hw, (416, 416))
        self.assertEqual(ratio, 1.0)
        self.assertEqual(pad, (0, 0))

        if labels_t.numel():
            self.assertEqual(labels_t.shape[1], 5)
            coords = labels_t[:, 1:].numpy()
            self.assertTrue((coords >= 0).all(), f"negative coord in sample {idx}")
            self.assertTrue((coords <= 1).all(), f"coord > 1 in sample {idx}")
            self.assertTrue((labels_t[:, 3] > 0).all(), "zero width box")
            self.assertTrue((labels_t[:, 4] > 0).all(), "zero height box")
            cls_ids = labels_t[:, 0].long()
            self.assertTrue((cls_ids >= 0).all())
            self.assertTrue((cls_ids < 80).all())

    def test_first_ten(self):
        for i in range(10):
            self._check_mosaic_item(i)

    def test_random_50(self):
        rng = np.random.default_rng(7)
        for idx in rng.integers(0, len(self.ds), size=50):
            self._check_mosaic_item(int(idx))

    def test_label_count_reasonable(self):
        """Mosaic of 4 images should usually produce more labels than one image alone."""
        total_mosaic, total_single = 0, 0
        with patch("dataset.random_perspective", _noop_rp):
            self.ds.mosaic_prob = 1.0
            for i in range(20):
                _, lb, *_ = self.ds[i]
                total_mosaic += lb.shape[0] if lb.numel() else 0
            self.ds.mosaic_prob = 0.0
            for i in range(20):
                _, lb, *_ = self.ds[i]
                total_single += lb.shape[0] if lb.numel() else 0
            self.ds.mosaic_prob = 1.0  # restore
        self.assertGreater(total_mosaic, total_single,
                           "mosaic should aggregate more labels than single-image")

    def test_image_not_all_gray(self):
        """Canvas fill (114) should be mostly covered by real image content."""
        with patch("dataset.random_perspective", _noop_rp):
            img_t, *_ = self.ds[0]
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gray_frac = (img_np == 114).all(axis=2).mean()
        self.assertLess(gray_frac, 0.30, "too much uncovered canvas (>30%)")


class TestCollateReal(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with patch("dataset.random_perspective", _noop_rp):
            from dataset import CocoYOLODataset
            cls.ds = CocoYOLODataset(COCO_ROOT, split=SPLIT, img_size=416, augment=True)
            cls.ds.mosaic_prob = 1.0

    def test_collate_4(self):
        from dataset import yolo_collate
        with patch("dataset.random_perspective", _noop_rp):
            batch = [self.ds[i] for i in range(4)]
        imgs, labels, meta = yolo_collate(batch)
        self.assertEqual(imgs.shape, (4, 3, 416, 416))
        self.assertEqual(labels.shape[1], 6)
        bi = labels[:, 0].long()
        self.assertTrue((bi >= 0).all())
        self.assertTrue((bi < 4).all())
        # each meta entry: (img_id, hw, ratio, pad)
        self.assertEqual(len(meta), 4)

    def test_collate_all_labels_normalised(self):
        from dataset import yolo_collate
        with patch("dataset.random_perspective", _noop_rp):
            batch = [self.ds[i] for i in range(8)]
        _, labels, _ = yolo_collate(batch)
        if labels.numel():
            coords = labels[:, 2:].numpy()
            self.assertTrue((coords >= 0).all())
            self.assertTrue((coords <= 1).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
