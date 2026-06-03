"""pycocotools wrapper for COCO mAP."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import torch


def predictions_to_coco_json(results, meta, idx_to_id, img_size: int):
    """results: list of (N, 6) tensors per image — boxes in xyxy on the
    letterboxed canvas, scores, class indices (0..nc-1).
    meta:    list of (img_id, (H0, W0), ratio, (pad_x, pad_y)) per image.
    Returns: list of COCO-format dicts.
    """
    out = []
    for res, m in zip(results, meta):
        img_id, (H0, W0), ratio, (pad_x, pad_y) = m
        if res.numel() == 0:
            continue
        boxes = res[:, :4].cpu().numpy()
        scores = res[:, 4].cpu().numpy()
        clss = res[:, 5].cpu().numpy().astype(int)

        # undo letterbox
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= ratio
        # clamp to image
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, W0)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, H0)
        # xywh
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        for i in range(len(boxes)):
            out.append({
                "image_id": int(img_id),
                "category_id": int(idx_to_id[int(clss[i])]),
                "bbox": [float(boxes[i, 0]), float(boxes[i, 1]), float(w[i]), float(h[i])],
                "score": float(scores[i]),
            })
    return out


def evaluate_coco(coco_gt_path: str, predictions: list[dict]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(coco_gt_path)
    if not predictions:
        return {"mAP@0.5:0.95": 0.0, "mAP@0.5": 0.0}
    coco_dt = coco_gt.loadRes(predictions)
    e = COCOeval(coco_gt, coco_dt, "bbox")
    with redirect_stdout(io.StringIO()):
        e.evaluate()
        e.accumulate()
        e.summarize()
    return {
        "mAP@0.5:0.95": float(e.stats[0]),
        "mAP@0.5": float(e.stats[1]),
    }