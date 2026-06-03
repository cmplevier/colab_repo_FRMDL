from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

import wandb

from dataset import CocoYOLODataset, yolo_collate
from models import EMSYOLO
from utils.loss import ComputeLoss
from utils.nms import decode_predictions
from utils.coco_eval import evaluate_coco, predictions_to_coco_json
from SNN_framework.firing_rate import FiringRateTracker


# ---------------------------------------------------------------------
# Hardcoded config: edit this directly instead of using a .yaml file
# ---------------------------------------------------------------------
CFG = {
    "model": {
        "name": "ems_yolo",
        "backbone": "ems_resnet34",
        "T": 4,
        "decay": 0.25,
        "num_classes": 80,
        "num_anchors": 3,

        # Use whatever your build_model(cfg) expects.
        # Keep/remove fields based on your models.py.
        "in_channels": [256, 512],
        "anchors": [
            [[10, 13], [16, 30], [33, 23]],      # P4 / higher-res scale
            [[30, 61], [62, 45], [59, 119]],     # P5 / lower-res scale
        ],
        "strides": [16, 32],
    },

    "data": {
        "img_size": 640,
        "num_workers": 4,
    },

    "train": {
        "epochs": 100,
        "batch_size": 32,
        "lr0": 1e-2,
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "warmup_epochs": 3,
        "accum_steps": 1,
        "amp": True,
        "eval_every": 5,
        "save_every": 5,
    },
}


def load_pretrained(model: nn.Module, ckpt_path: str) -> None:
    """Load weights from a checkpoint with possibly different T."""
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd["model"] if "model" in sd else sd

    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[pretrained] missing={len(missing)} unexpected={len(unexpected)}")

    if missing:
        print("  missing:", missing[:5], "..." if len(missing) > 5 else "")

    if unexpected:
        print("  unexpected:", unexpected[:5], "..." if len(unexpected) > 5 else "")


def cosine_lr(epoch: int, total_epochs: int, lr0: float, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return lr0 * (epoch + 1) / max(1, warmup_epochs)

    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)

    return lr0 * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, val_loader, cfg, output_dir, coco_gt_path, idx_to_id, device):
    model.eval()

    all_preds = []

    for imgs, _labels, meta in val_loader:
        imgs = imgs.to(device, non_blocking=True)

        preds = model(imgs)

        results = decode_predictions(
            preds,
            model.strides,
            model.nc,
            model.na,
            conf_thresh=0.001,
        )

        all_preds.extend(
            predictions_to_coco_json(
                results,
                meta,
                idx_to_id,
                cfg["data"]["img_size"],
            )
        )

    if not all_preds:
        return {"mAP@0.5:0.95": 0.0, "mAP@0.5": 0.0}

    pred_path = Path(output_dir) / "preds.json"

    with open(pred_path, "w") as f:
        json.dump(all_preds, f)

    return evaluate_coco(coco_gt_path, all_preds)


def _deep_update(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-root",
        required=True,
        help="COCO root with annotations/ and images/",
    )

    ap.add_argument(
        "--output",
        required=True,
    )

    ap.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config file (overrides hardcoded CFG defaults)",
    )

    ap.add_argument(
        "--pretrained",
        default=None,
        help="Optional checkpoint to init from",
    )

    ap.add_argument(
        "--resume",
        default=None,
        help="Resume training from checkpoint",
    )

    args = ap.parse_args()

    import copy
    cfg = copy.deepcopy(CFG)
    if args.config:
        with open(args.config) as f:
            _deep_update(cfg, yaml.safe_load(f))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Save the hardcoded config as JSON for reproducibility.
    with open(out / "config_hardcoded.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # -----------------------------------------------------------------
    # Wandb setup
    # -----------------------------------------------------------------
    wandb.login(key="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")

    wandb.init(        
        entity="adela-greganova-tu-delft",
        project="COCO-runs",
        name="run_full_data_1",
        config=cfg,
        dir=str(out))




    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------
    train_ds = CocoYOLODataset(
        args.data_root,
        "train2017",
        img_size=cfg["data"]["img_size"],
        augment=True,
    )

    val_ds = CocoYOLODataset(
        args.data_root,
        "val2017",
        img_size=cfg["data"]["img_size"],
        augment=False,
    )

    pin_memory = cfg["data"].get("pin_memory", False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=yolo_collate,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        collate_fn=yolo_collate,
        pin_memory=pin_memory,
    )

    coco_gt_path = str(
        Path(args.data_root) / "annotations" / "instances_val2017.json"
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    model = EMSYOLO(
        T=cfg["model"]["T"],
        decay=cfg["model"]["decay"],
        num_classes=cfg["model"]["num_classes"],
        num_anchors=cfg["model"]["num_anchors"],
    ).to(device)

    if args.pretrained:
        load_pretrained(model, args.pretrained)

    # Make sure T matches the hardcoded config.
    model.set_T(cfg["model"]["T"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[model] EMS-YOLO T={model.T} params={n_params / 1e6:.2f}M")

    wandb.log(
        {
            "model/params_millions": n_params / 1e6,
            "model/T": model.T,
        }
    )

    # -----------------------------------------------------------------
    # Loss + optimizer
    # -----------------------------------------------------------------
    loss_fn = ComputeLoss(model)

    optim = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr0"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
        nesterov=True,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"])

    start_epoch = 0
    best_map = 0.0

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")

        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])

        start_epoch = ck["epoch"] + 1
        best_map = ck.get("best_map", 0.0)

        print(f"[resume] from epoch {start_epoch}, best mAP {best_map:.3f}")

    epochs = cfg["train"]["epochs"]
    accum = cfg["train"]["accum_steps"]

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.train()

        lr = cosine_lr(
            epoch,
            epochs,
            cfg["train"]["lr0"],
            cfg["train"]["warmup_epochs"],
        )

        for pg in optim.param_groups:
            pg["lr"] = lr

        t0 = time.time()

        running = {
            "box": 0.0,
            "obj": 0.0,
            "cls": 0.0,
            "total": 0.0,
        }

        optim.zero_grad(set_to_none=True)

        max_steps = cfg["train"].get("max_steps", None)
        for step, (imgs, labels, _meta) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=cfg["train"]["amp"]):
                preds = model(imgs)
                loss, parts = loss_fn(preds, labels)
                loss = loss / accum

            scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            for k, v in parts.items():
                running[k] += float(v)

            if max_steps is not None and (step + 1) >= max_steps:
                break

            if (step + 1) % 100 == 0:
                msg = " ".join(
                    f"{k}={v / (step + 1):.4f}"
                    for k, v in running.items()
                )

                print(
                    f"[epoch {epoch} step {step + 1}/{len(train_loader)} "
                    f"lr={lr:.4g}] {msg}"
                )

        dt = time.time() - t0

        print(f"[epoch {epoch}] done in {dt / 60:.1f} min")

        wandb.log(
            {
                "epoch": epoch,
                "lr": lr,
                "time/epoch_min": dt / 60,
                "train/box": running["box"] / (step + 1),
                "train/obj": running["obj"] / (step + 1),
                "train/cls": running["cls"] / (step + 1),
                "train/total": running["total"] / (step + 1),
            }
        )

        # -------------------------------------------------------------
        # Evaluation
        # -------------------------------------------------------------
        if (epoch + 1) % cfg["train"]["eval_every"] == 0 or epoch == epochs - 1:
            with FiringRateTracker(model) as tracker:
                metrics = evaluate(
                    model,
                    val_loader,
                    cfg,
                    out,
                    coco_gt_path,
                    val_ds.idx_to_id,
                    device,
                )

                fr = tracker.firing_rates()
                fr_summary = tracker.summary()
                tracked_layers = tracker.num_tracked_layers()

            print(
                f"[epoch {epoch}] "
                f"mAP@0.5={metrics['mAP@0.5']:.4f} "
                f"mAP@0.5:0.95={metrics['mAP@0.5:0.95']:.4f} "
                f"firing_rate={fr['overall']:.6f} "
                f"tracked_lif_layers={tracked_layers}"
            )

            if tracked_layers == 0:
                print(
                    "[warning] FiringRateTracker found 0 LIFNeuron layers. "
                    "Check that the model uses the same LIFNeuron class imported by firing_rate.py."
                )

            wandb.log(
                {
                    "epoch": epoch,
                    "val/mAP@0.5": metrics["mAP@0.5"],
                    "val/mAP@0.5:0.95": metrics["mAP@0.5:0.95"],
                    "snn/firing_rate": fr["overall"],
                    "snn/tracked_lif_layers": tracked_layers,
                }
            )

            for layer_name, layer_rate in fr.items():
                if layer_name != "overall":
                    wandb.log(
                        {
                            "epoch": epoch,
                            f"snn/layer_firing_rate/{layer_name}": layer_rate,
                        }
                    )

            with open(out / f"firing_rates_epoch{epoch}.json", "w") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "overall": fr["overall"],
                        "tracked_lif_layers": tracked_layers,
                        "per_layer": fr_summary,
                    },
                    f,
                    indent=2,
                )

            if metrics["mAP@0.5"] > best_map:
                best_map = metrics["mAP@0.5"]

                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": optim.state_dict(),
                        "epoch": epoch,
                        "best_map": best_map,
                        "cfg": cfg,
                    },
                    out / "best.pt",
                )

                with open(out / "best_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

        # -------------------------------------------------------------
        # Periodic save
        # -------------------------------------------------------------
        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "epoch": epoch,
                    "best_map": best_map,
                    "cfg": cfg,
                },
                out / "last.pt",
            )

    wandb.finish()


if __name__ == "__main__":
    main()
    os._exit(0)
