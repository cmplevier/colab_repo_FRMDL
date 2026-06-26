"""Training script for EMS-YOLO on the Gen1 event-camera dataset.

Usage:
    python EMS-YOLO/train_gen1.py \\
        --train-dir /path/to/gen1_processed/train \\
        --val-dir   /path/to/gen1_processed/val \\
        --output    /path/to/runs/gen1-resnet10 \\
        --config    EMS-YOLO/configs/gen1_local.yaml

W&B key is read from the WANDB_API_KEY environment variable.
Set WANDB_MODE=disabled to run without W&B.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_gen1 import Gen1Dataset, gen1_collate
from models import EMSYOLO
from utils.loss import ComputeLoss
from utils.nms import decode_predictions
from utils.coco_eval import evaluate_coco, predictions_to_coco_json
from SNN_framework.firing_rate import FiringRateTracker
from SNN_framework.energy import EnergyTracker


CFG = {
    "model": {
        "backbone": "ems_resnet10",
        "T": 5,
        "decay": 0.25,
        "num_classes": 2,
        "num_anchors": 3,
        "anchors": [
            [[10, 14], [23, 27], [37, 58]],
            [[81, 82], [135, 169], [216, 269]],
        ],
        "strides": [16, 32],
    },
    "data": {
        "img_size": 320,
        "num_workers": 4,
        "pin_memory": False,
    },
    "train": {
        "epochs": 250,
        "batch_size": 8,
        "lr0": 1e-2,
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "warmup_epochs": 3,
        "accum_steps": 1,
        "amp": True,
        "eval_every": 5,
        "save_every": 10,
    },
    "wandb": {
        "entity": "adela-greganova-tu-delft",
        "project": "Gen1-runs",
        "name": None,
    },
}


def _deep_update(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def _wandb_safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def cosine_lr(epoch: int, total_epochs: int, lr0: float, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return lr0 * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return lr0 * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def load_pretrained(model: nn.Module, ckpt_path: str) -> None:
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd["model"] if "model" in sd else sd
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[pretrained] missing={len(missing)} unexpected={len(unexpected)}")


@torch.no_grad()
def evaluate(model, val_loader, cfg, output_dir, gt_json_path, idx_to_id, device):
    model.eval()
    all_preds = []

    for frames, _labels, meta in val_loader:
        # frames: (B, T, 3, H, W) -> (T, B, 3, H, W)
        imgs = frames.permute(1, 0, 2, 3, 4).to(device, non_blocking=True)

        preds = model(imgs)

        results = decode_predictions(
            preds,
            model.strides,
            model.nc,
            model.na,
            conf_thresh=0.001,
            anchors=cfg["model"].get("anchors"),
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

    return evaluate_coco(gt_json_path, all_preds)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--train-dir", required=True,
                    help="Directory of preprocessed train .npy/.txt files")
    ap.add_argument("--val-dir",   required=True,
                    help="Directory of preprocessed val .npy/.txt files")
    ap.add_argument("--output",    required=True,
                    help="Output directory for checkpoints and logs")
    ap.add_argument("--config",    default=None,
                    help="YAML config file (overrides built-in defaults)")
    ap.add_argument("--pretrained", default=None,
                    help="Optional checkpoint to initialise weights from")
    ap.add_argument("--resume",    default=None,
                    help="Resume training from checkpoint")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override train.batch_size")
    ap.add_argument("--epochs",    type=int, default=None,
                    help="Override train.epochs")
    args = ap.parse_args()

    cfg = copy.deepcopy(CFG)
    if args.config:
        with open(args.config) as f:
            _deep_update(cfg, yaml.safe_load(f))
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    wandb.login()
    wcfg = cfg.get("wandb", {})
    wandb.init(
        entity=wcfg.get("entity", "adela-greganova-tu-delft"),
        project=wcfg.get("project", "Gen1-runs"),
        name=wcfg.get("name", None),
        config=cfg,
        dir=str(out),
    )
    wandb.define_metric("epoch")
    for pat in ("lr", "time/*", "train/*", "val/*", "snn/*", "energy/*"):
        wandb.define_metric(pat, step_metric="epoch")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    img_size = cfg["data"]["img_size"]
    T        = cfg["model"]["T"]

    train_ds = Gen1Dataset(args.train_dir, T=T, img_size=img_size, augment=True)
    val_ds   = Gen1Dataset(args.val_dir,   T=T, img_size=img_size, augment=False)

    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    # Build and save COCO GT JSON from the val split (used by pycocotools at eval time)
    gt_json_path = str(out / "gt_val.json")
    print("[data] building GT JSON for evaluation ...", end=" ", flush=True)
    with open(gt_json_path, "w") as f:
        json.dump(val_ds.build_coco_gt(), f)
    print("done")

    pin_memory = cfg["data"].get("pin_memory", False)
    nw = cfg["data"]["num_workers"]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=nw,
        collate_fn=gen1_collate,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=nw,
        collate_fn=gen1_collate,
        pin_memory=pin_memory,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = EMSYOLO(
        backbone=cfg["model"]["backbone"],
        T=T,
        decay=cfg["model"]["decay"],
        num_classes=cfg["model"]["num_classes"],
        num_anchors=cfg["model"]["num_anchors"],
    ).to(device)

    if args.pretrained:
        load_pretrained(model, args.pretrained)

    model.set_T(T)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {cfg['model']['backbone']}  T={T}  params={n_params/1e6:.2f}M")
    wandb.log({"model/params_millions": n_params / 1e6, "model/T": T})

    # ------------------------------------------------------------------
    # Loss / optimiser / scaler
    # ------------------------------------------------------------------
    loss_fn = ComputeLoss(model, anchors=cfg["model"].get("anchors"))

    optim = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr0"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
        nesterov=True,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")

    start_epoch = 0
    best_map    = 0.0

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"] + 1
        best_map    = ck.get("best_map", 0.0)
        print(f"[resume] epoch {start_epoch}, best mAP {best_map:.3f}")

    epochs = cfg["train"]["epochs"]
    accum  = cfg["train"]["accum_steps"]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.train()

        lr = cosine_lr(epoch, epochs, cfg["train"]["lr0"], cfg["train"]["warmup_epochs"])
        for pg in optim.param_groups:
            pg["lr"] = lr

        t0 = time.time()
        running = {"box": 0.0, "obj": 0.0, "cls": 0.0, "total": 0.0}
        optim.zero_grad(set_to_none=True)

        max_steps = cfg["train"].get("max_steps", None)
        for step, (frames, labels, _meta) in enumerate(train_loader):
            # frames: (B, T, 3, H, W) -> model expects (T, B, 3, H, W)
            imgs   = frames.permute(1, 0, 2, 3, 4).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda"):
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

            if (step + 1) % 50 == 0:
                msg = " ".join(f"{k}={v/(step+1):.4f}" for k, v in running.items())
                print(f"[epoch {epoch} step {step+1}/{len(train_loader)} lr={lr:.4g}] {msg}")

        dt    = time.time() - t0
        steps = step + 1
        print(f"[epoch {epoch}] done in {dt/60:.1f} min")

        wandb.log({
            "epoch":           epoch,
            "lr":              lr,
            "time/epoch_min":  dt / 60,
            "train/box":       running["box"]   / steps,
            "train/obj":       running["obj"]   / steps,
            "train/cls":       running["cls"]   / steps,
            "train/total":     running["total"] / steps,
        })

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        if (epoch + 1) % cfg["train"]["eval_every"] == 0 or epoch == epochs - 1:
            with FiringRateTracker(model) as tracker:
                energy_tracker = EnergyTracker(model, T=T)
                metrics = evaluate(
                    model, val_loader, cfg, out, gt_json_path, val_ds.idx_to_id, device
                )
                fr          = tracker.firing_rates()
                fr_summary  = tracker.summary()
                tracked_lif = tracker.num_tracked_layers()
                energy      = energy_tracker.energy(fr)
                energy_tracker.remove()

            ratio_str = f" E_ratio={energy['ratio']:.4f}" if energy["ratio"] is not None else ""
            print(
                f"[epoch {epoch}] "
                f"mAP@0.5={metrics['mAP@0.5']:.4f} "
                f"mAP@0.5:0.95={metrics['mAP@0.5:0.95']:.4f} "
                f"firing_rate={fr['overall']:.6f}"
                f"{ratio_str}"
            )

            eval_log = {
                "epoch":                epoch,
                "val/map_50":           metrics["mAP@0.5"],
                "val/map_50_95":        metrics["mAP@0.5:0.95"],
                "snn/firing_rate":      fr["overall"],
                "snn/tracked_lif":      tracked_lif,
                "energy/E_SNN_pJ":      energy["E_SNN_J"] * 1e12,
                "energy/E_ANN_pJ":      energy["E_ANN_J"] * 1e12,
                "energy/tracked_pairs": energy_tracker.num_tracked_pairs(),
            }
            if energy["ratio"] is not None:
                eval_log["energy/ratio"] = energy["ratio"]
            eval_log.update({
                f"snn/layer_{_wandb_safe(name)}": rate
                for name, rate in fr.items() if name != "overall"
            })
            wandb.log(eval_log)

            with open(out / f"firing_rates_epoch{epoch}.json", "w") as f:
                json.dump({
                    "epoch":       epoch,
                    "overall":     fr["overall"],
                    "tracked_lif": tracked_lif,
                    "per_layer":   fr_summary,
                    "energy": {
                        "E_SNN_pJ":     energy["E_SNN_J"] * 1e12,
                        "E_ANN_pJ":     energy["E_ANN_J"] * 1e12,
                        "ratio":        energy["ratio"],
                        "tracked_pairs": energy_tracker.num_tracked_pairs(),
                        "per_layer": {
                            k: {**v, "E_SNN_pJ": v["E_SNN_J"]*1e12, "E_ANN_pJ": v["E_ANN_J"]*1e12}
                            for k, v in energy["per_layer"].items()
                        },
                    },
                }, f, indent=2)

            if metrics["mAP@0.5"] > best_map:
                best_map = metrics["mAP@0.5"]
                torch.save({
                    "model":    model.state_dict(),
                    "optim":    optim.state_dict(),
                    "epoch":    epoch,
                    "best_map": best_map,
                    "cfg":      cfg,
                }, out / "best.pt")
                with open(out / "best_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

        # ------------------------------------------------------------------
        # Periodic checkpoint
        # ------------------------------------------------------------------
        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            torch.save({
                "model":    model.state_dict(),
                "optim":    optim.state_dict(),
                "epoch":    epoch,
                "best_map": best_map,
                "cfg":      cfg,
            }, out / "last.pt")

    wandb.finish()


if __name__ == "__main__":
    main()
    os.exit(0)
