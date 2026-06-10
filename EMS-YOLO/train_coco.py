"""EMS-YOLO training pipeline for COCO 2017.

Usage:
    python EMS-YOLO/train_coco.py \\
        --data-root C:/coco \\
        --output    runs/coco-resnet34 \\
        --config    EMS-YOLO/configs/coco_local.yaml

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
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import CocoYOLODataset, yolo_collate
from models import EMSYOLO
from utils.loss import ComputeLoss
from utils.nms import decode_predictions
from utils.coco_eval import evaluate_coco, predictions_to_coco_json
from SNN_framework.firing_rate import FiringRateTracker
from SNN_framework.energy import EnergyTracker


CFG = {
    "model": {
        "backbone": "ems_resnet34",
        "T": 4,
        "decay": 0.25,
        "num_classes": 80,
        "num_anchors": 3,
        "anchors": [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
        ],
        "strides": [16, 32],
    },
    "data": {
        "img_size": 640,
        "num_workers": 4,
        "pin_memory": False,
    },
    "train": {
        "epochs": 300,
        "batch_size": 8,
        "lr0": 1e-2,
        "momentum": 0.937,
        "weight_decay": 5e-4,
        "warmup_epochs": 3,
        "accum_steps": 4,
        "amp": True,
        "eval_every": 5,
        "save_every": 10,
    },
    "wandb": {
        "entity": "adela-greganova-tu-delft",
        "project": "COCO-runs",
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
    if missing:
        print("  missing:", missing[:5], "..." if len(missing) > 5 else "")
    if unexpected:
        print("  unexpected:", unexpected[:5], "..." if len(unexpected) > 5 else "")


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
            anchors=cfg["model"].get("anchors"),
        )
        all_preds.extend(
            predictions_to_coco_json(results, meta, idx_to_id, cfg["data"]["img_size"])
        )

    if not all_preds:
        return {"mAP@0.5:0.95": 0.0, "mAP@0.5": 0.0}

    with open(Path(output_dir) / "preds.json", "w") as f:
        json.dump(all_preds, f)

    return evaluate_coco(coco_gt_path, all_preds)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="COCO root — must contain images/ and annotations/")
    ap.add_argument("--output",    required=True,
                    help="Directory for checkpoints and logs")
    ap.add_argument("--config",    default=None,
                    help="YAML config file (overrides built-in defaults)")
    ap.add_argument("--pretrained", default=None,
                    help="Checkpoint to initialise weights from")
    ap.add_argument("--resume",    default=None,
                    help="Resume from checkpoint")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override train.batch_size")
    ap.add_argument("--epochs",    type=int, default=None,
                    help="Override train.epochs")
    ap.add_argument("--run-name",  default=None,
                    help="W&B run name (overrides wandb.name in config)")
    args = ap.parse_args()

    cfg = copy.deepcopy(CFG)
    if args.config:
        with open(args.config) as f:
            _deep_update(cfg, yaml.safe_load(f))
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.run_name is not None:
        cfg["wandb"]["name"] = args.run_name

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["train"]["amp"] and device.type == "cuda"
    print(f"[device] {device}  amp={use_amp}")

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    wandb.login()
    wcfg = cfg["wandb"]
    wandb.init(
        entity=wcfg.get("entity"),
        project=wcfg.get("project", "COCO-runs"),
        name=wcfg.get("name"),
        config=cfg,
        dir=str(out),
    )
    wandb.define_metric("epoch")
    for pat in ("lr", "time/*", "train/*", "val/*", "snn/*", "energy/*"):
        wandb.define_metric(pat, step_metric="epoch")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    img_size = cfg["data"]["img_size"]
    pin_memory = cfg["data"].get("pin_memory", False)
    nw = cfg["data"]["num_workers"]

    train_ds = CocoYOLODataset(args.data_root, "train2017", img_size=img_size, augment=True)
    val_ds   = CocoYOLODataset(args.data_root, "val2017",   img_size=img_size, augment=False)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=True, num_workers=nw, collate_fn=yolo_collate,
        pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=nw, collate_fn=yolo_collate,
        pin_memory=pin_memory,
    )

    coco_gt_path = str(Path(args.data_root) / "annotations" / "instances_val2017.json")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = EMSYOLO(
        backbone=cfg["model"]["backbone"],
        T=cfg["model"]["T"],
        decay=cfg["model"]["decay"],
        num_classes=cfg["model"]["num_classes"],
        num_anchors=cfg["model"]["num_anchors"],
    ).to(device)

    if args.pretrained:
        load_pretrained(model, args.pretrained)

    model.set_T(cfg["model"]["T"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {cfg['model']['backbone']}  T={model.T}  params={n_params/1e6:.2f}M")
    wandb.log({"model/params_millions": n_params / 1e6, "model/T": model.T})

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

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

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

        max_steps        = cfg["train"].get("max_steps", None)
        max_consec_nans  = cfg["train"].get("max_consecutive_nans", 20)
        nan_count        = 0
        consecutive_nans = 0
        clean_steps      = 0  # steps that actually contributed to running loss

        # Wrap the epoch so firing rates are measured for free on training passes
        with FiringRateTracker(model) as train_fr_tracker, \
             EnergyTracker(model, T=cfg["model"]["T"]) as train_energy_tracker:

            for step, (imgs, labels, _meta) in enumerate(train_loader):
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(device.type, enabled=use_amp):
                    preds = model(imgs)
                    loss, parts = loss_fn(preds, labels)
                    loss = loss / accum

                if not torch.isfinite(loss):
                    nan_count        += 1
                    consecutive_nans += 1
                    # Discard any partial gradients accumulated before this batch so
                    # they don't mix with the next clean accumulation window.
                    optim.zero_grad(set_to_none=True)
                    if consecutive_nans >= max_consec_nans:
                        raise RuntimeError(
                            f"Training diverged: {consecutive_nans} consecutive "
                            f"non-finite losses at epoch {epoch} step {step}. "
                            f"Check LR, anchors, and label sanity."
                        )
                    continue

                consecutive_nans = 0  # reset — this batch was clean
                scaler.scale(loss).backward()

                if (step + 1) % accum == 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)

                for k, v in parts.items():
                    running[k] += float(v)
                clean_steps += 1

                if max_steps is not None and (step + 1) >= max_steps:
                    break

                if (step + 1) % 100 == 0:
                    msg = " ".join(f"{k}={v/clean_steps:.4f}" for k, v in running.items())
                    print(f"[epoch {epoch} step {step+1}/{len(train_loader)} lr={lr:.4g}] {msg}")

            train_fr     = train_fr_tracker.firing_rates()
            train_energy = train_energy_tracker.energy(train_fr)

        dt = time.time() - t0
        denom = max(clean_steps, 1)
        nan_str = f"  nan_batches={nan_count}" if nan_count else ""
        print(
            f"[epoch {epoch}] done in {dt/60:.1f} min  "
            f"fr={train_fr['overall']:.4f}  "
            + (f"E_ratio={train_energy['ratio']:.4f}" if train_energy["ratio"] is not None else "")
            + nan_str
        )

        train_log = {
            "epoch":                  epoch,
            "lr":                     lr,
            "time/epoch_min":         dt / 60,
            "train/box":              running["box"]   / denom,
            "train/obj":              running["obj"]   / denom,
            "train/cls":              running["cls"]   / denom,
            "train/total":            running["total"] / denom,
            "train/nan_batches":      nan_count,
            "snn/train_firing_rate":  train_fr["overall"],
            "energy/train_E_SNN_pJ":  train_energy["E_SNN_J"] * 1e12,
            "energy/train_E_ANN_pJ":  train_energy["E_ANN_J"] * 1e12,
        }
        if train_energy["ratio"] is not None:
            train_log["energy/train_ratio"] = train_energy["ratio"]
        wandb.log(train_log)

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        if (epoch + 1) % cfg["train"]["eval_every"] == 0 or epoch == epochs - 1:
            # Clear before eval: training leaves behind cached blocks sized for training
            # tensors; eval uses different shapes and won't be able to reuse them, causing
            # fragmentation. A clean slate avoids OOM from fragmented free memory.
            if device.type == "cuda":
                torch.cuda.empty_cache()

            with FiringRateTracker(model) as tracker:
                energy_tracker = EnergyTracker(model, T=cfg["model"]["T"])
                try:
                    metrics = evaluate(
                        model, val_loader, cfg, out, coco_gt_path, val_ds.idx_to_id, device
                    )
                finally:
                    energy_tracker.remove()
                    # Clear after eval: eval leaves cached blocks sized for val tensors;
                    # the next training epoch won't reuse them either.
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                fr         = tracker.firing_rates()
                fr_summary = tracker.summary()
                tracked    = tracker.num_tracked_layers()
                energy     = energy_tracker.energy(fr)

            ratio_str = f" E_ratio={energy['ratio']:.4f}" if energy["ratio"] is not None else ""
            print(
                f"[epoch {epoch}] "
                f"mAP@0.5={metrics['mAP@0.5']:.4f} "
                f"mAP@0.5:0.95={metrics['mAP@0.5:0.95']:.4f} "
                f"firing_rate={fr['overall']:.6f}"
                f"{ratio_str}"
            )

            if tracked == 0:
                print("[warning] FiringRateTracker found 0 LIFNeuron layers.")
            if energy_tracker.num_resolved_pairs() == 0:
                print("[warning] EnergyTracker resolved 0 LIF-Conv pairs.")

            eval_log = {
                "epoch":                epoch,
                "val/map_50":           metrics["mAP@0.5"],
                "val/map_50_95":        metrics["mAP@0.5:0.95"],
                "snn/firing_rate":      fr["overall"],
                "snn/tracked_lif":      tracked,
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
                    "tracked_lif": tracked,
                    "per_layer":   fr_summary,
                    "energy": {
                        "E_SNN_pJ":      energy["E_SNN_J"] * 1e12,
                        "E_ANN_pJ":      energy["E_ANN_J"] * 1e12,
                        "ratio":         energy["ratio"],
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
