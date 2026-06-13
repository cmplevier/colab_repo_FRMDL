"""EMS-YOLO training pipeline for COCO 2017.

Single-GPU:
    python EMS-YOLO/train_coco.py --data-root C:/coco --output runs/coco --config ...

Multi-GPU (DDP):
    torchrun --standalone --nproc_per_node=2 EMS-YOLO/train_coco.py --data-root ...

W&B key is read from the WANDB_API_KEY environment variable.
Set WANDB_MODE=disabled to run without W&B.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
        "compile": False,
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


def cosine_lr(epoch: int, total_epochs: int, lr0: float, warmup_epochs: int,
              lrf: float = 0.1) -> float:
    if epoch < warmup_epochs:
        return lr0 * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return lr0 * (lrf + (1 - lrf) * 0.5 * (1 + math.cos(math.pi * progress)))


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
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output",    required=True)
    ap.add_argument("--config",    default=None)
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--resume",    default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs",    type=int, default=None)
    ap.add_argument("--run-name",  default=None)
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

    # ------------------------------------------------------------------
    # DDP init — torchrun sets LOCAL_RANK; absent means single-GPU
    # ------------------------------------------------------------------
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp     = local_rank != -1
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = not is_ddp or local_rank == 0
    use_amp = cfg["train"]["amp"] and device.type == "cuda"

    if is_main:
        print(f"[device] {device}  amp={use_amp}  "
              f"world={dist.get_world_size() if is_ddp else 1}")

    # Resuming reuses the existing run directory; fresh runs get a timestamped subdir
    if args.resume:
        out = Path(args.resume).parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.output) / timestamp

    if is_main:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    # ------------------------------------------------------------------
    # W&B — only rank 0
    # ------------------------------------------------------------------
    if is_main:
        wandb.login()
        wcfg     = cfg["wandb"]
        base_name = wcfg.get("name") or cfg["model"]["backbone"]
        run_name  = f"{base_name}-{out.name}"  # e.g. ems_resnet34-20260611_120000
        wandb.init(
            entity=wcfg.get("entity"),
            project=wcfg.get("project", "COCO-runs"),
            name=run_name,
            config=cfg,
            dir=str(out),
        )
        wandb.define_metric("epoch")
        wandb.define_metric("step")
        wandb.define_metric("lr", step_metric="step")
        for pat in ("train/*",):
            wandb.define_metric(pat, step_metric="step")
        for pat in ("time/*", "val/*", "snn/*", "energy/*"):
            wandb.define_metric(pat, step_metric="epoch")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    img_size   = cfg["data"]["img_size"]
    pin_memory = cfg["data"].get("pin_memory", False)
    nw         = cfg["data"]["num_workers"]

    train_ds = CocoYOLODataset(args.data_root, "train2017", img_size=img_size, augment=True)
    val_ds   = CocoYOLODataset(args.data_root, "val2017",   img_size=img_size, augment=False)
    if is_main:
        print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
    train_loader  = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=nw, collate_fn=yolo_collate,
        pin_memory=pin_memory, drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=nw, collate_fn=yolo_collate,
        pin_memory=pin_memory,
        persistent_workers=nw > 0,
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

    if cfg["train"].get("compile", False):
        model = torch.compile(model)

    # raw_model gives attribute access (strides, nc, na) regardless of DDP/compile wrapping
    raw_model = model
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f"[model] {cfg['model']['backbone']}  T={raw_model.T}  params={n_params/1e6:.2f}M")
        wandb.log({"model/params_millions": n_params / 1e6, "model/T": raw_model.T})

    # ------------------------------------------------------------------
    # Loss / optimiser / scaler
    # ------------------------------------------------------------------
    loss_fn = ComputeLoss(raw_model, anchors=cfg["model"].get("anchors"))

    # Split into 3 groups matching YOLOv5/EMS-YOLO original:
    #   BN weights + biases → no weight_decay (TDBN weights must not be pulled toward 0)
    #   everything else → weight_decay
    g_wd, g_no_wd = [], []
    for m in model.modules():
        if hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
            g_no_wd.append(m.bias)
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            g_no_wd.append(m.weight)
        elif hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
            g_wd.append(m.weight)

    optim = torch.optim.SGD(
        g_no_wd,
        lr=cfg["train"]["lr0"],
        momentum=cfg["train"]["momentum"],
        nesterov=True,
    )
    optim.add_param_group({"params": g_wd, "weight_decay": cfg["train"]["weight_decay"]})

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    best_map    = 0.0

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        raw_model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"] + 1
        best_map    = ck.get("best_map", 0.0)
        if is_main:
            print(f"[resume] epoch {start_epoch}, best mAP {best_map:.3f}")

    epochs = cfg["train"]["epochs"]
    accum  = cfg["train"]["accum_steps"]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lr = cosine_lr(epoch, epochs, cfg["train"]["lr0"], cfg["train"]["warmup_epochs"],
                       lrf=cfg["train"].get("lrf", 0.1))
        for pg in optim.param_groups:
            pg["lr"] = lr

        t0 = time.time()
        running = {"box": 0.0, "obj": 0.0, "cls": 0.0, "total": 0.0}
        optim.zero_grad(set_to_none=True)

        max_steps        = cfg["train"].get("max_steps", None)
        max_consec_nans  = cfg["train"].get("max_consecutive_nans", 20)
        nan_count        = 0
        consecutive_nans = 0
        clean_steps      = 0

        with FiringRateTracker(raw_model) as train_fr_tracker, \
             EnergyTracker(raw_model, T=cfg["model"]["T"]) as train_energy_tracker:

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
                    optim.zero_grad(set_to_none=True)
                    if consecutive_nans >= max_consec_nans:
                        raise RuntimeError(
                            f"Training diverged: {consecutive_nans} consecutive "
                            f"non-finite losses at epoch {epoch} step {step}."
                        )
                    continue

                consecutive_nans = 0
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

                if is_main and (step + 1) % 100 == 0:
                    msg = " ".join(f"{k}={v/clean_steps:.4f}" for k, v in running.items())
                    print(f"[epoch {epoch} step {step+1}/{len(train_loader)} lr={lr:.4g}] {msg}")
                    wandb.log({
                        "epoch":             epoch,
                        "step":              epoch * len(train_loader) + step,
                        "lr":                lr,
                        "train/box":         running["box"]   / clean_steps,
                        "train/obj":         running["obj"]   / clean_steps,
                        "train/cls":         running["cls"]   / clean_steps,
                        "train/total":       running["total"] / clean_steps,
                        "train/nan_batches": nan_count,
                    })

            train_fr     = train_fr_tracker.firing_rates()
            train_energy = train_energy_tracker.energy(train_fr)

        dt    = time.time() - t0
        denom = max(clean_steps, 1)

        if is_main:
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
        # Evaluation — rank 0 only; other ranks wait at the barriers
        # ------------------------------------------------------------------
        if (epoch + 1) % cfg["train"]["eval_every"] == 0 or epoch == epochs - 1:
            if is_ddp:
                dist.barrier()

            if is_main:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                with FiringRateTracker(raw_model) as tracker:
                    energy_tracker = EnergyTracker(raw_model, T=cfg["model"]["T"])
                    try:
                        metrics = evaluate(
                            raw_model, val_loader, cfg, out,
                            coco_gt_path, val_ds.idx_to_id, device,
                        )
                    finally:
                        energy_tracker.remove()
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
                        "model":    raw_model.state_dict(),
                        "optim":    optim.state_dict(),
                        "epoch":    epoch,
                        "best_map": best_map,
                        "cfg":      cfg,
                    }, out / "best.pt")
                    with open(out / "best_metrics.json", "w") as f:
                        json.dump(metrics, f, indent=2)

            if is_ddp:
                dist.barrier()

        # ------------------------------------------------------------------
        # Periodic checkpoint — rank 0 only
        # ------------------------------------------------------------------
        if is_main and (epoch + 1) % cfg["train"]["save_every"] == 0:
            torch.save({
                "model":    raw_model.state_dict(),
                "optim":    optim.state_dict(),
                "epoch":    epoch,
                "best_map": best_map,
                "cfg":      cfg,
            }, out / "last.pt")

    if is_main:
        wandb.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
