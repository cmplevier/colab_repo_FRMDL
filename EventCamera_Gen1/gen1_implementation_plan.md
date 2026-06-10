# Gen1 Dataset Integration Plan
## Replicating Tables 2 and 3 of EMS-YOLO (ICCV 2023)

---

## What Tables 2 and 3 Measure

| Table | Backbone | T | Dataset | Goal |
|-------|----------|---|---------|------|
| Table 2 | EMS-ResNet10 | 5 | Gen1 | Main detection results (mAP@0.5, mAP@0.5:0.95, firing rate, #params) |
| Table 3 | EMS-ResNet18 | 5 | Gen1 | Ablation: impact of residual connections (same metrics + energy) |

Metrics to report for each row:
- mAP@0.5 and mAP@0.5:0.95
- Firing rate (fraction of neurons active per timestep)
- Number of parameters
- Energy efficiency (derived from firing rate × SynOps — already handled by `SNN_framework/energy.py`)

---

## Source Reference

All implementation details are sourced from the official repo:
**https://github.com/BICLab/EMS-YOLO** — specifically the `g1-resnet/` subdirectory.

---

## Key Technical Facts (from repo analysis)

### Dataset
- **2 classes:** car (0), pedestrian (1)
- **Native resolution:** 240×304 (Prophesee Gen1 monochrome event sensor)
- **Training resolution:** 320×320 (resized during preprocessing)
- **Raw format:** Prophesee `.dat` event files + `.npy` bounding box annotation files

### Event-to-Frame Encoding
Events are converted to frames **offline** (before training) using `get_gen1_data.py`:
- **Channels:** 3 (R=G=B, all identical — could be 1, but 3 matches our backbone stem)
- **Background:** 127 (gray, no event)
- **Positive polarity (p=1):** 255 (white)
- **Negative polarity (p=0):** 0 (black)
- **Output shape per sample:** `(T=5, 3, 320, 320)` saved as `.npy`
- **Time window:** 250,000 µs per sample → 5 bins × 50,000 µs each

### Model
- Table 2: `EMSResNet10` backbone, T=5
- Table 3: `EMSResNet18` backbone, T=5
- Head: same `EMSYOLOHead` as COCO runs (unchanged)
- Input to model: `(T, B, 3, 320, 320)` tensor

### Anchors (Gen1, 320×320 resolution)
```
P4/16:  [[10,14], [23,27], [37,58]]
P5/32:  [[81,82], [135,169], [216,269]]
```
These differ from the COCO anchors — must be updated in the config.

### Training Hyperparameters
| Param | Value |
|-------|-------|
| Epochs | 250 |
| Batch size | 8 |
| lr0 | 0.01 |
| momentum | 0.937 |
| weight_decay | 5e-4 |
| Warmup epochs | 3 |
| Image size | 320 |
| Auto-anchor | hardcode (re-clusters from data) |
| Mosaic augmentation | disabled |

---

## Disk Space Strategy

The Gen1 raw dataset is ~700 GB. Offline preprocessing would write additional `.npy` files
(~1.5 MB per sample × ~100K samples ≈ 150 GB extra), which is not feasible.

**Solution: on-the-fly preprocessing inside the DataLoader workers.**

The dataset loader reads directly from the raw `.dat` files and converts events to frames
in `__getitem__`. With `num_workers ≥ 4`, CPU preprocessing is fully parallelized and
hidden behind GPU compute — no disk writes, no extra space.

```
Raw Prophesee .dat files  (700 GB, read-only)
        │
        │  [EVERY BATCH — inside DataLoader workers]
        ▼
  Gen1Dataset.__getitem__
  (prophesee_utils psee_loader → frame conversion → resize → tensor)
        │
        ▼
  train.py  →  EMSYOLO  →  loss  →  backprop
```

**Trade-off:** First epoch is slightly slower than a pre-cached run because events are
re-parsed each time. In practice, with 4–8 workers on a fast NVMe `/scratch`, the
bottleneck remains the GPU, not the CPU preprocessing. If it does bottleneck, request
a temporary quota increase from the DelftBlue helpdesk to cache the val split only
(val is a small fraction of the total).

---

## Implementation Tasks

### Task 1 — Port Prophesee I/O utilities

**Files to create** (copy from `g1-resnet/prophesee_utils/io/`):
```
EMS-YOLO/prophesee_utils/__init__.py
EMS-YOLO/prophesee_utils/io/__init__.py
EMS-YOLO/prophesee_utils/io/psee_loader.py
EMS-YOLO/prophesee_utils/io/dat_events_tools.py
EMS-YOLO/prophesee_utils/io/npy_events_tools.py
```

These are used **at training time** inside `__getitem__` — no offline step needed.

**How:** Download / copy from the official repo. Do not modify.

---

### Task 2 — (Removed: no offline preprocessing needed)

~~Port the preprocessing script~~ — replaced by on-the-fly conversion in Task 3.
The `get_gen1_data.py` from the official repo is **not needed** with this approach.

---

### Task 3 — Write the Gen1 dataset loader (on-the-fly)

**File to create:** `EMS-YOLO/dataset_gen1.py`

Reads raw `.dat` + `.npy` annotation pairs directly. No preprocessing step required.

**Key design points:**
- Index-building pass on init: scan all `.dat`/`.npy` pairs, read annotation timestamps, build a list of `(dat_path, npy_path, sample_start_ts)` tuples — one entry per labeled sample.
- `__getitem__` opens the `.dat` file, seeks to `sample_start_ts`, reads `sample_size=250_000 µs` of events, splits into T=5 bins, renders each bin to `(240, 304, 3)` uint8 (gray=127, pos=255, neg=0), resizes to `(320, 320)`.
- Returns `(frames, labels, meta)` matching the same contract as `CocoYOLODataset`.
- Applies horizontal flip augmentation (p=0.5) consistently across all T frames and labels.

```python
from prophesee_utils.io.psee_loader import PSEELoader

class Gen1Dataset(Dataset):
    def __init__(self, root: str, split: str = "train",
                 T: int = 5, sample_size: int = 250_000,
                 img_size: int = 320, augment: bool = True):
        self.T = T
        self.sample_size = sample_size
        self.img_size = img_size
        self.augment = augment
        self.samples = self._index(Path(root) / split)

    def __getitem__(self, idx):
        dat_path, boxes, start_ts = self.samples[idx]
        loader = PSEELoader(str(dat_path))
        loader.seek_time(start_ts)
        events = loader.load_delta_t(self.sample_size)  # structured array

        bin_size = self.sample_size // self.T
        frames = np.full((self.T, 240, 304, 3), 127, dtype=np.uint8)
        for t in range(self.T):
            mask = (events['t'] >= start_ts + t * bin_size) & \
                   (events['t'] <  start_ts + (t+1) * bin_size)
            ev = events[mask]
            frames[t, ev['y'], ev['x'], :] = (ev['p'] * 255).reshape(-1, 1)

        # resize (240,304) → (320,320) for each frame
        frames = np.stack([
            np.array(Image.fromarray(f).resize((self.img_size, self.img_size)))
            for f in frames
        ])  # (T, 320, 320, 3)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        # shape: (T, 3, 320, 320)

        labels = _boxes_to_yolo(boxes, orig_hw=(240, 304))
        if self.augment:
            frames, labels = random_hflip_temporal(frames, labels, p=0.5)

        return frames, labels, ...
```

**Collate function** (`gen1_collate`): same shape contract as `yolo_collate` — batches frames to `(B, T, 3, H, W)` and labels to `(M, 6)` with batch index prepended.

**Note on model input shape:** `EMSYOLO.forward()` currently expects `(T, B, C, H, W)`. The collate function should output `(B, T, C, H, W)` and `train.py` should permute before the forward pass: `imgs = imgs.permute(1, 0, 2, 3, 4)`.

**Performance note:** Use `num_workers=8` on DelftBlue. Each worker independently opens and seeks `.dat` files; PSEELoader is not thread-safe across workers but safe when each worker holds its own file handle (which DataLoader guarantees with `fork`). On Linux `/scratch` (NVMe), I/O is fast enough that this does not bottleneck a single A100.

---

### Task 4 — Extend `models.py` with backbone selection

**File to modify:** `EMS-YOLO/models.py`

Add a `backbone` string argument so the same `EMSYOLO` class works for all three tables:

```python
from SNN_framework.ResNetBackbones import EMSResNet10, EMSResNet18, EMSResNet34

_BACKBONES = {
    "ems_resnet10": EMSResNet10,
    "ems_resnet18": EMSResNet18,
    "ems_resnet34": EMSResNet34,
}

class EMSYOLO(nn.Module):
    def __init__(self, backbone="ems_resnet34", T=5, decay=0.25,
                 num_classes=80, num_anchors=3):
        ...
        self.backbone = _BACKBONES[backbone](T=T, decay=decay)
```

The config YAML `model.backbone` field controls which backbone is used.

---

### Task 5 — Create Gen1 YAML configs

#### `EMS-YOLO/configs/gen1_resnet10.yaml` (Table 2)

```yaml
model:
  backbone: ems_resnet10
  T: 5
  decay: 0.25
  num_classes: 2
  num_anchors: 3
  anchors:
    - [[10, 14], [23, 27], [37, 58]]    # P4/16
    - [[81, 82], [135, 169], [216, 269]] # P5/32
  strides: [16, 32]

data:
  img_size: 320
  num_workers: 4
  pin_memory: true

train:
  epochs: 250
  batch_size: 8
  lr0: 1.0e-2
  momentum: 0.937
  weight_decay: 5.0e-4
  warmup_epochs: 3
  accum_steps: 1
  amp: true
  eval_every: 5
  save_every: 10
```

#### `EMS-YOLO/configs/gen1_resnet18.yaml` (Table 3)

Same as above, with:
```yaml
model:
  backbone: ems_resnet18
  T: 5
```

---

### Task 6 — Integrate Gen1 into `train.py`

**File to modify:** `EMS-YOLO/train.py`

Add a `--dataset` argument and branch on it:

```python
ap.add_argument("--dataset", choices=["coco", "gen1"], default="coco")
ap.add_argument("--gen1-root", default=None,
                help="Root of pre-processed Gen1 .npy dataset")
```

In `main()`, swap dataset and evaluator based on `args.dataset`:

```python
if args.dataset == "gen1":
    from dataset_gen1 import Gen1Dataset, gen1_collate
    train_ds = Gen1Dataset(args.gen1_root, "train", augment=True)
    val_ds   = Gen1Dataset(args.gen1_root, "val",   augment=False)
    collate_fn = gen1_collate
else:
    from dataset import CocoYOLODataset, yolo_collate
    train_ds = CocoYOLODataset(args.data_root, "train2017", ...)
    val_ds   = CocoYOLODataset(args.data_root, "val2017",   ...)
    collate_fn = yolo_collate
```

The evaluation loop (`evaluate()`) already uses `predictions_to_coco_json` + `evaluate_coco` which work on any COCO-format annotation file, so no evaluator change is needed — Gen1 annotations just need to be in COCO JSON format (see Task 7).

---

### Task 7 — Gen1 evaluation annotations

The `evaluate_coco()` function expects a COCO-format ground-truth JSON.
`get_gen1_data.py` should also emit a `instances_val.json` in COCO format
(images list, annotations list with `id`, `image_id`, `category_id`, `bbox`, `area`, `iscrowd`).

Alternatively, convert Gen1 `.npy` annotations to COCO JSON as a one-time step alongside the preprocessing in Task 2.

---

### Task 8 — SLURM job scripts for DelftBlue

No preprocessing job is needed. Both training jobs point directly at the raw `.dat` files.
Request `--cpus-per-task=8` so DataLoader workers keep the GPU fed.

#### `EMS-YOLO/run_gen1_resnet10.sh` (Table 2)
```bash
#!/bin/bash
#SBATCH --job-name=gen1-resnet10
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8          # 8 workers for on-the-fly frame conversion
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%j_resnet10.out
#SBATCH --error=logs/%j_resnet10.err

module load 2024r1
module load cuda/12.1
module load miniconda3
conda activate ems-yolo

export WANDB_API_KEY=<your_key>

cd /scratch/<netid>/FRMDL/EMS-YOLO

python train.py \
    --dataset  gen1 \
    --gen1-root /scratch/<netid>/gen1_raw \
    --output   /scratch/<netid>/runs/gen1-resnet10 \
    --config   configs/gen1_resnet10.yaml
```

#### `EMS-YOLO/run_gen1_resnet18.sh` (Table 3)
Same as above with `gen1_resnet18.yaml` and output dir `gen1-resnet18`.

---

## Execution Order

```
1. Download Gen1 raw data onto DelftBlue  (/scratch/<netid>/gen1_raw)
2. sbatch run_gen1_resnet10.sh            (Table 2, ~24-48 hours)
3. sbatch run_gen1_resnet18.sh            (Table 3, ~24-48 hours)
```

Steps 2 and 3 can run in parallel — they both read from the same read-only raw directory.

---

## Getting the Gen1 Raw Dataset

1. Register at **https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/**
2. Download the three splits: `train`, `val`, `test` (total ~60 GB)
3. Upload to DelftBlue:
   ```bash
   rsync -avz gen1_raw/ <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/gen1_raw/
   ```
   Or download directly from a login node once you have the download link.

Expected raw structure:
```
gen1_raw/
  train/
    *.dat    # event streams
    *.npy    # bounding box annotations
  val/
    *.dat
    *.npy
  test/
    *.dat
    *.npy
```

---

## Open Issues / Things to Verify

| # | Issue | Status |
|---|-------|--------|
| 1 | T value for Table 3 (ResNet18) — paper does not explicitly state it; assumed T=5 matching ResNet10 | Assumed |
| 2 | Whether auto-anchor re-clustering is needed or fixed anchors suffice | Verify empirically |
| 3 | COCO JSON generation for Gen1 val set (needed by `evaluate_coco()`) — must be generated in `_index()` during dataset init, not as a preprocessing step | To implement in Task 3 |
| 4 | W&B API key must be moved from hardcoded line 221 of `train.py` to `$WANDB_API_KEY` env var | To fix |
| 5 | `EMSYOLO.forward()` receives `(B, T, C, H, W)` from Gen1 collate but internally expects `(T, B, C, H, W)` — needs a permute | To fix in Task 6 |
| 6 | Firing rate and energy logging for Gen1 runs (needed for Tables 2/3) — verify `FiringRateTracker` and `EnergyTracker` are active in the training loop | Verify |
| 7 | On-the-fly I/O may bottleneck training if `/scratch` is slow or workers are insufficient — monitor GPU utilisation in first epoch; if <80%, increase `num_workers` or request quota for val cache | Monitor |
