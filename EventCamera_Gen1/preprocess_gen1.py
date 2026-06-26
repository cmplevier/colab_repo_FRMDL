"""
Preprocess a chunk of Gen1 raw data into frame-stacked .npy files for EMS-YOLO.

Encoding (matches the official EMS-YOLO repo):
    no event  -> 127 (gray)
    p = 0     ->   0 (black)
    p = 1     -> 255 (white)
All three channels are identical (1-channel signal in a 3-channel container).

Output per sample:
    <idx>.npy   shape (T, H, W, 3) uint8   — frame stack
    <idx>.txt   YOLO labels: one line per box: class cx cy w h (normalised to [0,1])

Usage:
    python EventCamera_Gen1/preprocess_gen1.py \\
        --raw-dir  /path/to/gen1_raw/train  \\
        --out-dir  /path/to/gen1_processed/train \\
        --T 5 \\
        --sample-size 250000 \\
        --img-size 320

Run once per split (train / val / test). Each split can be processed independently,
so you can process one downloaded archive at a time and delete the raw .dat files
after each run to reclaim space.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# resolve prophesee_utils relative to this file so the script works
# regardless of where it is called from
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prophesee_utils.io.psee_loader import PSEELoader

SENSOR_H = 240
SENSOR_W = 304


def _events_to_frame(events, ts_start, ts_end):
    """Render events in the half-open interval [ts_start, ts_end) into a uint8 HxWx3 frame."""
    frame = np.full((SENSOR_H, SENSOR_W, 3), 127, dtype=np.uint8)
    mask = (events['t'] >= ts_start) & (events['t'] < ts_end)
    ev = events[mask]
    if len(ev):
        # clip coordinates to sensor bounds (guard against occasional bad events)
        x = np.clip(ev['x'].astype(np.int32), 0, SENSOR_W - 1)
        y = np.clip(ev['y'].astype(np.int32), 0, SENSOR_H - 1)
        frame[y, x, :] = (ev['p'].astype(np.uint8) * 255).reshape(-1, 1)
    return frame


def _process_recording(dat_path, ann_path, out_dir, T, sample_size, img_size, start_idx):
    """
    Process one (recording, annotation) pair.
    Returns the number of samples written.
    """
    bboxes = np.load(str(ann_path))

    # annotation field name for timestamp varies between dataset versions
    ts_field = 't' if 't' in bboxes.dtype.names else 'ts'
    timestamps = np.unique(bboxes[ts_field])

    loader = PSEELoader(str(dat_path))
    n_written = 0

    for ts in timestamps:
        window_start = max(0, int(ts) - sample_size)

        loader.seek_time(window_start)
        events = loader.load_delta_t(sample_size)

        if len(events) == 0:
            continue

        # build T equal-duration bins
        bin_size = sample_size // T
        frames = []
        for t in range(T):
            bin_start = window_start + t * bin_size
            bin_end   = window_start + (t + 1) * bin_size
            frames.append(_events_to_frame(events, bin_start, bin_end))
        frames = np.stack(frames)  # (T, H, W, 3)

        # resize each frame if requested
        if img_size != SENSOR_H or img_size != SENSOR_W:
            frames = np.stack([
                np.array(Image.fromarray(f).resize((img_size, img_size), Image.BILINEAR))
                for f in frames
            ])  # (T, img_size, img_size, 3)

        # build YOLO labels from annotations at this timestamp
        ann_at_ts = bboxes[bboxes[ts_field] == ts]
        lines = []
        for ann in ann_at_ts:
            x  = float(ann['x'])
            y  = float(ann['y'])
            w  = float(ann['w'])
            h  = float(ann['h'])
            cls = int(ann['class_id'])
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2) / SENSOR_W
            cy = (y + h / 2) / SENSOR_H
            wn = w / SENSOR_W
            hn = h / SENSOR_H
            # clamp to [0, 1] in case of any annotation noise
            cx, cy, wn, hn = (min(max(v, 0.0), 1.0) for v in (cx, cy, wn, hn))
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")

        if not lines:
            continue

        sample_idx = start_idx + n_written
        np.save(str(out_dir / f"{sample_idx:07d}.npy"), frames)
        (out_dir / f"{sample_idx:07d}.txt").write_text("\n".join(lines))
        n_written += 1

    return n_written


def main():
    ap = argparse.ArgumentParser(
        description="Convert a Gen1 raw chunk to frame-stacked .npy files for EMS-YOLO training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--raw-dir",     required=True,
                    help="Directory containing .dat event files and paired _bbox.npy annotation files")
    ap.add_argument("--out-dir",     required=True,
                    help="Output directory — .npy and .txt files are written here")
    ap.add_argument("--T",           type=int, default=5,
                    help="Number of temporal bins per sample")
    ap.add_argument("--sample-size", type=int, default=250_000,
                    help="Duration of each sample in µs")
    ap.add_argument("--img-size",    type=int, default=320,
                    help="Output spatial resolution (square); sensor native is 240x304")
    ap.add_argument("--start-idx",   type=int, default=0,
                    help="Starting index for output filenames — useful when appending multiple chunks")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # match each .dat file with its _bbox.npy annotation file
    # naming convention: <base>_td.dat  <->  <base>_bbox.npy
    dat_files = sorted(raw_dir.glob("*.dat"))
    if not dat_files:
        print(f"ERROR: no .dat files found in {raw_dir}", file=sys.stderr)
        sys.exit(1)

    pairs = []
    for dat in dat_files:
        base = dat.stem[:-3] if dat.stem.endswith("_td") else dat.stem
        ann = dat.with_name(base + "_bbox.npy")
        if ann.exists():
            pairs.append((dat, ann))
        else:
            print(f"  WARNING: no annotation file for {dat.name} (expected {ann.name}), skipping")

    if not pairs:
        print("ERROR: no matched (dat, annotation) pairs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} recording(s) in {raw_dir}")
    print(f"Output: {out_dir}  |  T={args.T}  sample_size={args.sample_size}µs  img_size={args.img_size}px\n")

    global_idx = args.start_idx
    for i, (dat, ann) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {dat.name}", end="  ", flush=True)
        n = _process_recording(dat, ann, out_dir, args.T, args.sample_size, args.img_size,
                               start_idx=global_idx)
        print(f"-> {n} samples  (cumulative: {global_idx + n})")
        global_idx += n

    total = global_idx - args.start_idx
    npy_bytes = sum(f.stat().st_size for f in out_dir.glob("*.npy"))
    print(f"\nDone.  {total} samples written to {out_dir}")
    print(f"Output size: {npy_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
