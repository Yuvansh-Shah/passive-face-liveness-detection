"""Download a balanced subset of CelebA-Spoof from HuggingFace via streaming.

Streams from nguyenkhoa/celeba-spoof-for-face-antispoofing-test (67k cropped
face images, binary live/spoof labels) and saves a balanced subset as PNGs
into train/val/test class folders. Avoids downloading the full 5 GB parquet.
"""

import argparse
import random
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "nguyenkhoa/celeba-spoof-for-face-antispoofing-test"
DEFAULT_DATA_DIR = str(Path.home() / "khoji_liveness_data")
MIN_SIDE = 48  # skip tiny crops that carry no usable texture signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--per-class", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data_dir)
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    for split in splits:
        for cls in ("live", "spoof"):
            (root / split / cls).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    ds = load_dataset(DATASET_ID, split="test", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=2000)

    counts = {"live": 0, "spoof": 0}
    skipped = 0
    for ex in ds:
        if all(v >= args.per_class for v in counts.values()):
            break
        cls = ex["labelNames"]
        if cls not in counts or counts[cls] >= args.per_class:
            continue
        img = ex.get("cropped_image")
        try:
            if img is None or min(img.size) < MIN_SIDE:
                skipped += 1
                continue
            img = img.convert("RGB")
        except Exception:
            skipped += 1
            continue

        r = rng.random()
        if r < 0.8:
            split = "train"
        elif r < 0.9:
            split = "val"
        else:
            split = "test"
        idx = counts[cls]
        img.save(root / split / cls / f"{cls}_{idx:05d}.png")
        counts[cls] += 1
        total = counts["live"] + counts["spoof"]
        if total % 500 == 0:
            print(f"saved {total}: {counts}, skipped {skipped}", flush=True)

    print(f"DONE: {counts}, skipped {skipped}", flush=True)
    for split in splits:
        for cls in ("live", "spoof"):
            n = len(list((root / split / cls).glob("*.png")))
            print(f"{split}/{cls}: {n}", flush=True)


if __name__ == "__main__":
    main()
