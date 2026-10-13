"""Extract a balanced live/spoof subset from locally downloaded parquet shards.

Reads the CelebA-Spoof parquet shards in ~/khoji_liveness_raw,
decodes images, filters tiny/broken ones, and saves a balanced subset as PNGs
into train/val/test (80/10/10) class folders. Rows in the source are already
shuffled, so sequential fill is an unbiased sample.
"""

import io
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

RAW_DIR = Path.home() / "khoji_liveness_raw" / "data"
DATA_DIR = Path.home() / "khoji_liveness_data"
PER_CLASS = 6000
MIN_SIDE = 48
SEED = 42


def main():
    for split in ("train", "val", "test"):
        for cls in ("live", "spoof"):
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    counts = {"live": 0, "spoof": 0}
    skipped = 0
    pool = ThreadPoolExecutor(max_workers=8)
    pending = []

    def handle(img_bytes, cls, idx, split):
        nonlocal skipped
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.load()
            if min(img.size) < MIN_SIDE:
                return
            img.convert("RGB").save(DATA_DIR / split / cls / f"{cls}_{idx:05d}.png")
        except Exception:
            pass

    shards = sorted(RAW_DIR.glob("*.parquet"))
    print(f"shards: {[s.name for s in shards]}", flush=True)
    for shard in shards:
        if all(v >= PER_CLASS for v in counts.values()):
            break
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=256, columns=["cropped_image", "labelNames"]):
            if all(v >= PER_CLASS for v in counts.values()):
                break
            imgs = batch.column("cropped_image").to_pylist()
            labels = batch.column("labelNames").to_pylist()
            for img_struct, cls in zip(imgs, labels):
                if cls not in counts or counts[cls] >= PER_CLASS:
                    continue
                if img_struct is None or img_struct.get("bytes") is None:
                    skipped += 1
                    continue
                r = rng.random()
                split = "train" if r < 0.8 else ("val" if r < 0.9 else "test")
                pending.append(pool.submit(handle, img_struct["bytes"], cls, counts[cls], split))
                counts[cls] += 1
            total = counts["live"] + counts["spoof"]
            if total % 2000 < 256:
                print(f"queued {counts}, skipped {skipped}", flush=True)

    for f in pending:
        f.result()
    pool.shutdown()
    print(f"DONE queued {counts}, null-skipped {skipped}", flush=True)
    for split in ("train", "val", "test"):
        for cls in ("live", "spoof"):
            n = len(list((DATA_DIR / split / cls).glob("*.png")))
            print(f"{split}/{cls}: {n}", flush=True)


if __name__ == "__main__":
    main()
