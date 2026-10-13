"""Fine-tune a pretrained MobileNetV3 as a binary live/spoof classifier.

Speed-first prototype: standard cross-entropy, AdamW + cosine schedule,
basic augmentation, mixed precision. Saves the best checkpoint by val
accuracy to checkpoints/liveness_mnv3.pt.
"""

import argparse
import json
import random
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

DEFAULT_DATA_DIR = str(Path.home() / "khoji_liveness_data")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_SIZE = 224
ARCH = "mobilenetv3_large_100"


class RandomDownscale(torch.nn.Module):
    """Simulate a face captured from far away: it lands on few sensor pixels,
    picks up sensor noise AT THAT SCALE, and the face-crop pipeline then
    upscales it — turning i.i.d. noise into the spatially-correlated blotches
    real far-webcam crops show. Noise added after upscaling (the plain
    GaussianNoise stage) cannot reproduce that pattern, and it is exactly what
    flipped real medium/far webcam captures to spoof. Applied to both classes
    so low detail + correlated noise alone is not a spoof cue."""

    def __init__(self, lo=64, hi=192, noise_p=0.7, noise_sigma=(2.0, 10.0)):
        super().__init__()
        self.lo, self.hi = lo, hi
        self.noise_p, self.noise_sigma = noise_p, noise_sigma

    def forward(self, x):
        size = random.randint(self.lo, self.hi)
        small = v2.functional.resize(x, size, antialias=True)
        if random.random() < self.noise_p:
            sigma = random.uniform(*self.noise_sigma)  # on the 0-255 scale
            noisy = small.float() + torch.randn_like(small, dtype=torch.float32) * sigma
            small = noisy.clamp(0, 255).to(torch.uint8)
        return v2.functional.resize(small, IMG_SIZE, antialias=True)


class RandomGammaDark(torch.nn.Module):
    """Gamma-crush (out = in**g, g > 1): the v2 model's residual weak spot —
    underexposed faces read as spoof, which is what a webcam produces when the
    subject is far and auto-exposure meters on the bright background."""

    def __init__(self, lo=1.3, hi=2.5):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):
        return x.clamp(0.0, 1.0).pow(random.uniform(self.lo, self.hi))


def make_loaders(data_dir, batch_size, workers):
    # Degradation augmentation applied to BOTH classes: the v1 model learned
    # "sensor noise / dim frames = spoof" (webcam captures of real faces were
    # near-universally rejected). Injecting these conditions into live and
    # spoof alike forces the model onto recapture cues that survive them.
    # v3 adds RandomDownscale + RandomGammaDark: with the face-crop inference
    # pipeline, far-away subjects yield small, underexposed face crops.
    train_tf = v2.Compose([
        v2.PILToTensor(),
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.65, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([RandomDownscale(64, 192)], p=0.35),
        v2.ColorJitter(brightness=(0.45, 1.25), contrast=0.35, saturation=0.35, hue=0.04),
        v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.6))], p=0.3),
        v2.RandomApply([v2.JPEG((40, 92))], p=0.4),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomApply([RandomGammaDark(1.3, 2.2)], p=0.3),
        v2.RandomApply([v2.RandomChoice(
            [v2.GaussianNoise(sigma=s) for s in (0.02, 0.04, 0.06)]
        )], p=0.55),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    root = Path(data_dir)
    train_ds = datasets.ImageFolder(root / "train", train_tf)
    val_ds = datasets.ImageFolder(root / "val", eval_tf)
    assert train_ds.classes == ["live", "spoof"], train_ds.classes
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=workers, pin_memory=True, drop_last=True,
                          persistent_workers=workers > 0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True,
                        persistent_workers=workers > 0)
    return train_dl, val_dl, train_ds.classes


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    train_dl, val_dl, classes = make_loaders(args.data_dir, args.batch_size, args.workers)
    print(f"train batches: {len(train_dl)}, val images: {len(val_dl.dataset)}", flush=True)

    model = timm.create_model(ARCH, pretrained=True, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if (i + 1) % 50 == 0:
                print(f"epoch {epoch + 1} step {i + 1}/{len(train_dl)} "
                      f"loss {running / (i + 1):.4f}", flush=True)
        val_acc = evaluate(model, val_dl, device)
        print(f"epoch {epoch + 1}: train_loss {running / len(train_dl):.4f} "
              f"val_acc {val_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "arch": ARCH,
                "img_size": IMG_SIZE,
                "classes": classes,
                "val_acc": val_acc,
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
            }, ckpt_dir / "liveness_mnv3.pt")
            print(f"saved best checkpoint (val_acc {val_acc:.4f})", flush=True)

    (ckpt_dir / "train_summary.json").write_text(json.dumps({
        "arch": ARCH, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "best_val_acc": best_acc,
    }, indent=2))
    print(f"DONE best_val_acc {best_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
