"""Evaluate the trained liveness model on the held-out test split.

Reports accuracy, per-class recall (live recall = BPCER complement,
spoof recall = APCER complement), ROC-AUC, and a confusion matrix.
Saves a grid of correct and incorrect example predictions to
results/examples.png and metrics to results/metrics.json.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEFAULT_DATA_DIR = str(Path.home() / "khoji_liveness_data")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HERE = Path(__file__).parent
CKPT = HERE / "checkpoints" / "liveness_mnv3.pt"
RESULTS = HERE / "results"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    classes = ckpt["classes"]

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(ckpt["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_ds = datasets.ImageFolder(Path(DEFAULT_DATA_DIR) / "test", tf)
    assert test_ds.classes == classes
    loader = DataLoader(test_ds, batch_size=64, num_workers=4)

    all_probs, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x.to(device))
            all_probs.append(torch.softmax(logits.float(), 1).cpu())
            all_true.append(y)
    probs = torch.cat(all_probs).numpy()
    true = torch.cat(all_true).numpy()
    pred = probs.argmax(1)

    acc = float((pred == true).mean())
    cm = confusion_matrix(true, pred)  # rows = true, cols = pred
    live_recall = cm[0, 0] / cm[0].sum()   # % of real faces accepted
    spoof_recall = cm[1, 1] / cm[1].sum()  # % of attacks caught
    auc = float(roc_auc_score(true, probs[:, 1]))

    metrics = {
        "test_images": len(test_ds),
        "accuracy": round(acc, 4),
        "live_recall": round(float(live_recall), 4),
        "spoof_recall": round(float(spoof_recall), 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {"rows=true[live,spoof]": cm.tolist()},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)

    # Example grid: 8 correct (4 per class, most confident) + 8 worst mistakes
    conf_of_pred = probs[np.arange(len(pred)), pred]
    rows = []
    for want_correct in (True, False):
        picks = []
        if want_correct:
            for c in (0, 1):
                idx = np.where((pred == true) & (true == c))[0]
                picks.extend(idx[np.argsort(-conf_of_pred[idx])][:4])
        else:
            idx = np.where(pred != true)[0]
            picks.extend(idx[np.argsort(-conf_of_pred[idx])][:8])
        rows.append(picks)

    fig, axes = plt.subplots(2, 8, figsize=(22, 7))
    fig.suptitle("Top: correct predictions | Bottom: most confident mistakes", fontsize=14)
    for r, picks in enumerate(rows):
        for c in range(8):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(picks):
                continue
            i = int(picks[c])
            path, _ = test_ds.samples[i]
            ax.imshow(Image.open(path).convert("RGB"))
            ok = pred[i] == true[i]
            ax.set_title(f"true:{classes[true[i]]}\npred:{classes[pred[i]]} "
                         f"({conf_of_pred[i]:.2f})",
                         fontsize=9, color="green" if ok else "red")
    fig.tight_layout()
    fig.savefig(RESULTS / "examples.png", dpi=110)
    print(f"saved {RESULTS / 'examples.png'}", flush=True)


if __name__ == "__main__":
    main()
