"""Minimal FastAPI liveness-detection endpoint.

Run:  python api.py          (serves on http://127.0.0.1:8000)
Test: curl -F "file=@image.jpg" http://127.0.0.1:8000/predict
"""

import io
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from predict import load_detector, load_model, predict_pipeline

app = FastAPI(title="Khoji Liveness Detection (prototype)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tf, classes = load_model(device)
detector = load_detector()

# Debug log of received images + verdicts (local disk only) so misclassified
# real-world captures can be inspected and used to tune the model. Keeps the
# most recent ~200 uploads.
DEBUG_DIR = Path(__file__).parent / "debug_uploads"
DEBUG_DIR.mkdir(exist_ok=True)


def _log_upload(img: Image.Image, result: dict):
    try:
        existing = sorted(DEBUG_DIR.glob("*.jpg"))
        for old in existing[:max(0, len(existing) - 199)]:
            old.unlink(missing_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
        if result["status"] == "ok":
            tag = f"{result['label']}_{result['confidence']:.2f}"
            if result.get("face") is None:
                tag += "_fullframe"  # no-face fallback path — worth triaging
        else:
            tag = result["status"]  # e.g. face_too_small, no_face
        img.convert("RGB").save(DEBUG_DIR / f"{stamp}_{tag}.jpg", quality=90)
    except Exception:
        pass  # logging must never break the endpoint


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise HTTPException(400, "could not decode image")
    result = predict_pipeline(model, tf, classes, detector, img, device)
    _log_upload(img, result)
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
