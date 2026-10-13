"""Run liveness inference on a single image.

Pipeline: detect the face (YuNet, multi-scale), crop it to match the
training-data framing (CelebA-Spoof crops are face-filling), then classify
live vs spoof. Faces too small in the original frame return a
"face_too_small" status instead of an unreliable verdict.

Usage: python predict.py path/to/image.jpg
Prints JSON: {"status": "ok", "label": "live"|"spoof", "confidence": 0.97, ...}
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CKPT = Path(__file__).parent / "checkpoints" / "liveness_mnv3.pt"
DETECTOR_ONNX = Path(__file__).parent / "checkpoints" / "face_detection_yunet_2023mar.onnx"

# Faces narrower than this (px, in the original frame) are refused rather than
# classified: below it the crop carries too little detail for a reliable
# verdict (calibrated on the downscale sweep in results/distance_report.json).
MIN_FACE_PX = 64

# Detector box -> classifier crop. CelebA-Spoof training crops are
# face-filling (median face-box/image-width = 1.0), so keep the margin small;
# the top margin is larger because YuNet boxes sit at the eyebrows and the
# training crops include forehead/hair.
CROP_MARGIN = 0.15
CROP_TOP_EXTRA = 1.4


def load_model(device):
    ckpt = torch.load(CKPT, map_location=device, weights_only=True)
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(ckpt["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return model, tf, ckpt["classes"]


def load_detector():
    return cv2.FaceDetectorYN.create(str(DETECTOR_ONNX), "", (320, 320))


def detect_face(detector, img: Image.Image):
    """Find the largest face. Returns (box [x1,y1,x2,y2] in original px, conf)
    or (None, None).

    YuNet misses faces that are very large relative to its input (webcam
    close-ups), so try a downscaled copy first, then larger scales for small
    faces in high-resolution photos.
    """
    rgb = np.asarray(img.convert("RGB"))
    for det_size in (320, 640, None):
        scale = 1.0 if det_size is None else det_size / max(img.size)
        if scale >= 1.0 and det_size is not None:
            continue  # image already smaller than this pass
        if scale < 1.0:
            w, h = round(img.width * scale), round(img.height * scale)
            frame = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        else:
            frame = rgb
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = detector.detect(bgr)
        if faces is None:
            continue
        f = [float(v) for v in max(faces, key=lambda f: f[2] * f[3])]
        box = [f[0] / scale, f[1] / scale, (f[0] + f[2]) / scale, (f[1] + f[3]) / scale]
        return box, f[14]
    return None, None


def crop_face(img: Image.Image, box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    nx1 = max(0, int(x1 - CROP_MARGIN * w))
    nx2 = min(img.width, int(x2 + CROP_MARGIN * w))
    ny1 = max(0, int(y1 - CROP_MARGIN * h * CROP_TOP_EXTRA))
    ny2 = min(img.height, int(y2 + CROP_MARGIN * h))
    return img.crop((nx1, ny1, nx2, ny2))


@torch.no_grad()
def predict(model, tf, classes, img: Image.Image, device):
    """Raw classifier call on an image (no detection/cropping)."""
    x = tf(img.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x).float(), 1)[0].cpu()
    i = int(probs.argmax())
    return {
        "label": classes[i],
        "confidence": round(float(probs[i]), 4),
        "probs": {c: round(float(p), 4) for c, p in zip(classes, probs)},
    }


def predict_pipeline(model, tf, classes, detector, img: Image.Image, device):
    """Full inference pipeline: detect -> gate on face size -> crop -> classify."""
    # phone JPEGs store pixels rotated + an EXIF orientation tag; YuNet is not
    # rotation-invariant, so apply the tag before anything else
    img = ImageOps.exif_transpose(img).convert("RGB")
    box, det_conf = detect_face(detector, img)

    if box is None:
        if max(img.size) >= 640:
            # A camera-sized frame with no detectable face: refuse rather
            # than classify background (or a face too small even to detect).
            return {
                "status": "no_face",
                "face": None,
                "message": "No clear face detected — face the camera directly "
                           "in decent lighting, or move closer.",
            }
        # Small images are usually already tight face crops (the detector
        # can't box a face that overflows the frame). Classify the full
        # frame as before, but say so.
        result = predict(model, tf, classes, img, device)
        result.update(status="ok", face=None,
                      note="no face detected; classified the full frame")
        return result

    face_w = box[2] - box[0]
    face = {
        "box": [round(v, 1) for v in box],
        "detector_confidence": round(det_conf, 3),
        "width_px": round(face_w),
    }
    if face_w < MIN_FACE_PX:
        return {
            "status": "face_too_small",
            "face": face,
            "message": "Face is too small for a reliable decision — "
                       "please move closer to the camera.",
        }

    result = predict(model, tf, classes, crop_face(img, box), device)
    result.update(status="ok", face=face)
    return result


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python predict.py <image_path>")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, classes = load_model(device)
    detector = load_detector()
    result = predict_pipeline(model, tf, classes, detector,
                              Image.open(sys.argv[1]), device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
