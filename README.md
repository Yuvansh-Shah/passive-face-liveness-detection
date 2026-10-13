# Khoji Liveness Detection — Prototype

Passive liveness detection: given a single face image, classify **live** (real
person in front of the camera) vs **spoof** (re-photographed photo — printed
copy or phone/screen replay). Built as a speed-first proof of concept for the
Khoji missing-persons platform, to catch the attack where someone points a
live camera at a downloaded/screenshotted photo of a missing person.

## Results (held-out test split, 1,154 images) — v3 distance-robust model

| Metric | Value |
|---|---|
| Accuracy | **98.6%** |
| Spoof recall (attacks caught) | **98.6%** (561/569) |
| Live recall (real faces accepted) | **98.6%** (577/585) |
| ROC-AUC | 0.9985 |
| APCER (spoofs accepted as live) | 1.41% (8/569) |
| BPCER (live faces rejected) | 1.37% (8/585) |
| ACER | 1.39% |

Model: MobileNetV3-Large (4.2M params, ImageNet-pretrained via `timm`),
fine-tuned 14 epochs on ~9.5k images with degradation + distance
augmentation (see the lessons below). Full metrics in
`liveness/results/metrics.json`. Two earlier models came first: v1 (no
degradation augmentation, 98.9%) and v2 (degradation augmentation, 99.6%
in-distribution but broken at distance). v3 trades ~1 point of
in-distribution accuracy for working at realistic camera distances.

Inference now runs a pipeline, not a bare classifier: **YuNet face
detection** (multi-scale, `checkpoints/face_detection_yunet_2023mar.onnx`,
~230 KB) → crop to training-style framing → classify. Faces narrower than
64 px return `face_too_small` ("please move closer") and undetectable faces
in camera-sized frames return `no_face`, instead of an unreliable verdict.

### The webcam lesson (v1 → v2)

The v1 model classified essentially every real webcam capture as spoof.
Diagnosis: it had learned **sensor noise as a spoof shortcut** (spoof
images pass through two cameras, so they're noisier in this dataset) —
adding mild noise to live test images flipped 82/100 to spoof, and a
simulated "webcam in a dim room" flipped 94/100. Retraining with noise /
low-light / blur / JPEG augmentation on *both* classes fixed it
(1/100 and 7/100 respectively) and also closed the reverse hole: v1 let
61% of *degraded spoofs* through as live; v2 lets ~10% through. Moral for
production: in-distribution accuracy says nothing about capture-condition
shortcuts; test with the real camera pipeline early.

### The distance lesson (v2 → v3)

v2 classified real faces as live **only when very close to the camera**;
at normal sitting distance the same face read spoof 1.00. Two stacked causes:

1. **No face detection.** The model saw the whole frame (`Resize(256)` +
   `CenterCrop(224)`), but the training crops are face-filling (median
   face-box/image-width = 1.0). A far face was a small blob in background —
   nothing like training data. Cropping the detected face fixed distant
   *photos* instantly (a real far-away photo went spoof 0.98 → live 1.00).
2. **Upscaled sensor noise read as spoof.** Even after cropping, real
   webcam captures at distance still read spoof. Isolation tests showed
   denoising alone flipped them decisively to live (0.33 → 0.999): a far
   face lands on few sensor pixels, and upscaling the crop turns i.i.d.
   sensor noise into spatially-correlated blotches — a pattern the v2
   noise augmentation (added at full resolution) never produced, so the
   noise-shortcut hole was still open for it. The v3 recipe injects noise
   *at the small scale inside a downscale-upscale augmentation*
   (`RandomDownscale`), plus gamma-darkening (`RandomGammaDark`) for the
   underexposure webcam auto-exposure causes at distance. Both applied to
   both classes, same philosophy as the v2 fix.

Notably, the obvious hypothesis — "upscale blur reads as spoof" — was
**wrong**: downscale-upscaling a live capture *raised* p(live). Measure,
don't guess. After the fix, the same real webcam captures read live at
close (0.93), medium (0.72), and far (0.91) distance, printed-photo spoofs
still read spoof, and simulated far faces are accepted with **zero false
spoofs** down to the 64 px gate (failures route to "move closer"/"no face"
guidance instead of wrong verdicts).

### Qualitative failure pattern

- **Missed spoofs** are clean, well-lit, in-focus re-captures with almost no
  visible moiré, glare, or paper texture — the hardest and most realistic case.
- **Falsely rejected live faces** are blurry, low-resolution, or washed-out
  real photos whose softness resembles a print.
- The model appears to key heavily on sharpness/color-cast/texture artifacts.
  That is expected for a single-frame passive approach and works well
  in-distribution, but see limitations below.

## Run the demo

On Windows, double-click `run_demo.bat`. It sets up an isolated Python
environment, installs the packages and opens the demo in your browser.
On any platform:

```bash
pip install -r demo_requirements.txt
python liveness/demo.py
```

Then upload a photo or use the webcam at http://127.0.0.1:8000. The trained
checkpoint ships with the repo, so no training is needed to try it. A CPU is
enough.

To retrain or re-run the evaluation, install `requirements-train.txt` and see
Data below.

## Files

All code lives in `liveness/`; the paths in this table are relative to it.

| File | Purpose |
|---|---|
| `download_data.py` / `extract_data.py` | Get the dataset (see Data below) |
| `train.py` | Fine-tune MobileNetV3 (`--epochs`, `--batch-size`, `--lr` flags) |
| `evaluate.py` | Test-split metrics + example grid |
| `predict.py` | Inference pipeline (detect → crop → classify). CLI: `python predict.py path/to/image.jpg` → JSON status + label + confidence + face box |
| `api.py` | FastAPI demo: `python api.py`, then `curl -F "file=@img.jpg" http://127.0.0.1:8000/predict` |
| `demo.py` + `static/` | Web demo UI: `python demo.py` → opens http://127.0.0.1:8000 (upload or webcam capture → live/spoof verdict, or "move closer" / "no face" guidance) |
| `checkpoints/liveness_mnv3.pt` | Trained classifier checkpoint (~16 MB) |
| `checkpoints/face_detection_yunet_2023mar.onnx` | YuNet face detector (~230 KB, from opencv_zoo) |

## Data

Balanced ~12k-image subset (5,863 live / 5,986 spoof after filtering, split
80/10/10) of **CelebA-Spoof**, taken from the HuggingFace mirror
`nguyenkhoa/celeba-spoof-for-face-antispoofing-test` (cropped faces, binary
labels). The images are not in this repository; the scripts keep them in
`~/khoji_liveness_data` (override with `--data-dir`). To rebuild: run
`extract_data.py` after downloading 3 parquet shards with the snippet in that
file's docstring, or `download_data.py` to stream directly (slower).

## Honest limitations — read before trusting the numbers

1. **Single-source data.** Train and test both come from CelebA-Spoof, so the
   98.6% is *in-distribution* accuracy. Real Khoji uploads (different phones,
   compression, lighting, Indian demographics) will score noticeably worse
   until we fine-tune on matched data. Treat the number as "the approach
   works," not "the system is 98.6% accurate in production."
2. **OULU-NPU not included.** It requires a signed EULA/registration, so this
   prototype couldn't download it. A second dataset would let us measure
   cross-dataset generalization — the standard way FAS models are stress-tested
   (and where accuracy typically drops 10–30 points).
3. **No print-vs-screen breakdown.** The HF subset only carries binary
   live/spoof labels, so we can't report separate recall for printed photos vs
   screen replays. Qualitatively the model nails obvious moiré/glare screen
   shots and crumpled prints, and misses the cleanest re-captures.
4. **Subset, not the full dataset.** Trained on ~9.5k of 625k available
   images (speed-first choice). More data + the full CelebA-Spoof train split
   is a cheap next win.
5. **Face-cropping can remove spoof context, and digital originals are out
   of scope by design.** The v3 pipeline crops to the face before
   classifying, which also crops *out* frame-level attack evidence (phone
   bezels, hands holding a print, screen edges, document layouts). Verdicts
   rest on face-texture recapture cues alone. Consequence, verified on a
   real case (2026-07-07): a **clean digital scan of an e-Aadhaar card
   passes as live 1.00** — the embedded photo is a first-generation digital
   image with zero recapture artifacts, so *both* v2 and v3 pass its face
   crop; a phone photo of a physical ID card is still caught (the recapture
   leaves evidence). This is an injection-class attack, not a presentation
   attack: any clean digital original (including an ordinary selfie photo)
   defeats pixel-based liveness, because there is nothing in the pixels to
   distinguish. Production needs channel controls (liveness verdicts only on
   frames the system itself captured) and/or a context-trained model —
   note that naively also classifying a wider context crop with the current
   model does NOT work: real far-away captures read spoof on context views
   (that is exactly the fixed distance bug), so the "spoof" it emits on a
   document context is OOD bias, not document understanding. Relatedly,
   when the detector finds no face in a small image the API falls back to
   classifying the whole frame (tight dataset-style crops need this —
   YuNet misses ~16% of them), and heavy phone-ISP denoising is untested
   (the v3 model reads *absence* of sensor noise as print-like).
6. **Untested against attacks that aren't in the data**: high-quality prints
   on matte paper, 4K/OLED screen replays at optimal distance, masks, and
   deliberate adversarial attempts.
7. **Residual photometric weak spots (v3, measured through the pipeline on
   100 live test images)**: strong gamma-darkening flips 9/100 live faces to
   spoof (was 65/100 in v2), warm cast 8/100 (was 23), dim+cast+noise 14/100,
   heavy i.i.d. noise 14/100 (v2 was better on this synthetic case at 1/100 —
   the price of fixing real webcam noise). Degraded-spoof leakage improved:
   4/100 dim+noisy spoofs pass as live (was ~10) and 1/100 gamma-dark.
   Live acceptance in the far-face simulation is ≥90% down to ~90 px faces.
8. **Every API upload is saved to `debug_uploads/`** (last ~200, local disk
   only) with its verdict — useful for debugging misclassifications; delete
   the folder if that's not wanted.

## Licences and attribution

- The classifier weights were fine-tuned on CelebA-Spoof, which is released
  for non-commercial research. Treat `liveness_mnv3.pt` the same way.
- The face detector is YuNet from
  [opencv_zoo](https://github.com/opencv/opencv_zoo) (MIT licence).
- The backbone is `mobilenetv3_large_100` from
  [timm](https://github.com/huggingface/pytorch-image-models), ImageNet-pretrained.

## Suggested next steps

- Fine-tune/evaluate on a self-collected set matching Khoji's real capture
  conditions (few hundred phone re-captures of screens/prints goes a long way).
- Add cross-dataset eval (OULU-NPU after EULA, or Kaggle LCC-FASD).
- Calibrate a confidence threshold for the product decision (e.g. flag for
  human review between 0.3–0.7 rather than hard accept/reject).
