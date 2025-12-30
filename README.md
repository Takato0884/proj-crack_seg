# Crack Segmentation (Ultralytics YOLO) with W&B

This repo trains a YOLO segmentation model on the `datasets/crack-seg` dataset and optionally logs metrics and artifacts to Weights & Biases (W&B).

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Set your W&B API key:

```bash
export WANDB_API_KEY=your_api_key_here
```

If you prefer to avoid network calls, you can log offline:

```bash
export WANDB_MODE=offline
```

## Run

Edit `src/main.py` to set `use_wandb=True` (and `wandb_mode="offline"` if desired), then run:

```bash
python src/main.py
```

Artifacts (best model) and metrics will appear under `runs/` and, if W&B is enabled, in your W&B project.

## Single image inference

Use the CLI in `src/inference.py` to generate an annotated image (bounding boxes/masks) from a trained model.

1) Auto-derived output directory (saves to `<runs>/<project>/<exp>_inference/image/`):

```bash
python src/inference.py \
  --weights /Users/takato/proj-crack_seg/runs/crack_seg/exp01/weights/best.pt \
  --image /Users/takato/proj-crack_seg/datasets/kanazawa/1_165711-165712_frame_2408.png
```

2) Explicit output directory (still saves inside `<out-dir>/image/`):

```bash
python src/inference.py \
  --weights /Users/takato/proj-crack_seg/runs/crack_seg/exp01/weights/best.pt \
  --image /Users/takato/proj-crack_seg/datasets/kanazawa/1_165711-165712_frame_2408.png \
  --out-dir /Users/takato/proj-crack_seg/runs/crack_seg/exp01_inference
```

Optional flags:
- `--conf 0.25` to adjust confidence threshold
- `--line-width 2` to change box/mask thickness

Output files will be written under the chosen directory’s `image/` folder.

## Notes
- W&B fields in `ExpConfig`:
  - `use_wandb`: enable/disable W&B logging
  - `wandb_project`: defaults to `project` if not provided
  - `wandb_entity`: optional team/org name
  - `wandb_run_name`: defaults to `name`
  - `wandb_mode`: e.g., `offline` to avoid network calls
- On first run, Ultralytics will create `runs/<project>/segment/<name>`.
