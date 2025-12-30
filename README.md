## Commit/Message Prefix Rules

Use the following prefixes in commit messages, PR titles, or change logs to make intent clear:

### Prefix List

- feat: Add new features or modules (e.g., new model, function, CLI)
- fix: Fix bugs or issues
- refactor: Reorganize internal structure or code (without changing behavior)
- exp: Add or update experiment-related files (changes under `experiments/`)
- data: Add or update data files (e.g., `data/`, `processed/`, etc.)
- docs: Documentation updates (e.g., README, comments, reports)
- conf: Modify configuration files (e.g., `configs/`, environment settings)
- chore: Miscellaneous tasks (e.g., dependency updates, `.gitignore` fixes)
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

## Inference

Use the CLI in `src/inference.py` to generate annotated images (bounding boxes/masks) from a trained model.

1) Single image (auto-derived output directory; saves to `<runs>/<project>/<exp>_inference/image/>`):

```bash
python src/inference.py \
  --weights /home/hayashi0884/proj-crack_seg/runs/crack_seg/exp01/weights/best.pt \
  --image /home/hayashi0884/proj-crack_seg/datasets/kanazawa/images/13_166356-165133_frame_1697.png
```

Optional flags:
- `--conf 0.25` to adjust confidence threshold
- `--line-width 2` to change box/mask thickness

Output files will be written under the chosen directory’s `image/` folder.

2) Directory of images (non-recursive; processes all images directly under the folder):

```bash
python src/inference.py \
  --weights /home/hayashi0884/proj-crack_seg/runs/crack_seg/exp01/weights/best.pt \
  --image /home/hayashi0884/proj-crack_seg/datasets/kanazawa/images
```

Notes:
- Supported extensions: .jpg, .jpeg, .png, .bmp, .tif, .tiff
- The script does not search subdirectories (non-recursive). If you need recursion, let me know and I’ll extend it.
- For each image, the script prints the number of detections, e.g.,
  - `No detections for <filename>`
  - `Detections for <filename>: N`

3) Save outputs to a custom directory (recommended)

When you want to store all annotated images directly under a specific folder (no extra nested subfolders), use `--out-dir` and `--out-name`:

```bash
python src/inference.py \
  --weights /home/hayashi0884/proj-crack_seg/runs/crack_seg/exp01/weights/best.pt \
  --image /home/hayashi0884/proj-crack_seg/datasets/kanazawa/images \
  --out-dir /home/hayashi0884/proj-crack_seg/output \
  --out-name exp01
```

Behavior:
- Outputs are saved directly in `/home/hayashi0884/proj-crack_seg/output/exp01/`.
- If `--out-name` is omitted, it defaults to `inference` (e.g., `/output/inference/`).

## Notes
- W&B fields in `ExpConfig`:
  - `use_wandb`: enable/disable W&B logging
  - `wandb_project`: defaults to `project` if not provided
  - `wandb_entity`: optional team/org name
  - `wandb_run_name`: defaults to `name`
  - `wandb_mode`: e.g., `offline` to avoid network calls
- On first run, Ultralytics will create `runs/<project>/segment/<name>`.
