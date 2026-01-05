## How to Run (Training & Inference)

This section explains how to run training (`src/main.py`) and inference (`src/inference.py`) in this repository. The environment assumes Linux and bash.

### Prerequisites

- Install dependencies

```bash
pip install -r requirements.txt
```

- Location of the dataset config YAML
	- `datasets/crack-seg/crack-seg.yaml`
- Location of pretrained models
	- Segmentation: `weight/yolo11n-seg.pt`

---

### Train + Test (`src/main.py`)

`src/main.py` trains with the given `data.yaml`, then evaluates the test split using the resulting `best.pt`. Outputs are saved under `runs/<project>/<segment>/<exp_name>/`.

Minimal example (using defaults):

```bash
python src/main.py \
	--data-path datasets/crack-seg/crack-seg.yaml \
	--pretrained-model weight/yolo11n-seg.pt \
	--runs-dir runs \
	--project crack_seg \
	--name exp01 \
	--epochs 200 \
	--batch 64 \
	--patience 10 \
	--seed 0 \
	--imgsz 640
```

Key arguments:
- `--data-path`: Dataset config (yaml), e.g., `datasets/crack-seg/crack-seg.yaml`
- `--pretrained-model`: Pretrained weights, e.g., `weight/yolo11n-seg.pt`
- `--runs-dir`: Root output dir, default `runs`
- `--project`: Project name under runs, default `crack_seg`
- `--name`: Experiment name, default `exp01`
- `--exist-ok`: Overwrite existing experiment directory if present
- `--epochs`, `--batch`, `--patience`, `--seed`, `--imgsz`: Ultralytics training settings
- `--use-wandb`: Enable Weights & Biases logging (optional)
- `--wandb-mode`: W&B mode (e.g., `offline`)

Typical outputs:
- Train directory: `runs/crack_seg/segment/exp01/`
- Weights: `runs/crack_seg/segment/exp01/weights/best.pt`
- Test evaluation: `runs/crack_seg/exp01_test/` (curves, confusion matrix, etc.)

---

### Inference (`src/inference.py`)

`src/inference.py` runs inference on a single image or all images directly under a directory (non-recursive). You can either let Ultralytics save to its default output location or save to a custom directory.

1) Save to Ultralytics default output (e.g., `runs/predict-seg/...`)

```bash
python src/inference.py \
	--weights runs/crack_seg/segment/exp01/weights/best.pt \
	--image datasets/crack-seg/images/test/1819.rf.d2d41865c85e1019dc3e8b9daf73c434.jpg
```

2) Save to a custom output directory (`--out-dir`)

```bash
python src/inference.py \
	--weights runs/crack_seg/segment/exp01/weights/best.pt \
	--image datasets/crack-seg/images/test \
	--out-dir runs/crack_seg \
	--out-name exp01_test_infer
```

The example above saves annotated images to `runs/crack_seg/exp01_test_infer/`.

Optional:
- `--save-polygons <path>`: Save predicted polygons in JSON Lines format (one JSON per image). Example:

```bash
python src/inference.py \
	--weights runs/crack_seg/segment/exp01/weights/best.pt \
	--image datasets/crack-seg/images/test \
	--out-dir runs/crack_seg \
	--out-name exp01_test_infer \
	--save-polygons runs/crack_seg/exp01_test_infer/polygons.jsonl
```

Notes:
- When a directory is provided, only the files directly under it are processed (non-recursive).
- Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`
- Saving uses OpenCV (`cv2`), which should be present via `requirements.txt`.

---

### Common paths

- Dataset config: `datasets/crack-seg/crack-seg.yaml`
- Pretrained weights: `weight/yolo11n-seg.pt`
- Training outputs: `runs/crack_seg/segment/<exp_name>/`
- Best weights: `runs/crack_seg/segment/<exp_name>/weights/best.pt`

Adjust paths/arguments as needed for your environment.

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