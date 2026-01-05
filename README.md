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

For crack-seg dataset:
```bash
python src/main.py \
	--data-path datasets/crack-seg/crack-seg.yaml \
	--pretrained-model weight/yolo11n-seg.pt \
	--runs-dir runs \
	--project crack_seg \
	--name exp01 \
	--epochs 200 \
	--batch 16 \
	--imgsz 640 \
	--use-wandb \
	--wandb-mode online
```

For subset_kanazawa dataset:
```bash
python src/main.py \
	--data-path datasets/subset_kanazawa/data.yaml \
	--pretrained-model weight/yolo11n-seg.pt \
	--runs-dir runs \
	--project subset_kanazawa \
	--name yolo_seg-11n \
	--epochs 200 \
	--batch 16 \
	--imgsz 640 \
	--use-wandb \
	--wandb-mode online
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

### Segmentation label summary (`src/preprocessing.py`)

Summarize how many label files contain segmentation annotations (non-empty .txt) under `labels/train|val|test`:

```bash
python src/preprocessing.py summary-seg-labels \
	--labels-root /home/hayashi0884/proj-crack_seg/datasets/subset_kanazawa/labels
```

Output fields per split and overall:
- total: number of .txt files in the split
- with_annotations: files that contain at least one non-empty line
- without_annotations: empty files
- percent_with: ratio of annotated files

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