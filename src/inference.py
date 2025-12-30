from __future__ import annotations

from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def infer_and_save(
	weights_path: str,
	image_path: str,
	out_dir: Optional[str] = None,
	line_width: int = 2,
	conf: float = 0.25,
) -> str:
	"""
	Run inference on a single image using a trained Ultralytics YOLO model and
	save an annotated image (with bounding boxes/masks) to the requested folder.

	Inputs:
	- weights_path: Absolute path to best.pt
	- image_path: Absolute path to an input image
	- out_dir: Output directory where the annotated image will be saved.
			   If None, it's derived from the weights path as '<exp>_inference'.
	- line_width: Line width for the drawn boxes/masks.
	- conf: Confidence threshold.

	Returns the folder path containing the saved image.
	"""

	w = Path(weights_path).resolve()
	img = Path(image_path).resolve()

	if not w.exists():
		raise FileNotFoundError(f"weights not found: {w}")
	if not img.exists():
		raise FileNotFoundError(f"image not found: {img}")

	# Derive default output directory from the weights path if not provided
	if out_dir is None:
		# e.g. /.../runs/crack_seg/exp01/weights/best.pt -> /.../runs/crack_seg/exp01_inference
		exp_dir = w.parent.parent  # .../exp01
		out_root = exp_dir.parent  # .../runs/crack_seg
		out_dir_path = out_root / f"{exp_dir.name}_inference"
	else:
		out_dir_path = Path(out_dir)

	# We want images inside '<out_dir>/image/' as requested
	project_dir = out_dir_path
	name_dir = "image"

	project_dir.mkdir(parents=True, exist_ok=True)

	# Load model and run prediction
	model = YOLO(str(w))

	# Ultralytics will save into {project}/{name} when save=True
	# For segmentation models, masks will be drawn; boxes shown as well.
	model.predict(
		source=str(img),
		save=True,
		save_conf=True,
		conf=conf,
		project=str(project_dir),
		name=name_dir,
		exist_ok=True,
		line_width=line_width,
		imgsz=640,
		verbose=False,
	)

	return str(project_dir / name_dir)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Run YOLO inference and save annotated image")
	parser.add_argument("--weights", required=True, help="Path to best.pt")
	parser.add_argument("--image", required=True, help="Path to input image")
	parser.add_argument(
		"--out-dir",
		required=False,
		default=None,
		help="Output directory (will save into <out-dir>/image/). If omitted, derived from weights.",
	)
	parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
	parser.add_argument("--line-width", type=int, default=2, help="Line width for boxes/masks")

	args = parser.parse_args()

	saved_dir = infer_and_save(
		weights_path=args.weights,
		image_path=args.image,
		out_dir=args.out_dir,
		line_width=args.line_width,
		conf=args.conf,
	)
	print(f"Saved annotated image(s) to: {saved_dir}")

