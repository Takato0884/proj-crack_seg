from __future__ import annotations
from ultralytics import YOLO
import argparse
import os
from pathlib import Path


def is_image_file(path: Path) -> bool:
	return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
	parser = argparse.ArgumentParser(description="Run YOLOv8 inference on an image or a directory of images.")
	parser.add_argument("--weights", type=str, required=True, help="Path to the model weights (e.g., best.pt)")
	parser.add_argument("--image", type=str, required=True, help="Path to an image file or a directory (non-recursive)")
	parser.add_argument("--out-dir", type=str, default=None, help="Output root directory (e.g., /home/.../output)")
	parser.add_argument("--out-name", type=str, default=None, help="Subfolder name to create under out-dir (e.g., exp01). If omitted, uses 'inference'.")
	args = parser.parse_args()

	model = YOLO(args.weights)

	src_path = Path(args.image)
	sources = []
	if src_path.is_dir():
		# Non-recursive: only files directly under the directory
		for p in sorted(src_path.iterdir()):
			if p.is_file() and is_image_file(p):
				sources.append(str(p))
	else:
		# Single file
		if not is_image_file(src_path):
			raise ValueError(f"Not a supported image file: {src_path}")
		sources.append(str(src_path))

	# Prepare custom output path when requested
	custom_save = args.out_dir is not None
	save_root = None
	if custom_save:
		save_root = Path(args.out_dir).expanduser().resolve()
		save_name = args.out_name if args.out_name else "inference"
		save_dir = save_root / save_name
		save_dir.mkdir(parents=True, exist_ok=True)

	# Run prediction: if custom saving, let us handle saving; else let Ultralytics save to its default
	results = model.predict(source=sources, save=not custom_save)

	# Print detections per image and optionally save outputs directly under save_dir
	for r in results:
		# Count detections: prefer masks if available, otherwise boxes
		det_count = 0
		try:
			if hasattr(r, "masks") and r.masks is not None and hasattr(r.masks, "data") and r.masks.data is not None:
				det_count = len(r.masks.data)
			elif hasattr(r, "boxes") and r.boxes is not None:
				det_count = len(r.boxes)
		except Exception:
			det_count = 0

		# Get filename
		im_path = r.path if hasattr(r, "path") else None
		filename = Path(im_path).name if im_path else "image.jpg"

		if det_count == 0:
			print(f"No detections for {filename}")
		else:
			print(f"Detections for {filename}: {det_count}")

		# Save annotated image when custom output is requested
		if custom_save:
			try:
				plotted = r.plot()  # returns a numpy array (BGR)
				out_path = save_dir / filename
				import cv2
				cv2.imwrite(str(out_path), plotted)
			except Exception as e:
				print(f"Failed to save {filename} to {save_dir}: {e}")


if __name__ == "__main__":
	main()

