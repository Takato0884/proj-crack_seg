from __future__ import annotations
import os
import zipfile
import shutil
from pathlib import Path
from typing import Iterable, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}


def list_zip_files(base_dir: Path, keyword: str | None = None) -> list[Path]:
	"""Return zip files directly under base_dir; filter by keyword in filename when provided."""
	zips = [p for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() == ".zip"]
	if keyword:
		zips = [p for p in zips if keyword in p.name]
	return sorted(zips)


def safe_move(src: Path, dest_dir: Path) -> Tuple[Path, bool]:
	"""Move src into dest_dir; if name conflict exists, append numeric suffix. Returns (final_path, conflict)."""
	dest_dir.mkdir(parents=True, exist_ok=True)
	target = dest_dir / src.name
	conflict = False
	if target.exists():
		conflict = True
		base = target.stem
		ext = target.suffix
		i = 1
		while True:
			alt = dest_dir / f"{base}_{i}{ext}"
			if not alt.exists():
				target = alt
				break
			i += 1
	shutil.move(str(src), str(target))
	return target, conflict


def extract_and_collect(base_dir: Path, zip_paths: Iterable[Path]) -> list[Path]:
	"""Extract each zip under a folder named by its stem. Return list of created/extracted directories."""
	extracted_dirs: list[Path] = []
	for z in zip_paths:
		dest_dir = base_dir / z.stem
		dest_dir.mkdir(exist_ok=True)
		with zipfile.ZipFile(z, 'r') as zip_ref:
			zip_ref.extractall(dest_dir)
		extracted_dirs.append(dest_dir)
	return extracted_dirs


def move_images_to(images_root: Path, from_dirs: Iterable[Path]) -> Tuple[int, int, int]:
	"""Walk from_dirs, move images into images_root. Returns (moved_count, skipped_count, conflict_count)."""
	moved = 0
	skipped = 0
	conflicts = 0
	for d in from_dirs:
		for root, dirs, files in os.walk(d):
			for fname in files:
				src = Path(root) / fname
				if src.suffix.lower() in IMAGE_EXTS:
					try:
						_, conflicted = safe_move(src, images_root)
						moved += 1
						conflicts += int(conflicted)
					except Exception:
						# continue on errors
						pass
				else:
					skipped += 1
	return moved, skipped, conflicts


def cleanup_zips(zip_paths: Iterable[Path]) -> int:
	"""Delete zip files. Returns count removed."""
	removed = 0
	for z in zip_paths:
		try:
			z.unlink()
			removed += 1
		except Exception:
			pass
	return removed


def cleanup_empty_dirs(dirs: Iterable[Path]) -> int:
	"""Attempt to remove empty directories in the list (and their empty subdirs). Returns count removed."""
	removed = 0
	for d in dirs:
		try:
			for root, subdirs, files in os.walk(d, topdown=False):
				for sub in subdirs:
					p = Path(root) / sub
					try:
						p.rmdir()
					except OSError:
						pass
			d.rmdir()
			removed += 1
		except OSError:
			# not empty; leave it
			pass
	return removed


def process_kanazawa_zips(
	base_dir: str | Path,
	images_dir: str | Path,
	keyword: str = "クラック",
	delete_zips: bool = False,
	remove_empty_dirs: bool = False,
) -> dict:
	"""
	Extract zip files under base_dir whose names contain keyword, then move all images into images_dir.

	Returns a summary dict with counts.
	"""
	base = Path(base_dir).expanduser().resolve()
	images_root = Path(images_dir).expanduser().resolve()
	assert base.exists(), f"Directory not found: {base}"
	images_root.mkdir(parents=True, exist_ok=True)

	zips = list_zip_files(base, keyword=keyword)
	extracted_dirs = extract_and_collect(base, zips)
	moved, skipped, conflicts = move_images_to(images_root, extracted_dirs)

	zip_removed = cleanup_zips(zips) if delete_zips else 0
	dirs_removed = cleanup_empty_dirs(extracted_dirs) if remove_empty_dirs else 0

	return {
		"zip_count": len(zips),
		"extracted_dir_count": len(extracted_dirs),
		"moved_images": moved,
		"skipped_non_images": skipped,
		"name_conflicts": conflicts,
		"zip_removed": zip_removed,
		"dirs_removed": dirs_removed,
	}


def split_images_by_video(
	image_root: str | Path,
	ratio: tuple[int, int, int] = (6, 2, 2),
	seed: int | None = 42,
	copy: bool = True,
) -> dict:
	"""
	Split images in image_root into train/val/test by unique video IDs inferred from filename pattern:
	"<camera>_<start>-<end>_frame_<n>.<ext>". All images sharing the same prefix before "_frame" belong to one video.

	- ratio: (train, val, test) weights. Defaults to 6:2:2.
	- seed: random seed for reproducibility (None to disable shuffling).
	- copy: when True copy files; when False move files.

	Creates subdirs train/, val/, test/ under image_root and copies/moves grouped images so groups do not overlap.
	Returns counts summary.
	"""
	import re
	import random

	root = Path(image_root).expanduser().resolve()
	assert root.exists(), f"Image directory not found: {root}"

	# Collect images (only files directly under root)
	images = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
	if not images:
		return {"groups": 0, "train": 0, "val": 0, "test": 0}

	# Extract group key: prefix up to "_frame"
	pat = re.compile(r"^(?P<prefix>.+?)_frame_\d+\.[^.]+$", re.IGNORECASE)
	groups: dict[str, list[Path]] = {}
	for p in images:
		m = pat.match(p.name)
		if not m:
			# If doesn't match, use stem as its own group to avoid mixing
			key = p.stem
		else:
			key = m.group("prefix")
		groups.setdefault(key, []).append(p)

	unique_keys = list(groups.keys())
	if seed is not None:
		random.Random(seed).shuffle(unique_keys)

	# Compute split indices by ratio weights
	train_w, val_w, test_w = ratio
	total_w = train_w + val_w + test_w
	assert total_w > 0, "Invalid ratio"

	n = len(unique_keys)
	train_n = int(round(n * train_w / total_w))
	val_n = int(round(n * val_w / total_w))
	# ensure totals sum to n
	test_n = max(0, n - train_n - val_n)

	train_keys = set(unique_keys[:train_n])
	val_keys = set(unique_keys[train_n:train_n + val_n])
	test_keys = set(unique_keys[train_n + val_n:train_n + val_n + test_n])

	# Prepare destination dirs
	d_train = root / "train"
	d_val = root / "val"
	d_test = root / "test"
	for d in (d_train, d_val, d_test):
		d.mkdir(exist_ok=True)

	# Move or copy files for each split
	import shutil as _shutil

	def _place(files: list[Path], dest: Path) -> int:
		count = 0
		for src in files:
			# skip if src is already inside a split dir
			if src.parent == d_train or src.parent == d_val or src.parent == d_test:
				continue
			target = dest / src.name
			try:
				if copy:
					_shutil.copy2(str(src), str(target))
				else:
					_shutil.move(str(src), str(target))
				count += 1
			except Exception:
				pass
		return count

	train_files = sum((_place(groups[k], d_train) for k in train_keys), 0)
	val_files = sum((_place(groups[k], d_val) for k in val_keys), 0)
	test_files = sum((_place(groups[k], d_test) for k in test_keys), 0)

	return {
		"groups": n,
		"train_groups": len(train_keys),
		"val_groups": len(val_keys),
		"test_groups": len(test_keys),
		"train": train_files,
		"val": val_files,
		"test": test_files,
	}


def generate_labels_from_image_splits(
	labels_root: str | Path,
	image_root: str | Path,
	xml_path: str | Path | None = None,
	allowed_exts: set[str] | None = None,
) -> dict:
	"""
	Create labels/train|val|test folders under labels_root and for each image found in
	image_root/train|val|test, create a corresponding .txt file (same stem, .txt extension).

	- allowed_exts: set of image extensions to consider; defaults to IMAGE_EXTS.
	Returns a dict of counts per split.
	"""
	labels_base = Path(labels_root).expanduser().resolve()
	images_base = Path(image_root).expanduser().resolve()
	assert images_base.exists(), f"Image root not found: {images_base}"
	labels_base.mkdir(parents=True, exist_ok=True)
	# Optional: parse XML masks into a mapping filename -> list of RLE sequences (as floats)
	polygons_map: dict[str, list[list[float]]] = {}
	if xml_path is not None:
		import xml.etree.ElementTree as ET
		xml_file = Path(xml_path).expanduser().resolve()
		try:
			tree = ET.parse(str(xml_file))
			root = tree.getroot()
			for img in root.iter():
				fname = None
				for attr in ("name", "file_name", "filename", "FileName"):
					val = img.attrib.get(attr)
					if val:
						fname = Path(val).name
						break
				if not fname:
					continue
				plist: list[list[float]] = []
				# Extract raw RLE number sequences from <mask rle="...">
				for mask in img.findall(".//mask"):
					rle = mask.attrib.get("rle")
					if rle:
						try:
							nums = []
							for token in rle.split(","):
								token = token.strip()
								if token:
									nums.append(float(token))
							if len(nums) >= 2:
								plist.append(nums)
						except Exception:
							pass
				if plist:
					polygons_map.setdefault(fname, []).extend(plist)
		except Exception:
			polygons_map = {}

	exts = allowed_exts or IMAGE_EXTS

	def _prepare(split: str) -> tuple[Path, Path]:
		# Write labels directly under labels_root/split
		lbl_dir = labels_base / split
		lbl_dir.mkdir(parents=True, exist_ok=True)
		# Read images directly under image_root/split
		img_dir = images_base / split
		assert img_dir.exists(), f"Images for split '{split}' not found: {img_dir}"
		return lbl_dir, img_dir

	def _generate(lbl_dir: Path, img_dir: Path) -> int:
		count = 0
		for p in sorted(img_dir.iterdir()):
			if p.is_file() and p.suffix.lower() in exts:
				txt_path = lbl_dir / (p.stem + ".txt")
				# write RLE sequences if available; else create empty
				polys = polygons_map.get(p.name) or polygons_map.get(p.stem) or []
				with open(txt_path, "w", encoding="utf-8") as f:
					if polys:
						for poly in polys:
							# Write class id followed by raw RLE numbers (space-separated)
							line = "0 " + " ".join(f"{int(v)}" if float(v).is_integer() else f"{v:.6f}" for v in poly)
							f.write(line + "\n")
					else:
						# empty file to indicate no annotation
						f.write("")
				count += 1
		return count

	res = {}
	for split in ("train", "val", "test"):
		lbl_dir, img_dir = _prepare(split)
		res[split] = _generate(lbl_dir, img_dir)

	return res


def summarize_segmentation_labels(labels_root: str | Path) -> dict:
	"""
	Summarize how many samples contain segmentation annotations under
	labels_root/train|val|test. A sample is counted as "with_annotations"
	if its corresponding .txt file contains at least one non-empty line.

	Returns a dict with per-split and overall counts:
	{
	  'train': {total, with_annotations, without_annotations, percent_with},
	  'val': {..},
	  'test': {..},
	  'overall': {total, with_annotations, without_annotations, percent_with}
	}
	"""
	base = Path(labels_root).expanduser().resolve()
	splits = ("train", "val", "test")
	result: dict[str, dict[str, float | int]] = {}

	grand_total = 0
	grand_with = 0

	for s in splits:
		d = base / s
		total = 0
		with_ann = 0
		if d.exists():
			for p in sorted(d.glob("*.txt")):
				total += 1
				has = False
				try:
					with open(p, "r", encoding="utf-8", errors="ignore") as f:
						for line in f:
							if line.strip():
								has = True
								break
				except Exception:
					pass
				with_ann += int(has)
		without = total - with_ann
		percent = (with_ann / total * 100.0) if total else 0.0
		result[s] = {
			"total": total,
			"with_annotations": with_ann,
			"without_annotations": without,
			"percent_with": round(percent, 2),
		}
		grand_total += total
		grand_with += with_ann

	grand_without = grand_total - grand_with
	grand_percent = (grand_with / grand_total * 100.0) if grand_total else 0.0
	result["overall"] = {
		"total": grand_total,
		"with_annotations": grand_with,
		"without_annotations": grand_without,
		"percent_with": round(grand_percent, 2),
	}

	return result


def convert_coco_to_yolo(
	annotations_dir: str | Path | None = None,
	annotations_file: str | Path | None = None,
	use_segments: bool = False,
	verbose: bool = False,
	images_root: str | Path | None = None,
) -> dict:
	"""
	Convert COCO annotations (instances_*.json) to YOLOv8 segmentation labels.

	- annotations_dir: directory containing COCO JSON annotations (e.g., datasets/subset_kanazawa)
	- annotations_file: specific COCO JSON file (e.g., instances_default.json)
	- use_segments: use segmentation polygons when provided in COCO
	- verbose: print additional logs

	Returns a summary dict including paths and counters.
	"""
	import json
	from collections import defaultdict
	from typing import Any

	try:
		import numpy as np  # type: ignore
		import cv2  # type: ignore
		from pycocotools import mask as maskUtils  # type: ignore
	except Exception as e:
		raise RuntimeError(
			"pycocotools and opencv-python are required for RLE conversion. Please install dependencies from requirements.txt"
		) from e

	# Resolve paths
	ann_file: Path | None = None
	ann_dir: Path
	if annotations_file:
		ann_file = Path(annotations_file).expanduser().resolve()
		ann_dir = ann_file.parent
	elif annotations_dir:
		ann_dir = Path(annotations_dir).expanduser().resolve()
		# Try to find a single instances_*.json in the directory
		candidates = sorted([p for p in ann_dir.glob("instances*.json")])
		if not candidates:
			raise FileNotFoundError(f"No instances*.json found in {ann_dir}")
		ann_file = candidates[0]
	else:
		raise ValueError("Provide annotations_file or annotations_dir for convert_coco_to_yolo")

	assert ann_file and ann_file.exists(), f"Annotation JSON not found: {ann_file}"

	# Output labels directory: create 'labels' sibling folder under the annotations dir
	out_labels_dir = ann_dir / "labels"
	out_labels_dir.mkdir(parents=True, exist_ok=True)
	# Prepare split mapping if images_root is provided
	split_names = ("train", "val", "test")
	file_to_split: dict[str, str] = {}
	if images_root is not None:
		img_root = Path(images_root).expanduser().resolve()
		if not img_root.exists():
			raise FileNotFoundError(f"Images root not found: {img_root}")
		# Pre-create split subfolders under labels
		for s in split_names:
			(out_labels_dir / s).mkdir(parents=True, exist_ok=True)
		# Build filename -> split mapping by scanning images_root/train|val|test
		for s in split_names:
			split_dir = img_root / s
			if not split_dir.exists():
				continue
			for root, dirs, files in os.walk(split_dir):
				for fname in files:
					# first-come mapping; warn on conflicts
					if fname in file_to_split and file_to_split[fname] != s:
						if verbose:
							print(f"Warning: filename '{fname}' found in multiple splits ({file_to_split[fname]} and {s}); keeping first")
						continue
					file_to_split.setdefault(fname, s)

	if verbose:
		print(f"Loading COCO annotations: {ann_file}")

	with open(ann_file, "r", encoding="utf-8") as f:
		data = json.load(f)

	# Build lookups
	images: dict[int, dict[str, Any]] = {}
	for img in data.get("images", []):
		images[img["id"]] = img

	categories: dict[int, dict[str, Any]] = {}
	for cat in data.get("categories", []):
		categories[cat["id"]] = cat

	anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
	for ann in data.get("annotations", []):
		anns_by_image[ann["image_id"]].append(ann)

	def rle_to_contours(seg: dict, img_h: int, img_w: int) -> list[np.ndarray]:
		# seg: {"counts": ..., "size": [h, w]} possibly encoded as bytes or str
		# Ensure RLE is in the expected format for pycocotools
		rle = {
			"counts": seg.get("counts"),
			"size": seg.get("size", [img_h, img_w]),
		}
		if isinstance(rle["counts"], list):
			# Uncompressed RLE (rare) – convert to compressed
			rle = maskUtils.frPyObjects(rle, rle["size"][0], rle["size"][1])
		m = maskUtils.decode(rle)  # (H, W, 1) or (H, W)
		if m.ndim == 3:
			m = m[:, :, 0]
		m = (m > 0).astype(np.uint8) * 255
		# Find contours
		contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		return contours

	def contour_to_normalized_points(cnt: np.ndarray, img_h: int, img_w: int) -> list[tuple[float, float]]:
		# cnt shape: (N, 1, 2)
		pts = cnt.reshape(-1, 2)
		# Optionally approximate to reduce points
		eps = 0.002 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2) if len(pts) > 50 else pts
		# Normalize to [0,1]
		return [(float(x) / img_w, float(y) / img_h) for (x, y) in approx]

	written = 0
	skipped = 0
	for img_id, img in images.items():
		file_name = img.get("file_name") or img.get("filename") or img.get("name")
		if not file_name:
			skipped += 1
			continue
		stem = Path(file_name).stem
		img_w = int(img.get("width") or (img.get("w") or (img.get("size", [0, 0])[1])))
		img_h = int(img.get("height") or (img.get("h") or (img.get("size", [0, 0])[0])))
		if not img_w or not img_h:
			# fallback from any associated annotation RLE size later
			pass

		lines: list[str] = []
		for ann in anns_by_image.get(img_id, []):
			cat_id = int(ann.get("category_id", 0))
			cls = 0  # default single class
			# If multiple categories exist, map to index by sorted cat IDs
			if len(categories) > 1:
				cls = sorted(categories.keys()).index(cat_id)

			seg = ann.get("segmentation")
			if seg is None:
				continue

			try:
				if isinstance(seg, dict) and "counts" in seg and "size" in seg:
					# RLE path
					if not img_h or not img_w:
						img_h, img_w = int(seg["size"][0]), int(seg["size"][1])
					contours = rle_to_contours(seg, img_h, img_w)
					for cnt in contours:
						pts = contour_to_normalized_points(cnt, img_h, img_w)
						if len(pts) < 3:
							continue
						# YOLOv8 segmentation: class followed by x y pairs
						line = str(cls) + " " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
						lines.append(line)
				elif isinstance(seg, list) and seg and isinstance(seg[0], list):
					# Polygon path already provided: [[x1,y1,...]]
					# Use as-is if use_segments requested; otherwise skip
					if use_segments:
						for poly in seg:
							coords = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
							if not (img_w and img_h):
								# cannot normalize without size
								continue
							pts = [(float(x) / img_w, float(y) / img_h) for (x, y) in coords]
							if len(pts) < 3:
								continue
							line = str(cls) + " " + " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
							lines.append(line)
					else:
						# user did not request segments; skip
						pass
				else:
					# Unsupported segmentation format
					continue
			except Exception as e:
				if verbose:
					print(f"Skip ann {ann.get('id')} for {file_name}: {e}")
				continue

		# Decide destination by split mapping when available
		dest_dir = out_labels_dir
		if images_root is not None:
			split = file_to_split.get(Path(file_name).name)
			if split in split_names:
				dest_dir = out_labels_dir / split
			elif verbose:
				print(f"Info: '{file_name}' not found under {images_root}/train|val|test; writing to labels root")

		# Write label file (empty allowed)
		out_txt = dest_dir / f"{stem}.txt"
		with open(out_txt, "w", encoding="utf-8") as f:
			f.write("\n".join(lines))
		written += 1

	summary = {
		"annotations_json": str(ann_file),
		"labels_dir": str(out_labels_dir),
		"images_processed": written,
		"images_skipped": skipped,
	}

	print(
		f"COCO RLE→YOLO conversion completed. JSON: {ann_file}, labels: {out_labels_dir}, images processed: {written}, skipped: {skipped}"
	)
	return summary



if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Preprocessing utilities")
	sub = parser.add_subparsers(dest="cmd", required=True)

	# Subcommand: kanazawa zips
	p_z = sub.add_parser("kanazawa-zips", help="Extract zips and move images")
	p_z.add_argument("--base-dir", required=True, help="Base directory containing zip files (e.g., datasets/kanazawa)")
	p_z.add_argument("--images-dir", required=True, help="Target images directory")
	p_z.add_argument("--keyword", default="クラック", help="Keyword to filter zip filenames")
	p_z.add_argument("--delete-zips", action="store_true", help="Delete zip files after extraction")
	p_z.add_argument("--remove-empty-dirs", action="store_true", help="Remove empty extracted directories")

	# (Removed COCO-related subcommands)

	# Subcommand: split by video groups
	p_split = sub.add_parser("split-by-video", help="Split images into train/val/test by unique video groups")
	p_split.add_argument("--image-root", required=True, help="Root directory containing images (files directly under this)")
	p_split.add_argument("--ratio", nargs=3, type=int, default=(6, 2, 2), metavar=("TRAIN", "VAL", "TEST"), help="Split ratio, e.g., 6 2 2")
	p_split.add_argument("--seed", type=int, default=42, help="Random seed for shuffling groups")
	p_split.add_argument("--move", action="store_true", help="Move files instead of copying")

	# Subcommand: labels from image splits
	p_lfi = sub.add_parser("labels-from-images", help="Create labels/train|val|test and txt files by mirroring image filenames (images directly under split), optionally filling from annotations.xml")
	p_lfi.add_argument("--labels-root", required=True, help="Output labels root (train/val/test will be created)")
	p_lfi.add_argument("--image-root", required=True, help="Root containing train/val/test image folders (images directly under these folders)")
	p_lfi.add_argument("--xml", help="Path to annotations XML to populate polygons into txt files")

	# Subcommand: COCO -> YOLO conversion (using ultralytics)
	p_c2y = sub.add_parser("coco-to-yolo", help="Convert COCO annotations (instances_*.json) to YOLO format using ultralytics")
	p_c2y.add_argument("--annotations-dir", help="Directory containing COCO JSON annotations (e.g., datasets/subset_kanazawa)")
	p_c2y.add_argument("--annotations-file", help="Specific COCO JSON file (e.g., instances_default.json)")
	p_c2y.add_argument("--use-segments", action="store_true", help="Use segmentation polygons for YOLO labels")
	p_c2y.add_argument("--verbose", action="store_true", help="Print additional conversion logs")
	p_c2y.add_argument("--images-root", help="Images root containing train/val/test folders to route labels accordingly (e.g., datasets/crack-seg/images)")

	# Subcommand: summarize segmentation labels
	p_sum = sub.add_parser("summary-seg-labels", help="Summarize counts of samples with segmentation annotations under labels/train|val|test")
	p_sum.add_argument("--labels-root", required=True, help="Root directory containing labels/train|val|test")


	# (Removed Laplacian stats subcommand)

	args = parser.parse_args()

	if args.cmd == "kanazawa-zips":
		summary = process_kanazawa_zips(
			base_dir=args.base_dir,
			images_dir=args.images_dir,
			keyword=args.keyword,
			delete_zips=args.delete_zips,
			remove_empty_dirs=args.remove_empty_dirs,
		)
		print("Summary:", summary)
	# (Removed COCO-related command branches)
	elif args.cmd == "split-by-video":
		train, val, test = tuple(args.ratio)
		res = split_images_by_video(
			image_root=args.image_root,
			ratio=(train, val, test),
			seed=args.seed,
			copy=(not args.move),
		)
		print("Summary:", res)
	elif args.cmd == "labels-from-images":
		res = generate_labels_from_image_splits(labels_root=args.labels_root, image_root=args.image_root, xml_path=args.xml)
		print("Labels generated:", res)
	elif args.cmd == "coco-to-yolo":
			# Call the refactored converter
			summary = convert_coco_to_yolo(
				annotations_dir=args.annotations_dir,
				annotations_file=args.annotations_file,
				use_segments=args.use_segments,
				verbose=args.verbose,
				images_root=args.images_root,
			)
			print("Summary:", summary)
	elif args.cmd == "summary-seg-labels":
		res = summarize_segmentation_labels(labels_root=args.labels_root)
		print("Summary:", res)
