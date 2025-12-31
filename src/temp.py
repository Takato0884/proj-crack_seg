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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract kanazawa zips and move images to images dir")
    parser.add_argument("--base-dir", required=True, help="Base directory containing zip files (e.g., datasets/kanazawa)")
    parser.add_argument("--images-dir", required=True, help="Target images directory")
    parser.add_argument("--keyword", default="クラック", help="Keyword to filter zip filenames")
    parser.add_argument("--delete-zips", action="store_true", help="Delete zip files after extraction")
    parser.add_argument("--remove-empty-dirs", action="store_true", help="Remove empty extracted directories")

    args = parser.parse_args()

    summary = process_kanazawa_zips(
        base_dir=args.base_dir,
        images_dir=args.images_dir,
        keyword=args.keyword,
        delete_zips=args.delete_zips,
        remove_empty_dirs=args.remove_empty_dirs,
    )
    print("Summary:", summary)
