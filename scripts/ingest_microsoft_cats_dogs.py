"""
scripts/ingest_microsoft_cats_dogs.py
=====================================
Milestone 7.1–7.3: Download, clean, split, and register the Microsoft
Cats vs Dogs dataset into the app's internal storage.

Steps:
  1. Download via kagglehub (uses cached copy if already present).
  2. Sanitise: remove non-JPG/JPEG files; move corrupt images to storage/trash/.
  3. Copy cleaned images into storage/datasets/microsoft_cats_dogs/raw/<class>/.
  4. Split 80/10/10 using DatasetTransformer.split_train_val_test().
  5. Run MetadataManager.generate_metadata() and print a summary.

Usage:
    cd /home/clayton/workspace/git/classifier_app
    source venv/bin/activate
    python scripts/ingest_microsoft_cats_dogs.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import kagglehub
from PIL import Image, UnidentifiedImageError

from vision_app.core.data_manager import DatasetBuilder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
KAGGLE_HANDLE = "shaunthesheep/microsoft-catsvsdogs-dataset"
DATASET_NAME  = "microsoft_cats_dogs"
STORAGE_ROOT  = PROJECT_ROOT / "storage"
SPLIT_RATIOS  = (0.8, 0.1, 0.1)

# PetImages contains Cat/ and Dog/ directly.
SOURCE_CLASSES = {"Cat": "Cat", "Dog": "Dog"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg"}


def _copy_and_clean(
    src_class_dir: Path,
    dest_class_dir: Path,
    trash_dir: Path,
    class_name: str,
) -> dict:
    """
    Iterate src_class_dir, skip non-JPG files, move corrupt images to trash/,
    copy valid images to dest_class_dir.

    Returns a summary dict.
    """
    dest_class_dir.mkdir(parents=True, exist_ok=True)
    trash_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "copied": 0, "skipped_type": 0, "corrupt": 0}

    for f in sorted(src_class_dir.iterdir()):
        stats["total"] += 1

        # 7.2a — skip non-image files (e.g. Thumbs.db, *.txt)
        if not _is_image_file(f):
            print(f"  [skip]    {f.name}  (not a JPG)")
            stats["skipped_type"] += 1
            continue

        # 7.2b — PIL corruption check
        try:
            with Image.open(f) as img:
                img.verify()
        except (UnidentifiedImageError, Exception) as exc:
            print(f"  [corrupt] {f.name}  ({exc})")
            shutil.move(str(f), str(trash_dir / f"{class_name}_{f.name}"))
            stats["corrupt"] += 1
            continue

        # Valid — copy to raw/<class>/
        dest = dest_class_dir / f.name
        if not dest.exists():
            shutil.copy2(str(f), str(dest))
        stats["copied"] += 1

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Milestone 7 — Microsoft Cats vs Dogs Ingestion")
    print("=" * 60)

    # 7.1 — Download (returns cache path; skips re-download if already cached)
    print("\n[7.1] Resolving dataset via kagglehub…")
    download_path = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
    pet_images_path = download_path / "PetImages"
    print(f"      Source: {pet_images_path}")

    if not pet_images_path.exists():
        print("ERROR: PetImages/ not found in downloaded path. Aborting.")
        sys.exit(1)

    builder = DatasetBuilder(STORAGE_ROOT)
    dataset_path = builder.datasets_root / DATASET_NAME
    # Raw images live in raw/<class>/ — split_train_val_test() expects this layout
    raw_path  = dataset_path / "raw"
    trash_dir = STORAGE_ROOT / "trash" / DATASET_NAME

    # Skip copy phase if already done
    already_copied = all(
        (raw_path / cls).exists() and any((raw_path / cls).iterdir())
        for cls in SOURCE_CLASSES.values()
    )
    if already_copied:
        print("\n[7.2] Raw images already present — skipping copy/clean phase.")
    else:
        print(f"\n[7.2] Cleaning and copying into {raw_path} …")

    total_stats: dict[str, dict] = {}

    for src_name, dest_name in SOURCE_CLASSES.items():
        src_dir  = pet_images_path / src_name
        dest_dir = raw_path / dest_name

        if not src_dir.exists():
            print(f"  WARNING: {src_dir} not found, skipping.")
            continue

        if not already_copied:
            print(f"\n  Processing '{src_name}' → '{dest_name}' …")
            stats = _copy_and_clean(src_dir, dest_dir, trash_dir, dest_name)
            total_stats[dest_name] = stats
            print(
                f"    Total: {stats['total']}  |  Copied: {stats['copied']}  |  "
                f"Skipped (type): {stats['skipped_type']}  |  Corrupt: {stats['corrupt']}"
            )
        else:
            count = sum(1 for _ in dest_dir.iterdir() if _.is_file())
            total_stats[dest_name] = {"copied": count}
            print(f"  {dest_name}: {count} images already in {dest_dir.name}/")

    # 7.3a — Split into train/val/test if not already split
    train_path = dataset_path / "train"
    if train_path.exists() and any(train_path.iterdir()):
        print("\n[7.3] Split already exists — skipping split step.")
    else:
        print(f"\n[7.3] Splitting {SPLIT_RATIOS} → train/val/test …")
        builder.dataset_transformer.split_train_val_test(raw_path, ratios=SPLIT_RATIOS)
        print("      Split complete.")

    # 7.3b — Generate metadata
    print(f"\n[7.3] Generating metadata for '{DATASET_NAME}' …")
    metadata = builder.metadata_manager.generate_metadata(dataset_path)

    print("\n  Metadata summary:")
    print(f"    Dataset name  : {metadata.get('name')}")
    print(f"    Num classes   : {metadata.get('num_classes')}")
    print(f"    Classes       : {metadata.get('classes')}")
    print(f"    Total images  : {metadata.get('total_images')}")

    dist = metadata.get("split_distribution", {})
    for split, class_counts in dist.items():
        total = sum(class_counts.values())
        print(f"    {split:5s}         : {total} images  {dict(class_counts)}")

    # 7.3c — Class balance check
    print("\n[7.3] Running class balance validation …")
    result = builder.dataset_validator.check_class_balance(dataset_path, split="train")
    for issue in result.get("warnings", []):
        print(f"  WARNING: {issue}")
    if not result.get("warnings"):
        print("  Class balance OK.")

    print("\n" + "=" * 60)
    print("Ingestion complete.")
    print(f"  Dataset: {dataset_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
