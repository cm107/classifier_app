"""
data_manager.py — Dataset creation, organization, and pruning logic.

Architecture:
    DatasetBuilder (parent)
        ├── FileSystemHandler   — disk I/O (mkdir, move, scan)
        ├── MetadataManager     — metadata.json "Source of Truth"
        ├── DatasetTransformer  — splitting, merging, pruning
        └── DatasetValidator    — corruption detection, class balance
"""

import hashlib
import json
import random
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

from vision_app.core.logger import log


# ---------------------------------------------------------------------------
# Constants shared across submodules
# ---------------------------------------------------------------------------
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# FileSystemHandler
# ---------------------------------------------------------------------------
class FileSystemHandler:
    """Abstract all os/shutil operations into a safe, high-level API."""

    def __init__(self, parent: "DatasetBuilder"):
        self._parent = parent

    def create_dataset_structure(self, name: str, classes: list) -> Path:
        """
        Create <datasets_root>/<name>/{train,val,test}/<class>/ directories.
        Returns the dataset root path.
        """
        try:
            dataset_path = self._parent.datasets_root / name
            log.verbose("FileSystemHandler", f"Creating dataset structure: {dataset_path}")
            for split in ("train", "val", "test"):
                for cls in classes:
                    (dataset_path / split / cls).mkdir(parents=True, exist_ok=True)
            log.info("FileSystemHandler", f"Dataset structure created: {name} with {len(classes)} classes")
            return dataset_path
        except Exception as e:
            log.exception("FileSystemHandler", f"Failed to create dataset structure: {e}")
            raise

    def move_sample(self, src_path: Path, dest_folder: Path) -> Path:
        """
        Move a single image file into dest_folder, avoiding filename collisions
        by appending _1, _2, … to the stem if needed.
        """
        try:
            src_path = Path(src_path)
            dest_folder = Path(dest_folder)
            dest_folder.mkdir(parents=True, exist_ok=True)

            dest_path = dest_folder / src_path.name
            counter = 1
            while dest_path.exists():
                dest_path = dest_folder / f"{src_path.stem}_{counter}{src_path.suffix}"
                counter += 1

            log.verbose("FileSystemHandler", f"Moving {src_path.name} to {dest_path}")
            shutil.move(str(src_path), str(dest_path))
            return dest_path
        except Exception as e:
            log.exception("FileSystemHandler", f"Failed to move sample from {src_path} to {dest_folder}: {e}")
            raise

    def get_class_distribution(self, root_path: Path) -> dict:
        """
        Scan a single split directory and return {class_name: image_count}.
        root_path should be e.g. datasets/MyData/train/
        """
        root_path = Path(root_path)
        if not root_path.exists():
            return {}

        distribution = {}
        for class_dir in sorted(root_path.iterdir()):
            if class_dir.is_dir():
                count = sum(
                    1 for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
                )
                distribution[class_dir.name] = count
        return distribution

    def delete_empty_classes(self, dataset_path: Path):
        """
        Remove any class subdirectory that contains zero image files,
        across all splits inside dataset_path.
        """
        dataset_path = Path(dataset_path)
        for split_dir in dataset_path.iterdir():
            if not split_dir.is_dir():
                continue
            for class_dir in list(split_dir.iterdir()):
                if class_dir.is_dir() and not any(class_dir.iterdir()):
                    class_dir.rmdir()


# ---------------------------------------------------------------------------
# MetadataManager
# ---------------------------------------------------------------------------
class MetadataManager:
    """Maintain a metadata.json 'Source of Truth' for each dataset."""

    _FILENAME = "metadata.json"

    def __init__(self, parent: "DatasetBuilder"):
        self._parent = parent

    def generate_metadata(self, dataset_path: Path) -> dict:
        """
        Create (or overwrite) metadata.json with current stats.
        Includes: name, created_at, num_classes, classes, total_images,
        and per-split distribution.
        """
        try:
            dataset_path = Path(dataset_path)
            log.verbose("MetadataManager", f"Generating metadata for {dataset_path.name}")
            metadata = self._build_stats(dataset_path)
            self._write(dataset_path, metadata)
            log.info("MetadataManager", f"Metadata generated: {metadata['num_classes']} classes, {metadata['total_images']} total images")
            return metadata
        except Exception as e:
            log.exception("MetadataManager", f"Failed to generate metadata for {dataset_path}: {e}")
            raise

    def update_stats(self, dataset_path: Path) -> dict:
        """
        Refresh metadata.json stats while preserving the original created_at.
        """
        try:
            dataset_path = Path(dataset_path)
            log.verbose("MetadataManager", f"Updating stats for {dataset_path.name}")
            existing = self.load_metadata(dataset_path)

            metadata = self._build_stats(dataset_path)
            metadata["created_at"] = existing.get("created_at", metadata["created_at"])
            self._write(dataset_path, metadata)
            log.info("MetadataManager", f"Stats updated: {metadata['num_classes']} classes, {metadata['total_images']} total images")
            return metadata
        except Exception as e:
            log.exception("MetadataManager", f"Failed to update stats for {dataset_path}: {e}")
            raise

    def load_metadata(self, dataset_path: Path) -> dict:
        """Read metadata.json and return as a dict (empty dict if missing)."""
        try:
            meta_path = Path(dataset_path) / self._FILENAME
            if not meta_path.exists():
                log.debug("MetadataManager", f"Metadata not found at {meta_path}")
                return {}
            log.verbose("MetadataManager", f"Loading metadata from {meta_path}")
            with meta_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            log.debug("MetadataManager", f"Metadata loaded: {metadata.get('name', 'unknown')} dataset")
            return metadata
        except Exception as e:
            log.exception("MetadataManager", f"Failed to load metadata from {dataset_path}: {e}")
            return {}

    def get_label_map(self, dataset_path: Path) -> dict:
        """
        Return an {int: str} label map derived from sorted class folder names
        inside the train split — e.g. {0: "Apple", 1: "Orange"}.
        """
        train_path = Path(dataset_path) / "train"
        if not train_path.exists():
            return {}
        classes = sorted(d.name for d in train_path.iterdir() if d.is_dir())
        return {i: cls for i, cls in enumerate(classes)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_stats(self, dataset_path: Path) -> dict:
        fsh = self._parent.file_system_handler
        train_path = dataset_path / "train"

        split_distribution = {}
        total_images = 0
        for split in ("train", "val", "test"):
            split_path = dataset_path / split
            if split_path.exists():
                dist = fsh.get_class_distribution(split_path)
                split_distribution[split] = dist
                total_images += sum(dist.values())

        classes = (
            sorted(d.name for d in train_path.iterdir() if d.is_dir())
            if train_path.exists()
            else []
        )

        return {
            "name": dataset_path.name,
            "created_at": datetime.now().isoformat(),
            "num_classes": len(classes),
            "classes": classes,
            "total_images": total_images,
            "split_distribution": split_distribution,
        }

    def _write(self, dataset_path: Path, metadata: dict):
        try:
            meta_path = dataset_path / self._FILENAME
            log.verbose("MetadataManager", f"Writing metadata to {meta_path}")
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            log.debug("MetadataManager", f"Metadata written successfully")
        except Exception as e:
            log.exception("MetadataManager", f"Failed to write metadata to {dataset_path}: {e}")
            raise


# ---------------------------------------------------------------------------
# DatasetTransformer
# ---------------------------------------------------------------------------
class DatasetTransformer:
    """Perform structural dataset operations: splitting, merging, pruning."""

    def __init__(self, parent: "DatasetBuilder"):
        self._parent = parent

    def split_train_val_test(
        self, raw_path: Path, ratios: tuple = (0.8, 0.1, 0.1)
    ):
        """
        Move images from a flat raw/<class>/ structure into
        <dataset_root>/{train,val,test}/<class>/.

        raw_path — directory whose immediate subdirs are class folders.
        The parent of raw_path is treated as the dataset root.
        Files are shuffled with a fixed seed (42) for reproducibility.
        """
        try:
            raw_path = Path(raw_path)
            dataset_path = raw_path.parent

            if abs(sum(ratios) - 1.0) > 1e-6:
                raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios):.4f}")

            log.info("DatasetTransformer", f"Starting train/val/test split with ratios {ratios}")
            fsh = self._parent.file_system_handler

            total_moved = 0
            for class_dir in sorted(raw_path.iterdir()):
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                files = [
                    f for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
                ]
                random.seed(42)
                random.shuffle(files)

                n = len(files)
                n_train = int(n * ratios[0])
                n_val = int(n * ratios[1])

                log.verbose("DatasetTransformer", f"Splitting {class_name}: {n_train} train, {n_val} val, {n - n_train - n_val} test")

                split_files = {
                    "train": files[:n_train],
                    "val": files[n_train: n_train + n_val],
                    "test": files[n_train + n_val:],
                }

                for split, split_file_list in split_files.items():
                    dest_dir = dataset_path / split / class_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for f in split_file_list:
                        fsh.move_sample(f, dest_dir)
                        total_moved += 1

            log.info("DatasetTransformer", f"Split complete: {total_moved} files organized into train/val/test")
        except Exception as e:
            log.exception("DatasetTransformer", f"Failed to split dataset: {e}")
            raise

    def merge_datasets(self, source_paths: list, target_path: Path):
        """
        Merge one or more source datasets into target_path.
        Handles overlapping class names safely via move_sample collision avoidance.
        """
        target_path = Path(target_path)
        fsh = self._parent.file_system_handler

        for source_path in source_paths:
            source_path = Path(source_path)
            for split_dir in source_path.iterdir():
                if not split_dir.is_dir():
                    continue
                for class_dir in split_dir.iterdir():
                    if not class_dir.is_dir():
                        continue
                    dest_class = target_path / split_dir.name / class_dir.name
                    dest_class.mkdir(parents=True, exist_ok=True)
                    for img_file in class_dir.iterdir():
                        if (
                            img_file.is_file()
                            and img_file.suffix.lower() in _IMAGE_EXTENSIONS
                        ):
                            fsh.move_sample(img_file, dest_class)

    def prune_dataset(self, dataset_path: Path, criteria: dict):
        """
        Move bad files to storage/trash/<dataset_name>/ rather than deleting.

        criteria keys:
            remove_duplicates (bool) — hash-based MD5 deduplication
            remove_corrupt    (bool) — PIL.Image verification
        """
        dataset_path = Path(dataset_path)
        trash_root = (
            self._parent.storage_root / "trash" / dataset_path.name
        )
        trash_root.mkdir(parents=True, exist_ok=True)

        seen_hashes: set = set()

        for img_path in sorted(dataset_path.rglob("*")):
            if not (
                img_path.is_file()
                and img_path.suffix.lower() in _IMAGE_EXTENSIONS
            ):
                continue

            relative = img_path.relative_to(dataset_path)

            if criteria.get("remove_corrupt", False):
                if not self._is_valid_image(img_path):
                    dest = trash_root / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(img_path), str(dest))
                    continue

            if criteria.get("remove_duplicates", False):
                file_hash = self._md5(img_path)
                if file_hash in seen_hashes:
                    dest = trash_root / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(img_path), str(dest))
                else:
                    seen_hashes.add(file_hash)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_image(file_path: Path) -> bool:
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    @staticmethod
    def _md5(file_path: Path) -> str:
        h = hashlib.md5()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# DatasetValidator
# ---------------------------------------------------------------------------
class DatasetValidator:
    """Check for corruptions and class balance issues."""

    def __init__(self, parent: "DatasetBuilder"):
        self._parent = parent

    def find_corrupt_files(self, dataset_path: Path) -> list:
        """
        Walk the entire dataset tree and return a list of Paths that
        PIL cannot open/verify.
        """
        try:
            log.verbose("DatasetValidator", f"Scanning for corrupt files in {dataset_path}")
            corrupt = []
            for img_path in sorted(Path(dataset_path).rglob("*")):
                if img_path.is_file() and img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                    except Exception as e:
                        log.debug("DatasetValidator", f"Corrupt image detected: {img_path} ({e})")
                        corrupt.append(img_path)
            if corrupt:
                log.warning("DatasetValidator", f"Found {len(corrupt)} corrupt files")
            else:
                log.info("DatasetValidator", "No corrupt files detected")
            return corrupt
        except Exception as e:
            log.exception("DatasetValidator", f"Failed to scan for corrupt files: {e}")
            raise

    def check_class_balance(self, dataset_path: Path, split: str = "train") -> dict:
        """
        Analyse the class distribution for a given split.

        Returns:
            {
                "distribution": {class_name: count, ...},
                "warnings":     [str, ...]
            }
        """
        try:
            log.verbose("DatasetValidator", f"Checking class balance for {split} split in {dataset_path}")
            fsh = self._parent.file_system_handler
            distribution = fsh.get_class_distribution(Path(dataset_path) / split)

            if not distribution:
                log.warning("DatasetValidator", "No classes found in dataset")
                return {"distribution": {}, "warnings": ["No classes found."]}

            counts = list(distribution.values())
            max_count = max(counts)
            min_count = min(counts)
            warnings = []

            empty_classes = [k for k, v in distribution.items() if v == 0]
            if empty_classes:
                msg = f"Empty classes detected: {empty_classes}"
                log.warning("DatasetValidator", msg)
                warnings.append(msg)

            if min_count > 0 and (max_count / min_count) > 5:
                msg = (
                    f"Severe class imbalance detected "
                    f"(max={max_count}, min={min_count}). "
                    "Consider collecting more samples for under-represented classes."
                )
                log.warning("DatasetValidator", msg)
                warnings.append(msg)

            log.info("DatasetValidator", f"Class balance check: {len(distribution)} classes, balance ratio {max_count/min_count if min_count > 0 else 'N/A':.2f}")
            return {"distribution": distribution, "warnings": warnings}
        except Exception as e:
            log.exception("DatasetValidator", f"Failed to check class balance: {e}")
            raise


# ---------------------------------------------------------------------------
# DatasetBuilder — parent class that owns all submodules
# ---------------------------------------------------------------------------
class DatasetBuilder:
    """
    Top-level manager for dataset operations.

    Submodules are accessed via:
        self.file_system_handler
        self.metadata_manager
        self.dataset_transformer
        self.dataset_validator

    Args:
        storage_root: Path to the top-level storage/ directory.
                      Typically resolved as Path(main.py).parent / "storage".
    """

    def __init__(self, storage_root: Path):
        self.storage_root = Path(storage_root)
        self.datasets_root = self.storage_root / "datasets"
        self.datasets_root.mkdir(parents=True, exist_ok=True)

        # Submodules
        self.file_system_handler = FileSystemHandler(self)
        self.metadata_manager = MetadataManager(self)
        self.dataset_transformer = DatasetTransformer(self)
        self.dataset_validator = DatasetValidator(self)
