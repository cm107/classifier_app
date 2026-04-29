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
        dataset_path = self._parent.datasets_root / name
        for split in ("train", "val", "test"):
            for cls in classes:
                (dataset_path / split / cls).mkdir(parents=True, exist_ok=True)
        return dataset_path

    def move_sample(self, src_path: Path, dest_folder: Path) -> Path:
        """
        Move a single image file into dest_folder, avoiding filename collisions
        by appending _1, _2, … to the stem if needed.
        """
        src_path = Path(src_path)
        dest_folder = Path(dest_folder)
        dest_folder.mkdir(parents=True, exist_ok=True)

        dest_path = dest_folder / src_path.name
        counter = 1
        while dest_path.exists():
            dest_path = dest_folder / f"{src_path.stem}_{counter}{src_path.suffix}"
            counter += 1

        shutil.move(str(src_path), str(dest_path))
        return dest_path

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
        dataset_path = Path(dataset_path)
        metadata = self._build_stats(dataset_path)
        self._write(dataset_path, metadata)
        return metadata

    def update_stats(self, dataset_path: Path) -> dict:
        """
        Refresh metadata.json stats while preserving the original created_at.
        """
        dataset_path = Path(dataset_path)
        existing = self.load_metadata(dataset_path)

        metadata = self._build_stats(dataset_path)
        metadata["created_at"] = existing.get("created_at", metadata["created_at"])
        self._write(dataset_path, metadata)
        return metadata

    def load_metadata(self, dataset_path: Path) -> dict:
        """Read metadata.json and return as a dict (empty dict if missing)."""
        meta_path = Path(dataset_path) / self._FILENAME
        if not meta_path.exists():
            return {}
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

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
        meta_path = dataset_path / self._FILENAME
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


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
        raw_path = Path(raw_path)
        dataset_path = raw_path.parent

        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios):.4f}")

        fsh = self._parent.file_system_handler

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
        corrupt = []
        for img_path in sorted(Path(dataset_path).rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in _IMAGE_EXTENSIONS:
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception:
                    corrupt.append(img_path)
        return corrupt

    def check_class_balance(self, dataset_path: Path, split: str = "train") -> dict:
        """
        Analyse the class distribution for a given split.

        Returns:
            {
                "distribution": {class_name: count, ...},
                "warnings":     [str, ...]
            }
        """
        fsh = self._parent.file_system_handler
        distribution = fsh.get_class_distribution(Path(dataset_path) / split)

        if not distribution:
            return {"distribution": {}, "warnings": ["No classes found."]}

        counts = list(distribution.values())
        max_count = max(counts)
        min_count = min(counts)
        warnings = []

        empty_classes = [k for k, v in distribution.items() if v == 0]
        if empty_classes:
            warnings.append(f"Empty classes detected: {empty_classes}")

        if min_count > 0 and (max_count / min_count) > 5:
            warnings.append(
                f"Severe class imbalance detected "
                f"(max={max_count}, min={min_count}). "
                "Consider collecting more samples for under-represented classes."
            )

        return {"distribution": distribution, "warnings": warnings}


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
