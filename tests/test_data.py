"""
tests/test_data.py — Milestone 1 Verification Script

Runs through the full Milestone 1.2 checklist:
    1. Create a dataset named "TestProject" with classes "A" and "B".
    2. Verify metadata.json exists and correctly lists 2 classes.
    3. Split data and confirm file counts match the requested ratio.
    4. Load one sample via ClassificationDataset and confirm it is a Tensor.

Usage:
    python tests/test_data.py
"""

import shutil
import sys
from pathlib import Path

# Ensure the project root is on the path regardless of where the script is run.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vision_app.core.data_manager import DatasetBuilder
from vision_app.core.dataset import ClassificationDataset, StandardTransformations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STORAGE_ROOT = PROJECT_ROOT / "storage"
TEST_DATASET_NAME = "TestProject"


def _cleanup(builder: DatasetBuilder):
    dataset_path = builder.datasets_root / TEST_DATASET_NAME
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    print("  [cleanup] removed TestProject dataset")


def _populate_dummy_images(dataset_path: Path, classes: list, n_per_class: int = 20):
    """Create small synthetic PNG files for testing without real images."""
    from PIL import Image
    import random

    raw_path = dataset_path / "raw"
    for cls in classes:
        (raw_path / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = Image.new(
                "RGB",
                (64, 64),
                color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            )
            img.save(raw_path / cls / f"img_{i:04d}.png")
    return raw_path


# ---------------------------------------------------------------------------
# Test 1: Create dataset structure
# ---------------------------------------------------------------------------
def test_create_structure():
    print("\n[Test 1] Creating dataset structure ...")
    builder = DatasetBuilder(STORAGE_ROOT)
    _cleanup(builder)

    classes = ["A", "B"]
    dataset_path = builder.file_system_handler.create_dataset_structure(
        TEST_DATASET_NAME, classes
    )

    for split in ("train", "val", "test"):
        for cls in classes:
            d = dataset_path / split / cls
            assert d.exists(), f"Missing directory: {d}"

    print(f"  PASS — structure created at {dataset_path}")
    return builder, dataset_path


# ---------------------------------------------------------------------------
# Test 2: Metadata generation
# ---------------------------------------------------------------------------
def test_metadata(builder: DatasetBuilder, dataset_path: Path):
    print("\n[Test 2] Generating and loading metadata ...")

    # Populate raw images so stats are non-trivial
    raw_path = _populate_dummy_images(dataset_path, ["A", "B"], n_per_class=20)

    # Split into train/val/test so metadata has real counts
    builder.dataset_transformer.split_train_val_test(raw_path, ratios=(0.8, 0.1, 0.1))

    metadata = builder.metadata_manager.generate_metadata(dataset_path)
    meta_path = dataset_path / "metadata.json"

    assert meta_path.exists(), "metadata.json was not created"
    assert metadata["num_classes"] == 2, f"Expected 2 classes, got {metadata['num_classes']}"
    assert set(metadata["classes"]) == {"A", "B"}, f"Unexpected classes: {metadata['classes']}"
    assert metadata["total_images"] > 0, "total_images is 0"

    print(f"  PASS — metadata.json: {metadata['num_classes']} classes, "
          f"{metadata['total_images']} images")
    return metadata


# ---------------------------------------------------------------------------
# Test 3: Train/val/test split ratio
# ---------------------------------------------------------------------------
def test_split_ratio(builder: DatasetBuilder, dataset_path: Path, n_per_class: int = 20):
    print("\n[Test 3] Verifying 80/10/10 split ratio ...")

    fsh = builder.file_system_handler
    for cls in ("A", "B"):
        train_count = fsh.get_class_distribution(dataset_path / "train").get(cls, 0)
        val_count = fsh.get_class_distribution(dataset_path / "val").get(cls, 0)
        test_count = fsh.get_class_distribution(dataset_path / "test").get(cls, 0)
        total = train_count + val_count + test_count

        assert total == n_per_class, (
            f"Class {cls}: expected {n_per_class} total, got {total}"
        )
        assert train_count == int(n_per_class * 0.8), (
            f"Class {cls}: expected {int(n_per_class * 0.8)} train samples, "
            f"got {train_count}"
        )

        print(f"  {cls}: train={train_count}, val={val_count}, test={test_count}")

    print("  PASS — split counts match 80/10/10")


# ---------------------------------------------------------------------------
# Test 4: Load a sample as a torch.Tensor
# ---------------------------------------------------------------------------
def test_dataset_load(dataset_path: Path):
    import torch

    print("\n[Test 4] Loading sample through ClassificationDataset ...")

    transform = StandardTransformations(image_size=64, augment=False)
    ds = ClassificationDataset(dataset_path / "train", transform=transform)

    assert len(ds) > 0, "Dataset is empty"

    sample, label = ds[0]

    assert isinstance(sample, torch.Tensor), f"Expected Tensor, got {type(sample)}"
    assert sample.shape == (3, 64, 64), f"Unexpected shape: {sample.shape}"
    assert isinstance(label, int), f"Expected int label, got {type(label)}"

    print(f"  PASS — sample shape={tuple(sample.shape)}, label={label} "
          f"({ds.class_to_idx})")


# ---------------------------------------------------------------------------
# Test 5: Label smoothing
# ---------------------------------------------------------------------------
def test_label_smoothing(dataset_path: Path):
    import torch

    print("\n[Test 5] Label smoothing output ...")

    transform = StandardTransformations(image_size=64, augment=False)
    ds = ClassificationDataset(
        dataset_path / "train",
        transform=transform,
        label_smoothing=0.1,
    )

    _, soft_label = ds[0]

    assert isinstance(soft_label, torch.Tensor), "Expected soft label Tensor"
    assert soft_label.shape == (2,), f"Unexpected shape: {soft_label.shape}"
    assert abs(soft_label.sum().item() - 1.0) < 1e-5, "Soft label should sum to 1"

    print(f"  PASS — soft label: {soft_label.tolist()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Milestone 1.2 — Verification Checklist")
    print("=" * 60)

    N_PER_CLASS = 20

    builder, dataset_path = test_create_structure()
    test_metadata(builder, dataset_path)
    test_split_ratio(builder, dataset_path, n_per_class=N_PER_CLASS)
    test_dataset_load(dataset_path)
    test_label_smoothing(dataset_path)

    print("\n" + "=" * 60)
    print("All tests passed. Milestone 1 core logic is verified.")
    print("=" * 60)
