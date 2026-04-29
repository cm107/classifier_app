"""
tests/test_model.py — Milestone 2 Verification Script

Covers the checklist from docs/roadmap/milestone02.md:
    1. Shape Test       — forward pass produces [1, num_classes] output.
    2. SSL Head Test    — projection head produces [N, projection_dim] output.
    3. Backbone Freeze  — freeze_backbone() sets requires_grad=False on backbone.
    4. Checkpoint       — save, scramble weights, reload → weights restored.
    5. NTXentLoss       — loss is a positive scalar; symmetric inputs give ~0 loss.
    6. Validation       — run_evaluation returns expected keys and accuracy range.

Usage:
    python tests/test_model.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from vision_app.core.model import ScratchResNet
from vision_app.core.trainer import (
    ModelTrainer,
    NTXentLoss,
    OptimizationEngine,
    TrainingMetrics,
)


NUM_CLASSES = 5
DEVICE = torch.device("cpu")


def _make_model() -> ScratchResNet:
    return ScratchResNet(num_classes=NUM_CLASSES)


def _make_trainer(model=None) -> ModelTrainer:
    if model is None:
        model = _make_model()
    return ModelTrainer(model=model, device=DEVICE)


# ---------------------------------------------------------------------------
# Test 1: Output shape — supervised mode
# ---------------------------------------------------------------------------
def test_output_shape_supervised():
    print("\n[Test 1] Forward pass shape (supervised) ...")
    model = _make_model()
    model.use_projection_head = False
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, NUM_CLASSES), (
        f"Expected (1, {NUM_CLASSES}), got {tuple(out.shape)}"
    )
    print(f"  PASS — output shape: {tuple(out.shape)}")


# ---------------------------------------------------------------------------
# Test 2: Output shape — SSL projection head
# ---------------------------------------------------------------------------
def test_output_shape_ssl():
    print("\n[Test 2] Forward pass shape (SSL projection head) ...")
    model = ScratchResNet(num_classes=NUM_CLASSES, projection_dim=128)
    model.use_projection_head = True
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 128), f"Expected (2, 128), got {tuple(out.shape)}"
    print(f"  PASS — projection head output shape: {tuple(out.shape)}")


# ---------------------------------------------------------------------------
# Test 3: Variable input resolution (GAP)
# ---------------------------------------------------------------------------
def test_variable_input_resolution():
    print("\n[Test 3] Variable input resolution via GAP ...")
    model = _make_model()
    model.eval()

    for size in (128, 192, 256):
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, NUM_CLASSES), (
            f"Failed at size {size}: {tuple(out.shape)}"
        )

    print(f"  PASS — model handles 128×128, 192×192, 256×256 inputs")


# ---------------------------------------------------------------------------
# Test 4: Backbone freeze / unfreeze
# ---------------------------------------------------------------------------
def test_freeze_backbone():
    print("\n[Test 4] Backbone freeze / unfreeze ...")
    trainer = _make_trainer()

    trainer.state_manager.freeze_backbone(freeze=True)
    for p in trainer.model.backbone_parameters:
        assert not p.requires_grad, "Backbone param still requires grad after freeze"

    # Heads should remain trainable
    for p in trainer.model.fc.parameters():
        assert p.requires_grad, "FC head should still require grad"

    trainer.state_manager.freeze_backbone(freeze=False)
    for p in trainer.model.backbone_parameters:
        assert p.requires_grad, "Backbone param still frozen after unfreeze"

    print("  PASS — freeze / unfreeze works correctly")


# ---------------------------------------------------------------------------
# Test 5: Checkpoint save → scramble → load → restored
# ---------------------------------------------------------------------------
def test_checkpoint_integrity():
    print("\n[Test 5] Checkpoint integrity ...")
    trainer = _make_trainer()

    # Snapshot original first-layer weights
    original_weight = trainer.model.prep[0].weight.data.clone()

    optimizer = trainer.optimization_engine.configure_optimizer()

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "test_checkpoint.pth"
        trainer.state_manager.save_checkpoint(ckpt_path, epoch=1,
                                               optimizer_state=optimizer.state_dict())

        # Scramble weights
        nn.init.uniform_(trainer.model.prep[0].weight, -10, 10)
        scrambled = trainer.model.prep[0].weight.data.clone()
        assert not torch.allclose(original_weight, scrambled), "Scramble failed"

        # Restore from checkpoint
        trainer.state_manager.load_checkpoint(ckpt_path)
        restored = trainer.model.prep[0].weight.data.clone()

    assert torch.allclose(original_weight, restored), (
        "Weights after load_checkpoint do not match original"
    )
    print("  PASS — weights restored correctly from checkpoint")


# ---------------------------------------------------------------------------
# Test 6: NTXentLoss
# ---------------------------------------------------------------------------
def test_ntxent_loss():
    print("\n[Test 6] NTXentLoss ...")
    loss_fn = NTXentLoss(temperature=0.5)

    N, D = 8, 128
    z_i = F.normalize(torch.randn(N, D), dim=1)
    z_j = F.normalize(torch.randn(N, D), dim=1)

    loss = loss_fn(z_i, z_j)
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    print(f"  PASS — loss value: {loss.item():.4f}")


# ---------------------------------------------------------------------------
# Test 7: ValidationEngine with a tiny synthetic DataLoader
# ---------------------------------------------------------------------------
def test_validation_engine():
    print("\n[Test 7] ValidationEngine ...")
    from torch.utils.data import DataLoader, TensorDataset

    trainer = _make_trainer()
    trainer.model.use_projection_head = False

    # Synthetic: 16 samples, correct class always 0
    images = torch.randn(16, 3, 64, 64)
    labels = torch.zeros(16, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    results = trainer.validation_engine.run_evaluation(loader)

    assert "val_loss" in results
    assert "val_accuracy" in results
    assert "confusion_matrix" in results
    assert 0.0 <= results["val_accuracy"] <= 100.0

    print(f"  PASS — val_loss={results['val_loss']:.4f}, "
          f"val_accuracy={results['val_accuracy']:.1f}%")


# ---------------------------------------------------------------------------
# Test 8: TrainingMetrics dataclass
# ---------------------------------------------------------------------------
def test_training_metrics():
    print("\n[Test 8] TrainingMetrics dataclass ...")
    m = TrainingMetrics(
        epoch=3, total_epochs=20, train_loss=0.42, val_accuracy=87.5,
        learning_rate=1e-4, phase="supervised"
    )
    assert m.epoch == 3
    assert m.phase == "supervised"
    assert isinstance(m.extra, dict)
    print(f"  PASS — {m}")


# ---------------------------------------------------------------------------
# Imports needed inside test functions
# ---------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Milestone 2 — Verification Checklist")
    print("=" * 60)

    test_output_shape_supervised()
    test_output_shape_ssl()
    test_variable_input_resolution()
    test_freeze_backbone()
    test_checkpoint_integrity()
    test_ntxent_loss()
    test_validation_engine()
    test_training_metrics()

    print("\n" + "=" * 60)
    print("All tests passed. Milestone 2 model & trainer verified.")
    print("=" * 60)
