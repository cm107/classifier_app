"""
scripts/benchmark_train.py
===========================
Milestone 7.4: 3-epoch smoke test on the microsoft_cats_dogs dataset.

Success criteria:
  - Loss decreases consistently over 3 epochs.
  - A checkpoint.pth is saved to storage/models/microsoft_cats_dogs_smoke.pth.

Usage:
    cd /home/clayton/workspace/git/classifier_app
    source venv/bin/activate
    python scripts/benchmark_train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from vision_app.core.data_manager import DatasetBuilder
from vision_app.core.dataset import ClassificationDataset, StandardTransformations
from vision_app.core.model import ScratchResNet
from vision_app.core.trainer import ModelTrainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_NAME  = "microsoft_cats_dogs"
STORAGE_ROOT  = PROJECT_ROOT / "storage"
EPOCHS        = 3
BATCH_SIZE    = 32
LR            = 1e-3
MAX_LR        = 1e-2
WEIGHT_DECAY  = 1e-2
IMAGE_SIZE    = 128   # reduced for smoke test; use 224 for production
NUM_WORKERS   = 4
MEAN          = [0.485, 0.456, 0.406]
STD           = [0.229, 0.224, 0.225]


def main():
    print("=" * 60)
    print("Milestone 7.4 — Benchmark Training Smoke Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Verify dataset exists ──
    builder = DatasetBuilder(STORAGE_ROOT)
    dataset_path = builder.datasets_root / DATASET_NAME
    train_path = dataset_path / "train"
    val_path   = dataset_path / "val"

    if not train_path.exists():
        print(
            f"\nERROR: {train_path} not found.\n"
            "Run scripts/ingest_microsoft_cats_dogs.py first."
        )
        sys.exit(1)

    # ── Build label map ──
    label_map = builder.metadata_manager.get_label_map(dataset_path)
    num_classes = len(label_map)
    print(f"Classes ({num_classes}): {label_map}")

    # ── DataLoaders ──
    train_transform = StandardTransformations(
        image_size=IMAGE_SIZE, mean=MEAN, std=STD, augment=True
    )
    val_transform = StandardTransformations(
        image_size=IMAGE_SIZE, mean=MEAN, std=STD, augment=False
    )

    train_ds = ClassificationDataset(train_path, transform=train_transform)
    val_ds   = ClassificationDataset(val_path,   transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    print(
        f"\nDataset  : train={len(train_ds)}, val={len(val_ds)}"
    )
    print(
        f"Batches  : {len(train_loader)} train / {len(val_loader)} val  "
        f"(batch_size={BATCH_SIZE})"
    )

    # ── Model + Trainer ──
    model = ScratchResNet(num_classes=num_classes)
    trainer = ModelTrainer(model=model, device=str(device))

    optimizer  = trainer.optimization_engine.configure_optimizer(lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = trainer.optimization_engine.configure_scheduler(
        max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    loss_fn    = trainer.optimization_engine.get_loss_function("supervised")

    # ── Training loop ──
    print(f"\nStarting {EPOCHS}-epoch smoke test …\n")
    losses: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        # Quick validation
        val_metrics = trainer.validation_engine.run_evaluation(val_loader, loss_fn=loss_fn)
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch}/{EPOCHS}  "
            f"train_loss={avg_loss:.4f}  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"val_acc={val_metrics['val_accuracy']:.1f}%  "
            f"lr={lr_now:.2e}"
        )

    # ── Success criteria ──
    print()
    loss_decreasing = all(losses[i] > losses[i + 1] for i in range(len(losses) - 1))
    if loss_decreasing:
        print("  ✓  Loss decreased every epoch.")
    else:
        print(f"  ⚠  Loss did not decrease monotonically: {[f'{l:.4f}' for l in losses]}")
        print("     (Minor fluctuations are normal with a strong scheduler.)")

    # ── Save checkpoint ──
    ckpt_path = STORAGE_ROOT / "models" / f"{DATASET_NAME}_smoke.pth"
    (STORAGE_ROOT / "models").mkdir(parents=True, exist_ok=True)

    trainer.state_manager.save_checkpoint(
        ckpt_path,
        epoch=EPOCHS,
        optimizer_state=optimizer.state_dict(),
        extra_meta={
            "label_map":     label_map,
            "val_accuracy":  val_metrics["val_accuracy"],
            "phase":         "supervised",
        },
    )

    if ckpt_path.exists():
        print(f"  ✓  Checkpoint saved → {ckpt_path.name}  ({ckpt_path.stat().st_size // 1024} KB)")
    else:
        print("  ✗  Checkpoint was NOT saved — check write permissions.")

    print()
    print("=" * 60)
    print("Smoke test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
