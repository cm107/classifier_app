"""
train_worker.py — Background QThread wrapper for the ModelTrainer.

Architecture (submodule pattern):
    TrainWorker
        ├── SignalDispatcher  — all PySide6 Signal definitions
        └── LifecycleManager  — abort / pause flags and helpers

The run() method lives on TrainWorker and is invoked by QThread.start().
It owns DataLoader and ModelTrainer construction so those objects are
created on the worker thread, not the GUI thread.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QMutex, QMutexLocker, QObject, QThread, QWaitCondition, Signal

from vision_app.core.dataset import ClassificationDataset, ContrastiveTransformations, StandardTransformations
from vision_app.core.model import ScratchResNet
from vision_app.core.trainer import ModelTrainer, TrainingMetrics


# ---------------------------------------------------------------------------
# SignalDispatcher
# ---------------------------------------------------------------------------
class SignalDispatcher(QObject):
    """
    All PySide6 Signals for the training pipeline.

    Signals are defined on a QObject subclass so they can be safely
    connected to slots in the GUI thread.
    """

    # Emitted at the end of every epoch with a TrainingMetrics payload
    progress_updated = Signal(object)   # payload: TrainingMetrics

    # Human-readable status strings: "Starting SSL…", "Epoch 3/50", etc.
    status_changed = Signal(str)

    # True  → completed successfully
    # False → aborted or error
    finished = Signal(bool)

    # Emitted if a recoverable error occurs (e.g. CUDA OOM on a batch)
    error_occurred = Signal(str)

    def __init__(self, parent: "TrainWorker"):
        super().__init__()
        self._parent = parent


# ---------------------------------------------------------------------------
# LifecycleManager
# ---------------------------------------------------------------------------
class LifecycleManager:
    """
    Manages the abort and pause state of an active training run.

    Thread-safety:
        _abort and _paused are written from the GUI thread and read from the
        worker thread. A QMutex protects the flag reads/writes.
        Pause uses a QWaitCondition so the worker thread sleeps without
        busy-waiting.
    """

    def __init__(self, parent: "TrainWorker"):
        self._parent = parent
        self._abort: bool = False
        self._paused: bool = False
        self._mutex = QMutex()
        self._pause_condition = QWaitCondition()

    # --- Called from the GUI thread ---

    def request_abort(self):
        """Signal the worker to stop at the next batch boundary."""
        with QMutexLocker(self._mutex):
            self._abort = True
            # Wake the worker if it is currently paused
            self._pause_condition.wakeAll()

    def set_paused(self, paused: bool):
        """Pause or resume the training loop."""
        with QMutexLocker(self._mutex):
            self._paused = paused
            if not paused:
                self._pause_condition.wakeAll()

    def reset(self):
        """Reset flags before a new training run."""
        with QMutexLocker(self._mutex):
            self._abort = False
            self._paused = False

    # --- Called from the worker thread ---

    @property
    def should_abort(self) -> bool:
        with QMutexLocker(self._mutex):
            return self._abort

    def check_pause(self):
        """
        Block the worker thread if paused. Returns immediately if not paused
        or if an abort has been requested while waiting.
        """
        self._mutex.lock()
        while self._paused and not self._abort:
            self._pause_condition.wait(self._mutex)
        self._mutex.unlock()


# ---------------------------------------------------------------------------
# TrainWorker
# ---------------------------------------------------------------------------
class TrainWorker(QObject):
    """
    Runs the full training pipeline (SSL → linear probe → fine-tune) on a
    QThread so the GUI stays responsive.

    Usage:
        worker = TrainWorker(config)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.signals.finished.connect(thread.quit)
        thread.start()

    Args:
        config : dict with the following keys:
            dataset_path   (Path)  — path to the dataset root (contains train/ val/)
            model_path     (Path|None) — resume from checkpoint if provided
            num_classes    (int)
            epochs         (int)
            batch_size     (int)
            learning_rate  (float)
            max_lr         (float)
            weight_decay   (float)
            image_size     (int)
            num_workers    (int)
            phase          (str)   — "ssl" | "linear_probe" | "supervised"
            label_smoothing(float) — passed to ClassificationDataset
            mean           (list)  — channel mean for normalisation
            std            (list)  — channel std for normalisation
            storage_root   (Path)  — for saving checkpoints
    """

    def __init__(self, config: dict, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.config = config

        # Submodules
        self.signals = SignalDispatcher(self)
        self.lifecycle = LifecycleManager(self)

    # ------------------------------------------------------------------
    # Public helpers (called from GUI thread before/after run)
    # ------------------------------------------------------------------

    def abort(self):
        """
        Request the training loop to stop at the next batch boundary.

        Thread-safety note: Because the worker is moved to a QThread that runs
        no event loop, connecting a GUI signal to this slot with AutoConnection
        (the default) will queue the call and it will never be delivered.

        Always connect with DirectConnection from the GUI:
            button.clicked.connect(worker.abort, Qt.DirectConnection)
        Or call worker.lifecycle.request_abort() directly from any thread.
        """
        self.lifecycle.request_abort()

    def pause(self):
        self.lifecycle.set_paused(True)

    def resume(self):
        self.lifecycle.set_paused(False)

    # ------------------------------------------------------------------
    # Main entry point — runs on the worker QThread
    # ------------------------------------------------------------------

    def run(self):
        """
        Full training loop. Emits signals throughout.
        Catches RuntimeError (CUDA OOM) and generic exceptions so the worker
        always emits finished() even on failure.
        """
        self.lifecycle.reset()
        cfg = self.config
        phase = cfg.get("phase", "supervised")

        try:
            self.signals.status_changed.emit(f"Initialising ({phase} phase)…")

            # --- Build DataLoaders on the worker thread ---
            train_loader, val_loader, label_map = self._build_loaders(cfg)

            # --- Build model and trainer ---
            model = ScratchResNet(
                num_classes=cfg["num_classes"],
                projection_dim=cfg.get("projection_dim", 128),
            )
            model.use_projection_head = (phase == "ssl")

            trainer = ModelTrainer(model=model)

            # Optionally resume from checkpoint
            if cfg.get("model_path"):
                self.signals.status_changed.emit("Loading checkpoint…")
                trainer.state_manager.load_checkpoint(Path(cfg["model_path"]))

            # Freeze backbone for linear probe
            if phase == "linear_probe":
                trainer.state_manager.freeze_backbone(freeze=True)

            # --- Configure optimiser + scheduler ---
            optimizer = trainer.optimization_engine.configure_optimizer(
                lr=cfg.get("learning_rate", 1e-3),
                weight_decay=cfg.get("weight_decay", 1e-2),
            )
            scheduler = trainer.optimization_engine.configure_scheduler(
                max_lr=cfg.get("max_lr", 1e-2),
                steps_per_epoch=len(train_loader),
                epochs=cfg["epochs"],
            )
            loss_fn = trainer.optimization_engine.get_loss_function(phase)

            # --- Training loop ---
            epochs = cfg["epochs"]
            for epoch in range(1, epochs + 1):

                if self.lifecycle.should_abort:
                    break

                self.signals.status_changed.emit(f"Epoch {epoch}/{epochs}")
                model.train()

                epoch_loss = self._run_epoch(
                    trainer, train_loader, optimizer, scheduler, loss_fn, phase
                )

                if self.lifecycle.should_abort:
                    break

                # Validation (skip for SSL — no class labels)
                val_metrics = {}
                if phase != "ssl":
                    model.use_projection_head = False
                    val_metrics = trainer.validation_engine.run_evaluation(
                        val_loader, loss_fn=loss_fn
                    )

                current_lr = optimizer.param_groups[0]["lr"]
                metrics = TrainingMetrics(
                    epoch=epoch,
                    total_epochs=epochs,
                    train_loss=epoch_loss,
                    val_loss=val_metrics.get("val_loss", 0.0),
                    val_accuracy=val_metrics.get("val_accuracy", 0.0),
                    learning_rate=current_lr,
                    phase=phase,
                    extra={"confusion_matrix": val_metrics.get("confusion_matrix", [])},
                )
                self.signals.progress_updated.emit(metrics)

                # Save a checkpoint each epoch
                ckpt_dir = Path(cfg.get("storage_root", "storage")) / "models"
                ckpt_path = ckpt_dir / f"checkpoint_epoch{epoch:04d}.pth"
                trainer.state_manager.save_checkpoint(
                    ckpt_path, epoch=epoch,
                    optimizer_state=optimizer.state_dict(),
                    extra_meta={
                        "label_map": label_map,
                        "val_accuracy": val_metrics.get("val_accuracy", 0.0),
                        "phase": phase,
                    },
                )

            success = not self.lifecycle.should_abort
            msg = "Training complete." if success else "Training aborted."
            self.signals.status_changed.emit(msg)
            self.signals.finished.emit(success)

        except RuntimeError as exc:
            # Catches CUDA OOM and similar recoverable errors
            err = str(exc)
            self.signals.error_occurred.emit(f"RuntimeError: {err}")
            self.signals.status_changed.emit(f"Error: {err}")
            self.signals.finished.emit(False)

        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            self.signals.error_occurred.emit(f"Unexpected error: {err}")
            self.signals.status_changed.emit(f"Fatal: {err}")
            self.signals.finished.emit(False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_epoch(self, trainer, train_loader, optimizer, scheduler, loss_fn, phase) -> float:
        """Run one epoch; return average loss. Respects abort and pause flags."""
        import torch

        model = trainer.model
        device = trainer.device
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            self.lifecycle.check_pause()
            if self.lifecycle.should_abort:
                break

            optimizer.zero_grad()

            if phase == "ssl":
                # Batch: ((view1, view2), _labels)  — ContrastiveTransformations
                (view1, view2), _ = batch
                view1, view2 = view1.to(device), view2.to(device)
                z_i = model(view1)
                z_j = model(view2)
                loss = loss_fn(z_i, z_j)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches if n_batches > 0 else 0.0

    def _build_loaders(self, cfg: dict):
        """Construct train and val DataLoaders from config."""
        from torch.utils.data import DataLoader

        dataset_path = Path(cfg["dataset_path"])
        image_size = cfg.get("image_size", 224)
        batch_size = cfg.get("batch_size", 32)
        num_workers = cfg.get("num_workers", 4)
        phase = cfg.get("phase", "supervised")
        mean = cfg.get("mean", [0.485, 0.456, 0.406])
        std = cfg.get("std", [0.229, 0.224, 0.225])

        if phase == "ssl":
            transform = ContrastiveTransformations(image_size=image_size, mean=mean, std=std)
        else:
            transform = StandardTransformations(image_size=image_size, mean=mean, std=std, augment=True)

        val_transform = StandardTransformations(image_size=image_size, mean=mean, std=std, augment=False)

        label_smoothing = cfg.get("label_smoothing", 0.0) if phase != "ssl" else 0.0

        train_ds = ClassificationDataset(
            dataset_path / "train",
            transform=transform,
            label_smoothing=label_smoothing,
        )
        val_ds = ClassificationDataset(
            dataset_path / "val",
            transform=val_transform,
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=(phase == "ssl"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        # Build int→class_name label map from the dataset's class_to_idx
        label_map = {v: k for k, v in train_ds.class_to_idx.items()}
        return train_loader, val_loader, label_map
