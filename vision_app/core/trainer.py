"""
trainer.py — Model training logic, optimization, and validation.

Classes:
    NTXentLoss        — NT-Xent contrastive loss for SSL pre-training.
    TrainingMetrics   — Dataclass payload passed from TrainWorker to the UI.
    StateManager      — Checkpoint save/load and backbone freeze/unfreeze.
    OptimizationEngine— AdamW + OneCycleLR + loss function factory.
    ValidationEngine  — Top-1 accuracy and confusion matrix evaluation.
    ModelTrainer      — Parent class owning all submodules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from vision_app.core.logger import log


# ---------------------------------------------------------------------------
# NTXentLoss
# ---------------------------------------------------------------------------
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).

    Given two sets of projections z_i and z_j (each shape [N, D]) from two
    augmented views of the same batch, the loss maximises the cosine similarity
    between paired views while pushing all other pairs apart.

    Args:
        temperature : Scaling factor τ. Lower → sharper distribution. Default 0.5.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i, z_j : Projection head outputs, shape (N, D).
        Returns:
            Scalar loss.
        """
        n = z_i.size(0)
        device = z_i.device

        # Concatenate: [z_i; z_j] → shape (2N, D)
        z = torch.cat([z_i, z_j], dim=0)

        # L2-normalise before cosine similarity
        z = F.normalize(z, dim=1)

        # Full (2N × 2N) cosine similarity matrix, scaled by temperature
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Build the mask: True where entries are NOT self-comparisons
        mask = ~torch.eye(2 * n, dtype=torch.bool, device=device)

        # Positive pair indices:
        #   row i (from z_i) pairs with row i+n (from z_j), and vice-versa.
        labels = torch.arange(n, device=device)
        labels = torch.cat([labels + n, labels], dim=0)  # (2N,)

        # For numerical stability: subtract the max before exp (log-sum-exp trick)
        # Remove self-similarity by setting diagonal to -inf
        sim = sim.masked_fill(~mask, float("-inf"))

        return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# TrainingMetrics
# ---------------------------------------------------------------------------
@dataclass
class TrainingMetrics:
    """
    Payload emitted by TrainWorker at the end of each epoch/batch.
    Keeps the Worker ↔ UI contract explicit and type-safe.
    """

    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0        # Top-1 accuracy on the validation split
    learning_rate: float = 0.0
    phase: str = "supervised"        # "ssl" | "linear_probe" | "supervised"
    extra: dict = field(default_factory=dict)  # Confusion matrix etc.


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------
class StateManager:
    """
    Checkpoint save/load and backbone freeze helpers.

    Designed as a submodule of ModelTrainer; accesses the model via
    self._parent.model.
    """

    def __init__(self, parent: "ModelTrainer"):
        self._parent = parent

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer_state: dict,
        extra_meta: Optional[dict] = None,
    ):
        """
        Save model weights, optimizer state, and optional metadata to a .pth file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "epoch": epoch,
            "model_state_dict": self._parent.model.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "num_classes": self._parent.model.fc.out_features,
        }
        if extra_meta:
            payload["meta"] = extra_meta

        torch.save(payload, str(path))
        log.info("StateManager", f"Checkpoint saved: {path.name}, epoch={epoch}")

    def load_checkpoint(self, path: Path, strict: bool = True) -> dict:
        """
        Load a checkpoint file.

        When the saved num_classes differs from the current model (e.g. during
        transfer / fine-tuning), pass strict=False to skip the head weights.

        Returns the full checkpoint dict so the caller can restore optimizer
        state, epoch, etc.
        """
        path = Path(path)
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)

        saved_classes = checkpoint.get("num_classes")
        current_classes = self._parent.model.fc.out_features

        if saved_classes != current_classes:
            # Drop the classifier head weights — backbone-only restore
            state = {
                k: v for k, v in checkpoint["model_state_dict"].items()
                if not k.startswith("fc.") and not k.startswith("projection_head.")
            }
            self._parent.model.load_state_dict(state, strict=False)
        else:
            self._parent.model.load_state_dict(
                checkpoint["model_state_dict"], strict=strict
            )

        log.info("StateManager", f"Checkpoint loaded: {path.name}, epoch={checkpoint.get('epoch', 'unknown')}")
        return checkpoint

    def freeze_backbone(self, freeze: bool = True):
        """
        Toggle requires_grad on backbone parameters (everything except the heads).
        Call with freeze=False to unfreeze for full fine-tuning.
        """
        model = self._parent.model
        for p in model.backbone_parameters:
            p.requires_grad = not freeze

    def save_label_map(self, path: Path, label_map: dict):
        """Persist the int→str label map alongside the model weights."""
        label_map_path = Path(path).with_suffix(".labels.json")
        with label_map_path.open("w", encoding="utf-8") as f:
            json.dump(label_map, f, indent=2)

    def load_label_map(self, path: Path) -> dict:
        label_map_path = Path(path).with_suffix(".labels.json")
        if not label_map_path.exists():
            return {}
        with label_map_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        # JSON keys are always strings; convert back to int
        return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# OptimizationEngine
# ---------------------------------------------------------------------------
class OptimizationEngine:
    """
    Builds the optimizer, scheduler, and selects the appropriate loss function.
    """

    def __init__(self, parent: "ModelTrainer"):
        self._parent = parent
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler = None

    def configure_optimizer(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ) -> optim.AdamW:
        """
        Initialise AdamW over all parameters that require gradients.
        Stores and returns the optimizer.
        """
        params = [p for p in self._parent.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return self.optimizer

    def configure_scheduler(
        self,
        max_lr: float,
        steps_per_epoch: int,
        epochs: int,
    ) -> OneCycleLR:
        """
        Initialise OneCycleLR (steps per batch, not per epoch).
        Must be called after configure_optimizer.
        """
        if self.optimizer is None:
            raise RuntimeError("Call configure_optimizer before configure_scheduler.")

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
        return self.scheduler

    def get_loss_function(self, mode: str = "supervised") -> nn.Module:
        """
        Return the appropriate loss for the training phase.

        mode:
            "ssl"        → NTXentLoss
            "supervised" → CrossEntropyLoss with label_smoothing=0.1
        """
        if mode == "ssl":
            return NTXentLoss(temperature=0.5)
        return nn.CrossEntropyLoss(label_smoothing=0.1)


# ---------------------------------------------------------------------------
# ValidationEngine
# ---------------------------------------------------------------------------
class ValidationEngine:
    """
    Evaluate model performance on a validation DataLoader.
    """

    def __init__(self, parent: "ModelTrainer"):
        self._parent = parent

    def run_evaluation(
        self,
        val_loader: DataLoader,
        loss_fn: Optional[nn.Module] = None,
    ) -> dict:
        """
        Run a full validation pass with torch.no_grad().

        Returns:
            {
                "val_loss"     : float,
                "val_accuracy" : float (0–100),
                "confusion_matrix": list[list[int]]  (num_classes × num_classes)
            }
        """
        model = self._parent.model
        device = self._parent.device
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        num_classes = model.fc.out_features
        conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                for t, p in zip(labels.cpu(), preds.cpu()):
                    conf_matrix[t, p] += 1

        val_accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        return {
            "val_loss": avg_loss,
            "val_accuracy": val_accuracy,
            "confusion_matrix": conf_matrix.tolist(),
        }


# ---------------------------------------------------------------------------
# ModelTrainer — parent that owns all submodules
# ---------------------------------------------------------------------------
class ModelTrainer:
    """
    Orchestrates model training across all phases (SSL, linear probe, fine-tune).

    Submodules:
        self.state_manager       — checkpoint + freeze logic
        self.optimization_engine — optimizer, scheduler, loss
        self.validation_engine   — evaluation loop

    Args:
        model       : A ScratchResNet instance.
        device      : torch.device to run on.
    """

    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(device)
        self.device = device

        # Submodules
        self.state_manager = StateManager(self)
        self.optimization_engine = OptimizationEngine(self)
        self.validation_engine = ValidationEngine(self)
