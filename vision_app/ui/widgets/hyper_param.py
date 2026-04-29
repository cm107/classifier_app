"""
hyper_param.py — Hyperparameter configuration widget.

Classes:
    HyperParameterWidget — QFormLayout form wired to config.yaml.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vision_app.core.utils import ConfigLoader


# ---------------------------------------------------------------------------
# HyperParameterWidget
# ---------------------------------------------------------------------------
class HyperParameterWidget(QWidget):
    """
    Clean form for configuring training hyperparameters.

    All values are persisted to config.yaml on every change so they survive
    app restarts. The widget exposes a `config_changed` signal that the
    TrainingMonitorWidget listens to when building the TrainWorker config.

    Sections:
        Training Mode   — SSL / Linear Probe / Supervised
        Optimisation    — LR, max LR, weight decay
        Schedule        — Epochs, batch size, image size
        SSL             — Temperature (only enabled in SSL mode)
        Dataset         — dataset path selector (read from storage/)
    """

    config_changed = Signal(dict)   # emitted whenever any value changes

    def __init__(self, storage_root: Path, parent=None):
        super().__init__(parent)
        self._storage_root = Path(storage_root)
        self._config_loader = ConfigLoader()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        title = QLabel("Hyperparameters")
        title.setObjectName("section_title")
        layout.addWidget(title)

        # ── Training Mode ──
        mode_group = QGroupBox("Training Mode")
        mode_form = QFormLayout(mode_group)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Supervised", "SSL (Pre-train)", "Linear Probe"])
        mode_form.addRow("Phase:", self._mode_combo)

        self._dataset_combo = QComboBox()
        self._refresh_datasets()
        mode_form.addRow("Dataset:", self._dataset_combo)
        layout.addWidget(mode_group)

        # ── Optimisation ──
        opt_group = QGroupBox("Optimisation")
        opt_form = QFormLayout(opt_group)

        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setDecimals(6)
        self._lr_spin.setRange(1e-6, 1.0)
        self._lr_spin.setSingleStep(1e-4)

        self._max_lr_spin = QDoubleSpinBox()
        self._max_lr_spin.setDecimals(4)
        self._max_lr_spin.setRange(1e-5, 1.0)
        self._max_lr_spin.setSingleStep(1e-3)

        self._wd_spin = QDoubleSpinBox()
        self._wd_spin.setDecimals(5)
        self._wd_spin.setRange(0.0, 1.0)
        self._wd_spin.setSingleStep(1e-3)

        opt_form.addRow("Learning Rate:", self._lr_spin)
        opt_form.addRow("Max LR (OneCycle):", self._max_lr_spin)
        opt_form.addRow("Weight Decay:", self._wd_spin)
        layout.addWidget(opt_group)

        # ── Schedule ──
        sched_group = QGroupBox("Schedule")
        sched_form = QFormLayout(sched_group)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 10000)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 2048)

        self._img_size_spin = QSpinBox()
        self._img_size_spin.setRange(32, 1024)
        self._img_size_spin.setSingleStep(32)

        self._label_smooth_spin = QDoubleSpinBox()
        self._label_smooth_spin.setDecimals(2)
        self._label_smooth_spin.setRange(0.0, 0.5)
        self._label_smooth_spin.setSingleStep(0.05)

        sched_form.addRow("Epochs:", self._epochs_spin)
        sched_form.addRow("Batch Size:", self._batch_spin)
        sched_form.addRow("Image Size:", self._img_size_spin)
        sched_form.addRow("Label Smoothing:", self._label_smooth_spin)
        layout.addWidget(sched_group)

        # ── SSL ──
        ssl_group = QGroupBox("SSL (NT-Xent)")
        ssl_form = QFormLayout(ssl_group)

        self._temp_spin = QDoubleSpinBox()
        self._temp_spin.setDecimals(2)
        self._temp_spin.setRange(0.05, 2.0)
        self._temp_spin.setSingleStep(0.05)

        ssl_form.addRow("Temperature:", self._temp_spin)
        layout.addWidget(ssl_group)

        # ── Buttons ──
        btn_row = QHBoxLayout()
        self._save_btn = QPushButton("Save to config.yaml")
        self._reset_btn = QPushButton("Reset Defaults")
        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._reset_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        layout.addStretch()

        # ── Load saved values ──
        self._load_from_config()

        # ── SSL section only active in SSL mode ──
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._on_mode_changed(self._mode_combo.currentIndex())

        # ── Auto-save on any change ──
        for widget in (self._lr_spin, self._max_lr_spin, self._wd_spin,
                       self._epochs_spin, self._batch_spin, self._img_size_spin,
                       self._label_smooth_spin, self._temp_spin):
            widget.valueChanged.connect(self._auto_save)

        self._mode_combo.currentIndexChanged.connect(self._auto_save)
        self._dataset_combo.currentIndexChanged.connect(self._auto_save)
        self._save_btn.clicked.connect(self._save_to_config)
        self._reset_btn.clicked.connect(self._reset_defaults)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_train_config(self) -> dict:
        """Return a dict suitable for passing directly to TrainWorker."""
        phase_map = {0: "supervised", 1: "ssl", 2: "linear_probe"}
        phase = phase_map.get(self._mode_combo.currentIndex(), "supervised")
        dataset_name = self._dataset_combo.currentText()
        dataset_path = self._storage_root / "datasets" / dataset_name

        return {
            "dataset_path": dataset_path,
            "num_classes": self._count_classes(dataset_path),
            "epochs": self._epochs_spin.value(),
            "batch_size": self._batch_spin.value(),
            "learning_rate": self._lr_spin.value(),
            "max_lr": self._max_lr_spin.value(),
            "weight_decay": self._wd_spin.value(),
            "image_size": self._img_size_spin.value(),
            "label_smoothing": self._label_smooth_spin.value(),
            "temperature": self._temp_spin.value(),
            "phase": phase,
            "num_workers": 4,
            "storage_root": self._storage_root,
        }

    def refresh_datasets(self):
        self._refresh_datasets()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _refresh_datasets(self):
        self._dataset_combo.blockSignals(True)
        current = self._dataset_combo.currentText()
        self._dataset_combo.clear()
        ds_root = self._storage_root / "datasets"
        if ds_root.exists():
            for d in sorted(ds_root.iterdir()):
                if d.is_dir():
                    self._dataset_combo.addItem(d.name)
        # Restore previous selection if still available
        idx = self._dataset_combo.findText(current)
        if idx >= 0:
            self._dataset_combo.setCurrentIndex(idx)
        self._dataset_combo.blockSignals(False)

    def _on_mode_changed(self, index: int):
        ssl_active = (index == 1)  # "SSL (Pre-train)"
        self._temp_spin.setEnabled(ssl_active)
        self._label_smooth_spin.setEnabled(index != 1)

    def _load_from_config(self):
        try:
            cfg = self._config_loader.load()
            d = cfg.get("defaults", {})
            self._lr_spin.setValue(d.get("learning_rate", 1e-3))
            self._max_lr_spin.setValue(d.get("max_lr", 1e-2))
            self._wd_spin.setValue(d.get("weight_decay", 1e-2))
            self._epochs_spin.setValue(d.get("epochs", 50))
            self._batch_spin.setValue(d.get("batch_size", 32))
            self._img_size_spin.setValue(d.get("image_size", 224))
            self._label_smooth_spin.setValue(d.get("label_smoothing", 0.1))
            self._temp_spin.setValue(d.get("temperature", 0.5))
        except Exception:
            self._reset_defaults()

    def _save_to_config(self):
        try:
            cfg = self._config_loader.load()
            cfg.setdefault("defaults", {}).update({
                "learning_rate": self._lr_spin.value(),
                "max_lr": self._max_lr_spin.value(),
                "weight_decay": self._wd_spin.value(),
                "epochs": self._epochs_spin.value(),
                "batch_size": self._batch_spin.value(),
                "image_size": self._img_size_spin.value(),
                "label_smoothing": self._label_smooth_spin.value(),
                "temperature": self._temp_spin.value(),
            })
            self._config_loader.save(cfg)
        except Exception:
            pass  # non-critical; UI still works without persistence

    def _auto_save(self, *_):
        self._save_to_config()
        self.config_changed.emit(self.get_train_config())

    def _reset_defaults(self):
        self._lr_spin.setValue(1e-3)
        self._max_lr_spin.setValue(1e-2)
        self._wd_spin.setValue(1e-2)
        self._epochs_spin.setValue(50)
        self._batch_spin.setValue(32)
        self._img_size_spin.setValue(224)
        self._label_smooth_spin.setValue(0.1)
        self._temp_spin.setValue(0.5)

    @staticmethod
    def _count_classes(dataset_path: Path) -> int:
        train = dataset_path / "train"
        if not train.exists():
            return 2  # safe fallback
        return max(len([d for d in train.iterdir() if d.is_dir()]), 1)
