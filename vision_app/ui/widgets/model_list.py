"""
model_list.py — Model Management UI.

Classes:
    ModelItemDelegate  — Paints a coloured status circle in the first column.
    ModelManagerWidget — Scans storage/models/ for checkpoints; load/export/delete.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PySide6.QtCore import QModelIndex, QRect, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vision_app.core.utils import ModelExporter
from vision_app.core.logger import log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_ckpt_meta(path: Path) -> dict:
    """
    Load just the scalar metadata from a checkpoint without touching the full
    model weights (but we do need weights_only=False for the meta dict).

    Returns a dict with keys: epoch, num_classes, val_accuracy, label_map.
    Missing keys fall back to safe defaults.
    """
    try:
        # Load to CPU; we only need the metadata, not the state dict tensors.
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        meta = ckpt.get("meta", {})
        return {
            "epoch":        ckpt.get("epoch", 0),
            "num_classes":  ckpt.get("num_classes", 0),
            "val_accuracy": meta.get("val_accuracy", 0.0),
            "phase":        meta.get("phase", "—"),
            "label_map":    meta.get("label_map", {}),
        }
    except Exception:
        return {"epoch": 0, "num_classes": 0, "val_accuracy": 0.0,
                "phase": "—", "label_map": {}}


# ---------------------------------------------------------------------------
# ModelItemDelegate
# ---------------------------------------------------------------------------
_COL_STATUS = 0

class ModelItemDelegate(QStyledItemDelegate):
    """
    Renders the first column of the model table as a coloured circle.

    Colour rules:
        green  — val_accuracy >= 90 %
        yellow — val_accuracy >= 70 %
        red    — val_accuracy <  70 % (or unknown)
    """

    _COLORS = {
        "green":  QColor("#a6e3a1"),
        "yellow": QColor("#f9e2af"),
        "red":    QColor("#f38ba8"),
    }

    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index: QModelIndex):
        if index.column() != _COL_STATUS:
            super().paint(painter, option, index)
            return

        acc = index.data(Qt.ItemDataRole.UserRole)
        if acc is None:
            acc = 0.0

        if acc >= 90.0:
            color = self._COLORS["green"]
        elif acc >= 70.0:
            color = self._COLORS["yellow"]
        else:
            color = self._COLORS["red"]

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx = option.rect.center().x()
        cy = option.rect.center().y()
        radius = min(option.rect.width(), option.rect.height()) // 2 - 4
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(28, 28)


# ---------------------------------------------------------------------------
# ModelManagerWidget
# ---------------------------------------------------------------------------
class ModelManagerWidget(QWidget):
    """
    "Library" tab for viewing, loading, exporting, and deleting trained models.

    Signals:
        model_selected(Path) — emitted when the user clicks "Load for Fine-Tune".
                               Connect this to HyperParameterWidget.set_model_path().
    """

    model_selected = Signal(Path)

    # Table columns
    _COLS = ["●", "Name", "Type", "Classes", "Accuracy", "Phase", "Epoch", "Modified"]
    _C_STATUS   = 0
    _C_NAME     = 1
    _C_TYPE     = 2
    _C_CLASSES  = 3
    _C_ACC      = 4
    _C_PHASE    = 5
    _C_EPOCH    = 6
    _C_MODIFIED = 7

    def __init__(self, storage_root: Path, parent=None):
        super().__init__(parent)
        self._storage_root = Path(storage_root)
        self._models_root  = self._storage_root / "models"
        self._exporter = ModelExporter()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Trained Models")
        title.setObjectName("section_title")
        layout.addWidget(title)

        # ── Table ──
        self._table = QTableWidget(0, len(self._COLS))
        self._table.setHorizontalHeaderLabels(self._COLS)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setItemDelegateForColumn(_COL_STATUS, ModelItemDelegate(self))
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setColumnWidth(self._C_STATUS, 32)
        self._table.setColumnWidth(self._C_TYPE,   72)
        self._table.setColumnWidth(self._C_CLASSES, 64)
        self._table.setColumnWidth(self._C_ACC,    80)
        self._table.setColumnWidth(self._C_PHASE,  90)
        self._table.setColumnWidth(self._C_EPOCH,  56)
        layout.addWidget(self._table)

        # ── Buttons ──
        btn_row = QHBoxLayout()
        self._load_btn   = QPushButton("Load for Fine-Tune")
        self._load_btn.setObjectName("primary_button")
        self._export_btn = QPushButton("Export to TorchScript")
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setObjectName("danger_button")
        self._refresh_btn = QPushButton("Refresh")

        for btn in (self._load_btn, self._export_btn,
                    self._delete_btn, self._refresh_btn):
            btn_row.addWidget(btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Status ──
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #a6adc8;")
        layout.addWidget(self._status_label)

        # ── Wire ──
        self._load_btn.clicked.connect(self._on_load)
        self._export_btn.clicked.connect(self._on_export)
        self._delete_btn.clicked.connect(self._on_delete)
        self._refresh_btn.clicked.connect(self.refresh)

        self.refresh()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def refresh(self):
        """Re-scan storage/models/ and repopulate the table."""
        self._table.setRowCount(0)
        if not self._models_root.exists():
            return

        for path in sorted(self._models_root.iterdir()):
            if path.suffix not in (".pth", ".pt"):
                continue
            self._add_row(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_row(self, path: Path):
        file_type = "PyTorch" if path.suffix == ".pth" else "TorchScript"
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )

        meta = _read_ckpt_meta(path) if path.suffix == ".pth" else {
            "epoch": 0, "num_classes": 0, "val_accuracy": 0.0, "phase": "—", "label_map": {}
        }
        acc = meta["val_accuracy"]

        row = self._table.rowCount()
        self._table.insertRow(row)

        # Status circle — accuracy stored in UserRole
        status_item = QTableWidgetItem()
        status_item.setData(Qt.ItemDataRole.UserRole, acc)
        status_item.setData(Qt.ItemDataRole.UserRole + 1, str(path))  # stash full path
        self._table.setItem(row, self._C_STATUS, status_item)

        def _cell(text: str) -> QTableWidgetItem:
            item = QTableWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole + 1, str(path))
            return item

        self._table.setItem(row, self._C_NAME,     _cell(path.name))
        self._table.setItem(row, self._C_TYPE,     _cell(file_type))
        self._table.setItem(row, self._C_CLASSES,  _cell(str(meta["num_classes"])))
        self._table.setItem(row, self._C_ACC,      _cell(f"{acc:.1f}%"))
        self._table.setItem(row, self._C_PHASE,    _cell(meta["phase"]))
        self._table.setItem(row, self._C_EPOCH,    _cell(str(meta["epoch"])))
        self._table.setItem(row, self._C_MODIFIED, _cell(mtime))

        self._table.setRowHeight(row, 30)

    def _selected_path(self) -> Optional[Path]:
        row = self._table.currentRow()
        if row < 0:
            return None
        item = self._table.item(row, self._C_NAME)
        if item is None:
            return None
        raw = item.data(Qt.ItemDataRole.UserRole + 1)
        return Path(raw) if raw else None

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_load(self):
        path = self._selected_path()
        if path is None:
            self._status_label.setText("Select a model first.")
            return
        if path.suffix != ".pth":
            self._status_label.setText("Only .pth checkpoints can be loaded for fine-tuning.")
            return
        self.model_selected.emit(path)
        log.info("ModelManagerWidget", f"Model loaded for fine-tuning: {path.name}")
        self._status_label.setText(f"Loaded: {path.name}")

    def _on_export(self):
        path = self._selected_path()
        if path is None:
            self._status_label.setText("Select a .pth model first.")
            return
        if path.suffix != ".pth":
            self._status_label.setText("Select a .pth checkpoint to export.")
            return

        meta = _read_ckpt_meta(path)
        output_path = path.with_suffix(".pt")

        self._status_label.setText("Exporting…")
        try:
            # Reconstruct model for tracing
            from vision_app.core.model import ScratchResNet
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            model = ScratchResNet(num_classes=meta["num_classes"])
            model.load_state_dict(ckpt["model_state_dict"], strict=False)

            img_size = 224  # standard; could read from meta if stored
            self._exporter.export_torchscript(
                model, output_path,
                example_input_size=(1, 3, img_size, img_size),
            )
            self._status_label.setText(f"Exported → {output_path.name}")
            self.refresh()
        except Exception as exc:
            log.exception("ModelManagerWidget", f"Failed to export model {path.name}: {exc}")
            self._status_label.setText(f"Export failed: {exc}")

    def _on_delete(self):
        path = self._selected_path()
        if path is None:
            self._status_label.setText("Select a model first.")
            return

        reply = QMessageBox.question(
            self,
            "Delete Model",
            f"Permanently delete '{path.name}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            path.unlink()
            log.info("ModelManagerWidget", f"Model deleted: {path.name}")
            self._status_label.setText(f"Deleted: {path.name}")
            self.refresh()
        except OSError as exc:
            log.exception("ModelManagerWidget", f"Failed to delete model {path.name}: {exc}")
            self._status_label.setText(f"Delete failed: {exc}")
