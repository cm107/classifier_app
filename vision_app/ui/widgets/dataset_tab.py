"""
dataset_tab.py — Dataset Management UI.

Classes:
    DatasetManagerWidget — QTreeWidget view of datasets + action buttons.
    PruningDialog        — Modal dialog for duplicate/corrupt pruning.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vision_app.core.data_manager import DatasetBuilder


# ---------------------------------------------------------------------------
# PruningWorker — runs prune_dataset off the GUI thread
# ---------------------------------------------------------------------------
class _PruningWorker(QThread):
    """Thin QThread wrapper so pruning doesn't block the UI."""

    finished = Signal(int)   # number of files moved to trash

    def __init__(self, builder: DatasetBuilder, dataset_path: Path, criteria: dict):
        super().__init__()
        self._builder = builder
        self._dataset_path = dataset_path
        self._criteria = criteria

    def run(self):
        # Count files before
        before = sum(1 for _ in self._dataset_path.rglob("*") if _.is_file())
        self._builder.dataset_transformer.prune_dataset(
            self._dataset_path, self._criteria
        )
        after = sum(1 for _ in self._dataset_path.rglob("*") if _.is_file())
        self.finished.emit(before - after)


# ---------------------------------------------------------------------------
# PruningDialog
# ---------------------------------------------------------------------------
class PruningDialog(QDialog):
    """
    Modal dialog with checkboxes for duplicate / corrupt detection
    and a progress bar that animates during the background prune operation.
    """

    def __init__(self, builder: DatasetBuilder, dataset_path: Path, parent=None):
        super().__init__(parent)
        self._builder = builder
        self._dataset_path = dataset_path
        self._worker: _PruningWorker | None = None

        self.setWindowTitle(f"Prune — {dataset_path.name}")
        self.setFixedSize(380, 240)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        layout.addWidget(QLabel(f"<b>Dataset:</b> {dataset_path.name}"))

        self._dup_cb = QCheckBox("Remove duplicate files (MD5 hash)")
        self._dup_cb.setChecked(True)
        layout.addWidget(self._dup_cb)

        self._corrupt_cb = QCheckBox("Remove corrupt / unreadable images")
        self._corrupt_cb.setChecked(True)
        layout.addWidget(self._corrupt_cb)

        self._status_label = QLabel("Ready.")
        layout.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self._start_prune)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    def _start_prune(self):
        criteria = {
            "remove_duplicates": self._dup_cb.isChecked(),
            "remove_corrupt": self._corrupt_cb.isChecked(),
        }
        if not any(criteria.values()):
            self.accept()
            return

        self._buttons.setEnabled(False)
        self._progress.setVisible(True)
        self._status_label.setText("Pruning… (files moved to trash/)")

        self._worker = _PruningWorker(self._builder, self._dataset_path, criteria)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_done(self, removed: int):
        self._progress.setVisible(False)
        self._status_label.setText(f"Done — {removed} file(s) moved to trash.")
        self._buttons.setEnabled(True)
        self._buttons.button(QDialogButtonBox.StandardButton.Cancel).setVisible(False)
        self._buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Close")
        self._buttons.accepted.disconnect(self._start_prune)
        self._buttons.accepted.connect(self.accept)


# ---------------------------------------------------------------------------
# DatasetManagerWidget
# ---------------------------------------------------------------------------
class DatasetManagerWidget(QWidget):
    """
    Main Dataset tab UI.

    Layout:
        ┌─ toolbar ─────────────────────────────┐
        │ [New Dataset] [Import] [Split] [Prune] │
        ├─ tree ────────────────────────────────┤
        │ MyDataset/                             │
        │   train/   A: 120  B: 115             │
        │   val/     A:  30  B:  28             │
        │   test/    A:  15  B:  14             │
        └───────────────────────────────────────┘
    """

    # Emitted after any operation that modifies the dataset tree
    datasets_changed = Signal()

    def __init__(self, storage_root: Path, parent=None):
        super().__init__(parent)
        self._storage_root = Path(storage_root)
        self._builder = DatasetBuilder(self._storage_root)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # --- Title ---
        title = QLabel("Dataset Manager")
        title.setObjectName("section_title")
        layout.addWidget(title)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self._btn_new = QPushButton("New Dataset")
        self._btn_import = QPushButton("Import Images")
        self._btn_split = QPushButton("Split")
        self._btn_prune = QPushButton("Prune")
        self._btn_refresh = QPushButton("↻ Refresh")

        for btn in (self._btn_new, self._btn_import, self._btn_split,
                    self._btn_prune, self._btn_refresh):
            toolbar.addWidget(btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # --- Tree ---
        self._tree = QTreeWidget()
        self._tree.setObjectName("dataset_tree")
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["Name", "Split", "Images"])
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.setAlternatingRowColors(True)
        layout.addWidget(self._tree)

        # --- Status line ---
        self._status = QLabel("")
        layout.addWidget(self._status)

        # --- Wire buttons ---
        self._btn_new.clicked.connect(self._on_new_dataset)
        self._btn_import.clicked.connect(self._on_import_images)
        self._btn_split.clicked.connect(self._on_split)
        self._btn_prune.clicked.connect(self._on_prune)
        self._btn_refresh.clicked.connect(self.refresh_tree)

        self.datasets_changed.connect(self.refresh_tree)

        # Initial load
        self.refresh_tree()

    # ------------------------------------------------------------------
    # Tree population
    # ------------------------------------------------------------------

    def refresh_tree(self):
        """Re-scan storage/datasets/ and rebuild the QTreeWidget."""
        self._tree.clear()
        datasets_root = self._storage_root / "datasets"
        if not datasets_root.exists():
            return

        for ds_dir in sorted(datasets_root.iterdir()):
            if not ds_dir.is_dir():
                continue

            metadata = self._builder.metadata_manager.load_metadata(ds_dir)
            total = metadata.get("total_images", "—")

            root_item = QTreeWidgetItem([ds_dir.name, "", str(total)])
            root_item.setData(0, Qt.ItemDataRole.UserRole, str(ds_dir))

            for split in ("train", "val", "test"):
                split_path = ds_dir / split
                if not split_path.exists():
                    continue
                dist = self._builder.file_system_handler.get_class_distribution(split_path)
                count = sum(dist.values())
                split_item = QTreeWidgetItem(["", split, str(count)])
                split_item.setForeground(1, self._split_color(split))
                for cls, n in sorted(dist.items()):
                    cls_item = QTreeWidgetItem(["", f"  {cls}", str(n)])
                    split_item.addChild(cls_item)
                root_item.addChild(split_item)

            self._tree.addTopLevelItem(root_item)
            root_item.setExpanded(True)

    @staticmethod
    def _split_color(split: str):
        from PySide6.QtGui import QColor
        colors = {"train": "#a6e3a1", "val": "#89b4fa", "test": "#fab387"}
        return QColor(colors.get(split, "#cdd6f4"))

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_new_dataset(self):
        name, ok = QInputDialog.getText(self, "New Dataset", "Dataset name:")
        if not ok or not name.strip():
            return
        name = name.strip()

        classes_raw, ok = QInputDialog.getText(
            self, "Classes", "Comma-separated class names (e.g. Cat,Dog):"
        )
        if not ok or not classes_raw.strip():
            return

        classes = [c.strip() for c in classes_raw.split(",") if c.strip()]
        if not classes:
            QMessageBox.warning(self, "Error", "Please enter at least one class name.")
            return

        self._builder.file_system_handler.create_dataset_structure(name, classes)
        self._builder.metadata_manager.generate_metadata(
            self._storage_root / "datasets" / name
        )
        self._status.setText(f"Created dataset '{name}' with {len(classes)} classes.")
        self.datasets_changed.emit()

    def _on_import_images(self):
        """Import a folder of images into a class of the selected dataset."""
        ds_path = self._selected_dataset_path()
        if ds_path is None:
            QMessageBox.information(self, "Select Dataset",
                                    "Please select a dataset in the tree first.")
            return

        src_dir = QFileDialog.getExistingDirectory(
            self, "Select Source Folder", str(Path.home())
        )
        if not src_dir:
            return

        class_name, ok = QInputDialog.getText(
            self, "Class Name", "Assign to class:"
        )
        if not ok or not class_name.strip():
            return

        class_name = class_name.strip()
        dest = ds_path / "train" / class_name

        count = 0
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for img in sorted(Path(src_dir).iterdir()):
            if img.is_file() and img.suffix.lower() in exts:
                self._builder.file_system_handler.move_sample(img, dest)
                count += 1

        self._builder.metadata_manager.update_stats(ds_path)
        self._status.setText(f"Imported {count} image(s) → {ds_path.name}/train/{class_name}")
        self.datasets_changed.emit()

    def _on_split(self):
        ds_path = self._selected_dataset_path()
        if ds_path is None:
            QMessageBox.information(self, "Select Dataset",
                                    "Please select a dataset in the tree first.")
            return

        raw_path = ds_path / "raw"
        if not raw_path.exists():
            QMessageBox.warning(
                self, "No Raw Folder",
                f"No 'raw/' folder found in {ds_path.name}.\n"
                "Import images first; they land in train/ by default.\n"
                "To re-split, place images in raw/<class>/ subdirectories."
            )
            return

        ratio_str, ok = QInputDialog.getText(
            self, "Split Ratio",
            "Enter train/val/test ratios (e.g. 0.8,0.1,0.1):",
            text="0.8,0.1,0.1",
        )
        if not ok:
            return

        try:
            parts = [float(x.strip()) for x in ratio_str.split(",")]
            assert len(parts) == 3 and abs(sum(parts) - 1.0) < 1e-4
        except Exception:
            QMessageBox.warning(self, "Invalid Ratio",
                                "Please enter three numbers that sum to 1.0.")
            return

        self._builder.dataset_transformer.split_train_val_test(
            raw_path, ratios=tuple(parts)
        )
        self._builder.metadata_manager.update_stats(ds_path)
        self._status.setText(
            f"Split '{ds_path.name}' → "
            f"{int(parts[0]*100)}/{int(parts[1]*100)}/{int(parts[2]*100)}"
        )
        self.datasets_changed.emit()

    def _on_prune(self):
        ds_path = self._selected_dataset_path()
        if ds_path is None:
            QMessageBox.information(self, "Select Dataset",
                                    "Please select a dataset in the tree first.")
            return

        dlg = PruningDialog(self._builder, ds_path, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._builder.metadata_manager.update_stats(ds_path)
            self._status.setText(f"Pruning complete for '{ds_path.name}'.")
            self.datasets_changed.emit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _selected_dataset_path(self) -> Path | None:
        """Return the dataset root path of the currently selected tree item."""
        item = self._tree.currentItem()
        if item is None:
            return None
        # Walk up to top-level item
        while item.parent() is not None:
            item = item.parent()
        data = item.data(0, Qt.ItemDataRole.UserRole)
        return Path(data) if data else None
