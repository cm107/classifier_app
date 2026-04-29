
# Project Working Progress: Vision App

**Current Milestone:** Milestone 6 Complete ✅
**Status:** 🟢 All Milestones Done

---

## ✅ Completed Tasks
- [x] Created `.github/copilot-instructions.md` and specialized instruction files
- [x] Initialized Git repository and environment
- [x] Defined Project Directory Structure (MVC/MVVM)
- [x] Created all `__init__.py` files and skeleton scripts for every module
- [x] Implemented `FileSystemHandler` in `data_manager.py`
- [x] Implemented `MetadataManager` for `metadata.json` management
- [x] Implemented `DatasetTransformer` (split, merge, prune with trash)
- [x] Implemented `DatasetValidator` (corrupt detection, class balance)
- [x] Implemented `ClassificationDataset` with label smoothing
- [x] Implemented `ContrastiveTransformations` (SSL dual-view)
- [x] Implemented `StandardTransformations` (train/val)
- [x] Implemented `StatsCalculator` in `utils.py`
- [x] Created `tests/test_data.py` — all 5 tests pass (Milestone 1.2 ✅)
- [x] Implemented `ResidualBlock` with He initialization and shortcut path
- [x] Implemented `ScratchResNet` with GAP, dual head switch, backbone_parameters
- [x] Implemented `NTXentLoss` (NT-Xent with temperature scaling and diagonal mask)
- [x] Implemented `TrainingMetrics` dataclass
- [x] Implemented `StateManager` (save/load checkpoint, freeze/unfreeze backbone)
- [x] Implemented `OptimizationEngine` (AdamW, OneCycleLR, loss factory)
- [x] Implemented `ValidationEngine` (Top-1 accuracy, confusion matrix)
- [x] Created `tests/test_model.py` — all 8 tests pass (Milestone 2 ✅)
- [x] Implemented `TrainWorker` with `SignalDispatcher` and `LifecycleManager` submodules
- [x] Implemented `StreamWorker` with FPS limiting and QImage emit
- [x] Created `config.yaml` schema (paths, defaults, hardware)
- [x] Implemented `ConfigLoader` in `utils.py` (load/save/get)
- [x] Created `tests/test_workers.py` — all 5 tests pass (Milestone 3 ✅)
- [x] Implemented `NavigationController` (sidebar QListWidget + QStackedWidget)
- [x] Implemented `StatusBarManager` (GPU label + auto-clear notifications)
- [x] Implemented `ThemeEngine` (loads style.qss)
- [x] Created `vision_app/ui/resources/style.qss` dark mode stylesheet
- [x] Implemented `DatasetManagerWidget` (QTreeWidget, New/Import/Split/Prune buttons)
- [x] Implemented `PruningDialog` (checkboxes, QProgressBar, background QThread)
- [x] Implemented `main.py` entry point (High-DPI, QApplication, storage dirs)
- [x] Milestone 4 imports verified ✅
- [x] Implemented `HyperParameterWidget` (QFormLayout, QSpinBox/QDoubleSpinBox, mode selector, config.yaml auto-save)
- [x] Implemented `LiveGraph` (pyqtgraph dual-axis: loss left, accuracy right)
- [x] Implemented `TrainingMonitorWidget` (metric cards, epoch progress bar, Start/Abort wired to TrainWorker)
- [x] Wired Train page into MainWindow (HyperParameter panel + monitor side-by-side)
- [x] Milestone 5 imports verified ✅
- [x] Updated `TrainWorker` to save `label_map` + `val_accuracy` in checkpoint `meta`
- [x] Added "Resume / Transfer" section to `HyperParameterWidget` (`set_model_path()`, clear button)
- [x] Implemented `ModelExporter` — TorchScript tracing + ONNX export
- [x] Implemented `InferenceEngine` — loads checkpoint, `predict_batch()`, `predict_single()`
- [x] Implemented `ModelManagerWidget` + `ModelItemDelegate` — table view, load/export/delete buttons
- [x] Implemented `InferenceViewWidget` — `SourceController`, `CameraViewport`, `OverlayPainter`, `RecordingManager`
- [x] Wired pages 2 (Models) and 3 (Inference) into `MainWindow`
- [x] Implemented `closeEvent` — aborts TrainWorker + stops StreamWorker before exit
- [x] Installed `opencv-python-headless` for `cv2` support
- [x] Milestone 6 imports verified ✅

---

## 🚀 Next Immediate Tasks
- All milestones complete. Ready for end-to-end system verification:
  - [ ] Train a small dataset and verify checkpoint appears in Models tab
  - [ ] Test "Load for Fine-Tune" flow (Models → Train page)
  - [ ] Test webcam / video file stream in Inference tab
  - [ ] Test Export to TorchScript
  - [ ] Monitor GPU usage during inference, verify idle after stop

---

## 🗺️ Master Roadmap

### Milestone 1: The Data Foundation (Core Logic)
- [x] **1.1 Data Management & Dataset Integration**
    - [x] `FileSystemHandler` (mkdir, move, scan)
    - [x] `MetadataManager` (metadata.json "Source of Truth")
    - [x] `DatasetTransformer` (Splitting, Merging, Pruning)
    - [x] `DatasetValidator` (Corrupt file detection, class balance checking)
    - [x] `ClassificationDataset` class
    - [x] `ContrastiveTransformations` (SSL dual-views)
    - [x] `StandardTransformations` (Supervised training/validation transforms)
    - [x] `StatsCalculator` in `utils.py` (per-channel mean/std for custom normalization)
- [x] **1.2 Verification** — `test_data.py` (create, split, load batch)

### Milestone 2: The Model & Training Engine
- [x] **2.1 Architecture**
    - [x] `ResidualBlock` (He Init, Shortcut Path)
    - [x] `ScratchResNet` (Global Average Pooling, Dual Head Switch)
- [x] **2.2 Training Logic**
    - [x] `StateManager` (Checkpointing, freeze_backbone)
    - [x] `OptimizationEngine` (AdamW, OneCycleLR)
    - [x] `ValidationEngine` (Top-1 Accuracy, Confusion Matrix)
    - [x] `TrainingMetrics` data class (loss, accuracy, lr, epoch)
    - [x] `NTXentLoss` (similarity matrix, diagonal mask)
- [x] **2.3 Verification** — shape test, freeze test, checkpoint integrity

### Milestone 3: Multithreading & Async Operations
- [x] **3.1 Workers**
    - [x] `TrainWorker` (SignalDispatcher, LifecycleManager, run() loop)
    - [x] `StreamWorker` (OpenCV capture, FPS limiting, QImage emit)
- [x] **3.2 Configuration**
    - [x] `config.yaml` schema (paths, defaults, hardware)
    - [x] `ConfigLoader` utility in `utils.py`
- [x] **3.3 Verification** — signal test, thread-safety test, abort test

### Milestone 4: The GUI Shell & Dataset UI
- [x] **4.1 Main Window**
    - [x] `NavigationController` (QStackedWidget sidebar)
    - [x] `StatusBarManager` (GPU info, notifications)
    - [x] `ThemeEngine` (QSS dark mode, style.qss)
- [x] **4.2 Dataset Tab**
    - [x] `DatasetManagerWidget` (QTreeWidget logic)
    - [x] `PruningDialog` (duplicate/corrupt checkboxes, QProgressBar)
- [x] **4.3 Entry Point**
    - [x] `main.py` (High-DPI, QApplication, storage dir creation)
- [x] **4.4 Verification** — imports verified, ready for visual testing

### Milestone 5: Training Monitor & Hyperparameters
- [x] **5.1 Monitor UI**
    - [x] `HyperParameterWidget` (QFormLayout, QSpinBox, mode selector, config.yaml persistence)
    - [x] `LiveGraph` (pyqtgraph integration, dynamic scaling, dual y-axis)
    - [x] `TrainingMonitorWidget` (QProgressBar, metric cards, Start/Abort buttons)
- [x] **5.2 Controller Wiring**
    - [x] Worker instantiation on Start, signal connections, cleanup on finish/abort
- [x] **5.3 Verification** — imports verified, ready for visual testing

### Milestone 6: Management & Live Inference
- [x] **6.1 Model Library**
    - [x] `InferenceEngine` (predict_batch, predict_single, checkpoint loading with label_map)
    - [x] `ModelExporter` in `utils.py` (TorchScript tracing + optional ONNX)
    - [x] `ModelManagerWidget` (QTableWidget, metadata display, Load/Export/Delete actions)
    - [x] `ModelItemDelegate` (coloured accuracy circle: green ≥90%, yellow ≥70%, red <70%)
    - [x] "Resume / Transfer" field added to `HyperParameterWidget`
    - [x] `TrainWorker` updated to persist `label_map` + `val_accuracy` in checkpoint `meta`
- [x] **6.2 Live View**
    - [x] `SourceController` (Webcam 0/1, Video File, Image Directory, Play/Stop toggle)
    - [x] `CameraViewport` (QLabel, aspect-ratio scaling, overlay-in-sync resizeEvent)
    - [x] `OverlayPainter` (transparent overlay, top-k predictions + probability bars)
    - [x] `RecordingManager` (cv2.VideoWriter, toggle record/stop, saves to storage/logs/)
    - [x] `InferenceViewWidget` (all submodules wired, StreamWorker integration)
- [x] **6.3 App Cleanup**
    - [x] `MainWindow.closeEvent` (abort TrainWorker + stop StreamWorker before exit)
    - [x] Pages 2-3 wired into `MainWindow._build_pages()`
- [x] **6.4 Verification** — all imports verified ✅, ready for end-to-end testing
