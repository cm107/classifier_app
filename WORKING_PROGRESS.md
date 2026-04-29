
# Project Working Progress: Vision App

**Current Milestone:** Milestone 5: Training Monitor & Hyperparameters
**Current Todo:** 5.1.1 HyperParameterWidget
**Status:** 🟡 In Progress

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

---

## 🚀 Next Immediate Tasks
- [ ] Implement `HyperParameterWidget` (QFormLayout, QSpinBox, mode selector, config.yaml persistence)
- [ ] Implement `LiveGraph` (pyqtgraph integration, dynamic scaling)
- [ ] Implement `TrainingMonitorWidget` (QProgressBar, metric cards, Start/Abort)
- [ ] Wire `TrainWorker` signals to UI (progress bar, graph, status)
- [ ] Verify input bounds, graph performance, persistence, graceful abort

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
- [ ] **5.1 Monitor UI**
    - [ ] `HyperParameterWidget` (QFormLayout, QSpinBox, mode selector, config.yaml persistence)
    - [ ] `LiveGraph` (pyqtgraph integration, dynamic scaling)
    - [ ] `TrainingMonitorWidget` (QProgressBar, metric cards, Start/Abort buttons)
- [ ] **5.2 Controller Wiring**
    - [ ] Worker instantiation on Start, signal connections, cleanup on finish/abort
- [ ] **5.3 Verification** — input bounds, graph performance, persistence, graceful abort

### Milestone 6: Management & Live Inference
- [ ] **6.1 Model Library**
    - [ ] `ModelManagerWidget` (QListView, metadata display, Load/Export/Delete actions)
    - [ ] `ModelItemDelegate` (status icon styling)
- [ ] **6.2 Live View**
    - [ ] `SourceController` (Webcam / Video / Image Directory switcher)
    - [ ] `CameraViewport` (QLabel, high-frequency setPixmap, scaling)
    - [ ] `OverlayPainter` (Top-3 predictions with percentage bars)
    - [ ] `RecordingManager` (Save inference session to video / export snapshots)
- [ ] **6.3 Inference Logic**
    - [ ] `InferenceEngine` (predict_batch, Softmax logits)
    - [ ] `StreamWorker` integration (frame_ready → CameraViewport, results_ready → OverlayPainter)
- [ ] **6.4 Final Polish**
    - [ ] `ModelExporter` in `utils.py` (TorchScript tracing, optional ONNX)
    - [ ] `MainWindow` close event (abort all workers, thread.wait())
- [ ] **6.5 Verification** — model persistence, webcam switch, overlay accuracy, GPU idle after stop
