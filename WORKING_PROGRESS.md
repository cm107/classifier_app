
# Project Working Progress: Vision App

**Current Milestone:** Milestone 1: The Data Foundation (Core Logic)
**Current Todo:** 1.1.1 FileSystemHandler Submodule
**Status:** 🟡 In Progress

---

## ✅ Completed Tasks
- [x] Created `.github/copilot-instructions.md` and specialized instruction files
- [x] Initialized Git repository and environment
- [x] Defined Project Directory Structure (MVC/MVVM)

---

## 🚀 Next Immediate Tasks
- [ ] Create the `__init__.py` files and empty scripts for each module listed in the roadmap
- [ ] Implement `FileSystemHandler` in `data_manager.py`
- [ ] Implement `MetadataManager` for `metadata.json` management
- [ ] Implement `DatasetTransformer` for 80/20 train/val splitting
- [ ] Create `test_data.py` to verify Milestone 1 logic

---

## 🗺️ Master Roadmap

### Milestone 1: The Data Foundation (Core Logic)
- [ ] **1.1 Data Management & Dataset Integration**
    - [ ] `FileSystemHandler` (mkdir, move, scan)
    - [ ] `MetadataManager` (metadata.json "Source of Truth")
    - [ ] `DatasetTransformer` (Splitting, Merging, Pruning)
    - [ ] `DatasetValidator` (Corrupt file detection, class balance checking)
    - [ ] `ClassificationDataset` class
    - [ ] `ContrastiveTransformations` (SSL dual-views)
    - [ ] `StandardTransformations` (Supervised training/validation transforms)
    - [ ] `StatsCalculator` in `utils.py` (per-channel mean/std for custom normalization)
- [ ] **1.2 Verification** — `test_data.py` (create, split, load batch)

### Milestone 2: The Model & Training Engine
- [ ] **2.1 Architecture**
    - [ ] `ResidualBlock` (He Init, Shortcut Path)
    - [ ] `ScratchResNet` (Global Average Pooling, Dual Head Switch)
- [ ] **2.2 Training Logic**
    - [ ] `StateManager` (Checkpointing, freeze_backbone)
    - [ ] `OptimizationEngine` (AdamW, OneCycleLR)
    - [ ] `ValidationEngine` (Top-1 Accuracy, Confusion Matrix)
    - [ ] `TrainingMetrics` data class (loss, accuracy, lr, epoch)
    - [ ] `NTXentLoss` (similarity matrix, diagonal mask)
- [ ] **2.3 Verification** — shape test, freeze test, checkpoint integrity

### Milestone 3: Multithreading & Async Operations
- [ ] **3.1 Workers**
    - [ ] `TrainWorker` (SignalDispatcher, LifecycleManager, run() loop)
    - [ ] `StreamWorker` (OpenCV capture, FPS limiting, QImage emit)
- [ ] **3.2 Configuration**
    - [ ] `config.yaml` schema (paths, defaults, hardware)
    - [ ] `ConfigLoader` utility in `utils.py`
- [ ] **3.3 Verification** — signal test, thread-safety test, abort test

### Milestone 4: The GUI Shell & Dataset UI
- [ ] **4.1 Main Window**
    - [ ] `NavigationController` (QStackedWidget sidebar)
    - [ ] `StatusBarManager` (GPU info, notifications)
    - [ ] `ThemeEngine` (QSS dark mode, style.qss)
- [ ] **4.2 Dataset Tab**
    - [ ] `DatasetManagerWidget` (QTreeWidget logic)
    - [ ] `PruningDialog` (duplicate/corrupt checkboxes, QProgressBar)
- [ ] **4.3 Entry Point**
    - [ ] `main.py` (High-DPI, QApplication, storage dir creation)
- [ ] **4.4 Verification** — tab nav, live QTreeWidget update, path resolution

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
