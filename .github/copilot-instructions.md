# GitHub Copilot Instructions: High-Accuracy Vision Application

## 1. Project Overview & Architectural Boundaries
This project is a high-accuracy image classification suite built from scratch using **PyTorch** and **PySide6**. It follows a strict **MVC/MVVM** architecture to ensure the separation of heavy AI logic from the user interface.

### Core Boundary Rules:
- **vision_app/core/**: The "Engine." Must contain **zero PySide6 or GUI code**. It should be capable of running as a standalone CLI.
- **vision_app/ui/**: The "Interface." Handles presentation and user events. It must not perform heavy file I/O or AI math directly.
- **vision_app/worker/**: The "Bridge." All long-running tasks (training, streaming, file pruning) must run in **QThreads** and communicate with the UI exclusively via **PySide6 Signals**.

## 2. Submodule Implementation Pattern
To prevent "God Classes," complex logic centers must be divided into submodules using a parent-reference pattern.
- **Syntax**: Instantiate submodules in the parent constructor: `self.submodule = SubmoduleClass(self)`.
- **Reference**: Submodules store the parent as `self._parent` to access shared state or sibling submodules.
- **Standard Classes**:
    - `DatasetBuilder`: Includes `FileSystemHandler`, `MetadataManager`, `DatasetTransformer`, and `DatasetValidator`.
    - `ModelTrainer`: Includes `StateManager`, `OptimizationEngine`, and `ValidationEngine`.
    - `MainWindow`: Includes `NavigationController`, `StatusBarManager`, and `ThemeEngine`.
    - `InferenceViewWidget`: Includes `SourceController`, `OverlayPainter`, and `RecordingManager`.
    - `TrainWorker`: Includes `SignalDispatcher` and `LifecycleManager`.

Refer to [docs/coding_patterns/submodule_structure.md](../docs/coding_patterns/submodule_structure.md) for a more detailed explanation and example code snippets on this pattern.

## 3. AI & Mathematical Standards (Scratch Training)
Because the model is trained without a pretrained backbone, specific initialization and architectural choices are mandatory for convergence.
- **Weight Initialization**: Use **He (Kaiming) Initialization** (`kaiming_normal_`) for all convolutional layers in `ResidualBlock` and `ScratchResNet`.
- **Global Average Pooling (GAP)**: Use `nn.AdaptiveAvgPool2d(1)` instead of flattening to dense layers to prevent overfitting and ensure input size flexibility.
- **Dual Head Switch**: Implement logic in the model's `forward()` method to toggle between a **Projection Head** (for Self-Supervised Learning) and a **Linear Classifier** (for fixed classes).
- **Optimization**: Default to **AdamW** with a **OneCycleLR** scheduler. Use **Label Smoothing (0.1)** for supervised training. Clip gradients at **max_norm=1.0**.
- **Loss Functions**: Use **NT-Xent Loss** for SSL pre-training and **CrossEntropyLoss** for the supervised phase.
- **Multi-Phase Training**: Follow the 3-phase sequence — (1) SSL Pre-train, (2) Linear Probing (frozen backbone), (3) Full Fine-Tune with discriminative LRs. Refer to [docs/core_concepts/08_ssl_to_sl_transition.md](../docs/core_concepts/08_ssl_to_sl_transition.md).

## 4. UI & Multithreading Rules
- **Non-Blocking UI**: Never run loops in the main thread. Use `TrainWorker` or `StreamWorker`.
- **Safety**: Every worker must implement a `LifecycleManager` with an `_abort = False` flag that is checked at every batch/frame to allow for emergency stops.
- **Streaming**: The `StreamWorker` should limit frame rates (e.g., 30 FPS) to prevent UI lag.

## 5. Data & Path Management
- **ImageFolder Standard**: Maintain the hierarchy: `storage/datasets/[name]/train/[class_name]/image.jpg`.
- **Path Resolution**: Always use `pathlib` for cross-platform compatibility. Resolve the `storage/` directory relative to the `main.py` location.
- **Metadata**: Every dataset modification must be reflected in a `metadata.json` "Source of Truth" file.
- **Safe Deletion**: Never permanently delete user data. Move corrupt or duplicate files to `storage/trash/` instead.

## 6. Progress Tracking Protocol
- **State-Sync**: Before starting a task, read `WORKING_PROGRESS.md` and the specific roadmap file (e.g., [docs/roadmap/milestone01.md](../docs/roadmap/milestone01.md)) to identify the current TODO.
- **Verification**: After completing code, refer to the **Verification Checklist** in the roadmap to suggest a test script.
- **Updates**: Upon task completion, propose an update to the checkboxes in `WORKING_PROGRESS.md`.
