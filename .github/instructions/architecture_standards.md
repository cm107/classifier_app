# Architecture Standards: MVC/MVVM & Boundary Rules

## 1. Core Philosophy: Separation of Concerns
This project follows a strict **Model-View-Controller (MVC)** or **MVVM** architecture. The goal is to isolate the "Mathematical Engine" (PyTorch) from the "Presentation Layer" (PySide6).

### The Three Pillars:
1.  **Core (Model/Logic)**: Pure high-performance AI and data logic.
2.  **UI (View/Presentation)**: Reactive user interface and event handling.
3.  **Worker (Controller/Bridge)**: The multithreaded layer connecting the two.

## 2. Directory & Boundary Rules

### `vision_app/core/` (The Engine)
- **Constraint**: **Zero PySide6 or GUI imports allowed.**
- **Purpose**: Contains all PyTorch models, training loops, and data manipulation logic.
- **CLI Compatibility**: This directory must be able to run as a standalone CLI or script without the GUI.
- **Logic**: Implements `ScratchResNet`, `ModelTrainer`, and `DatasetBuilder`.

### `vision_app/ui/` (The Interface)
- **Constraint**: **Zero heavy computation or direct file-system manipulation.**
- **Purpose**: Handles layout, user input, and visual feedback.
- **Logic**: Implements `MainWindow`, `DatasetManagerWidget`, and `InferenceViewWidget`.
- **Communication**: Must trigger logic in `core` via the `worker` layer.

### `vision_app/worker/` (The Bridge)
- **Constraint**: **All heavy tasks must reside here.**
- **Purpose**: Prevents the UI from freezing by running heavy logic in `QThreads`.
- **Logic**: Implements `TrainWorker` (training/testing) and `StreamWorker` (webcam/video inference).
- **Communication**: Uses **PySide6 Signals** to pass data from `core` back to `ui`.

## 3. The Submodule Architecture Pattern
To prevent "God Classes" (classes that are too large and handle too many responsibilities), complex classes must be broken into submodules.

- **Implementation**: Instantiate submodules in the parent constructor using `self.submodule = SubmoduleClass(self)`.
- **Reference**: Submodules must store a reference to the parent as `self._parent` to access shared state, other submodules, or global paths.
- **Application**: Mandatory for `DatasetBuilder`, `ModelTrainer`, `MainWindow`, `InferenceViewWidget`, and `TrainWorker`.

## 4. Multithreading & UI Stability
- **Non-Blocking Rule**: Any loop that takes longer than 16ms (e.g., file scanning, training an epoch, processing a frame) **must** run in a background thread.
- **Safety**: Workers must implement an `_abort` flag checked at every batch or frame to allow for immediate user-requested stops.
- **Resource Management**: Threads must be explicitly stopped and cleaned up during the `MainWindow` close event to prevent orphan processes or GPU memory leaks.

## 5. Entry Point (`main.py`)
- Responsible for environment initialization (High-DPI scaling, folder creation).
- Checks for the existence of `storage/` subdirectories (`datasets/`, `models/`, `logs/`).
- Instantiates the `QApplication` and `MainWindow`.