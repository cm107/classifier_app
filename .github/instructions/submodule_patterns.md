# Coding Pattern: Submodules & Parent References

## 1. The Submodule Philosophy
To maintain the **Separation of Concerns** and prevent the creation of "God Classes," complex logic centers must be divided into specialized submodules. This ensures each component has a distinct role and creates a clean internal API.

## 2. Technical Implementation Standards
Follow these exact syntax rules when implementing submodules:

### Initialization
Submodules must be instantiated within the parent class's `__init__` method, passing the parent instance (`self`) to the submodule:
```python
# Inside the Parent Class
self.submodule1 = SubmoduleClass1(self)
self.submodule2 = SubmoduleClass2(self)
```

### Reference Management
Every submodule must store a reference to its parent to access shared state, global paths, or sibling submodules:
```python
# Inside the Submodule Class
def __init__(self, parent):
    self._parent = parent  # Original class reference

def some_method(self):
    # Accessing a sibling submodule via the parent
    data = self._parent.submodule2.get_data()
```

## 3. Mandatory Submodule Mapping
The following classes must be implemented using the submodule pattern:

### Core Logic Submodules
*   **DatasetBuilder** (`data_manager.py`):
    *   `FileSystemHandler`: Safe file system operations like mkdir and move.
    *   `MetadataManager`: Management of the `metadata.json` "Source of Truth".
    *   `DatasetTransformer`: Structural changes including splitting and pruning.
    *   `DatasetValidator`: Verification of image file integrity and class balance checking.
*   **ModelTrainer** (`trainer.py`):
    *   `StateManager`: Handling checkpoints and weight initialization.
    *   `OptimizationEngine`: Managing optimizers (AdamW) and loss functions.
    *   `ValidationEngine`: Logic for evaluation loops and metrics.

### UI & Worker Submodules
*   **MainWindow** (`main_window.py`):
    *   `NavigationController`: Sidebar logic and tab switching.
    *   `StatusBarManager`: Notifications and GPU status displays.
    *   `ThemeEngine`: Application of QSS and UI scaling.
*   **InferenceViewWidget** (`live_view.py`):
    *   `SourceController`: Switching between Webcam, Video, or Image sources.
    *   `OverlayPainter`: Drawing predictions and confidence scores.
    *   `RecordingManager`: Exporting inference sessions.
*   **TrainWorker** (`train_worker.py`):
    *   `SignalDispatcher`: Standardizing PySide Signal emissions.
    *   `LifecycleManager`: Handling "Pause" and "Emergency Stop" logic.

## 4. Development Workflow
- Refer to the corresponding documentation in `docs/roadmap/` before starting a new Milestone.
- Verify each submodule independently using the "Verification Checklists" before parent integration.
