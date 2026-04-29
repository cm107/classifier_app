# UI & Threading Rules: Multithreading with PySide6 and PyTorch

## 1. Non-Blocking UI Philosophy
To ensure a professional-grade experience, the main GUI thread must remain reactive at all times. Any task that is computationally expensive or involves blocking I/O **must** be offloaded to a background thread.
- **The 16ms Rule**: Any operation taking longer than 16ms (e.g., training an epoch, scanning 1000+ files, or processing a video frame) must reside in a `worker/` module.
- **Reactive UI**: The UI thread's only jobs are handling user events, updating layouts, and rendering visuals from data received via Signals.

## 2. Worker Implementation Standards
All background tasks must inherit from `QObject` and be moved to a `QThread`.
- **Worker Submodules**:
    - `SignalDispatcher`: Standardize all communication using `PySide6.QtCore.Signal`.
    - `LifecycleManager`: Manage the internal state of the thread (Running, Paused, Aborted).
- **Thread Affinity**: Initialize heavy objects (like `ModelTrainer`, `DataLoader`, or `cv2.VideoCapture`) **inside** the worker's `run()` method or a dedicated startup slot. This ensures they belong to the background thread and do not cause "cross-thread" access crashes.

## 3. Communication via Signals & Slots
Never modify UI widgets directly from a background thread. All data transfer must happen through the Signal/Slot mechanism.
- **Training Signals**: Use signals like `progress_updated(dict)`, `status_changed(str)`, and `finished(bool)` to pass metrics (loss, accuracy) back to the `LiveGraph` or `StatusBar`.
- **Inference Signals**: The `StreamWorker` must emit a `frame_ready(QImage)` for display on a `QLabel` and a `results_ready(list)` for the `OverlayPainter`.
- **Safety**: Wrap worker logic in try-except blocks to catch exceptions (like CUDA Out-Of-Memory) and report them via a `status_changed` signal rather than letting the thread crash silently.

## 4. Specific Worker Roles
### TrainWorker (`train_worker.py`)
- **Lifecycle Control**: Must implement an `_abort` flag checked at every batch loop to allow for "Emergency Stops".
- **Pause/Resume**: Use a `QWaitCondition` in the `LifecycleManager` to safely pause training without destroying the thread state.

### StreamWorker (`stream_worker.py`)
- **Frame Rate Limiting**: Use `QThread.msleep()` to cap the loop at 30 FPS. This prevents the worker from overwhelming the UI thread with too many `setPixmap()` calls.
- **Inference Integration**: The worker should call the `InferenceEngine` in `eval` mode with `torch.no_grad()` to conserve GPU resources while streaming.

## 5. Lifecycle & Resource Cleanup
- **Controller Pattern**: The UI (e.g., `TrainingMonitorWidget`) is responsible for instantiating workers, connecting signals, and starting the thread.
- **Safe Exit**: The `MainWindow` must implement a `closeEvent` that explicitly triggers the `_abort` flags of all active workers and waits for them to quit (`thread.wait()`) before the application terminates. This prevents orphan processes and GPU memory leaks.
