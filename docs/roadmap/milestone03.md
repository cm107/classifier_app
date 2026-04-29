Milestone 3 is the "Bridge" phase. In PySide6, running a heavy training loop or a webcam stream on the main thread will cause the GUI to "Not Responding." We will use **QThreads** and **Signals** to keep the UI reactive while the PyTorch engine runs at full speed in the background.

Here is the implementation TODO list for **Step 1.3: Multithreading & Async Operations**.

---

### 1.3.1: TrainWorker Implementation (`train_worker.py`)
*Goal: Create a background thread that manages the `ModelTrainer` and talks to the GUI.*

* [ ] **`class SignalDispatcher(QObject)`**:
    * Define `progress_updated = Signal(dict)`: Sends a dictionary containing current `epoch`, `loss`, `accuracy`, and `lr`.
    * Define `status_changed = Signal(str)`: Sends messages like "Starting SSL...", "Epoch 5/50", or "Training Interrupted".
    * Define `finished = Signal(bool)`: Emits `True` on success, `False` if an error occurred.
* [ ] **`class LifecycleManager`**:
    * Implement **`self._abort = False`**: A flag that the UI can toggle.
    * Implement **`self._is_paused = False`**: Logic to "wait" at the start of a batch loop using a `QWaitCondition`.
* [ ] **The `run()` Method**:
    * Initialize the `ModelTrainer` and `DataLoader` within this method (to ensure they belong to the worker thread).
    * Wrap the training loop in a `try-except` block to catch CUDA Out-Of-Memory (OOM) errors and report them back via signals.



---

### 1.3.2: StreamWorker Implementation (`stream_worker.py`)
*Goal: High-frequency frame capture for the live inference UI.*

* [ ] **`class StreamWorker(QObject)`**:
    * Define `frame_ready = Signal(QImage)`: Emits the processed frame ready for the `QLabel` display.
    * Define `results_ready = Signal(list)`: Emits the raw classification scores and class names for the UI overlays.
* [ ] **The Processing Loop**:
    * Initialize `cv2.VideoCapture`.
    * Implement **Frame Rate Limiting**: Use `QThread.msleep()` to ensure you don't overwhelm the UI thread (aim for 30 FPS).
    * **Preprocessing Pipeline**: Convert OpenCV's BGR format to RGB, apply `transforms.ToTensor()`, and run the `InferenceEngine`.
    * **Signal Emission**: Convert the final `numpy` array/tensor back to a `QImage`.

---

### 1.3.3: Global Configuration (`config.yaml`)
*Goal: Provide a persistent store for hyperparameters and paths.*

* [ ] **Define Schema**:
    * `paths`: `dataset_root`, `model_export_root`, `log_dir`.
    * `defaults`: `learning_rate: 1e-3`, `batch_size: 32`, `epochs: 50`, `temperature: 0.5`.
    * `hardware`: `use_cuda: True`, `num_workers: 4`.
* [ ] **Helper Class in `utils.py`**:
    * Implement `ConfigLoader`: A simple utility to read/write this YAML file using `PyYAML`.

---

### Verification Checklist for 1.3
Before you start building the actual GUI windows, verify the "piping" works:
1.  **Signal Test**: Create a temporary PySide6 script that launches the `TrainWorker`. Print the `progress_updated` dictionary to the console.
2.  **Thread Safety**: While the `TrainWorker` is "training" (even with dummy data), try to print a message from the main thread. If both print simultaneously without crashing, your threading is correct.
3.  **Abort Test**: Trigger the `self._abort` flag while training. Verify that the `TrainWorker` stops its loop and emits the `finished` signal within 1 second.

**Ready to move to Milestone 4, where we design the Main Window and the Dataset Management UI?**