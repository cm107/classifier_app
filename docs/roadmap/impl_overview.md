This is a comprehensive development roadmap designed to take you from a blank directory to a fully functional, high-accuracy AI classification suite. 

To ensure the "Separation of Concerns" and your submodule architecture work correctly, we will build from the **Core (Logic)** outward to the **UI (Presentation)**. This allows you to verify the "math" and "file handling" via scripts before worrying about PySide6 signal-slot complexity.

---

## Milestone 1: The Data Foundation (Core Logic)
*Goal: Establish the ability to create, manipulate, and verify datasets without a GUI.*

### 1.1 Data Management Implementation
* **`data_manager.py` $\rightarrow$ `DatasetBuilder`**:
    * Implement **`FileSystemHandler`**: Logic for `mkdir`, `move`, and `os.walk` to scan for images.
    * Implement **`MetadataManager`**: Create a `dataset_info.json` in each folder to store class names and counts.
    * Implement **`DatasetTransformer`**: Write the logic for the 80/20 Train/Val split and merging two directories.
* **`dataset.py`**:
    * Implement **`ClassificationDataset`**: A standard `torch.utils.data.Dataset` that reads from your `DatasetBuilder` structure.
    * Implement **`ContrastiveTransformations`**: The logic for the "two-view" SSL augmentations.

### 1.2 Verification Step (Non-Implementation)
* **Test Script**: Create a `test_data.py`. 
    * Run `DatasetBuilder` to create a dummy structure.
    * Verify that `DatasetTransformer` correctly moves files into `train/` and `val/` folders.
    * Load the result into `ClassificationDataset` and print the shape of the first batch to ensure the tensors are correct.

---

## Milestone 2: The Model & Training Engine
*Goal: Build the "Brain" of the app and verify it can learn on a small sample.*

### 2.1 Model Architecture
* **`model.py`**:
    * Implement **`ResidualBlock`** and **`ScratchResNet`**. Ensure the `mode` switch (SSL vs. Classifier) works.

### 2.2 Training Logic
* **`trainer.py` $\rightarrow$ `ModelTrainer`**:
    * Implement **`StateManager`**: Logic for `torch.save` and `torch.load` of `.pth` files.
    * Implement **`OptimizationEngine`**: Setup for AdamW and the `NTXentLoss`.
    * Implement **`ValidationEngine`**: The loop that calculates top-1 accuracy on the validation set.

### 2.3 Verification Step (Non-Implementation)
* **Overfitting Test**: Use a tiny dataset (10 images). Run the `ModelTrainer` for 50 epochs.
* **Success Criteria**: The loss should trend toward zero. This confirms your gradients are flowing and your `ResidualBlocks` are initialized correctly.

---

## Milestone 3: Multithreading & Async Operations
*Goal: Prepare the bridge between the heavy AI logic and the PySide6 UI.*

### 3.1 Worker Implementation
* **`worker/train_worker.py`**:
    * Implement **`SignalDispatcher`**: Define custom PySide `Signal` objects (e.g., `epoch_finished`, `progress_update`).
    * Implement **`LifecycleManager`**: Wrap the `ModelTrainer` in a loop that checks a `self._is_running` flag every batch so the user can "Stop" training safely.
* **`worker/stream_worker.py`**:
    * Implement the OpenCV `VideoCapture` loop that emits a `QImage` signal for the UI to display.

### 3.2 Verification Step (Non-Implementation)
* **CLI Thread Test**: Run the `TrainWorker` from a script. Manually trigger a "Stop" signal after 5 seconds to ensure the thread terminates without crashing the Python interpreter.

---

## Milestone 4: The GUI Shell & Dataset UI
*Goal: Build the first visual components for the user to interact with data.*

### 4.1 Main Window & Navigation
* **`main_window.py`**:
    * Implement **`NavigationController`**: Set up a `QStackedWidget` to swap between views.
    * Implement **`StatusBarManager`**: Display "System Ready" and "GPU Active: Yes/No".

### 4.2 Dataset Management Tab
* **`widgets/dataset_tab.py`**:
    * Implement the UI buttons for "Create Dataset," "Prune," and "Split."
    * Connect these buttons to the `DatasetBuilder` submodules created in Milestone 1.

### 4.3 Verification Step (Non-Implementation)
* **UI Integration Test**: Launch `main.py`. Try to create a folder structure using the UI. Verify that the folders actually appear on your hard drive in the `storage/datasets/` directory.

---

## Milestone 5: Training Monitor & Hyperparameters
*Goal: Create the "Dashboard" for launching and watching AI training.*

### 5.1 Configuration & Monitoring
* **`widgets/hyper_param.py`**:
    * Create input fields for Learning Rate, Batch Size, and Epochs. Link these to a `config.yaml` file.
* **`widgets/monitor.py`**:
    * Implement **`LiveGraph`**: Use `pyqtgraph` or a custom `QWidget` to plot the loss values received from the `TrainWorker`.

### 5.2 Verification Step (Non-Implementation)
* **Live Feedback Test**: Start a "Dummy Training" session (high-speed loop with random data). Ensure the graphs update in real-time without the UI lagging or becoming "Not Responding."

---

## Milestone 6: Management & Live Inference
*Goal: The final stage—managing saved weights and running real-world tests.*

### 6.1 Model Library
* **`widgets/model_list.py`**:
    * Scan the `storage/models/` folder and list files. 
    * Implement the "Delete" and "Load for Fine-tune" logic using the `StateManager` submodule.

### 6.2 Inference UI
* **`widgets/live_view.py`**:
    * Implement **`SourceController`**: Logic to toggle between Webcam and Video.
    * Implement **`OverlayPainter`**: Use `QPainter` to draw the prediction text over the incoming video frames from the `StreamWorker`.

### 6.3 Final Verification (Full System Test)
* **The End-to-End Run**: 
    1. Create a dataset in the UI.
    2. Set hyperparameters.
    3. Run SSL Pre-training (Watch the monitor).
    4. Run Supervised Fine-tuning.
    5. Switch to the Inference Tab and point the webcam at an object to see the classification.

---

**Would you like me to begin by detailing the implementation of the `FileSystemHandler` and `MetadataManager` for Milestone 1?**