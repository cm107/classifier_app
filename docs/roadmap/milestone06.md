This is the final milestone, where you turn your trained weights into a functional tool. Milestone 6 covers **Model Management** (the library of your previous work) and **Live Inference** (the real-time application of your AI).

Here is the implementation TODO list for **Step 1.6: Management & Live Inference**.

---

### 1.6.1: Model Management UI (`model_list.py`)
*Goal: Create a "Library" to view, delete, and select models for fine-tuning.*

* [ ] **`class ModelManagerWidget(QWidget)`**:
    * **The List View**: Implement a `QListView` or `QTableWidget` that scans the `storage/models/` directory for `.pth` (PyTorch) and `.pt` (TorchScript) files.
    * **Metadata Display**: When a model is selected, show its "Birth Certificate" (creation date, number of classes, and the best validation accuracy achieved).
    * **Action Buttons**:
        * **"Load for Fine-Tune"**: Sends the model path back to the `TrainingTab`.
        * **"Export to TorchScript"**: Triggers the `ModelExporter` utility to create a portable version.
        * **"Delete"**: Removes the file from disk with a `QMessageBox` confirmation.
* [ ] **`class ModelItemDelegate`**:
    * Customize the look of the list items to show a "Status Icon" (e.g., a green checkmark for models with >90% accuracy).

---

### 1.6.2: Live Inference UI (`live_view.py`)
*Goal: The "Action" tab where the model processes live video or image folders.*

* [ ] **`class SourceController`**:
    * Implement a `QComboBox` to select the input source: `Webcam (0)`, `Video File`, or `Image Directory`.
    * Implement a "Play/Pause" toggle to control the `StreamWorker`.
* [ ] **`class CameraViewport(QLabel)`**:
    * Optimize the `setPixmap()` calls to handle high-frequency updates without flickering.
    * Implement **Scaling**: Ensure the video scales correctly when the user resizes the application window.
* [ ] **`class OverlayPainter`**:
    * Implement the `paintEvent` logic to draw a semi-transparent box at the top of the frame.
    * Display the **Top-3 Predictions** with percentage bars (e.g., "Apple: 98%").



---

### 1.6.3: Inference Logic & Streaming (`inference.py` & `stream_worker.py`)
*Goal: Bridging the model's math to the UI's pixels.*

* [ ] **`class InferenceEngine`**:
    * Implement the `predict_batch()` method to handle a single frame or a folder of images.
    * Integrate **Softmax** logic to convert raw model "logits" into human-readable percentages.
* [ ] **The `StreamWorker` Integration**:
    * Connect the `frame_ready` signal to the `CameraViewport`.
    * Connect the `results_ready` signal to the `OverlayPainter`.
    * Ensure the model is moved to `eval()` mode and `torch.no_grad()` is active to save GPU memory during the stream.

---

### 1.6.4: Final Polish & Export (`utils.py`)
*Goal: Making the tool professional and portable.*

* [ ] **`class ModelExporter`**:
    * Implement **TorchScript Tracing**: Convert the model for use in C# or Kotlin as we planned.
    * Implement **ONNX Export**: (Optional) For maximum compatibility with other inference engines.
* [ ] **Final App Cleanup**:
    * Implement a "Close Event" in `MainWindow` to ensure all `QThreads` (Training and Streaming) are killed safely before the app exits.

---

### Final System Verification Checklist
1.  **Model Persistence**: Train a small model, save it, and verify it appears in the "Models" tab with the correct name.
2.  **Webcam Switch**: Start the webcam, then switch to a "Video File" source. Ensure the old stream is closed and the new one starts smoothly.
3.  **Accuracy Check**: Point the webcam at an object from your dataset. Verify the "Overlay" updates in real-time with high confidence.
4.  **Resource Usage**: Monitor Task Manager/NVIDIA-SMI. Ensure that stopping the inference stream actually drops the GPU usage back to idle.

**This concludes the full development plan for your PySide6 Vision App! Would you like me to provide the starting code for the `main_window.py` shell to get the project's visual structure running?**