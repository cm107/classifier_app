Milestone 5 is the "Command Center" of your application. This is where the user configures the AI's "brain" and watches the learning process in real-time. This phase heavily relies on the **Signals** we defined in Milestone 3 to update the UI without lagging.

Here is the implementation TODO list for **Step 1.5: Training Monitor & Hyperparameters**.

---

### 1.5.1: Hyperparameter Configuration (`hyper_param.py`)
*Goal: Create a clean, validated form for setting training variables.*

* [ ] **`class HyperParameterWidget(QWidget)`**:
    * **Form Layout**: Use a `QFormLayout` to align labels and inputs (Learning Rate, Batch Size, Epochs, Temperature).
    * **Input Validation**: Use `QDoubleSpinBox` for floats (LR) and `QSpinBox` for integers (Batch Size) to prevent invalid text input.
    * **Mode Selector**: Add a `QComboBox` to switch between "Self-Supervised (SSL)" and "Supervised" modes.
    * **Save/Load Logic**: Connect these inputs to your `config.yaml` so that settings persist even after closing the app.

---

### 1.5.2: Training Monitor UI (`monitor.py`)
*Goal: Provide visual feedback that the model is converging.*

* [ ] **`class LiveGraph(QWidget)`**:
    * **Library Integration**: Use `pyqtgraph` (highly recommended for performance) or `matplotlib`'s Qt backend.
    * **Dynamic Scaling**: Implement an `update_plot(epoch, loss, acc)` method that appends new data points to a running list and refreshes the canvas.
* [ ] **`class TrainingMonitorWidget(QWidget)`**:
    * **Progress Bars**: Add a `QProgressBar` for the current Epoch and another for the overall Training Session.
    * **Metric Cards**: Create large, readable `QLabels` that display "Current Loss," "Top-1 Accuracy," and "Time Remaining."
    * **Control Buttons**: Add a "Start Training" button and an "Abort" button.

---

### 1.5.3: Logic Wiring (The Controller Pattern)
*Goal: Connecting the UI buttons to the background `TrainWorker`.*

* [ ] **Start Logic**:
    * Instantiate the `TrainWorker` when the "Start" button is clicked.
    * Pull the current values from the `HyperParameterWidget` and pass them to the worker's constructor.
* [ ] **Signal Connections**:
    * `worker.progress_updated.connect(monitor_widget.update_plot)`
    * `worker.status_changed.connect(main_window.status_bar.show_message)`
    * `worker.finished.connect(self.on_training_finished)`
* [ ] **Clean Up**: Ensure the `TrainWorker` is properly deleted and the thread is quit when training ends or is aborted to prevent memory leaks.

---

### Verification Checklist for 1.5
1.  **Input Bounds**: Try to enter a negative Learning Rate or a Batch Size of 0. Ensure the UI prevents this or highlights it as an error.
2.  **Graph Performance**: Run a "mock" training session that emits 1,000 data points per second. Verify the UI remains responsive and the graph doesn't flicker.
3.  **Persistence**: Change the Epochs to 100, restart the app, and verify that it still says 100.
4.  **Graceful Abort**: Hit the "Abort" button mid-graph-update. Ensure the graph stops immediately and the "Start" button becomes clickable again.

**Would you like me to move on to the final TODO list: Milestone 6, covering Model Management and Live Inference (Webcam/Video)?**