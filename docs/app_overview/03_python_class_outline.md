To keep this PySide6 application modular and maintainable, I have mapped out the class structure for each file. This design follows a "Separation of Concerns" principle where the UI stays reactive and the Core logic stays computationally focused.

Here are the classes to be implemented in each `.py` file:

---

### `vision_app/core/` (The Engine)
* **`model.py`**
    * `ResidualBlock(nn.Module)`
    * `ScratchResNet(nn.Module)`
* **`trainer.py`**
    * `ModelTrainer` (Handles the loops, optimization, and SSL/Supervised logic)
    * `TrainingMetrics` (Data class for tracking loss/accuracy)
* **`dataset.py`**
    * `ClassificationDataset(Dataset)`
    * `ContrastiveTransformations` (The dual-view logic for SSL)
    * `StandardTransformations` (For supervised training and validation)
* **`data_manager.py`**
    * `DatasetBuilder` (Handles creation, merging, and pruning logic)
    * `DatasetValidator` (Checks for corruptions and class balance)
* **`inference.py`**
    * `InferenceEngine` (Universal wrapper for Image/Video/Stream prediction)
* **`utils.py`**
    * `StatsCalculator` (Mean/Std calculations)
    * `ModelExporter` (TorchScript/ONNX conversion logic)

---

### `vision_app/ui/` (The Interface)
* **`main_window.py`**
    * `MainWindow(QMainWindow)`
* **`widgets/dataset_tab.py`**
    * `DatasetManagerWidget(QWidget)`
    * `PruningDialog(QDialog)`
* **`widgets/hyper_param.py`**
    * `HyperParameterWidget(QWidget)`
* **`widgets/monitor.py`**
    * `TrainingMonitorWidget(QWidget)`
    * `LiveGraph(QWidget)` (Custom plotting component)
* **`widgets/model_list.py`**
    * `ModelManagerWidget(QWidget)`
    * `ModelItemDelegate(QStyledItemDelegate)` (For custom list styling)
* **`widgets/live_view.py`**
    * `InferenceViewWidget(QWidget)`
    * `CameraViewport(QLabel)` (The frame-rendering surface)

---

### `vision_app/worker/` (The Multi-threading)
* **`train_worker.py`**
    * `TrainWorker(QObject)` (Communicates via PySide Signals to prevent UI freezing)
* **`stream_worker.py`**
    * `StreamWorker(QObject)` (Handles high-frequency frame capture and processing)

---

### `main.py` (The Entry Point)
* *Typically no classes here; contains the `if __name__ == "__main__":` block to instantiate `QApplication` and `MainWindow`.*

---

**Would you like me to start by detailing the implementation for the `DatasetBuilder` in `data_manager.py` so you can begin the data-handling foundation?**