Moving into Milestone 2, we transition from the "File System" to the "Mathematical Engine." This is where the model is defined and the training logic is standardized. Because you are building from scratch, the **Initialization** and **State Management** steps are the most critical to prevent the model from failing to converge.

Here is the implementation TODO list for **Step 1.2: The Model & Training Engine**.

---

### 1.2.1: Model Architecture (`model.py`)
*Goal: Define the "Silicon" structure of the classifier.*

* [ ] **`class ResidualBlock(nn.Module)`**:
    * Implement the **Main Path**: Two $3 \times 3$ convolutions with `BatchNorm` and `ReLU`.
    * Implement the **Shortcut Path**: A $1 \times 1$ convolution logic to match dimensions if `stride > 1`.
    * Implement **He Initialization**: Specifically apply `kaiming_normal_` to all convolutional weights within the `__init__`.
* [ ] **`class ScratchResNet(nn.Module)`**:
    * Define the **Input Layer**: Initial $3 \times 3$ conv to handle the 3-channel RGB input.
    * Layer Stacking: Create three distinct stages using the `ResidualBlock` with increasing filter sizes ($64 \rightarrow 128 \rightarrow 256 \rightarrow 512$).
    * Implement **Global Average Pooling (GAP)**: Use `nn.AdaptiveAvgPool2d(1)` to ensure the model accepts any input resolution.
    * Implement the **Dual Head Switch**: Create a logic flag in `forward()` to toggle between the **Projection Head** (for SSL) and the **Linear Classifier** (for final labels).



---

### 1.2.2: ModelTrainer Submodules (`trainer.py`)
*Goal: Create the "Manager" that dictates how the model learns.*

* [ ] **`class StateManager`**:
    * **`save_checkpoint(path, epoch, optimizer_state)`**: Save a `.pth` file containing the model state dict and current optimizer metadata.
    * **`load_checkpoint(path)`**: Safely load weights, handling cases where the number of classes in the file might differ from the current model (for fine-tuning).
    * **`freeze_backbone(toggle=True)`**: A helper to set `requires_grad = False` for all layers except the final head.
* [ ] **`class OptimizationEngine`**:
    * **`configure_optimizer(lr, weight_decay)`**: Initialize `optim.AdamW`.
    * **`configure_scheduler(max_lr, steps_per_epoch)`**: Set up `OneCycleLR`.
    * **`get_loss_function(mode)`**: Return `NTXentLoss` for 'ssl' mode or `CrossEntropyLoss(label_smoothing=0.1)` for 'supervised' mode.
* [ ] **`class ValidationEngine`**:
    * **`run_evaluation(model, val_loader)`**: Implement a `with torch.no_grad()` loop.
    * **`calculate_metrics()`**: Compute Top-1 accuracy and a basic Confusion Matrix (useful for the UI later).

---

### 1.2.3: Specialized Logic & Metrics
*Goal: Standardize the data being passed back to the UI.*

* [ ] **`class TrainingMetrics` (Data Class)**:
    * Define a structure to hold `loss`, `accuracy`, `learning_rate`, and `epoch_index`. This object will be the primary payload sent from the `TrainWorker` to the UI.
* [ ] **`NTXentLoss` Implementation**:
    * Write the similarity matrix logic.
    * Ensure the "mask" correctly ignores the diagonal (self-similarity) to prevent the model from cheating.

---

### Verification Checklist for 1.2
Before moving to Milestone 3, you should verify these points via a Python terminal:
1.  **Shape Test**: Pass a random tensor `torch.randn(1, 3, 224, 224)` through the `ScratchResNet`. Ensure the output is `[1, num_classes]`.
2.  **Backbone Freeze**: Call `freeze_backbone(True)` and verify that `param.requires_grad` is `False` for the early layers.
3.  **Checkpoint Integrity**: Save a model, change its weights randomly, then load the saved weights. Verify the weights returned to their original values.

**Would you like me to move on to detailing the TODO list for Milestone 3 (Multithreading & Async Operations), which bridges this logic to your PySide6 UI?**