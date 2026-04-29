# AI & Mathematical Logic Standards

## 1. Model Architecture Foundations
Since this model is built **from scratch** without a pretrained backbone, the architecture must be designed to effectively extract features and prevent signal degradation.

### Core Architecture Rules:
- **Residual Blocks**: Implement `ResidualBlock` using a **Main Path** (two $3 \times 3$ convolutions with BatchNorm and ReLU) and a **Shortcut Path** (identity connection or $1 \times 1$ convolution for dimension matching).
- **Weight Initialization**: Explicitly apply **He (Kaiming) Initialization** (`kaiming_normal_`) to all convolutional weights to prevent gradient death during the start of scratch training.
- **Global Average Pooling (GAP)**: Use `nn.AdaptiveAvgPool2d(1)` at the end of the feature extractor instead of flattening to dense layers. This regularizes the model and prevents the "dense layer explosion" common in scratch models.
- **Dual Head Switch**: Implement a logic flag in the `forward()` method of `ScratchResNet` to toggle between the **Projection Head** (MLP for SSL) and the **Linear Classifier** (for final labels).

## 2. Optimization & Loss Functions
Training from scratch is highly sensitive to hyperparameters; follow these strict configurations for stability.

- **Optimizer**: Default to **AdamW** for superior weight decay handling.
- **Scheduler**: Use the **OneCycleLR** policy to provide a warm-up and annealing phase.
- **Gradient Clipping**: Clip gradients at **1.0** to maintain stability during the initial random-weight iterations.
- **Supervised Loss**: Use **CrossEntropyLoss** with **0.1 Label Smoothing** to prevent overconfidence and aid generalization.
- **Self-Supervised Loss**: Use **NT-Xent (Normalized Temperature-scaled Cross Entropy)** for pre-training. Ensure the similarity matrix mask correctly ignores the diagonal (self-similarity).
- **LR Discriminative Layers**: During Phase 3 (Full Fine-Tune), use a **10x smaller learning rate** for the backbone layers than for the classification head (e.g., backbone `lr=1e-5`, head `lr=1e-4`). This prevents high-quality SSL features from being overwritten.

## 3. Data Math & Pre-processing
The "math" of the data must reflect the specific environment of the dataset rather than general-purpose defaults.

- **Custom Normalization**: Do not use ImageNet defaults. Use `StatsCalculator` (in `utils.py`) to compute the per-channel **Mean and Standard Deviation** of the target dataset and apply them via `transforms.Normalize`.
- **SSL Transformations**: For Self-Supervised Learning, implement `ContrastiveTransformations` to return **two views** of the same image using **Random Resized Crop**, **Color Jitter**, and **Gaussian Blur**.
- **Standard Transformations**: For supervised training and validation, implement `StandardTransformations` (resize, normalize with custom stats, basic augmentation for train only).
- **Batch Size**: For scratch training, prefer larger batch sizes (e.g., **64 or 128**) to provide more stable gradient estimates.
- **SSL DataLoader**: Set `drop_last=True` in the `DataLoader` during SSL pre-training. `NTXentLoss` depends on a fixed batch size for the similarity matrix; a partial final batch will cause a dimension mismatch.
- **Class Imbalance**: When class counts are unequal (surfaced by `DatasetValidator`), use `WeightedRandomSampler` in the `DataLoader` to ensure minority classes appear as frequently as majority classes during training.

## 4. Multi-Phase Training Logic
To achieve high accuracy, training must follow a staged deployment.

1.  **Phase 1 (SSL Pre-train)**: Train the backbone + Projection Head on all data (labeled and unlabeled) using **NT-Xent Loss**.
2.  **Phase 2 (Linear Probing)**: Freeze the backbone and train only the new Linear Classifier head for 2–5 epochs to prevent "gradient shock".
3.  **Phase 3 (Full Fine-Tune)**: Unfreeze the entire model and train with discriminative learning rates (backbone `1e-5`, head `1e-4`) using **AdamW**.

## 5. Validation & Deployment
- **Early Stopping**: Monitor **Validation Loss**, not accuracy. Stop training when validation loss stops decreasing for N consecutive epochs to prevent overfitting.
- **TTA (Test Time Augmentation)**: At inference time, run 5 augmented versions of each input (e.g., original, horizontal flip, 4 random crops) and **average the softmax outputs**. This reliably boosts accuracy on edge-case samples.
- **Half-Precision (FP16)**: Use `model.half()` during inference to halve GPU memory usage with negligible accuracy impact. Particularly important when running the `StreamWorker` continuously.
- **TorchScript Export**: Use `torch.jit.trace()` in `ModelExporter` to serialize the model for use outside Python (e.g., C++ / LibTorch). Always set `model.eval()` before tracing.
