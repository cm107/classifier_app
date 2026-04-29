To complete the architecture, we replace the traditional, memory-heavy "Flatten + Multiple Dense Layers" approach with **Global Average Pooling (GAP)**. 

When training from scratch, GAP is a powerful regularizer. It reduces the millions of parameters typically found in the output head down to a single vector, forcing the convolutional layers to learn the actual "essence" of the classes rather than just memorizing spatial coordinates.

### The Final Output Head

Here is how you integrate the `ResidualBlock` into a full model with a GAP-based classifier head:

```python
class ScratchClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ScratchClassifier, self).init__()
        
        # Initial Feature Extraction
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Stacking Residual Blocks (Increasing depth)
        self.layer1 = ResidualBlock(64, 128, stride=2)
        self.layer2 = ResidualBlock(128, 256, stride=2)
        self.layer3 = ResidualBlock(256, 512, stride=2)
        
        # Global Average Pooling Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final Linear Classifier
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.gap(x)           # Shape: [Batch, 512, 1, 1]
        x = torch.flatten(x, 1)   # Shape: [Batch, 512]
        x = self.fc(x)            # Shape: [Batch, num_classes]
        
        return x
```



---

### Why this design yields high accuracy:

* **AdaptiveAvgPool2d(1)**: No matter the input image size (e.g., $224 \times 224$ or $256 \times 256$), this layer always squashes the spatial dimensions to $1 \times 1$. This makes your model flexible and prevents the "Dense Layer Explosion" that often leads to overfitting in scratch models.
* **Feature Bottleneck**: By gradually increasing channels ($64 \rightarrow 128 \rightarrow 256 \rightarrow 512$), you allow the model to move from simple edge detection to complex object part recognition.
* **Initialization Consistency**: Because every `ResidualBlock` handles its own Kaiming initialization (as seen in the previous code), the entire stack is ready for high-learning-rate training immediately.

### One Final Tip: Data Normalization
Since you are in Ibaraki/Osaka, if you are collecting your own image data locally (e.g., for a computer vision project), remember to calculate the **mean and standard deviation** of your specific dataset. 

Using the standard ImageNet values (0.485, 0.456, 0.406) is a common mistake for scratch models. Your model will converge much faster if the input data is normalized specifically to *your* environment's lighting and colors.

---

**Would you like me to show you a quick script to calculate the Mean and Std Dev of your custom image folder?**