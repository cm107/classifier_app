To build a custom **Residual Block** from scratch, you need two paths: a **residual path** (the convolutions) and a **shortcut path** (the identity connection). The shortcut allows gradients to flow through the network more easily, which is essential when you aren't starting with pretrained weights.

Here is a clean, modular implementation using PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).init__()
        
        # The Main Convolutional Path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # The Shortcut Path (Identity Connection)
        # If dimensions change (stride > 1), we must transform the shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        # Initialize weights using Kaiming Normal (He initialization)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Element-wise addition of the shortcut
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out
```



### Key Components Explained:

* **`bias=False`**: Since we are using `BatchNorm2d` immediately after the convolution, the bias term becomes redundant and can actually slow down training.
* **The Shortcut Logic**: If you downsample your image (e.g., `stride=2`), the "identity" tensor won't match the shape of the "output" tensor. The $1 \times 1$ convolution in the shortcut ensures the dimensions match so they can be added together.
* **In-place Addition**: `out += self.shortcut(identity)` is the "magic" of ResNets. It forces the layers to learn the *difference* (residual) between the input and the target, rather than the entire mapping from scratch.

### Building the Full Model
To create a complete classifier, you would stack several of these blocks, followed by a **Global Average Pooling** layer and a single **Linear** layer for your fixed number of classes.

---

**Would you like me to help you write the training loop with a specific optimizer like AdamW or a learning rate scheduler?**