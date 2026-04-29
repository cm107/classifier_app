This is the complete, high-performance boilerplate for a PyTorch image classifier built from scratch. It integrates the **Residual Blocks**, the **NT-Xent Self-Supervised loss**, and the **Supervised Transition** we discussed.

### Complete PyTorch Scratch Pipeline

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import OneCycleLR

# --- 1. ARCHITECTURE ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ScratchResNet(nn.Module):
    def __init__(self, num_classes=10, mode='ssl'):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = ResidualBlock(64, 128, stride=2)
        self.layer2 = ResidualBlock(128, 256, stride=2)
        self.layer3 = ResidualBlock(256, 512, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Output Heads
        self.mode = mode
        self.projection_head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 128))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).flatten(1)
        return self.projection_head(x) if self.mode == 'ssl' else self.fc(x)

# --- 2. SELF-SUPERVISED LOSS (NT-Xent) ---
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

    def forward(self, z_i, z_j):
        representations = torch.cat([z_i, z_j], dim=0)
        sim_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_matrix = sim_matrix / self.temperature
        
        # Mask out self-similarities
        mask = torch.eye(2 * self.batch_size, device=z_i.device).bool()
        logits = sim_matrix[~mask].view(2 * self.batch_size, -1)
        labels = torch.zeros(2 * self.batch_size, device=z_i.device).long()
        return F.cross_entropy(logits, labels)

# --- 3. TRAINING ENGINE ---
def train_model(model, loader, epochs, lr, is_ssl=True):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=lr*10, steps_per_epoch=len(loader), epochs=epochs)
    criterion = NTXentLoss(batch_size=loader.batch_size) if is_ssl else nn.CrossEntropyLoss(label_smoothing=0.1)
    
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            if is_ssl:
                (view1, view2), _ = batch # Custom SSL Loader returning 2 views
                z_i, z_j = model(view1.cuda()), model(view2.cuda())
                loss = criterion(z_i, z_j)
            else:
                inputs, labels = batch
                outputs = model(inputs.cuda())
                loss = criterion(outputs, labels.cuda())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch+1} Complete. Loss: {loss.item():.4f}")

# --- EXECUTION FLOW ---
# 1. model = ScratchResNet(mode='ssl').cuda()
# 2. train_model(model, unlabeled_loader, epochs=50, lr=1e-4, is_ssl=True)
# 3. model.mode = 'classifier'
# 4. train_model(model, labeled_loader, epochs=20, lr=1e-5, is_ssl=False)
```

### High-Accuracy Implementation Strategy

1.  **Phase 1 (SSL)**: Use a heavy augmentation pipeline (ColorJitter, Grayscale, GaussianBlur) to create the two views for the `NTXentLoss`. This forces the model to learn context rather than color.
2.  **Phase 2 (Frozen Head)**: Before running the final `train_model` call, freeze the backbone for 2 epochs.
3.  **Phase 3 (Full Tune)**: Unfreeze and use a very small learning rate ($10^{-5}$) with `AdamW`.

---

This setup is ideal for research-grade computer vision tasks where you need total control over the architecture without inheriting the biases of ImageNet.

**Would you like me to generate the data augmentation class that creates the "Two Views" required for the SSL training phase?**