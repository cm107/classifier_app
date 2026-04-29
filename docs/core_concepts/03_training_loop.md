To ensure your custom model converges efficiently from scratch, the training loop needs to be more robust than a standard "fine-tuning" script. We will use **AdamW** for its superior weight decay handling and **OneCycleLR** to aggressively find the optimal gradient path.

### The Specialized Training Loop

This structure includes a "warm-up" and "cool-down" phase, which is critical when weights are initialized randomly.

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

def train_custom_model(model, train_loader, val_loader, num_epochs, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Loss Function: Label smoothing helps with scratch training generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer: AdamW handles weight decay better than standard Adam
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # Scheduler: OneCycleLR manages the "warm-up" automatically
    scheduler = OneCycleLR(optimizer, max_lr=1e-2, 
                           steps_per_epoch=len(train_loader), 
                           epochs=num_epochs)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping: Prevents exploding gradients in early scratch training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step() # Step per batch for OneCycleLR
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss/len(train_loader):.4f}")
        # Add validation logic here...
```



---

### Why this configuration works for your case:

1.  **Label Smoothing ($0.1$):** Since the model starts with zero knowledge, it can easily overfit to the noise in your specific dataset. Label smoothing prevents the output layer from becoming "overconfident," forcing the internal blocks to learn more generalized features.
2.  **Gradient Clipping:** In the first few iterations, random weights can produce massive gradients. Clipping them at $1.0$ keeps the updates stable.
3.  **OneCycleLR:** This scheduler starts with a very low learning rate, ramps it up to `max_lr` to find a broad local minimum, and then decays it to nearly zero. This "annealing" process is often the difference between $85\%$ and $95\%$ accuracy when training from scratch.

### Implementation Checklist
* **Batch Size:** For scratch training, a larger batch size (e.g., 64 or 128) provides more stable gradient estimates.
* **Validation:** Monitor the gap between training and validation loss. If training loss drops but validation stays flat, increase your **Dropout** or **Weight Decay**.

Would you like to see how to implement the **Global Average Pooling** and **Output Head** to finish the model architecture?