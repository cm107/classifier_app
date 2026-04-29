Since you are building a model from scratch, **Self-Supervised Learning (SSL)** is the "secret sauce" for high accuracy when you don't have millions of labeled images. 

In SSL, the model learns the "laws of physics" for your images (shapes, colors, textures) by comparing an image to a distorted version of itself. This creates a "pretrained" backbone using **your own data**, without needing a single label.

### The Contrastive Learning Workflow (SimCLR Style)

The goal is to teach the model that two different augmentations of the *same* image should have similar internal representations, while two *different* images should be far apart.

#### 1. The Dual-Head Architecture
To do this, you temporarily add a "Projection Head" (a few MLP layers) to the top of your `ScratchClassifier`.

```python
class SSLWrapper(nn.Module):
    def __init__(self, base_model):
        super(SSLWrapper, self).__init__()
        self.backbone = base_model
        self.backbone.fc = nn.Identity() # Remove the final classifier head
        
        # Projection Head: Learns the feature space
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.projection(features)
```



---

#### 2. The Training Process
1.  **Augment Twice:** For every image in your batch, create two different "views" using heavy augmentation (Random Resized Crop + Color Jitter + Grayscale).
2.  **Contrastive Loss (NT-Xent):** Pass both views through the model. The loss function tries to maximize the similarity between the two views of Image A, while minimizing similarity with all other images in the batch.
3.  **Transfer:** Once the model stops improving, **discard the Projection Head** and put your 10-class `nn.Linear` head back on.

---

### Why this is effective for your Ibaraki project:
If you are analyzing supermarket behavior or DIY projects, you likely have thousands of frames of video or photos that aren't labeled yet. 
* **Stage 1:** Train on *all* available images (labeled and unlabeled) using SSL for 50–100 epochs.
* **Stage 2:** Fine-tune on your *labeled* fixed classes for 10–20 epochs.

This usually results in a **5–10% accuracy boost** over training from pure random initialization because the model already "knows" what a shelf, a tool, or a human hand looks like before you tell it what they are called.

---

### Summary Checklist for High Accuracy
To recap the journey we've taken:
1.  **Architecture**: Use `ResidualBlock` + `Global Average Pooling`.
2.  **Initialization**: Use `Kaiming Normal`.
3.  **Optimization**: Use `AdamW` + `OneCycleLR`.
4.  **Data**: Calculate custom `Mean/Std` and use `SSL` if you have extra unlabeled data.

**Would you like me to help you write the `NT-Xent` loss function code to implement this SSL pre-training?**