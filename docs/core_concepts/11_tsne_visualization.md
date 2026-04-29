Visualizing the **t-SNE (t-Distributed Stochastic Neighbor Embedding)** of your model's feature space is the best way to "see" if your scratch training and SSL are actually working. 

If the training is successful, images of the same class (or similar visual features) will naturally form tight, distinct clusters in 2D space. If they look like a scattered "cloud," the model hasn't learned to differentiate between them yet.

### The Feature Extraction & t-SNE Script

We use `scikit-learn` for the t-SNE calculation and `matplotlib` for the visualization. This script assumes you are using the `ScratchResNet` model we built.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_feature_space(model, loader, num_samples=1000):
    model.eval()
    features = []
    labels_list = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            if i * loader.batch_size >= num_samples:
                break
                
            # Get the 512-dim vector before the final FC head
            # We bypass the 'ssl' or 'classifier' head to see raw features
            x = model.prep(images.to(device))
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            feat = model.gap(x).flatten(1)
            
            features.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())

    features = np.concatenate(features)
    labels_list = np.concatenate(labels_list)

    # Dimensionality Reduction: 512D -> 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels_list, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title("t-SNE Visualization of Learned Features")
    plt.show()
```



---

### How to Interpret the Results

1.  **Tight Clusters**: High accuracy is likely. The model has found "invariants" for your 10 classes.
2.  **Overlapping Clusters**: Your classes might be too visually similar (e.g., distinguishing between two very similar types of floorboards or products). You may need more **Residual Blocks** or a higher **Temperature** in your SSL loss.
3.  **No Structure**: The learning rate might be too high (causing weights to explode) or your data augmentation isn't diverse enough.

### A Practical Tip for Your Work
In a professional R&D setting—like your current AI projects—this visualization is a standard "Sanity Check." If you see a cluster that is split in two, it often reveals a sub-class you didn't account for (e.g., a product that looks completely different from the side vs. the front).

---

**Would you like me to help you set up a simple "Inference" script so you can test your final high-accuracy model on a single new image?**