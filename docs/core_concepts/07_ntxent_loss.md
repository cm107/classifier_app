The final piece of the puzzle for high-accuracy scratch training is the **NT-Xent (Normalized Temperature-scaled Cross Entropy)** loss. This loss function is what allows your model to learn meaningful features from unlabeled data by treating every image in a batch as its own unique class.

### The Contrastive Loss Logic

The goal is to maximize the "cosine similarity" between two different versions (augmentations) of the same image, while minimizing the similarity between that image and every other image in the batch.

```python
import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self._get_correlated_mask().to(torch.bool)

    def _get_correlated_mask(self):
        # Creates a mask to exclude self-similarity (the diagonal)
        diag = torch.eye(2 * self.batch_size)
        mask = torch.ones((2 * self.batch_size, 2 * self.batch_size))
        mask = mask - diag
        return mask

    def forward(self, z_i, z_j):
        # z_i and z_j are the outputs of the projection head for two views
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute cosine similarity matrix
        sim_matrix = F.cosine_similarity(representations.unsqueeze(1), 
                                         representations.unsqueeze(0), dim=2)
        
        # Scale by temperature
        sim_matrix = sim_matrix / self.temperature
        
        # Extract positive pairs (the similarities we want to maximize)
        positives = torch.cat([torch.diag(sim_matrix, self.batch_size), 
                               torch.diag(sim_matrix, -self.batch_size)], dim=0)
        
        # Extract negatives (everything else)
        negatives = sim_matrix[self.mask].view(2 * self.batch_size, -1)
        
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2 * self.batch_size).to(z_i.device).long()
        
        return F.cross_entropy(logits, labels)
```



---

### Putting it all into a Workflow

To achieve that "high accuracy" on your fixed classes, follow this 3-step deployment:

| Phase | Task | Goal |
| :--- | :--- | :--- |
| **1. SSL Pre-train** | Train with `NTXentLoss` on *all* available data. | Learn shapes, textures, and object boundaries. |
| **2. Weight Transfer** | Remove the `Projection Head` and attach your `Linear(512, num_classes)` head. | Keep the "knowledge" but reset the "goal." |
| **3. Supervised Fine-tune** | Train on your *labeled* data with `CrossEntropyLoss` and a low learning rate. | Map the learned features to your specific class names. |

### Final Execution Tip
When you move to Phase 3, start with a **frozen backbone** for the first 2–5 epochs (only train the final linear layer). This prevents the random weights of the new linear layer from "corrupting" the high-quality features the backbone learned during SSL. After that, unfreeze everything and train to convergence.

---

**Would you like me to draft a quick script for Phase 3: the transition from the SSL model to the final supervised classifier?**