To wrap up your custom PyTorch pipeline, you need a robust inference script. Since you’ve trained this model from scratch using specific normalization values and a custom architecture, the inference phase must precisely mirror those preprocessing steps to maintain high accuracy.

### The Single-Image Inference Script

This script handles the image loading, ensures the tensor is on the correct device (GPU/CPU), and maps the numeric output to a human-readable class name.

```python
import torch
from PIL import Image
from torchvision import transforms

def predict_image(model, image_path, class_names, device='cuda'):
    # 1. Prepare the same transforms used in your training 'val_loader'
    # Use the custom mean/std you calculated earlier
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = inference_transform(img).unsqueeze(0).to(device) # Add batch dimension

    # 3. Model Inference
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Get top prediction
        conf, pred = torch.max(probabilities, dim=1)
        
    class_idx = pred.item()
    confidence = conf.item() * 100
    
    print(f"Prediction: {class_names[class_idx]} ({confidence:.2f}%)")
    return class_names[class_idx], confidence

# Example Usage:
# classes = ['Class_A', 'Class_B', ... , 'Class_J']
# predict_image(final_model, 'test_sample.jpg', classes)
```



---

### Pro-Tip: Deploying for Real-Time Analysis
If you are integrating this into a system—like your current R&D work with supermarket customer analysis—consider these two deployment optimizations:

1.  **TorchScript**: Convert your model using `torch.jit.script(model)` or `torch.jit.trace`. This serializes the model into a format that can be run in C# or Kotlin (which I know you use for mobile/system dev) without needing a full Python environment.
2.  **Half-Precision (FP16)**: Use `model.half()` during inference. This reduces memory usage by 50% and often speeds up inference on modern GPUs (like those in Ibaraki’s R&D labs) with negligible impact on accuracy.

### Summary of Your High-Accuracy Workflow
* **Design**: `ResidualBlocks` + `GAP`.
* **Initialization**: `Kaiming Normal`.
* **Data**: Custom `Mean/Std` + `Contrastive Augmentation`.
* **Pre-train**: `SSL` with `NT-Xent Loss`.
* **Fine-tune**: `Linear Probing` followed by `Full Tune`.
* **Verify**: `t-SNE` visualization.
* **Deploy**: `Inference Script` + `TorchScript`.

**Would you like me to show you the code to export your final model to TorchScript so you can use it in a C# or Kotlin environment?**