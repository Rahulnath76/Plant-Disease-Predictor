"""
Simple Image Prediction Helper
"""

import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from .models import build_model
from .config import Config
from .utils import get_disease_remedy

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(image_input, model_path="tomato_model.pth", model_name="custom_cnn"):
    """Predicts disease class for a given image file or PIL Image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = Config.CLASSES

    # Load model
    model = build_model(model_name, num_classes=len(classes))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and transform image
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    else:
        image = image_input.convert('RGB')

    tensor_img = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = F.softmax(outputs, dim=1)[0]
        idx = torch.argmax(probs).item()

    predicted_class = classes[idx]
    confidence = round(probs[idx].item() * 100, 2)
    remedy = get_disease_remedy(predicted_class)

    return {
        'class_name': predicted_class,
        'confidence': confidence,
        'remedy': remedy
    }
