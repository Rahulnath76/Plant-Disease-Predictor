"""
Simple Single-Image Prediction Function
"""

import os
import torch
from PIL import Image
from .models import build_model
from .dataset import val_transform
from .config import Config
from .utils import get_disease_remedy


def predict(image_input, model_path="tomato_model.pth", model_name="custom_cnn"):
    """Predicts plant disease class for an image using notebook syntax."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = Config.CLASSES

    # Load model
    model = build_model(model_name, num_classes=len(classes)).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Open image and apply val_transform
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    else:
        image = image_input.convert('RGB')

    tensor_img = val_transform(image).unsqueeze(0).to(device)

    # Model inference using torch.max (exact notebook code)
    with torch.no_grad():
        outputs = model(tensor_img)
        _, predicted = torch.max(outputs, 1)

    predicted_class = classes[predicted.item()]
    remedy = get_disease_remedy(predicted_class)

    return {
        'class_name': predicted_class,
        'remedy': remedy
    }
