"""
PyTorch Model Architectures (Custom CNN & ResNet18 Transfer Learning)
"""

import torch.nn as nn
from torchvision import models


class Model(nn.Module):
    """Custom 3-layer CNN Architecture."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Alias CustomCNN to Model
CustomCNN = Model


def get_resnet18_model(num_classes=10):
    """ResNet-18 Transfer Learning Model."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_fits = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_fits, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model


def build_model(model_name="custom_cnn", num_classes=10):
    """Model Factory Function."""
    if model_name.lower() in ["resnet18", "resnet", "transfer"]:
        return get_resnet18_model(num_classes=num_classes)
    return Model(num_classes=num_classes)
