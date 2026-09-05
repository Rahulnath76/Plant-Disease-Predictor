"""
Utility Functions: Reproducibility, Visualizations, and Agricultural Disease Remedies
"""

import os
import random
from typing import Dict, List, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt

# Comprehensive Agricultural Treatment & Remedy Guide for Tomato Diseases
DISEASE_REMEDIES: Dict[str, Dict[str, str]] = {
    'Tomato___Bacterial_spot': {
        'name': 'Bacterial Spot',
        'symptoms': 'Small, dark, water-soaked spots on leaves that turn brown/black with yellow halos.',
        'treatment': 'Apply copper-based fungicides/bactericides early. Avoid overhead irrigation to prevent leaf wetness.',
        'prevention': 'Use certified disease-free seeds, practice 2-3 year crop rotation, and disinfect tools.'
    },
    'Tomato___Early_blight': {
        'name': 'Early Blight',
        'symptoms': 'Concentric target-like dark spots on older leaves, surrounded by yellowing area.',
        'treatment': 'Spray chlorothalonil, mancozeb, or copper fungicides. Remove and destroy infected lower leaves.',
        'prevention': 'Mulch around plant bases to prevent soil splash, ensure adequate plant spacing for airflow.'
    },
    'Tomato___Late_blight': {
        'name': 'Late Blight',
        'symptoms': 'Large dark brown/black water-soaked patches on leaves with white fungal mold under humid conditions.',
        'treatment': 'Apply systemic fungicides such as metalaxyl or copper-based sprays immediately.',
        'prevention': 'Destroy volunteer tomato plants and solanaceous weeds; plant resistant varieties.'
    },
    'Tomato___Leaf_Mold': {
        'name': 'Leaf Mold',
        'symptoms': 'Pale green or yellow spots on upper leaf surface with olive-green velvety mold underneath.',
        'treatment': 'Apply appropriate protective fungicides (copper, chlorothalonil) at early symptom notice.',
        'prevention': 'Improve greenhouse ventilation, reduce relative humidity below 85%, avoid wetting foliage.'
    },
    'Tomato___Septoria_leaf_spot': {
        'name': 'Septoria Leaf Spot',
        'symptoms': 'Numerous small circular spots with dark margins and gray centers containing tiny black dots.',
        'treatment': 'Apply copper or chlorothalonil fungicides. Prune infected bottom leaves near ground level.',
        'prevention': 'Stake plants, remove crop debris after harvest, rotate crops away from solanaceous plants.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'name': 'Spider Mites (Two-Spotted)',
        'symptoms': 'Fine yellow/white stippling on leaves, leaf bronzing, and fine webbing on leaf undersides.',
        'treatment': 'Apply insecticidal soap, neem oil, or specific miticides; wash leaves with water spray.',
        'prevention': 'Maintain proper moisture, avoid over-fertilization with high nitrogen, introduce predatory mites.'
    },
    'Tomato___Target_Spot': {
        'name': 'Target Spot',
        'symptoms': 'Circular brown lesions with light brown centers and dark concentric rings.',
        'treatment': 'Spray broad-spectrum fungicides (azoxystrobin, chlorothalonil).',
        'prevention': 'Avoid dense canopy foliage, manage weed hosts, improve field drainage.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'name': 'Tomato Yellow Leaf Curl Virus (TYLCV)',
        'symptoms': 'Stunted growth, upward leaf curling, yellow margins, and flower drop.',
        'treatment': 'No chemical cure for virus once infected; pull out and destroy severely infected plants.',
        'prevention': 'Control whitefly vector populations using sticky traps, insecticidal sprays, or reflective mulch.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'name': 'Tomato Mosaic Virus (ToMV)',
        'symptoms': 'Mottled dark green and light yellow leaf patterns, leaf distortion, and stunted stems.',
        'treatment': 'No chemical cure available. Remove infected plants to limit viral spread.',
        'prevention': 'Wash hands and sanitize tools with trisodium phosphate or milk solution before handling plants.'
    },
    'Tomato___healthy': {
        'name': 'Healthy Tomato Leaf',
        'symptoms': 'Vibrant green leaves with no spots, lesions, discoloration, or viral mottling.',
        'treatment': 'No treatment necessary. Maintain good agricultural practices.',
        'prevention': 'Continue balanced watering, proper fertilization, and routine monitoring.'
    }
}


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_graph(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    """Plots training and validation loss and accuracy graphs."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def get_disease_remedy(class_name):
    """Returns remedy information for a given disease class."""
    return DISEASE_REMEDIES.get(class_name, {
        'name': class_name,
        'symptoms': 'Symptom information unavailable.',
        'treatment': 'Consult an agricultural specialist.',
        'prevention': 'Practice proper sanitation and crop management.'
    })

