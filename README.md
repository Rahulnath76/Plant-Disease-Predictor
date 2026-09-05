# Plant Disease Predictor 🍃🍅

A Deep Learning solution designed to detect and diagnose plant diseases from leaf imagery. The system leverages custom CNN and ResNet-18 transfer learning models built in PyTorch, accompanied by an interactive Gradio web application that provides real-time disease diagnosis, confidence metrics, and actionable agricultural remedies.

---

## 🎯 Overview

Plant diseases post-harvest and during crop growth significantly reduce agricultural yield. This project provides an end-to-end machine learning workflow to classify tomato plant leaf conditions across **10 distinct categories** (9 disease types + healthy state). In addition to classification, the application delivers actionable advisory reports—including symptoms, chemical/organic treatments, and prevention guidelines—to aid farmers and growers in early crop management.

---

## ✨ Key Features

- 🧠 **Dual Architecture Support**:
  - **Custom CNN**: A light 3-block Convolutional Neural Network with Batch Normalization, ReLU activation, Dropout (0.4), and Adaptive Average Pooling.
  - **ResNet-18 Transfer Learning**: Pre-trained ResNet-18 backbone fine-tuned for transfer learning on agricultural leaf features.
- 🩺 **Integrated Agricultural Advisory**: Returns instant remedy advice for identified diseases (Symptoms, Treatment, and Prevention).
- ⚡ **Gradio Web Interface**: User-friendly drag-and-drop web dashboard for quick leaf photo uploads and diagnosis visualization.
- ⚙️ **Robust Training Engine**: Built-in early stopping mechanism, customizable hyperparameters via YAML configs, and metric plotting utilities.
- 📦 **Modular Package Design**: Modular Python structure (`src/plant_disease/`) separating datasets, models, training loops, and utility modules.

---

## 🦠 Supported Disease Classes

The model classifies images into the following 10 categories:

1. **Bacterial Spot** (`Tomato___Bacterial_spot`)
2. **Early Blight** (`Tomato___Early_blight`)
3. **Late Blight** (`Tomato___Late_blight`)
4. **Leaf Mold** (`Tomato___Leaf_Mold`)
5. **Septoria Leaf Spot** (`Tomato___Septoria_leaf_spot`)
6. **Spider Mites (Two-Spotted)** (`Tomato___Spider_mites Two-spotted_spider_mite`)
7. **Target Spot** (`Tomato___Target_Spot`)
8. **Yellow Leaf Curl Virus** (`Tomato___Tomato_Yellow_Leaf_Curl_Virus`)
9. **Mosaic Virus** (`Tomato___Tomato_mosaic_virus`)
10. **Healthy Leaf** (`Tomato___healthy`)

---

## 📊 Dataset Information

- **Dataset Source**: [New Plant Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) (`vipoooool/new-plant-diseases-dataset`)
- **Overview**: The dataset consists of over 87,000 RGB images of healthy and diseased crop leaves generated via offline augmentation from the original PlantVillage dataset.
- **Tomato Subset**: This project specifically extracts and utilizes the Tomato crop subset, which contains 18,000+ images categorized across 10 tomato leaf classes.
- **Data Splits**:
  - **Train Set**: Used for model parameter optimization with data augmentations (flips, rotations, normalization).
  - **Valid Set**: Used for validation evaluation and early stopping checks.
  - **Test Set**: Unseen test samples for final inference verification.
- **Download via Code**: The project integrates [`kagglehub`](https://github.com/kaggle/kagglehub) to automatically download and extract the dataset on demand.

---


## 📂 Directory Layout

```text
Plant Disease Prediction/
├── app.py                      # Interactive Gradio Web App entry point
├── requirements.txt            # Python environment dependencies
├── configs/
│   └── config.yaml             # Hyperparameters, dataset paths, & model settings
├── src/
│   └── plant_disease/
│       ├── __init__.py         # Package initialization
│       ├── config.py           # Python configuration class
│       ├── dataset.py          # PyTorch Datasets & DataLoaders with data transforms
│       ├── models.py           # Custom CNN & ResNet-18 architecture definitions
│       ├── predict.py          # Single/batch inference pipeline
│       ├── trainer.py          # Model training loop & Early Stopping logic
│       └── utils.py            # Disease remedy database & visualization functions
└── notebooks/
    └── plant_disease_prediction.ipynb # Interactive training & EDA notebook
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Ensure Python 3.8+ is installed. Clone the repository and install required packages:

```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application

To run the interactive web interface:

```bash
python app.py
```

Once running, open the local URL output by Gradio (typically `http://127.0.0.1:7860`) in your browser to upload leaf images and obtain real-time diagnoses.

### 3. Model Training & Exploration

To train models or explore data analysis interactively:
- Open `notebooks/plant_disease_prediction.ipynb` in Jupyter Notebook or VS Code.
- Alternatively, utilize modules from `src/plant_disease` in your own python scripts.

---

## 🛠️ Tech Stack

- **Deep Learning Framework**: [PyTorch](https://pytorch.org/) & [torchvision](https://pytorch.org/vision/stable/index.html)
- **Web UI Framework**: [Gradio](https://www.gradio.app/)
- **Data & Image Processing**: NumPy, Pillow, Matplotlib
- **Dataset Hosting**: Kaggle / KaggleHub
