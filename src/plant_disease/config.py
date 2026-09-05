"""
Simple Configuration File
"""

class Config:
    # Dataset Settings
    DATASET_NAME = "vipoooool/new-plant-diseases-dataset"
    DATA_DIR = "./tomato_dataset"
    NUM_CLASSES = 10
    CLASSES = [
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy"
    ]

    # Image & Transform Settings
    IMAGE_SIZE = (224, 224)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    # Training Settings
    BATCH_SIZE = 128
    NUM_WORKERS = 2
    LEARNING_RATE = 0.001
    EPOCHS = 20
    SEED = 42
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_DELTA = 0.001
    CHECKPOINT_DIR = "./checkpoints"
