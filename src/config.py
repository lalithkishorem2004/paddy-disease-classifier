"""
Configuration module for the Paddy Leaf Disease Classifier.
Contains path definitions, target classes, splitting ratios, and hyperparameters.
"""

from pathlib import Path
import torch

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Raw data paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "paddy_doctor"
RAW_IMAGES_DIR = RAW_DATA_DIR / "train_images"
RAW_CSV_PATH = RAW_DATA_DIR / "train.csv"

# Processed data paths
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIELD_PROCESSED_DIR = PROCESSED_DIR / "field"

# Models save directory
MODEL_SAVE_DIR = BASE_DIR / "models"

# Target classes to use (Subset of 5 classes out of 10)
TARGET_CLASSES = [
    "bacterial_leaf_blight",
    "blast",
    "brown_spot",
    "hispa",
    "normal"
]

# Split ratios for stratified split (Train: 80%, Val: 10%, Test: 10%)
SPLIT_RATIOS = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10
}

# Image parameters
IMAGE_SIZE = 224

# Hyperparameters
RANDOM_SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10  # Default number of epochs

# Compute device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
