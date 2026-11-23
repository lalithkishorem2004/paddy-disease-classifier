"""
Dataset module for the Paddy Leaf Disease Classifier.
Contains the custom PyTorch Dataset class (PaddyDataset) and 
the Albumentations transformation pipelines (light, heavy, and validation).
"""

import sys
import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Ensure config is importable
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

import config

class PaddyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading paddy leaf images.
    Filters to target classes and applies Albumentations transforms.
    """
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (Path or str): Path to the processed split directory (e.g., train, val, test)
            transform (albumentations.Compose, optional): Transformations to apply to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Create map from class name to label index
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(config.TARGET_CLASSES)}
        
        # Scan data directory for images corresponding to target classes
        self._load_samples()
        
    def _load_samples(self):
        for class_name in config.TARGET_CLASSES:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
                
            # Grab all image extensions
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                    
        # Sort samples to ensure consistent indexing and reproducibility
        self.samples.sort(key=lambda x: x[0].name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image in BGR format
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise IOError(f"Could not load image at path: {img_path}")
            
        # Convert to RGB format
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply transformation if provided
        if self.transform is not None:
            augmented = self.transform(image=img_rgb)
            img_tensor = augmented["image"]
        else:
            # Fallback to convert numpy to PyTorch tensor if no transform
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
            
        return img_tensor, label


def get_transforms(split_name, augmentation_type="light"):
    """
    Generates Albumentations transformation pipelines for training or evaluation.
    
    Args:
        split_name (str): One of "train", "val", or "test"
        augmentation_type (str): "light" or "heavy" (used only for train split)
        
    Returns:
        albumentations.Compose: Transformation pipeline
    """
    # Base transforms applied to validation and testing (no data augmentation)
    base_transform = A.Compose([
        A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    if split_name != "train":
        return base_transform
        
    if augmentation_type == "light":
        # Standard lighter augmentations (e.g. flipping, rotating)
        return A.Compose([
            A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
    elif augmentation_type == "heavy":
        # Heavier field-realistic augmentations: blur, brightness/contrast adjustments, shadows, occlusion
        return A.Compose([
            A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            
            # Brightness / Contrast adjustments
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            
            # Simulated field blur (camera motion or refocusing issues)
            A.OneOf([
                A.MotionBlur(p=1.0),
                A.GaussianBlur(p=1.0),
            ], p=0.3),
            
            # Simulated natural shadow cover from crop foliage
            A.RandomShadow(p=0.3),
            
            # Simulated occlusions/clutter (e.g., weeds, soil, overlapping leaves) using CoarseDropout
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(8, 24),
                hole_width_range=(8, 24),
                fill=0,
                p=0.3
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")
