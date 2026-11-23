"""
Unit tests for PaddyDataset and Albumentations transforms.
Uses temporary mock directories and files for validation.
"""

import sys
import pytest
import cv2
import numpy as np
import torch
from pathlib import Path

# Add src directory to system path
workspace_root = Path(__file__).resolve().parent.parent
src_dir = workspace_root / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

import config
from dataset import PaddyDataset, get_transforms

@pytest.fixture
def mock_dataset_dir(tmp_path):
    """Creates a temporary dataset directory with mock images for testing."""
    for class_name in config.TARGET_CLASSES:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        
        # Create a few dummy images
        for i in range(3):
            # Create a random image (300x300, 3 channels)
            img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"mock_img_{i}.jpg"), img)
            
    return tmp_path

def test_dataset_loading(mock_dataset_dir):
    """Tests that PaddyDataset correctly scans and loads mock image samples."""
    # Initialize with default/no transform
    dataset = PaddyDataset(mock_dataset_dir, transform=None)
    
    # We created 3 mock images for each of the 5 target classes
    expected_total = len(config.TARGET_CLASSES) * 3
    assert len(dataset) == expected_total
    
    # Check that the returned values are of correct type
    img_tensor, label = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert isinstance(label, int)
    assert label in range(len(config.TARGET_CLASSES))
    
    # Without transform, shape should be (C, H, W) of the original image (3, 300, 300)
    assert img_tensor.shape == (3, 300, 300)

def test_dataset_transforms(mock_dataset_dir):
    """Tests that the transform pipelines output the correct shape and type."""
    # Test base validation transform
    val_transform = get_transforms("val")
    dataset_val = PaddyDataset(mock_dataset_dir, transform=val_transform)
    img_tensor, label = dataset_val[0]
    assert img_tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert img_tensor.dtype == torch.float32

    # Test light training transform
    light_transform = get_transforms("train", augmentation_type="light")
    dataset_light = PaddyDataset(mock_dataset_dir, transform=light_transform)
    img_tensor_l, _ = dataset_light[0]
    assert img_tensor_l.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    # Test heavy training transform
    heavy_transform = get_transforms("train", augmentation_type="heavy")
    dataset_heavy = PaddyDataset(mock_dataset_dir, transform=heavy_transform)
    img_tensor_h, _ = dataset_heavy[0]
    assert img_tensor_h.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)
