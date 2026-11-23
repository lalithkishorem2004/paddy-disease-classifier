"""
Training script for the Paddy Leaf Disease Classifier.
Trains an EfficientNet-B0 model on the specified domain's processed split
using transfer learning and logs per-epoch training and validation metrics.
Checkpoints the best model based on validation accuracy.
"""

import os
import sys
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from pathlib import Path

# Add parent directory of this script to system path to ensure config and dataset are importable
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

import config
from dataset import PaddyDataset, get_transforms

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validates the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on Paddy Leaf Diseases.")
    parser.add_argument("--domain", type=str, default="field", choices=["field", "clean"],
                        help="The dataset domain to train on (field or clean).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs to train (overrides config).")
    parser.add_argument("--out", type=str, default=None,
                        help="Path to save the best model checkpoint (.pth).")
    parser.add_argument("--aug", type=str, default="light", choices=["light", "heavy"],
                        help="Type of data augmentation to use for training (light or heavy).")
    args = parser.parse_args()

    # Determine paths and options
    epochs = args.epochs if args.epochs is not None else config.EPOCHS
    domain_dir = config.PROCESSED_DIR / args.domain
    
    if args.out:
        checkpoint_path = Path(args.out)
    else:
        config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = config.MODEL_SAVE_DIR / f"best_model_{args.domain}_{args.aug}.pth"
        
    print("="*60)
    print(f"Starting training on domain: {args.domain.upper()}")
    print(f"Augmentation mode: {args.aug.upper()}")
    print(f"Device: {config.DEVICE}")
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Saving best model to: {checkpoint_path}")
    print("="*60)

    # Note on Hardware Acceleration
    # ----------------------------------------------------
    # EDUCATIONAL COMMENT:
    # Training deep neural networks (like EfficientNet-B0) on thousands of images
    # is extremely computationally expensive. If you run this script on a CPU,
    # training even one epoch might take 10-20 minutes or more.
    # RECOMMENDATION: For actual training runs, it is highly recommended to use
    # a GPU. You can run this code in a free Google Colab or Kaggle Notebook environment
    # by enabling the GPU accelerator (T4 or P100) and uploading this project's code.
    # ----------------------------------------------------
    if config.DEVICE.type == "cpu":
        print("\n[WARNING] You are training on a CPU. This will be slow!")
        print("To speed things up, consider training on Google Colab or Kaggle with a GPU.\n")

    set_seed(config.RANDOM_SEED)

    # Load datasets
    train_dir = domain_dir / "train"
    val_dir = domain_dir / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Processed data directories not found under {domain_dir}. "
            "Please run src/data_prep.py first."
        )
        
    train_dataset = PaddyDataset(train_dir, transform=get_transforms("train", args.aug))
    val_dataset = PaddyDataset(val_dir, transform=get_transforms("val"))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4 if os.name == 'posix' else 0,  # Use multiple workers on Linux, 0 on Windows
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4 if os.name == 'posix' else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Initialize Pretrained EfficientNet-B0
    print("Loading pretrained EfficientNet-B0...")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    # Replace the final fully connected classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(config.TARGET_CLASSES))
    
    model = model.to(config.DEVICE)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training Loop
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        
        elapsed = time.time() - start_time
        
        print(f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%", end="")
              
        # Save checkpoint if it's the best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': config.TARGET_CLASSES,
                'augmentation': args.aug,
                'domain': args.domain
            }, checkpoint_path)
            print(" [Saved Best Checkpoint]")
        else:
            print("")
            
    print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Best model saved to: {checkpoint_path}")

if __name__ == "__main__":
    main()
