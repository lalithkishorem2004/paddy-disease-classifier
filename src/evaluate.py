"""
Evaluation script for the Paddy Leaf Disease Classifier.
Loads a trained model checkpoint and evaluates it on a specified dataset domain and split.
Computes and prints accuracy, classification report (precision/recall/F1), and confusion matrix.
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

# Add parent directory of this script to system path to ensure config and dataset are importable
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

import config
from dataset import PaddyDataset, get_transforms

def load_model(checkpoint_path, device):
    """Loads model from checkpoint."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model architecture
    model = efficientnet_b0()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(config.TARGET_CLASSES))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Runs inference and collects labels and predictions."""
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        
    return np.array(all_labels), np.array(all_preds)

def print_metrics(y_true, y_pred, class_names, title="Evaluation Summary"):
    """Computes and prints metrics using scikit-learn."""
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("="*60 + "\n")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Paddy Leaf Classifier model.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model checkpoint file (.pth).")
    parser.add_argument("--domain", type=str, default="field", choices=["field", "clean"],
                        help="The dataset domain to evaluate on (field or clean).")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="The dataset split to evaluate on (train, val, test).")
    args = parser.parse_args()
    
    # Load dataset
    data_dir = config.PROCESSED_DIR / args.domain / args.split
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset split directory not found: {data_dir}")
        
    device = config.DEVICE
    
    # Load model
    try:
        model, checkpoint = load_model(args.model, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    dataset = PaddyDataset(data_dir, transform=get_transforms(args.split))
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Evaluating {args.split} split for domain '{args.domain.upper()}' ({len(dataset)} images)...")
    y_true, y_pred = evaluate_model(model, dataloader, device)
    
    title = f"Evaluation Results - Domain: {args.domain.upper()} ({args.split.upper()})"
    print_metrics(y_true, y_pred, config.TARGET_CLASSES, title=title)

if __name__ == "__main__":
    main()
