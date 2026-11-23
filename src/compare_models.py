"""
compare_models.py

Runs both trained checkpoints (light-augmentation vs heavy-augmentation)
against the same set of test images and prints a side-by-side comparison
table, including whether each prediction matches the true label.

Usage (run from the project root, or adjust the paths below):
    python src/compare_models.py

Requires: torch, torchvision, opencv-python, albumentations
    pip install torch torchvision opencv-python albumentations
"""

import os
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Adjust these paths to match your local project layout ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LIGHT_CHECKPOINT = PROJECT_ROOT / "models" / "best_model_field_light.pth"
HEAVY_CHECKPOINT = PROJECT_ROOT / "models" / "best_model_field_heavy.pth"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "field" / "test"

TARGET_CLASSES = ["bacterial_leaf_blight", "blast", "brown_spot", "hispa", "normal"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Set this to limit how many images per class to test (None = use all)
MAX_IMAGES_PER_CLASS = 3


def get_eval_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_model(num_classes):
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(checkpoint_path, device):
    """Load a model from a checkpoint saved by the Colab training notebook."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(len(TARGET_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    val_acc = checkpoint.get("val_acc", None)
    return model, val_acc


def predict(model, image_path, transform, device):
    """Run a single image through a model, return (predicted_class, confidence)."""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(transformed)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    return TARGET_CLASSES[pred_idx], confidence


def collect_test_images():
    """Gather a small labeled sample from each class folder in the test set."""
    samples = []
    for class_name in TARGET_CLASSES:
        class_dir = TEST_DIR / class_name
        if not class_dir.is_dir():
            print(f"Warning: test folder not found for class '{class_name}' at {class_dir}")
            continue
        image_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if MAX_IMAGES_PER_CLASS is not None:
            image_files = image_files[:MAX_IMAGES_PER_CLASS]
        for fname in image_files:
            samples.append((class_dir / fname, class_name))
    return samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Loading checkpoints...")
    light_model, light_val_acc = load_checkpoint(LIGHT_CHECKPOINT, device)
    heavy_model, heavy_val_acc = load_checkpoint(HEAVY_CHECKPOINT, device)
    print(f"  Light model  - reported val_acc during training: {light_val_acc}")
    print(f"  Heavy model  - reported val_acc during training: {heavy_val_acc}\n")

    transform = get_eval_transform()
    samples = collect_test_images()

    if not samples:
        print("No test images found. Check TEST_DIR path.")
        return

    print(f"Running comparison on {len(samples)} images...\n")

    light_correct = 0
    heavy_correct = 0

    header = f"{'Image':<20}{'True Label':<22}{'Light Pred':<22}{'Light Conf':<12}{'Heavy Pred':<22}{'Heavy Conf':<12}"
    print(header)
    print("-" * len(header))

    for image_path, true_label in samples:
        light_pred, light_conf = predict(light_model, image_path, transform, device)
        heavy_pred, heavy_conf = predict(heavy_model, image_path, transform, device)

        if light_pred == true_label:
            light_correct += 1
        if heavy_pred == true_label:
            heavy_correct += 1

        light_mark = "✓" if light_pred == true_label else "✗"
        heavy_mark = "✓" if heavy_pred == true_label else "✗"

        print(
            f"{image_path.name:<20}{true_label:<22}"
            f"{light_pred + ' ' + light_mark:<22}{light_conf * 100:>6.1f}%     "
            f"{heavy_pred + ' ' + heavy_mark:<22}{heavy_conf * 100:>6.1f}%"
        )

    total = len(samples)
    print("\n" + "=" * len(header))
    print(f"Light model accuracy on this sample: {light_correct}/{total} ({light_correct/total*100:.1f}%)")
    print(f"Heavy model accuracy on this sample: {heavy_correct}/{total} ({heavy_correct/total*100:.1f}%)")
    print("=" * len(header))
    print(
        "\nNote: this is a small labeled sample from the same field dataset's "
        "test split, so it mainly tells you which model handles held-out field "
        "images better. For the full clean-vs-field generalization gap "
        "experiment, you'll additionally test a model trained on the clean "
        "UCI dataset against this same field test set."
    )


if __name__ == "__main__":
    main()
