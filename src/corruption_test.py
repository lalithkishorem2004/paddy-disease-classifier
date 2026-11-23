"""
corruption_test.py

Simulates harder, non-textbook field conditions by applying synthetic
corruption (motion blur, low light, shadows, noise) to your existing
labeled test images, then compares how the light-augmentation model vs
the heavy-augmentation model hold up on both the original and corrupted
versions.

This is the core "generalization gap" evidence for your project: it
shows whether heavy field-realistic augmentation actually earns its
keep on harder images, even if both models look identical on clean
held-out data.

Usage (run from the project root):
    python src/corruption_test.py

Requires: torch, torchvision, opencv-python, albumentations
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
CORRUPTED_SAMPLES_DIR = PROJECT_ROOT / "data" / "corrupted_samples"

TARGET_CLASSES = ["bacterial_leaf_blight", "blast", "brown_spot", "hispa", "normal"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MAX_IMAGES_PER_CLASS = 15 # 4 x 5 classes = 20 images tested, each in 2 conditions


def get_corruption_transform():
    """
    Simulates a genuinely harder real-world photo: motion blur (shaky
    hands), reduced brightness (overcast/evening light), a cast shadow,
    and sensor noise (cheap phone camera). This is intentionally more
    aggressive than the 'heavy' training augmentation so it's a fair
    out-of-distribution stress test for both models.
    """
    return A.Compose([
        A.MotionBlur(blur_limit=9, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.15), contrast_limit=0.1, p=1.0),
        A.RandomShadow(num_shadows_limit=(1, 3), shadow_dimension=6, p=0.8),
        A.GaussNoise(var_limit=(20.0, 60.0), p=0.7),
    ])


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
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(len(TARGET_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_from_array(model, image_rgb, eval_transform, device):
    transformed = eval_transform(image=image_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(transformed)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
    return TARGET_CLASSES[pred_idx], confidence


def collect_test_images():
    samples = []
    for class_name in TARGET_CLASSES:
        class_dir = TEST_DIR / class_name
        if not class_dir.is_dir():
            print(f"Warning: test folder not found for class '{class_name}' at {class_dir}")
            continue
        image_files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:MAX_IMAGES_PER_CLASS]
        for fname in image_files:
            samples.append((class_dir / fname, class_name))
    return samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Loading checkpoints...")
    light_model = load_checkpoint(LIGHT_CHECKPOINT, device)
    heavy_model = load_checkpoint(HEAVY_CHECKPOINT, device)

    eval_transform = get_eval_transform()
    corruption_transform = get_corruption_transform()
    samples = collect_test_images()

    if not samples:
        print("No test images found. Check TEST_DIR path.")
        return

    # Save a few corrupted samples to disk so you can visually inspect
    # them / include one in your report as a before-after example.
    CORRUPTED_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running clean vs. corrupted comparison on {len(samples)} images...\n")

    results = {
        "light_clean": 0,
        "light_corrupted": 0,
        "heavy_clean": 0,
        "heavy_corrupted": 0,
    }
    total = len(samples)

    header = f"{'Image':<16}{'True':<20}{'Light-Clean':<16}{'Light-Corrupt':<16}{'Heavy-Clean':<16}{'Heavy-Corrupt':<16}"
    print(header)
    print("-" * len(header))

    for i, (image_path, true_label) in enumerate(samples):
        image_bgr = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        corrupted_rgb = corruption_transform(image=image_rgb)["image"]

        # Save a handful of corrupted examples for visual inspection / report
        if i < 5:
            corrupted_bgr = cv2.cvtColor(corrupted_rgb, cv2.COLOR_RGB2BGR)
            out_path = CORRUPTED_SAMPLES_DIR / f"corrupted_{image_path.name}"
            cv2.imwrite(str(out_path), corrupted_bgr)

        light_clean_pred, _ = predict_from_array(light_model, image_rgb, eval_transform, device)
        light_corrupt_pred, _ = predict_from_array(light_model, corrupted_rgb, eval_transform, device)
        heavy_clean_pred, _ = predict_from_array(heavy_model, image_rgb, eval_transform, device)
        heavy_corrupt_pred, _ = predict_from_array(heavy_model, corrupted_rgb, eval_transform, device)

        results["light_clean"] += int(light_clean_pred == true_label)
        results["light_corrupted"] += int(light_corrupt_pred == true_label)
        results["heavy_clean"] += int(heavy_clean_pred == true_label)
        results["heavy_corrupted"] += int(heavy_corrupt_pred == true_label)

        def mark(pred):
            return f"{pred} {'✓' if pred == true_label else '✗'}"

        print(
            f"{image_path.name:<16}{true_label:<20}"
            f"{mark(light_clean_pred):<16}{mark(light_corrupt_pred):<16}"
            f"{mark(heavy_clean_pred):<16}{mark(heavy_corrupt_pred):<16}"
        )

    print("\n" + "=" * len(header))
    print("SUMMARY")
    print("=" * len(header))
    light_clean_acc = results["light_clean"] / total * 100
    light_corrupt_acc = results["light_corrupted"] / total * 100
    heavy_clean_acc = results["heavy_clean"] / total * 100
    heavy_corrupt_acc = results["heavy_corrupted"] / total * 100

    print(f"Light model  - clean accuracy:     {light_clean_acc:.1f}%")
    print(f"Light model  - corrupted accuracy: {light_corrupt_acc:.1f}%  (drop: {light_clean_acc - light_corrupt_acc:.1f} points)")
    print(f"Heavy model  - clean accuracy:     {heavy_clean_acc:.1f}%")
    print(f"Heavy model  - corrupted accuracy: {heavy_corrupt_acc:.1f}%  (drop: {heavy_clean_acc - heavy_corrupt_acc:.1f} points)")
    print("=" * len(header))
    print(f"\nSample corrupted images saved to: {CORRUPTED_SAMPLES_DIR}")
    print(
        "\nThis is your headline result: whichever model has the SMALLER "
        "accuracy drop under corruption is the one that generalizes better "
        "to real-world conditions — that's the concrete evidence for your "
        "project writeup, not the validation accuracy alone."
    )


if __name__ == "__main__":
    main()
