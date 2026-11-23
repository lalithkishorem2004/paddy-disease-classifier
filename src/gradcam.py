"""
Grad-CAM Explainability Script.
Loads a trained model checkpoint and generates Grad-CAM visual explanations
for a given input image, saving the overlaid heatmap visualization.
"""

import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.models import efficientnet_b0

# Add parent directory of this script to system path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.append(str(script_dir))

import config
from evaluate import load_model

# Import pytorch-grad-cam components
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def preprocess_image_for_cam(image_path, size=224):
    """Loads and preprocesses image for EfficientNet-B0 Grad-CAM."""
    # Load original image
    bgr_img = cv2.imread(str(image_path))
    if bgr_img is None:
        raise IOError(f"Could not load image at path: {image_path}")
        
    # Resize and convert to RGB
    bgr_img = cv2.resize(bgr_img, (size, size))
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    # Scale to [0, 1] float32 (required by show_cam_on_image)
    rgb_img_float = np.float32(rgb_img) / 255.0
    
    # Normalize image tensor for model input
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_img = (rgb_img_float - mean) / std
    
    # Rearrange dimensions to (C, H, W) and add batch dimension (1, C, H, W)
    input_tensor = torch.from_numpy(normalized_img).permute(2, 0, 1).unsqueeze(0).float()
    
    return rgb_img_float, input_tensor

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanation for a leaf image.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model checkpoint file (.pth).")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image file.")
    parser.add_argument("--out", type=str, default="gradcam_output.jpg",
                        help="Path to save the generated visualization.")
    parser.add_argument("--target-class", type=int, default=None,
                        help="Target class index for Grad-CAM. Defaults to model's predicted class.")
    args = parser.parse_args()

    device = config.DEVICE
    
    # Load the trained model
    try:
        model, checkpoint = load_model(args.model, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # Load and preprocess image
    try:
        rgb_img, input_tensor = preprocess_image_for_cam(args.image, size=config.IMAGE_SIZE)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)
        
    # Send image tensor to device
    input_tensor = input_tensor.to(device)
    
    # Run model prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        pred_class_idx = torch.argmax(probabilities).item()
        
    pred_class_name = config.TARGET_CLASSES[pred_class_idx]
    pred_confidence = probabilities[pred_class_idx].item()
    
    print(f"\nModel Prediction:")
    print(f"  Class Index: {pred_class_idx}")
    print(f"  Class Name:  {pred_class_name}")
    print(f"  Confidence:  {pred_confidence*100:.2f}%")

    # Select target class for visualization
    target_idx = args.target_class if args.target_class is not None else pred_class_idx
    target_class_name = config.TARGET_CLASSES[target_idx]
    print(f"\nGenerating Grad-CAM for class: '{target_class_name}' (Index: {target_idx})...")
    
    # Define target layers. For EfficientNet-B0, model.features[-1] is the final convolutional layer.
    target_layers = [model.features[-1]]
    
    # Initialize GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Define target category output
    targets = [ClassifierOutputTarget(target_idx)]
    
    # Generate CAM mask
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Extract the single image CAM mask
    
    # Overlay heatmap on the original RGB image (rgb_img is float32 [0, 1])
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Save the output image
    out_path = Path(args.out)
    cv2.imwrite(str(out_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    print(f"Grad-CAM visual explanation saved successfully to: {out_path}\n")

if __name__ == "__main__":
    main()
