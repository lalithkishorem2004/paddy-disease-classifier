"""
FastAPI Inference Demo for Paddy Leaf Disease Classifier.
Exposes:
  POST /predict               - classification prediction only
  POST /predict-with-gradcam  - classification prediction + Grad-CAM heatmap overlay (base64 PNG)
"""

import sys
import os
import io
import base64
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException

# Add workspace root and src directory to system path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))
if str(workspace_root / "src") not in sys.path:
    sys.path.append(str(workspace_root / "src"))

import config
from dataset import get_transforms
from evaluate import load_model

from fastapi.middleware.cors import CORSMiddleware

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = FastAPI(
    title="Paddy Leaf Disease Classifier API",
    description="A minimal REST API to classify paddy leaf diseases from uploaded images.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine the model checkpoint path
MODEL_PATH = os.getenv("MODEL_PATH", str(workspace_root / "models" / "best_model_field_light.pth"))

model = None
device = config.DEVICE
gradcam_helper = None  # GradCAM object, built once the model is loaded

# EfficientNet-B0's final conv block - matches the architecture used in training/evaluate.py.
# If your model uses a different backbone, this line is the one to adjust.
GRADCAM_TARGET_LAYER_PATH = "features"

IMG_SIZE = getattr(config, "IMG_SIZE", 224)


def init_model():
    """Initializes and loads the model checkpoint if it exists, and builds the Grad-CAM helper."""
    global model, gradcam_helper
    if not Path(MODEL_PATH).exists():
        print(f"\n[WARNING] Model checkpoint not found at: {MODEL_PATH}")
        print("Predictions will fail until a model is trained and saved there.")
        print("Run: python src/train.py --domain field --aug light\n")
        return False

    try:
        model, _ = load_model(MODEL_PATH, device)
        model.eval()

        # Build the Grad-CAM helper once, reused across requests
        target_layer = getattr(model, GRADCAM_TARGET_LAYER_PATH)[-1]
        gradcam_helper = GradCAM(model=model, target_layers=[target_layer])

        print(f"\n[INFO] Successfully loaded model from: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to load model from {MODEL_PATH}: {e}")
        return False


# Initialize the model on script loading
model_loaded = init_model()


@app.get("/")
def read_root():
    """Returns basic API metadata and service status."""
    return {
        "app_name": "Paddy Leaf Disease Classifier",
        "status": "ready" if model is not None else "model_missing",
        "target_classes": config.TARGET_CLASSES,
        "device": str(device),
        "model_path": MODEL_PATH
    }


def _read_and_preprocess(file_bytes):
    """Shared preprocessing: bytes -> PIL -> numpy RGB -> transformed tensor."""
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_np = np.array(image)

    transform = get_transforms("test")
    augmented = transform(image=img_np)
    input_tensor = augmented["image"].unsqueeze(0).to(device)

    return img_np, input_tensor


def _validate_extension(filename):
    ext = Path(filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a JPG, JPEG, or PNG image."
        )


def _ensure_model_loaded():
    global model, model_loaded
    if model is None:
        model_loaded = init_model()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model checkpoint not found on server. Please train a model first."
            )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file, processes it,
    and returns model predictions with confidence scores.
    """
    _ensure_model_loaded()
    _validate_extension(file.filename)

    try:
        contents = await file.read()
        img_np, input_tensor = _read_and_preprocess(contents)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        probabilities_dict = {
            config.TARGET_CLASSES[i]: float(probabilities[i])
            for i in range(len(config.TARGET_CLASSES))
        }

        top_idx = torch.argmax(probabilities).item()
        pred_class = config.TARGET_CLASSES[top_idx]
        confidence = float(probabilities[top_idx])

        return {
            "filename": file.filename,
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": probabilities_dict
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict-with-gradcam")
async def predict_with_gradcam(file: UploadFile = File(...)):
    """
    Same as /predict, but also returns a Grad-CAM heatmap overlay as a
    base64-encoded PNG, showing which region of the leaf the model focused
    on to make its prediction.
    """
    _ensure_model_loaded()
    _validate_extension(file.filename)

    try:
        contents = await file.read()
        img_np, input_tensor = _read_and_preprocess(contents)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        top_idx = torch.argmax(probabilities).item()
        pred_class = config.TARGET_CLASSES[top_idx]
        confidence = float(probabilities[top_idx])

        probabilities_dict = {
            config.TARGET_CLASSES[i]: float(probabilities[i])
            for i in range(len(config.TARGET_CLASSES))
        }

        # --- Grad-CAM generation ---
        # Grad-CAM needs gradients, so this part runs outside torch.no_grad()
        targets = [ClassifierOutputTarget(top_idx)]
        grayscale_cam = gradcam_helper(input_tensor=input_tensor, targets=targets)[0]

        # Resize original image to match model input size for a clean overlay
        img_resized = cv2.resize(img_np, (IMG_SIZE, IMG_SIZE))
        img_float = img_resized.astype(np.float32) / 255.0
        overlay = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

        # Encode overlay as base64 PNG so it can be sent in a JSON response
        success, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("Failed to encode Grad-CAM overlay image")
        gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "filename": file.filename,
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": probabilities_dict,
            "gradcam_image_base64": gradcam_base64
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
