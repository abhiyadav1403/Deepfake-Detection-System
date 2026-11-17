
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "deepfake_efficientnet_b0_final.pth"
)


# ==========================================
# 2. DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = models.efficientnet_b0(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 2)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    print("EfficientNet-B0 loaded successfully!")

    return model


model = load_model()


# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# 5. IMAGE PREDICTION
# ==========================================

def predict_image(image: Image.Image):

    image = image.convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    fake_probability = probabilities[0].item()
    real_probability = probabilities[1].item()

    predicted_class = torch.argmax(
        probabilities
    ).item()

    class_names = ["FAKE", "REAL"]

    prediction = class_names[predicted_class]

    confidence = probabilities[predicted_class].item()

    return {
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "fake_probability": round(
            fake_probability * 100, 2
        ),
        "real_probability": round(
            real_probability * 100, 2
        )
    }
