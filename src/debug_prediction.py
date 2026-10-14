from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =========================================================
# Paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_DIR
    / "dataset"
    / "test"
    / "fake"
    / "easy_102_0101.jpg"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "efficientnet_b0_finetuned_best.pth"
)


# =========================================================
# Device
# =========================================================

device = torch.device("cpu")


# =========================================================
# Load model
# =========================================================

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)

model.to(device)
model.eval()


# =========================================================
# Transform
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# Prediction function
# =========================================================

def predict_pil(image):

    tensor = transform(image)

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    fake = probabilities[0].item()
    real = probabilities[1].item()

    if fake > real:
        label = "Fake"
        confidence = fake
    else:
        label = "Real"
        confidence = real

    return label, confidence, fake, real


# =========================================================
# Load original image
# =========================================================

print("=" * 60)
print("MODEL DIAGNOSTIC TEST")
print("=" * 60)

print(f"Image: {IMAGE_PATH}")
print(f"Exists: {IMAGE_PATH.exists()}")


image = Image.open(
    IMAGE_PATH
).convert("RGB")


# =========================================================
# TEST 1: FULL IMAGE
# =========================================================

label, confidence, fake, real = predict_pil(image)

print()
print("FULL IMAGE PREDICTION")
print("-" * 40)

print(f"Prediction : {label}")
print(f"Confidence : {confidence * 100:.2f}%")
print(f"Fake       : {fake * 100:.2f}%")
print(f"Real       : {real * 100:.2f}%")


# =========================================================
# Face detection
# =========================================================

image_np = np.array(image)

image_bgr = cv2.cvtColor(
    image_np,
    cv2.COLOR_RGB2BGR
)

gray = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2GRAY
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)


# =========================================================
# TEST 2: FACE CROP
# =========================================================

if len(faces) == 0:

    print()
    print("No face detected.")
    raise SystemExit


largest_face = max(
    faces,
    key=lambda rect: rect[2] * rect[3]
)

x, y, w, h = largest_face

face = image_bgr[
    y:y+h,
    x:x+w
]

face_rgb = cv2.cvtColor(
    face,
    cv2.COLOR_BGR2RGB
)

face_pil = Image.fromarray(face_rgb)


label, confidence, fake, real = predict_pil(
    face_pil
)


print()
print("FACE CROP PREDICTION")
print("-" * 40)

print(f"Face size  : {w}x{h}")
print(f"Prediction : {label}")
print(f"Confidence : {confidence * 100:.2f}%")
print(f"Fake       : {fake * 100:.2f}%")
print(f"Real       : {real * 100:.2f}%")


print()
print("=" * 60)
print("DIAGNOSTIC TEST COMPLETED")
print("=" * 60)