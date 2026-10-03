from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "efficientnet_b0_finetuned_best.pth"


# =========================================================
# 2. Device
# =========================================================

device = torch.device("cpu")


# =========================================================
# 3. Load EfficientNet-B0
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

model = model.to(device)
model.eval()


# =========================================================
# 4. Image preprocessing
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
# 5. Face detection
# =========================================================

def detect_face(image_path):

    # Read image using PIL
    pil_image = Image.open(image_path).convert("RGB")

    # Convert PIL -> NumPy
    image = np.array(pil_image)

    # RGB -> BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    # Select largest face
    largest_face = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    x, y, w, h = largest_face

    face = image_bgr[y:y+h, x:x+w]

    return face


# =========================================================
# 6. Predict face
# =========================================================

def predict_image(image_path):

    # Detect face
    face = detect_face(image_path)

    if face is None:
        return {
            "label": "No Face Detected",
            "confidence": 0.0,
            "fake_probability": 0.0,
            "real_probability": 0.0
        }

    # OpenCV BGR -> RGB
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # NumPy -> PIL
    face_pil = Image.fromarray(face_rgb)

    # Preprocess
    image_tensor = transform(face_pil)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # Prediction
    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    fake_probability = probabilities[0].item()
    real_probability = probabilities[1].item()

    # Class prediction
    predicted_class = torch.argmax(probabilities).item()

    class_names = [
        "Fake",
        "Real"
    ]

    label = class_names[predicted_class]

    confidence = probabilities[predicted_class].item()

    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": fake_probability,
        "real_probability": real_probability
    }


# =========================================================
# 7. Test
# =========================================================

if __name__ == "__main__":

    image_path = PROJECT_DIR / "test_images" / "test1.jpg"

    print("=" * 50)
    print("DEEPFAKE IMAGE DETECTION")
    print("=" * 50)

    print(f"Image: {image_path}")
    print(f"Exists: {image_path.exists()}")

    result = predict_image(image_path)

    print("\nPrediction Results")
    print("-" * 30)

    print(f"Prediction : {result['label']}")
    print(f"Confidence : {result['confidence'] * 100:.2f}%")
    print(f"Fake       : {result['fake_probability'] * 100:.2f}%")
    print(f"Real       : {result['real_probability'] * 100:.2f}%")

    print("=" * 50)