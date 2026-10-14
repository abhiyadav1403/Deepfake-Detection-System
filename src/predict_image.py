from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =========================================================
# 1. Project paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "efficientnet_b0_finetuned_best.pth"
)


# =========================================================
# 2. Device
# =========================================================

device = torch.device("cpu")


# =========================================================
# 3. Class names
# =========================================================

CLASS_NAMES = [
    "Fake",
    "Real"
]


# =========================================================
# 4. Load EfficientNet-B0
# =========================================================

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    2
)


# =========================================================
# 5. Load trained model
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)

model = model.to(device)

model.eval()


# =========================================================
# 6. Image preprocessing
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
# 7. Predict image
# =========================================================

def predict_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Open original FULL image
    image = Image.open(
        image_path
    ).convert("RGB")

    # Preprocess full image
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # Model prediction
    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    # Probabilities
    fake_probability = probabilities[0].item()

    real_probability = probabilities[1].item()

    # Predicted class
    predicted_class = torch.argmax(
        probabilities
    ).item()

    label = CLASS_NAMES[
        predicted_class
    ]

    confidence = probabilities[
        predicted_class
    ].item()

    return {
        "label": label,
        "confidence": confidence,
        "fake_probability": fake_probability,
        "real_probability": real_probability
    }


# =========================================================
# 8. Command-line execution
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "python src/predict_image.py <image_path>"
        )

        raise SystemExit

    input_path = Path(
        sys.argv[1]
    )


    # -----------------------------------------------------
    # If only filename is provided,
    # search inside test_images
    # -----------------------------------------------------

    if not input_path.is_absolute():

        input_path = (
            PROJECT_DIR
            / "test_images"
            / input_path
        )


    image_path = input_path


    # -----------------------------------------------------
    # Display information
    # -----------------------------------------------------

    print("=" * 50)

    print(
        "DEEPFAKE IMAGE DETECTION"
    )

    print("=" * 50)

    print(
        f"Image : {image_path}"
    )

    print(
        f"Exists: {image_path.exists()}"
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    result = predict_image(
        image_path
    )


    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print("\nPrediction Results")

    print("-" * 30)

    print(
        f"Prediction : "
        f"{result['label']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Fake       : "
        f"{result['fake_probability'] * 100:.2f}%"
    )

    print(
        f"Real       : "
        f"{result['real_probability'] * 100:.2f}%"
    )

    print("=" * 50)