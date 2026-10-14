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
# 3. Load final EfficientNet-B0 model
# =========================================================

model = models.efficientnet_b0(
    weights=None
)

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
# 5. Predict image
# =========================================================

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    )

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


    fake_probability = (
        probabilities[0].item()
    )

    real_probability = (
        probabilities[1].item()
    )


    predicted_class = torch.argmax(
        probabilities
    ).item()


    class_names = [
        "Fake",
        "Real"
    ]

    label = class_names[
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
# 6. Command-line testing
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python src/predict_image_final.py <image_path>"
        )

        raise SystemExit


    input_path = Path(
        sys.argv[1]
    )


    # If only filename is given,
    # search inside test_images
    if not input_path.is_absolute():

        input_path = (
            PROJECT_DIR
            / "test_images"
            / input_path
        )


    print("=" * 55)
    print("FINAL DEEPFAKE IMAGE DETECTION")
    print("=" * 55)

    print(
        f"Image  : {input_path}"
    )

    print(
        f"Exists : {input_path.exists()}"
    )


    if not input_path.exists():

        raise FileNotFoundError(
            f"Image not found: {input_path}"
        )


    result = predict_image(
        input_path
    )


    print()
    print("Prediction Results")
    print("-" * 35)

    print(
        f"Prediction : {result['label']}"
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

    print("=" * 55)