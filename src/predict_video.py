from pathlib import Path
import csv
import sys

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

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "efficientnet_b0_finetuned_best.pth"
)

RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
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

print("Loading EfficientNet-B0 model...")

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

print("Model loaded successfully.")


# =========================================================
# 5. Image preprocessing
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
# 6. Predict one full video frame
# =========================================================

def predict_frame(frame):

    # OpenCV BGR -> RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # NumPy -> PIL
    image = Image.fromarray(
        frame_rgb
    )

    # Same preprocessing used during training
    image_tensor = transform(
        image
    )

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

    fake_probability = probabilities[0].item()

    real_probability = probabilities[1].item()

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
# 7. Predict video
# =========================================================

def predict_video(
    video_path,
    sample_every_seconds=1
):

    video_path = Path(
        video_path
    )

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )


    # -----------------------------------------------------
    # Open video
    # -----------------------------------------------------

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )


    # -----------------------------------------------------
    # Video information
    # -----------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )


    print()
    print("=" * 60)
    print("DEEPFAKE VIDEO DETECTION")
    print("=" * 60)

    print(
        f"Video        : {video_path}"
    )

    print(
        f"FPS          : {fps:.2f}"
    )

    print(
        f"Total Frames : {total_frames}"
    )

    print(
        f"Duration     : {duration:.2f} seconds"
    )

    print(
        f"Sampling     : "
        f"1 frame every {sample_every_seconds} second"
    )

    print("=" * 60)


    # -----------------------------------------------------
    # Frame sampling
    # -----------------------------------------------------

    frame_interval = max(
        1,
        int(
            fps * sample_every_seconds
        )
    )


    frame_number = 0

    frames_checked = 0

    fake_probabilities = []

    real_probabilities = []

    fake_count = 0

    real_count = 0


    # -----------------------------------------------------
    # CSV output
    # -----------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / f"{video_path.stem}_video_results.csv"
    )

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow([
        "Frame",
        "Time_Seconds",
        "Fake_Probability",
        "Real_Probability",
        "Prediction"
    ])


    # -----------------------------------------------------
    # Process video
    # -----------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break


        # Sample approximately once per second
        if frame_number % frame_interval != 0:

            frame_number += 1

            continue


        frames_checked += 1


        # Predict FULL frame
        result = predict_frame(
            frame
        )


        fake_probability = result[
            "fake_probability"
        ]

        real_probability = result[
            "real_probability"
        ]


        # Store probabilities
        fake_probabilities.append(
            fake_probability
        )

        real_probabilities.append(
            real_probability
        )


        # Count predictions
        if result["label"] == "Fake":

            fake_count += 1

        else:

            real_count += 1


        time_seconds = (
            frame_number / fps
            if fps > 0
            else 0
        )


        print(
            f"Frame {frame_number:5d} | "
            f"Time {time_seconds:6.2f}s | "
            f"Fake {fake_probability * 100:6.2f}% | "
            f"Real {real_probability * 100:6.2f}% | "
            f"{result['label']}"
        )


        # Save result
        csv_writer.writerow([
            frame_number,
            round(time_seconds, 2),
            round(
                fake_probability * 100,
                2
            ),
            round(
                real_probability * 100,
                2
            ),
            result["label"]
        ])


        frame_number += 1


    # -----------------------------------------------------
    # Release resources
    # -----------------------------------------------------

    cap.release()

    csv_file.close()


    print()
    print(
        f"CSV results saved to:"
    )

    print(csv_path)


    # -----------------------------------------------------
    # Calculate final result
    # -----------------------------------------------------

    if frames_checked == 0:

        print(
            "No frames were sampled."
        )

        return


    average_fake = np.mean(
        fake_probabilities
    )

    average_real = np.mean(
        real_probabilities
    )


    # Final prediction
    if average_fake > average_real:

        final_label = "DEEPFAKE"

        final_confidence = average_fake

    else:

        final_label = "REAL"

        final_confidence = average_real


    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("VIDEO DETECTION SUMMARY")
    print("=" * 60)

    print(
        f"Total video frames : {total_frames}"
    )

    print(
        f"Frames checked     : {frames_checked}"
    )

    print(
        f"Fake predictions   : {fake_count}"
    )

    print(
        f"Real predictions   : {real_count}"
    )

    print()

    print(
        f"Average Fake Probability : "
        f"{average_fake * 100:.2f}%"
    )

    print(
        f"Average Real Probability : "
        f"{average_real * 100:.2f}%"
    )

    print()

    print("=" * 60)
    print("FINAL VIDEO RESULT")
    print("=" * 60)

    print(
        f"Prediction : {final_label}"
    )

    print(
        f"Confidence : "
        f"{final_confidence * 100:.2f}%"
    )

    print("=" * 60)


    return {
        "prediction": final_label,
        "confidence": final_confidence,
        "fake_probability": average_fake,
        "real_probability": average_real,
        "frames_checked": frames_checked
    }


# =========================================================
# 8. Command-line execution
# =========================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")
        print(
            "python src/predict_video.py <video_path>"
        )

        raise SystemExit


    input_path = Path(
        sys.argv[1]
    )


    # If only filename is supplied,
    # search inside test_videos
    if not input_path.is_absolute():

        input_path = (
            PROJECT_DIR
            / "test_videos"
            / input_path
        )


    predict_video(
        input_path
    )