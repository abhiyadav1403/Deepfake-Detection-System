from pathlib import Path

import csv
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

VIDEO_PATH = PROJECT_DIR / "test_videos" / "test1.mp4"

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

CSV_PATH = RESULTS_DIR / "test1_video_results.csv"


# =========================================================
# 2. Device
# =========================================================

device = torch.device("cpu")


# =========================================================
# 3. Load EfficientNet-B0
# =========================================================

print("Loading EfficientNet-B0 model...")

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

print("Model loaded successfully.")


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
# 5. Face detector
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# =========================================================
# 6. Detect largest face
# =========================================================

def detect_face(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(100, 100)
)

    if len(faces) == 0:
        return None

    largest_face = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    x, y, w, h = largest_face

    face = frame[y:y+h, x:x+w]

    return face


# =========================================================
# 7. Predict a face
# =========================================================

def predict_face(face):

    # OpenCV BGR -> RGB
    face_rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # NumPy -> PIL
    face_pil = Image.fromarray(face_rgb)

    # Preprocessing
    image_tensor = transform(face_pil)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    # Model prediction
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

    if predicted_class == 0:
        label = "Fake"
    else:
        label = "Real"

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
# 8. Open video
# =========================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():

    print("ERROR: Could not open video.")

    raise SystemExit


fps = cap.get(cv2.CAP_PROP_FPS)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

duration = (
    total_frames / fps
    if fps > 0
    else 0
)


# =========================================================
# 9. Display video information
# =========================================================

print()
print("=" * 60)
print("DEEPFAKE VIDEO DETECTION")
print("=" * 60)

print(f"Video        : {VIDEO_PATH}")
print(f"Exists       : {VIDEO_PATH.exists()}")
print(f"FPS          : {fps:.2f}")
print(f"Total Frames : {total_frames}")
print(f"Duration     : {duration:.2f} seconds")

print("=" * 60)


# =========================================================
# 10. Frame sampling
# =========================================================

# One frame approximately every 1 second
frame_interval = max(
    1,
    int(fps)
)


frame_number = 0

frames_checked = 0
faces_detected = 0

fake_probabilities = []
real_probabilities = []

fake_count = 0
real_count = 0


# =========================================================
# Create CSV file
# =========================================================

csv_file = open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "Frame",
    "Face_Width",
    "Face_Height",
    "Fake_Probability",
    "Real_Probability",
    "Prediction"
])

fake_probabilities = []
real_probabilities = []

fake_count = 0
real_count = 0


# =========================================================
# 11. Process video
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Sample approximately 1 frame per second
    if frame_number % frame_interval != 0:
        continue

    frames_checked += 1

    # Detect face
    face = detect_face(frame)

    if face is None:

        print(
            f"Frame {frame_number:4d} "
            f"-> No face detected"
        )

        continue


    faces_detected += 1

    # Predict face
    result = predict_face(face)

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


    print(
        f"Frame {frame_number:4d} "
        f"-> {result['label']:4s} | "
        f"Fake: {fake_probability * 100:6.2f}% | "
        f"Real: {real_probability * 100:6.2f}% | "
        f"Face: {face.shape[1]}x{face.shape[0]}"
    )
    
    csv_writer.writerow([
    frame_number,
    face.shape[1],
    face.shape[0],
    round(fake_probability * 100, 2),
    round(real_probability * 100, 2),
    result["label"]
])


# =========================================================
# 12. Release video
# =========================================================

cap.release()

csv_file.close()

print()
print(f"CSV results saved to:")
print(CSV_PATH)


# =========================================================
# 13. Calculate final result
# =========================================================

print()
print("=" * 60)
print("VIDEO DETECTION SUMMARY")
print("=" * 60)

print(f"Total video frames : {total_frames}")
print(f"Frames checked     : {frames_checked}")
print(f"Faces detected     : {faces_detected}")


if faces_detected == 0:

    print()
    print("FINAL RESULT: NO FACE DETECTED")
    print("Cannot classify this video.")

else:

    # Average probability across detected faces
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


    print(f"Fake frame predictions : {fake_count}")
    print(f"Real frame predictions : {real_count}")

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