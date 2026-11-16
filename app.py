
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "deepfake_efficientnet_b0_final.pth"
)

IMAGE_DIR = PROJECT_DIR / "test_images"


# =====================================================
# 2. DEVICE
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

model = models.efficientnet_b0(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 2)
)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")


# =====================================================
# 4. IMAGE PREPROCESSING
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# 5. PREDICT IMAGE
# =====================================================

def predict_image(image):

    image = image.convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Original model mapping: 0 = FAKE, 1 = REAL
    fake_probability = probabilities[0].item()
    real_probability = probabilities[1].item()

    predicted_class = torch.argmax(probabilities).item()

    if predicted_class == 0:
        prediction = "FAKE"
        confidence = fake_probability
    else:
        prediction = "REAL"
        confidence = real_probability

    return (
        prediction,
        confidence * 100,
        fake_probability * 100,
        real_probability * 100
    )


# =====================================================
# 6. DISPLAY PREDICTION
# =====================================================

def display_result(image_name, image):

    prediction, confidence, fake, real = predict_image(image)

    print("\n========== DETECTION RESULT ==========")
    print("Image:", image_name)
    print("Prediction:", prediction)
    print(f"Confidence: {confidence:.2f}%")
    print(f"Fake Probability: {fake:.2f}%")
    print(f"Real Probability: {real:.2f}%")
    print("======================================")

    return prediction


# =====================================================
# 7. IMAGE DETECTION USING FILENAME
# =====================================================

def image_detection():

    print("\nImage folder:", IMAGE_DIR)

    filename = input(
        "Enter image name: "
    ).strip().strip('"')

    # Accept only the filename
    if not filename or Path(filename).name != filename:
        print("Please enter only the image filename.")
        return

    image_file = IMAGE_DIR / filename

    if not image_file.is_file():
        print("Image not found:", filename)
        print("Check the test_images folder.")
        return

    try:
        with Image.open(image_file) as img:
            image = img.convert("RGB")
            display_result(filename, image)

    except Exception as error:
        print("Error processing image:", error)


# =====================================================
# 8. LIVE WEBCAM DETECTION
# =====================================================

def live_camera_detection():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam could not be opened.")
        print("Check camera permissions and connection.")
        return

    print("\nLive camera started!")
    print("Press SPACE to capture and analyze.")
    print("Press Q to quit the camera.")

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                print("Could not read camera frame.")
                break

            cv2.imshow("Live Deepfake Detection", frame)

            key = cv2.waitKey(1) & 0xFF

            # Quit camera
            if key == ord("q"):
                break

            # Capture and analyze
            elif key == 32:

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                image = Image.fromarray(rgb_frame)

                prediction = display_result(
                    "Live Webcam Capture",
                    image
                )

                print(
                    "Note: Prediction is a model estimate, "
                    "not proof of authenticity."
                )

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")



# =====================================================
# VIDEO DETECTION
# =====================================================

VIDEO_DIR = PROJECT_DIR / "test_videos"


def video_detection():

    print("\nVideo folder:", VIDEO_DIR)

    filename = input(
        "Enter video name: "
    ).strip().strip('"')

    # Accept filename only
    if not filename or Path(filename).name != filename:
        print("Please enter only the video filename.")
        return

    video_file = VIDEO_DIR / filename

    if not video_file.is_file():
        print("Video not found:", filename)
        return

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fake_probabilities = []
    real_probabilities = []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        # Approximately one frame per second
        frame_interval = max(1, int(fps))
        frame_number = 0

        print("\nAnalyzing video... Please wait.")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_number % frame_interval == 0:

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                image = Image.fromarray(rgb_frame)

                _, _, fake, real = predict_image(image)

                fake_probabilities.append(fake / 100)
                real_probabilities.append(real / 100)

            frame_number += 1

    finally:
        cap.release()

    if not fake_probabilities:
        print("No frames could be analyzed.")
        return

    # Average frame probabilities
    average_fake = (
        sum(fake_probabilities) / len(fake_probabilities)
    )

    average_real = (
        sum(real_probabilities) / len(real_probabilities)
    )

    if average_fake > average_real:
        prediction = "DEEPFAKE"
        confidence = average_fake
    else:
        prediction = "REAL"
        confidence = average_real

    print("\n======= VIDEO DETECTION RESULT =======")
    print("Video:", filename)
    print("Prediction:", prediction)
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Fake Probability: {average_fake * 100:.2f}%")
    print(f"Real Probability: {average_real * 100:.2f}%")
    print("Frames analyzed:", len(fake_probabilities))
    print("======================================")



# =====================================================
# 9. MAIN MENU
# =====================================================



# =====================================================
# MAIN MENU
# =====================================================

if __name__ == "__main__":

    while True:

        print("\n================================")
        print("   DEEPFAKE DETECTION SYSTEM")
        print("================================")
        print("1. Detect image from folder")
        print("2. Live webcam detection")
        print("3. Detect video from folder")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            image_detection()

        elif choice == "2":
            live_camera_detection()

        elif choice == "3":
            video_detection()

        elif choice == "4":
            print("Exiting Deepfake Detection System.")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
