
import cv2
from PIL import Image
from pathlib import Path

from .model_service import predict_image


def predict_video(video_path, sample_every_seconds=1):

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = total_frames / fps

    frame_interval = max(
        1,
        int(fps * sample_every_seconds)
    )

    frame_number = 0
    frames_checked = 0

    fake_probabilities = []
    real_probabilities = []

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if frame_number % frame_interval == 0:

                # OpenCV BGR to RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Convert frame to PIL image
                image = Image.fromarray(rgb_frame)

                # Reuse existing EfficientNet prediction
                result = predict_image(image)

                fake_probabilities.append(
                    result["fake_probability"]
                )

                real_probabilities.append(
                    result["real_probability"]
                )

                frames_checked += 1

            frame_number += 1

    finally:
        cap.release()

    if frames_checked == 0:
        raise ValueError("No video frames could be analyzed.")

    # Calculate average probabilities
    average_fake = (
        sum(fake_probabilities) / frames_checked
    )

    average_real = (
        sum(real_probabilities) / frames_checked
    )

    # Final prediction
    if average_fake > average_real:
        prediction = "FAKE"
        confidence = average_fake
    else:
        prediction = "REAL"
        confidence = average_real

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "fake_probability": round(average_fake, 2),
        "real_probability": round(average_real, 2),
        "frames_checked": frames_checked,
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2)
    }
