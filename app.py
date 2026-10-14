from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import gradio as gr

from PIL import Image
from torchvision import models, transforms


# =========================================================
# 1. PROJECT PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "deepfake_efficientnet_b0_final.pth"
)


# =========================================================
# 2. DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# =========================================================
# 3. LOAD MODEL
# =========================================================

print("Loading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=None
)

# Same classifier architecture used during training
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(
        in_features=1280,
        out_features=2
    )
)


# Check model file
if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )


# Load trained weights
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# =========================================================
# 4. IMAGE PREPROCESSING
# =========================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =========================================================
# 5. IMAGE PREDICTION
# =========================================================

def predict_image(image):

    if image is None:

        return (
            "No image uploaded.",
            0.0,
            0.0
        )


    # Convert input to PIL
    if not isinstance(image, Image.Image):

        image = Image.fromarray(image)


    # Convert to RGB
    image = image.convert("RGB")


    # Preprocess
    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)


    # Prediction
    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]


    # Class mapping
    # 0 = FAKE
    # 1 = REAL

    fake_probability = (
        probabilities[0].item()
    )

    real_probability = (
        probabilities[1].item()
    )


    predicted_class = torch.argmax(
        probabilities
    ).item()


    if predicted_class == 0:

        prediction = "FAKE"

        confidence = fake_probability

    else:

        prediction = "REAL"

        confidence = real_probability


    result_text = (
        f"Prediction: {prediction}\n"
        f"Confidence: {confidence * 100:.2f}%"
    )


    return (
        result_text,
        fake_probability * 100,
        real_probability * 100
    )


# =========================================================
# 6. VIDEO PREDICTION
# =========================================================

def predict_video(video_path):

    if video_path is None:

        return (
            "No video uploaded.",
            0.0,
            0.0
        )


    video_path = Path(
        video_path
    )


    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        return (
            "Could not open video.",
            0.0,
            0.0
        )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if fps <= 0:

        fps = 25


    # Analyze approximately
    # one frame per second

    frame_interval = max(
        1,
        int(fps)
    )


    frame_number = 0

    fake_probabilities = []

    real_probabilities = []


    while True:

        ret, frame = cap.read()


        if not ret:

            break


        # Sample frames
        if frame_number % frame_interval != 0:

            frame_number += 1

            continue


        # OpenCV BGR -> RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        image = Image.fromarray(
            frame_rgb
        )


        # Preprocess
        image_tensor = transform(
            image
        )

        image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(
            DEVICE
        )


        # Prediction
        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]


        fake_probabilities.append(
            probabilities[0].item()
        )

        real_probabilities.append(
            probabilities[1].item()
        )


        frame_number += 1


    cap.release()


    # No frames
    if len(fake_probabilities) == 0:

        return (
            "No frames could be processed.",
            0.0,
            0.0
        )


    # Average probabilities
    average_fake = np.mean(
        fake_probabilities
    )

    average_real = np.mean(
        real_probabilities
    )


    # Final prediction
    if average_fake > average_real:

        prediction = "DEEPFAKE"

        confidence = average_fake

    else:

        prediction = "REAL"

        confidence = average_real


    result_text = (
        f"Prediction: {prediction}\n"
        f"Confidence: {confidence * 100:.2f}%\n"
        f"Frames analyzed: "
        f"{len(fake_probabilities)}"
    )


    return (
        result_text,
        average_fake * 100,
        average_real * 100
    )


# =========================================================
# 7. GRADIO APPLICATION
# =========================================================

with gr.Blocks(
    title="Deepfake Detection System"
) as demo:


    # =====================================================
    # HEADER
    # =====================================================

    gr.Markdown(
        """
        # 🔍 Deepfake Detection System

        Detect whether an image or video is likely to be
        **Real** or **AI-generated / Deepfake**.

        ### Model Information

        **Architecture:** EfficientNet-B0  
        **Training:** Dataset 1 + Dataset 2  
        **Test Accuracy:** 89.88%  
        **ROC-AUC:** 96.98%  
        **Framework:** PyTorch
        """
    )


    # =====================================================
    # IMAGE DETECTION TAB
    # =====================================================

    with gr.Tab("🖼️ Image Detection"):


        gr.Markdown(
            """
            Upload an image to detect whether it is
            **REAL** or **FAKE**.
            """
        )


        with gr.Row():


            image_input = gr.Image(
                type="pil",
                label="Upload Image"
            )


            image_output = gr.Textbox(
                label="Prediction",
                lines=3
            )


        image_button = gr.Button(
            "🔍 Analyze Image",
            variant="primary"
        )


        with gr.Row():


            image_fake = gr.Number(
                label="Fake Probability (%)"
            )


            image_real = gr.Number(
                label="Real Probability (%)"
            )


        image_button.click(
            fn=predict_image,

            inputs=image_input,

            outputs=[
                image_output,
                image_fake,
                image_real
            ]
        )


    # =====================================================
    # VIDEO DETECTION TAB
    # =====================================================

    with gr.Tab("🎥 Video Detection"):


        gr.Markdown(
            """
            Upload a video for frame-based
            deepfake detection.
            """
        )


        video_input = gr.Video(
            label="Upload Video"
        )


        video_button = gr.Button(
            "🔍 Analyze Video",
            variant="primary"
        )


        video_output = gr.Textbox(
            label="Prediction",
            lines=4
        )


        with gr.Row():


            video_fake = gr.Number(
                label="Average Fake Probability (%)"
            )


            video_real = gr.Number(
                label="Average Real Probability (%)"
            )


        video_button.click(
            fn=predict_video,

            inputs=video_input,

            outputs=[
                video_output,
                video_fake,
                video_real
            ]
        )


    # =====================================================
    # PROJECT INFORMATION
    # =====================================================

    
    # =====================================================
    # PROJECT INFORMATION
    # =====================================================

    gr.Markdown(
        """
        ---

        ## 📊 Model Performance

        | Metric | Result |
        |---|---:|
        | Test Images | 1,433 |
        | Correct Predictions | 1,288 |
        | Test Accuracy | **89.88%** |
        | ROC-AUC | **96.98%** |
        | FAKE Precision | **93.58%** |
        | FAKE Recall | **84.74%** |
        | FAKE F1-Score | **88.94%** |
        | REAL Precision | **87.04%** |
        | REAL Recall | **94.63%** |
        | REAL F1-Score | **90.68%** |

        ### ⚠️ Disclaimer

        This system is a machine-learning research project.
        Predictions should not be treated as definitive proof
        that an image or video has been manipulated.

        **Video detection uses frame sampling and
        probability averaging.**
        """
    )

if __name__ == "__main__":

    demo.launch()