
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .model_service import predict_image

import tempfile
from pathlib import Path

from .video_service import predict_video


# Create FastAPI application
app = FastAPI(
    title="Deepfake Detection API",
    description="Deepfake detection using EfficientNet-B0",
    version="1.0.0"
)


# Connect React frontend with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Deepfake Detection API is running successfully"
    }


# Image detection endpoint
@app.post("/predict/image")
async def detect_image(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    try:
        contents = await file.read()

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

        result = predict_image(image)

        return result

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:
        await file.close()



# ==========================================
# VIDEO DETECTION ENDPOINT
# ==========================================

@app.post("/predict/video")
async def detect_video(file: UploadFile = File(...)):

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    }

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported video format."
        )

    temp_path = None

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty."
            )

        # Create temporary video file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        # Process video
        result = predict_video(temp_path)

        return result

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:

        if temp_path:
            Path(temp_path).unlink(missing_ok=True)

        await file.close()
