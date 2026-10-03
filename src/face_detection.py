import cv2
import numpy as np
from PIL import Image


def detect_face(image_path):
    """
    Detect the largest face in an image
    and return the cropped face.
    """

    # Read image using PIL
    pil_image = Image.open(image_path).convert("RGB")

    # Convert PIL image to NumPy array
    image = np.array(pil_image)

    # Convert RGB to OpenCV BGR format
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Load OpenCV Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # No face found
    if len(faces) == 0:
        return None

    # Select largest face
    largest_face = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    x, y, w, h = largest_face

    # Crop face
    face = image[
        y:y + h,
        x:x + w
    ]

    return face


if __name__ == "__main__":

    from pathlib import Path

    # Get project root
    project_dir = Path(__file__).resolve().parent.parent

    # Test image
    image_path = project_dir / "test_images" / "test2.jpg"

    print("Image path:", image_path)
    print("Image exists:", image_path.exists())

    face = detect_face(image_path)

    if face is None:

        print("No face detected.")

    else:

        print("Face detected!")
        print("Face shape:", face.shape)

        cv2.imshow(
            "Detected Face",
            face
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()