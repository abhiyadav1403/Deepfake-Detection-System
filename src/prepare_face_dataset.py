from pathlib import Path

import cv2


# =========================================================
# Project paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_DIR / "dataset"
TARGET_DIR = PROJECT_DIR / "face_dataset"


# =========================================================
# Face detector
# =========================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


# =========================================================
# Detect largest face
# =========================================================

def detect_face(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        return None

    largest_face = max(
        faces,
        key=lambda rect: rect[2] * rect[3]
    )

    x, y, w, h = largest_face

    return image[y:y+h, x:x+w]


# =========================================================
# Process one dataset split
# =========================================================

def process_split(split_name):

    source_split = SOURCE_DIR / split_name
    target_split = TARGET_DIR / split_name

    print()
    print("=" * 60)
    print(f"PROCESSING: {split_name.upper()}")
    print("=" * 60)

    total_images = 0
    successful_faces = 0
    failed_faces = 0

    for class_name in ["fake", "real"]:

        source_class = source_split / class_name
        target_class = target_split / class_name

        target_class.mkdir(
            parents=True,
            exist_ok=True
        )

        if not source_class.exists():

            print(
                f"Warning: {source_class} does not exist."
            )

            continue

        image_files = list(
            source_class.glob("*.jpg")
        )

        image_files += list(
            source_class.glob("*.jpeg")
        )

        image_files += list(
            source_class.glob("*.png")
        )

        print()
        print(
            f"{class_name.upper()} images: "
            f"{len(image_files)}"
        )

        for index, image_path in enumerate(
            image_files,
            start=1
        ):

            total_images += 1

            image = cv2.imread(
                str(image_path)
            )

            if image is None:

                failed_faces += 1

                print(
                    f"Could not read: "
                    f"{image_path.name}"
                )

                continue

            face = detect_face(image)

            if face is None:

                failed_faces += 1

                print(
                    f"No face: "
                    f"{split_name}/{class_name}/"
                    f"{image_path.name}"
                )

                continue

            # Save cropped face
            output_path = (
                target_class
                / image_path.name
            )

            cv2.imwrite(
                str(output_path),
                face
            )

            successful_faces += 1

            # Progress
            if index % 100 == 0:

                print(
                    f"Processed {index}/"
                    f"{len(image_files)}"
                )

    print()
    print("-" * 60)
    print(f"Total images     : {total_images}")
    print(f"Faces extracted  : {successful_faces}")
    print(f"Failed/no face   : {failed_faces}")
    print("-" * 60)

    return (
        total_images,
        successful_faces,
        failed_faces
    )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FACE-CROPPED DATASET CREATION")
    print("=" * 60)

    print(f"Source : {SOURCE_DIR}")
    print(f"Target : {TARGET_DIR}")

    total = 0
    success = 0
    failed = 0

    for split in [
        "train",
        "validation",
        "test"
    ]:

        result = process_split(split)

        total += result[0]
        success += result[1]
        failed += result[2]

    print()
    print("=" * 60)
    print("DATASET CREATION COMPLETED")
    print("=" * 60)

    print(f"Total images processed : {total}")
    print(f"Faces extracted        : {success}")
    print(f"Failed / no face       : {failed}")

    if total > 0:

        percentage = (
            success / total
        ) * 100

        print(
            f"Success rate           : "
            f"{percentage:.2f}%"
        )

    print("=" * 60)