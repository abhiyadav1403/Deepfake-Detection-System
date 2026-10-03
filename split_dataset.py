# import os
# import shutil
# import random

# # Dataset paths
# BASE_DIR = "dataset"

# # Source folders
# REAL_SRC = os.path.join(BASE_DIR, "train", "real")
# FAKE_SRC = os.path.join(BASE_DIR, "train", "fake")

# # Split ratios
# TRAIN_RATIO = 0.70
# VAL_RATIO = 0.15
# TEST_RATIO = 0.15

# # Reproducibility
# random.seed(42)


# def split_images(source_folder, class_name):
#     images = [
#         file for file in os.listdir(source_folder)
#         if file.lower().endswith((".jpg", ".jpeg", ".png"))
#     ]

#     random.shuffle(images)

#     total = len(images)

#     train_end = int(total * TRAIN_RATIO)
#     val_end = train_end + int(total * VAL_RATIO)

#     train_images = images[:train_end]
#     val_images = images[train_end:val_end]
#     test_images = images[val_end:]

#     splits = {
#         "train": train_images,
#         "validation": val_images,
#         "test": test_images
#     }

#     for split_name, split_images_list in splits.items():

#         destination = os.path.join(
#             BASE_DIR,
#             split_name,
#             class_name
#         )

#         os.makedirs(destination, exist_ok=True)

#         for image in split_images_list:
#             source = os.path.join(source_folder, image)
#             target = os.path.join(destination, image)

#             # Don't copy a file onto itself
#             if os.path.abspath(source) != os.path.abspath(target):
#                 shutil.copy2(source, target)

#     print(f"\n{class_name.upper()} dataset:")
#     print(f"Total      : {total}")
#     print(f"Training   : {len(train_images)}")
#     print(f"Validation : {len(val_images)}")
#     print(f"Testing    : {len(test_images)}")


# # Process both classes
# split_images(REAL_SRC, "real")
# split_images(FAKE_SRC, "fake")

# print("\nDataset splitting completed successfully!")


import os
import shutil
import random

BASE_DIR = "dataset"

# Temporary original folders
REAL_SRC = os.path.join(BASE_DIR, "train", "real")
FAKE_SRC = os.path.join(BASE_DIR, "train", "fake")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

random.seed(42)


def split_class(source_folder, class_name):
    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    print(f"\n{class_name.upper()}")
    print(f"Total      : {total}")
    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Test       : {len(test_images)}")


# IMPORTANT:
# We will first create a temporary backup of the original images.