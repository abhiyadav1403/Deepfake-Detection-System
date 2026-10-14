from pathlib import Path


root = Path("face_dataset")

for split in ["train", "validation", "test"]:
    for class_name in ["fake", "real"]:

        folder = root / split / class_name

        count = len(list(folder.glob("*.jpg")))

        print(f"{split}/{class_name}: {count}")