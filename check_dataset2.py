from pathlib import Path
from PIL import Image

DATASET = Path("dataset_2")

good = 0
bad = 0

for file in DATASET.rglob("*"):

    if file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
        continue

    try:
        with Image.open(file) as img:
            img.verify()

        good += 1

    except Exception:
        bad += 1
        print("BAD IMAGE:", file)

print("\n========== RESULT ==========")
print("Good images:", good)
print("Bad images:", bad)
print("Total:", good + bad)