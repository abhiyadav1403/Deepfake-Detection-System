

import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

CSV_FILE = "FINAL_DATASET.csv"
OUTPUT_DIR = Path("dataset_2")

df = pd.read_csv(CSV_FILE)

print("Total images:", len(df))
print(df["label"].value_counts())


headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/154.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Referer": "https://unsplash.com/"
}


def download_image(row):

    image_id = row["image_id"]
    url = row["image_url"]

    label = str(row["label"]).lower()
    split = str(row["dataset_split"]).lower()

    if split == "val":
        split = "validation"

    if label not in ["real", "fake"]:
        return f"Skipped {image_id}: unknown label"

    save_dir = OUTPUT_DIR / split / label
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{image_id}.jpg"

    if save_path.exists():
        return f"Already exists: {image_id}"

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:

            save_path.write_bytes(response.content)

            return f"Downloaded: {image_id}"

        else:

            return f"Failed: {image_id} - HTTP {response.status_code}"

    except Exception as e:

        return f"Error: {image_id} - {e}"


# First test only 10 images
test_rows = [row for _, row in df.iterrows()]

with ThreadPoolExecutor(max_workers=5) as executor:

    results = list(
        executor.map(download_image, test_rows)
    )


for result in results:
    print(result)


print("\nTest completed!")