from pathlib import Path
import hashlib

DATASET_1 = Path("dataset")
DATASET_2 = Path("dataset_2")


def get_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


# Hash Dataset 1
dataset1_hashes = {}

for file in DATASET_1.rglob("*"):
    if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        file_hash = get_hash(file)
        dataset1_hashes[file_hash] = file

print("Dataset 1 images:", len(dataset1_hashes))


# Compare Dataset 2
duplicates = []

for file in DATASET_2.rglob("*"):
    if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:

        file_hash = get_hash(file)

        if file_hash in dataset1_hashes:
            duplicates.append(
                (file, dataset1_hashes[file_hash])
            )


print("\n========== DUPLICATE CHECK ==========")
print("Duplicates found:", len(duplicates))

for dataset2_file, dataset1_file in duplicates[:20]:
    print("\nDataset 1:", dataset1_file)
    print("Dataset 2:", dataset2_file)