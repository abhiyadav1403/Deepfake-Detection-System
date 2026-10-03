from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Project paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_DIR
    / "results"
    / "test1_video_results.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "test1_probability_plot.png"
)


# =========================================================
# Load CSV
# =========================================================

df = pd.read_csv(CSV_PATH)


# =========================================================
# Create plot
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["Frame"],
    df["Fake_Probability"],
    marker="o",
    label="Fake Probability"
)

plt.plot(
    df["Frame"],
    df["Real_Probability"],
    marker="o",
    label="Real Probability"
)


# =========================================================
# Labels
# =========================================================

plt.xlabel("Video Frame")

plt.ylabel("Probability (%)")

plt.title(
    "Deepfake Detection Probability Across Video Frames"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


# =========================================================
# Save
# =========================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300
)

plt.show()

print("=" * 50)
print("Probability graph created successfully.")
print("=" * 50)

print(f"Input CSV : {CSV_PATH}")
print(f"Output    : {OUTPUT_PATH}")