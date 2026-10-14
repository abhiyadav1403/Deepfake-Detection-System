from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. Project paths
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
# 2. Load CSV
# =========================================================

df = pd.read_csv(
    CSV_PATH
)


print("CSV loaded successfully.")

print()
print("Columns:")
print(df.columns.tolist())

print()
print("Number of sampled frames:")
print(len(df))


# =========================================================
# 3. Create probability plot
# =========================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    df["Time_Seconds"],
    df["Fake_Probability"],
    label="Fake Probability"
)

plt.plot(
    df["Time_Seconds"],
    df["Real_Probability"],
    label="Real Probability"
)


# =========================================================
# 4. Labels and title
# =========================================================

plt.xlabel(
    "Time (seconds)"
)

plt.ylabel(
    "Probability (%)"
)

plt.title(
    "Deepfake Video Detection - Frame Probabilities"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# =========================================================
# 5. Save plot
# =========================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300
)

plt.show()


print()
print(
    f"Plot saved to:"
)

print(
    OUTPUT_PATH
)