import pandas as pd
import matplotlib.pyplot as plt

from src.config import BASE_DATASET_PATH


# Load dataset
df = pd.read_csv(BASE_DATASET_PATH)


# Correct predictions
df["correct"] = (
    df["bert_pred"] == df["true_label"]
).astype(int)


# Confidence bins
bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

df["confidence_bin"] = pd.cut(
    df["bert_confidence"],
    bins=bins
)


# Accuracy per bin
calibration = df.groupby(
    "confidence_bin"
)["correct"].mean()


# Plot
plt.figure(figsize=(8, 5))

calibration.plot(kind="bar")

plt.ylim(0, 1)

plt.ylabel("Accuracy")

plt.xlabel("Confidence Bin")

plt.title("BERT Confidence Calibration")

plt.tight_layout()

plt.show()