from src.models.vader_model import get_vader_prediction
from src.models.lr_model import get_lr_prediction
from src.models.bert_model import get_bert_prediction

from src.features.noise import add_noise
from src.features.intensifiers import has_negative_intensifier
from src.config import BASE_DATASET_PATH
from src.data.curated_hard_examples import CURATED_HARD_EXAMPLES

from datasets import load_dataset
import pandas as pd
import random


# Load SST-2 dataset
dataset = load_dataset("sst2")


# Take training split
train_data = dataset["train"]


# Convert to pandas dataframe
df = pd.DataFrame(train_data)


# Rename columns
df = df.rename(columns={
    "sentence": "text",
    "label": "true_label"
})


# Keep only needed columns
df = df[["text", "true_label"]]


# Take sample. Scaled up from the original 5,000 -- SST-2's train split
# has ~67,300 rows, so this still uses well under a third of what's
# available. At the same ~3.7% BERT failure rate as before, 18,000
# samples yields roughly 650-700 real failure examples instead of ~190,
# giving the meta-model meaningfully more signal to learn from.
df = df.sample(18000, random_state=42)


# ---------------- Add Noise (SST-2 sample only) ----------------

random.seed(42)

# Noise rate increased from the original 0.3 (30%) to 0.45 (45%) --
# more perturbed examples means more opportunities for BERT to actually
# fail, giving the meta-model more real failure signal per sample
# instead of being dominated by easy, unperturbed text.
df["text"] = df["text"].apply(
    lambda x: add_noise(x) if random.random() < 0.45 else x
)


# ---------------- Blend in curated hard examples ----------------
#
# These are added AFTER noise injection and are never themselves
# perturbed by add_noise() -- they're deliberately natural, clean
# language (negation, sarcasm-adjacent intensifiers, mixed sentiment)
# meant to plug a real gap found by inspecting the original dataset:
# almost all of its "failure" examples were synthetic negation-insertion
# artifacts (grammatically broken phrases like "the not closest thing"),
# not natural hard cases. See src/data/curated_hard_examples.py for the
# full rationale.

curated_df = pd.DataFrame(
    CURATED_HARD_EXAMPLES, columns=["text", "true_label"]
)

df = pd.concat([df, curated_df], ignore_index=True)


# ---------------- VADER ----------------

vader_outputs = df["text"].apply(get_vader_prediction)

df["vader_pred"] = vader_outputs.apply(lambda x: x[0])

df["vader_score"] = vader_outputs.apply(lambda x: x[1])


# ---------------- Logistic Regression ----------------

lr_outputs = df["text"].apply(get_lr_prediction)

df["lr_pred"] = lr_outputs.apply(lambda x: x[0])

df["lr_confidence"] = lr_outputs.apply(lambda x: x[1])


# ---------------- DistilBERT ----------------

bert_outputs = df["text"].apply(get_bert_prediction)

df["bert_pred"] = bert_outputs.apply(lambda x: x[0])

df["bert_confidence"] = bert_outputs.apply(lambda x: x[1])

df["bert_entropy"] = bert_outputs.apply(lambda x: x[2])


# ---------------- Disagreement Features ----------------

df["vader_lr_disagreement"] = (
    df["vader_pred"] != df["lr_pred"]
).astype(int)

df["lr_bert_disagreement"] = (
    df["lr_pred"] != df["bert_pred"]
).astype(int)

df["vader_bert_disagreement"] = (
    df["vader_pred"] != df["bert_pred"]
).astype(int)


# ---------------- Negative-Intensifier Feature ----------------
#
# Flags phrasing like "disgustingly good" or "stupidly entertaining" --
# see src/features/intensifiers.py for the full rationale. Computed
# here at dataset-build time so the meta-model trains on it, and must
# also be computed identically in src/predict.py at serving time, or
# the live predictions would be working off a feature the model never
# actually learned from.

df["has_negative_intensifier"] = df["text"].apply(
    has_negative_intensifier
)


# ---------------- Failure Label ----------------

df["bert_failed"] = (
    df["bert_pred"] != df["true_label"]
).astype(int)


# Save processed dataset
df.to_csv(
    BASE_DATASET_PATH,
    index=False
)


# Preview dataframe
print(df.head())

print("\nDataset saved successfully.")