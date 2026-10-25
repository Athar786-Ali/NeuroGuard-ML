import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from src.config import BASE_DATASET_PATH, META_MODEL_PATH


# Load dataset
df = pd.read_csv(BASE_DATASET_PATH)


# Features
feature_columns = [
    "vader_pred",
    "vader_score",
    "lr_pred",
    "lr_confidence",
    "bert_pred",
    "bert_confidence",
    "bert_entropy",
    "vader_lr_disagreement",
    "lr_bert_disagreement",
    "vader_bert_disagreement",
    "has_negative_intensifier"
]


# Target
y = df["bert_failed"]


# Split while preserving original rows
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)


# Test features
X_test = test_df[feature_columns]


# True labels
y_test = test_df["bert_failed"]


# Load model
meta_model = joblib.load(META_MODEL_PATH)


# Predictions
y_pred = meta_model.predict(X_test)


print("\nCorrectly Predicted Failures:\n")


count = 0


for i in range(len(y_test)):

    if y_test.iloc[i] == 1 and y_pred[i] == 1:

        row = test_df.iloc[i]

        print("=" * 60)

        print("TEXT:\n")

        print(row["text"])

        print("\nTRUE LABEL:", row["true_label"])

        print("BERT PREDICTION:", row["bert_pred"])

        print("BERT CONFIDENCE:",
              row["bert_confidence"])

        print("BERT ENTROPY:",
              row["bert_entropy"])

        print("META-MODEL WARNING: FAILURE DETECTED")

        print("=" * 60)

        count += 1

    if count == 5:
        break