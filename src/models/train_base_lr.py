"""
Trains the TF-IDF vectorizer + Logistic Regression base model.

This logic originally only existed inside notebooks/meta_sentiment_analysis.ipynb
(cells 19-21) -- extracted here as a standalone, re-runnable script so it
can be executed during the Docker build, the same way src/models/meta_model.py
already is. This avoids needing to commit tfidf.pkl/lr_model.pkl as binary
files to git (which HuggingFace Spaces rejects over a plain git push).

Trains on the SST-2 training split with an 80/20 train/val split,
matching the original notebook's parameters exactly:
  - TF-IDF: max_features=5000, ngram_range=(1, 2)
  - LogisticRegression: max_iter=1000, default class_weight (no balancing
    here -- this is a base model feeding into the meta-model, not the
    meta-model itself, so it doesn't need the same calibration treatment)
"""

import joblib

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import TFIDF_PATH, LR_MODEL_PATH


print("Loading SST-2 dataset...")

dataset = load_dataset("sst2")

texts = [x["sentence"] for x in dataset["train"]]
labels = [x["label"] for x in dataset["train"]]

X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

print(f"Training on {len(X_train)} samples...")


# ---------------- TF-IDF Vectorizer ----------------

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)


# ---------------- Logistic Regression ----------------

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_tfidf, y_train)

val_accuracy = lr.score(X_val_tfidf, y_val)

print(f"Validation accuracy: {val_accuracy:.4f}")


# ---------------- Save ----------------

joblib.dump(tfidf, TFIDF_PATH)

joblib.dump(lr, LR_MODEL_PATH)

print(f"Saved TF-IDF vectorizer to {TFIDF_PATH}")

print(f"Saved Logistic Regression model to {LR_MODEL_PATH}")
