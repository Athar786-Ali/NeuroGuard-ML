"""
Trains the meta-model that predicts whether the BERT sentiment classifier
will get a given input wrong.

Why RandomForest + calibration + a tuned threshold, instead of the original
plain LogisticRegression with class_weight="balanced":

  - The original setup flagged "may fail" on ~20% of inputs when real BERT
    failures only happen ~4% of the time (precision 0.16 on the failure
    class -- 84% of warnings were false alarms).
  - class_weight="balanced" overcorrects for that 96/4 imbalance and makes
    the model trigger-happy instead of well-calibrated.
  - A calibrated classifier with a threshold tuned on a held-out validation
    set (rather than the default 0.5 cutoff) gives a controllable precision/
    recall tradeoff instead of an arbitrary one.
  - RandomForest captures the (modest) nonlinear signal in the disagreement/
    confidence features better than linear LR for this problem, based on
    head-to-head comparison (see notes in README/commit message).

This script:
  1. Splits data into train / val / test (not just train / test).
  2. Trains RandomForest wrapped in CalibratedClassifierCV (sigmoid/Platt
     scaling) so predict_proba outputs are meaningful probabilities.
  3. Picks a decision threshold on the validation set that maximizes F1
     (a balanced precision/recall tradeoff), then locks it in.
  4. Reports honest metrics on the untouched test set.
  5. Saves both the model and the chosen threshold so serving code
     (src/predict.py) doesn't hardcode 0.5.
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, precision_recall_curve

from src.config import (
    BASE_DATASET_PATH,
    META_MODEL_PATH,
    META_THRESHOLD_PATH,
    METRICS_PATH
)


# Load dataset
df = pd.read_csv(BASE_DATASET_PATH)


# Meta-features
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


X = df[feature_columns]


# Target: did the transformer's prediction not match the real label
y = df["bert_failed"]


# 3-way split: train (60%) / val (20%) / test (20%).
# val is used ONLY for threshold tuning, test is touched only at the end,
# so the reported metrics aren't optimistic.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)


# ---------------- Meta-model: calibrated RandomForest ----------------

base_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

meta_model = CalibratedClassifierCV(
    base_model,
    method="sigmoid",
    cv=5
)

meta_model.fit(X_train, y_train)


# ---------------- Threshold tuning on validation set ----------------

val_probs = meta_model.predict_proba(X_val)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_val, val_probs)

f1_scores = 2 * precision * recall / (precision + recall + 1e-12)

# precision_recall_curve returns one more point than thresholds (the last
# point is precision=1, recall=0 with no corresponding threshold), so drop it
best_idx = np.argmax(f1_scores[:-1])

best_threshold = thresholds[best_idx]

print(f"Chosen threshold (max F1 on validation set): {best_threshold:.4f}")
print(
    f"At that threshold on validation -> "
    f"precision: {precision[best_idx]:.3f}, "
    f"recall: {recall[best_idx]:.3f}, "
    f"f1: {f1_scores[best_idx]:.3f}"
)


# ---------------- Honest evaluation on held-out test set ----------------

test_probs = meta_model.predict_proba(X_test)[:, 1]

test_preds = (test_probs >= best_threshold).astype(int)

print("\nTest set classification report (threshold applied):\n")

print(classification_report(
    y_test, test_preds, target_names=["No Failure", "Failure"]
))


# ---------------- Export metrics + PR curve for the web UI ----------------
#
# The frontend needs real numbers to render the precision-recall curve and
# the headline stats -- rather than hand-typing values into a template
# (which is how the old UI ended up showing fabricated 0.87/0.93 numbers
# that didn't match the actual model), we compute everything here, on the
# untouched test set, and write it to a small JSON file that app.py reads
# at startup. Retraining the model automatically updates what the UI shows.

test_precision, test_recall, test_pr_thresholds = precision_recall_curve(
    y_test, test_probs
)

# Downsample the curve to ~50 points for a lightweight chart -- the full
# curve can have hundreds of points (one per unique probability value),
# which is overkill for a UI chart and bloats the JSON for no visual gain.
n_points = len(test_precision)
if n_points > 50:
    sample_idx = np.linspace(0, n_points - 1, 50).astype(int)
else:
    sample_idx = np.arange(n_points)

pr_curve_points = [
    {
        "precision": round(float(test_precision[i]), 4),
        "recall": round(float(test_recall[i]), 4)
    }
    for i in sample_idx
]

test_precision_at_threshold = float(
    (test_preds[y_test == 1] == 1).sum() / max(test_preds.sum(), 1)
)
test_recall_at_threshold = float(
    (test_preds[y_test == 1] == 1).sum() / max((y_test == 1).sum(), 1)
)
test_f1_at_threshold = (
    2 * test_precision_at_threshold * test_recall_at_threshold
    / max(test_precision_at_threshold + test_recall_at_threshold, 1e-12)
)

metrics_export = {
    "threshold": round(float(best_threshold), 4),
    "test_set_size": int(len(y_test)),
    "test_failure_rate": round(float(y_test.mean()), 4),
    "precision": round(test_precision_at_threshold, 4),
    "recall": round(test_recall_at_threshold, 4),
    "f1": round(test_f1_at_threshold, 4),
    "pr_curve": pr_curve_points
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics_export, f, indent=2)

print(f"\nMetrics + PR curve exported to {METRICS_PATH}")


# ---------------- Save model + threshold ----------------

joblib.dump(meta_model, META_MODEL_PATH)

joblib.dump(best_threshold, META_THRESHOLD_PATH)

print(f"\nMeta-model saved to {META_MODEL_PATH}")

print(f"Decision threshold saved to {META_THRESHOLD_PATH}")
