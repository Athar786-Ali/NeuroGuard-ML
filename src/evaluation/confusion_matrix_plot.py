import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Load dataset
df = pd.read_csv(
    "C:\\Users\\chiya\\Documents\\meta-model-failure-prediction\\src\\data\\processed\\base_dataset.csv"
)


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


X = df[feature_columns]


# Target
y = df["bert_failed"]


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Load trained model
meta_model = joblib.load(
    "artifacts/meta_model.pkl"
)


# Predictions
y_pred = meta_model.predict(X_test)


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)


# Plot
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Meta-Model Confusion Matrix"
)

plt.show()