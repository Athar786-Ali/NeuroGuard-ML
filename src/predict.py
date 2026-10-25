import joblib
import pandas as pd

from src.models.vader_model import get_vader_prediction
from src.models.lr_model import get_lr_prediction
from src.models.bert_model import get_bert_prediction
from src.config import META_MODEL_PATH, META_THRESHOLD_PATH
from src.features.intensifiers import has_negative_intensifier


# Load trained meta-model
meta_model = joblib.load(META_MODEL_PATH)

# Load the decision threshold tuned on the validation set during training
# (see src/models/meta_model.py). Falls back to 0.5 only if a threshold
# file isn't present, e.g. on an older artifact set.
try:
    FAILURE_THRESHOLD = joblib.load(META_THRESHOLD_PATH)
except FileNotFoundError:
    FAILURE_THRESHOLD = 0.5


def predict_failure(text):


    vader_pred, vader_score = get_vader_prediction(text)

    lr_pred, lr_confidence = get_lr_prediction(text)

    bert_pred, bert_confidence, bert_entropy = (
        get_bert_prediction(text)
    )


    vader_lr_disagreement = int(
        vader_pred != lr_pred
    )

    lr_bert_disagreement = int(
        lr_pred != bert_pred
    )

    vader_bert_disagreement = int(
        vader_pred != bert_pred
    )

    negative_intensifier = has_negative_intensifier(text)


    features = pd.DataFrame([{
        "vader_pred": vader_pred,
        "vader_score": vader_score,
        "lr_pred": lr_pred,
        "lr_confidence": lr_confidence,
        "bert_pred": bert_pred,
        "bert_confidence": bert_confidence,
        "bert_entropy": bert_entropy,
        "vader_lr_disagreement": vader_lr_disagreement,
        "lr_bert_disagreement": lr_bert_disagreement,
        "vader_bert_disagreement": vader_bert_disagreement,
        "has_negative_intensifier": negative_intensifier
    }])


    failure_probability = meta_model.predict_proba(
        features
    )[0][1]


    warning = (
        "TRANSFORMER MAY FAIL"
        if failure_probability >= FAILURE_THRESHOLD
        else "Prediction appears reliable"
    )

    return {

    "text": text,

    "vader_prediction": int(vader_pred),

    "lr_prediction": int(lr_pred),

    "bert_prediction": int(bert_pred),

    "bert_label": "positive" if bert_pred == 1 else "negative",

    "vader_label": "positive" if vader_pred == 1 else "negative",

    "lr_label": "positive" if lr_pred == 1 else "negative",

    "bert_confidence": round(
        float(bert_confidence),
        4
    ),

    "lr_confidence": round(
        float(lr_confidence),
        4
    ),

    "vader_score": round(
        float(abs(vader_score)),
        4
    ),

    "bert_entropy": round(
        float(bert_entropy),
        4
    ),

    "failure_probability": round(
        float(failure_probability),
        4
    ),

    "failure_threshold": round(
        float(FAILURE_THRESHOLD),
        4
    ),

    "is_failure_risk": bool(failure_probability >= FAILURE_THRESHOLD),

    "vader_agrees": bool(vader_pred == bert_pred),

    "lr_agrees": bool(lr_pred == bert_pred),

    "has_negative_intensifier": bool(negative_intensifier),

    "warning": warning

}


if __name__ == "__main__":

    while True:

        text = input("\nEnter text: ")

        if text.lower() == "exit":
            break

        result = predict_failure(text)

        print("\nRESULT:\n")

        for key, value in result.items():

            print(f"{key}: {value}")