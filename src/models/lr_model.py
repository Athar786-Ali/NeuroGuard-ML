import joblib

from src.config import TFIDF_PATH, LR_MODEL_PATH


# Load saved TF-IDF vectorizer
tfidf = joblib.load(TFIDF_PATH)


# Load trained Logistic Regression model
lr_model = joblib.load(LR_MODEL_PATH)


def get_lr_prediction(text):

    text_tfidf = tfidf.transform([text])

    prediction = lr_model.predict(text_tfidf)[0]

    probability = lr_model.predict_proba(text_tfidf)[0].max()

    return prediction, probability

if __name__ == "__main__":

    text = "this movie was amazing"

    prediction, probability = get_lr_prediction(text)

    print("Prediction:", prediction)
    print("Confidence:", probability)
    