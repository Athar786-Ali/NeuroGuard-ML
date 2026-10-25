from transformers import pipeline


print("Loading DistilBERT sentiment model (first run downloads ~260MB)...")

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1  # force CPU; avoids trying to find a GPU that isn't there
              # on most deploy hosts, which can otherwise raise or stall
)

print("DistilBERT model loaded.")


def get_bert_prediction(text):

    result = classifier(text)[0]

    label = result["label"]

    confidence = result["score"]

    prediction = 1 if label == "POSITIVE" else 0

    entropy = 1 - confidence

    return prediction, confidence, entropy


if __name__ == "__main__":

    text = "this movie was amazing"

    prediction, confidence, entropy = (
        get_bert_prediction(text)
    )

    print("Prediction:", prediction)

    print("Confidence:", confidence)

    print("Entropy:", entropy)