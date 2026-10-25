from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def get_vader_prediction(text):

    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    prediction = 1 if compound >= 0 else 0

    return prediction, compound

if __name__ == "__main__":

    text = "this movie was amazing"

    prediction, score = get_vader_prediction(text)

    print("Prediction:", prediction)
    print("Score:", score)