FROM python:3.11-slim

WORKDIR /app

# Install dependencies. requirements-dev.txt is needed here (not just
# requirements.txt) because src/models/train_base_lr.py below needs the
# `datasets` library to download SST-2 -- that's the tradeoff for not
# committing tfidf.pkl/lr_model.pkl as binary files, which HuggingFace
# Spaces rejects on a plain git push.
COPY requirements.txt requirements-dev.txt ./

RUN pip install --no-cache-dir -r requirements-dev.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the app
COPY . .

# HuggingFace Spaces runs on port 7860
ENV PORT=7860

# Pre-download the DistilBERT model during build so it's cached in the
# image and the first request doesn't hit a cold-start download timeout.
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', device=-1)"

# Train the TF-IDF + Logistic Regression base model. Downloads SST-2
# (cached by HF's datasets library) and fits a small sklearn model --
# a couple of minutes, not BERT-inference-slow.
RUN python -m src.models.train_base_lr

# Train the meta-model from the already-computed dataset CSV
# (src/data/processed/base_dataset.csv, committed to the repo as plain
# text -- no Git LFS needed). Fast: pure sklearn on tabular data, no BERT
# inference involved here, just the previously-computed features.
RUN python -m src.models.meta_model

EXPOSE 7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
