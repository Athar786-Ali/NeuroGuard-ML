"""
Central path configuration.

Every path in the project is derived from PROJECT_ROOT, which is computed
from this file's location on disk. This means the code works no matter
which machine, OS, or working directory it's run from -- local dev,
a deploy server, a CI runner, all the same.

Nothing outside this file should ever hardcode a path. If you add a new
artifact or data file, add its path here and import it everywhere else.
"""

import os

# Repo root = two levels up from this file (src/config.py -> src -> root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "src", "data", "processed")

TFIDF_PATH = os.path.join(ARTIFACTS_DIR, "tfidf.pkl")
LR_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lr_model.pkl")
META_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "meta_model.pkl")
META_THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, "meta_threshold.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

BASE_DATASET_PATH = os.path.join(DATA_PROCESSED_DIR, "base_dataset.csv")
