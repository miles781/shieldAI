"""
config.py

Central place for file paths and settings used across the project.
Keeping these in one file means if a path changes, we only edit it here
instead of hunting through every script.
"""

import os

# --- Folder paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "clean_comments.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

# --- Dataset columns ---
TEXT_COLUMN = "comment_text"

# The 6 original label columns in the Jigsaw dataset
LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Phase A: the single column we use for the core Safe vs Flagged classifier
PRIMARY_LABEL = "toxic"

# --- Moderation decision thresholds (Phase A) ---
# These convert a model probability into a moderation action.
SAFE_THRESHOLD = 0.3      # below this -> Safe
HARMFUL_THRESHOLD = 0.7   # above this -> Harmful, between the two -> Flag for review

# --- Train/test split ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
