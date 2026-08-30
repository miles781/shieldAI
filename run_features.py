"""
run_features.py

Loads the cleaned dataset (from Section 2), converts the text into
TF-IDF numeric features, and saves the fitted vectorizer so it can be
reused later for evaluation and live predictions.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_DATA_PATH
from src.features import fit_transform_text, save_vectorizer

import pandas as pd


def main():
    print("Loading cleaned data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["clean_text"] = df["clean_text"].fillna("")  # safety net for any empty rows
    print(f"Loaded {len(df)} cleaned comments.")

    print("Building TF-IDF features...")
    feature_matrix, vectorizer = fit_transform_text(df["clean_text"], max_features=8000)

    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"(That's {feature_matrix.shape[0]} comments x {feature_matrix.shape[1]} TF-IDF word features)")

    save_vectorizer(vectorizer)


if __name__ == "__main__":
    main()
