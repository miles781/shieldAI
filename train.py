"""
train.py

Trains the Phase A model: a Logistic Regression classifier that
predicts whether a comment is toxic or safe (PRIMARY_LABEL in config.py).

Steps:
1. Load cleaned data
2. Split into train/test sets
3. Convert text to TF-IDF features (fit only on the training set!)
4. Train Logistic Regression
5. Save the trained model + vectorizer for later use
"""

import sys
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    PROCESSED_DATA_PATH, PRIMARY_LABEL, TEST_SIZE, RANDOM_STATE,
    MODEL_PATH, MODEL_DIR
)
from src.features import fit_transform_text, save_vectorizer


def main():
    print("Loading cleaned data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["clean_text"] = df["clean_text"].fillna("")

    X_text = df["clean_text"]
    y = df[PRIMARY_LABEL]

    print(f"Total comments: {len(df)}")
    print(f"Toxic comments: {y.sum()} ({y.mean() * 100:.1f}%)")

    # IMPORTANT: split BEFORE fitting TF-IDF, so the test set stays
    # truly "unseen" - the vectorizer must never learn from test data.
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train size: {len(X_train_text)}, Test size: {len(X_test_text)}")

    print("Fitting TF-IDF on training data only...")
    X_train, vectorizer = fit_transform_text(X_train_text, max_features=8000)
    X_test = vectorizer.transform(X_test_text)  # reuse same vocabulary, don't refit

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("Training complete.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_vectorizer(vectorizer)
    print(f"Saved trained model to {MODEL_PATH}")

    # Save test set too, so evaluate.py uses the exact same split (no leakage, no re-fitting)
    joblib.dump((X_test, y_test), os.path.join(MODEL_DIR, "test_split.joblib"))
    print("Saved test split for evaluation step.")


if __name__ == "__main__":
    main()
