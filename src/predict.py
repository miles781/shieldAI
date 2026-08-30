"""
predict.py

Responsible for one job: taking a brand-new piece of text and running
it through the SAME preprocessing + TF-IDF vectorizer + trained model
pipeline that was used during training.

This is what the web app (Section 7) will call every time a user
submits text.
"""

import sys
import os
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, VECTORIZER_PATH
from src.preprocessing import clean_text

_model = None
_vectorizer = None


def _load_artifacts():
    """Loads the trained model and vectorizer once, and reuses them
    on later calls instead of reloading from disk every time."""
    global _model, _vectorizer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _vectorizer is None:
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def predict_toxicity(raw_text):
    """
    Takes raw, unprocessed user text and returns the probability
    that it is toxic, using the exact same cleaning + TF-IDF steps
    as training.

    Returns:
        float: probability between 0 and 1
    """
    model, vectorizer = _load_artifacts()

    cleaned = clean_text(raw_text)
    features = vectorizer.transform([cleaned])  # reuse trained vocabulary, don't refit
    probability = model.predict_proba(features)[0][1]  # probability of class "1" (toxic)

    return float(probability)


if __name__ == "__main__":
    # Quick manual test with a few example sentences
    examples = [
        "Have a wonderful day, thank you for your help!",
        "You are a complete idiot and should shut up",
        "I disagree with your edit but let's discuss it calmly",
    ]
    for text in examples:
        prob = predict_toxicity(text)
        print(f"Text: {text!r}\nToxic probability: {prob:.3f}\n")
