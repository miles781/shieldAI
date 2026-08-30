"""
moderation.py

Responsible for one job: converting a raw toxicity probability into
an actual moderation decision (Safe / Flag for Review / Harmful),
using the thresholds defined in config.py.

This is intentionally a separate file from predict.py - the ML model
predicts a probability; this file encodes the POLICY of what to do
with that probability. Keeping them separate means we could change
the moderation policy later without touching the model at all.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SAFE_THRESHOLD, HARMFUL_THRESHOLD
from src.predict import predict_toxicity


def get_moderation_decision(probability):
    """
    Applies the three-tier moderation policy to a toxicity probability.

    Returns:
        dict with 'decision' and 'confidence'
    """
    if probability < SAFE_THRESHOLD:
        decision = "Safe"
    elif probability > HARMFUL_THRESHOLD:
        decision = "Harmful"
    else:
        decision = "Flag for Review"

    return {
        "decision": decision,
        "confidence": round(probability, 3)
    }


def moderate_text(raw_text):
    """
    Full pipeline: raw text -> probability -> moderation decision.
    This is the single function the web app (Section 7) will call.
    """
    probability = predict_toxicity(raw_text)
    result = get_moderation_decision(probability)
    result["text"] = raw_text
    return result


if __name__ == "__main__":
    examples = [
        "Have a wonderful day, thank you for your help!",
        "You are a complete idiot and should shut up",
        "I disagree with your edit but let's discuss it calmly",
        "This is somewhat annoying but whatever",
    ]
    for text in examples:
        result = moderate_text(text)
        print(f"Text: {result['text']!r}")
        print(f"  -> Decision: {result['decision']} (confidence: {result['confidence']})")
        print()
