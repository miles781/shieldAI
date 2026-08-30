"""
severe_terms.py

Responsible for one job: loading the severe-term lexicon (built by
build_severe_lexicon.py) and checking whether a piece of cleaned text
contains any of those terms.

This is a hard safety net, not a classifier - see the docstring in
build_severe_lexicon.py for why it exists. It is deliberately dumb and
literal: it does not score, weight, or interpret anything. That's the
point - it exists precisely because the ML model's scoring can be
diluted by sentence length, and a plain "did this word appear at all"
check cannot be diluted.

Matching is whole-word only (via a set lookup on split() tokens), so
this will not match substrings inside unrelated words. It's meant to
run on text that has already been through preprocessing.clean_text(),
same as the TF-IDF pipeline.
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR

SEVERE_TERMS_PATH = os.path.join(MODEL_DIR, "severe_terms.json")

_severe_terms = None


def _load_severe_terms():
    global _severe_terms
    if _severe_terms is None:
        if not os.path.exists(SEVERE_TERMS_PATH):
            raise FileNotFoundError(
                f"Could not find {SEVERE_TERMS_PATH}.\n"
                "Run build_severe_lexicon.py first to generate it."
            )
        with open(SEVERE_TERMS_PATH) as f:
            _severe_terms = set(json.load(f))
    return _severe_terms


def contains_severe_term(clean_text_value):
    """
    clean_text_value should already be the output of
    preprocessing.clean_text() - lowercase, letters-only, single-spaced -
    so a plain split() gives whole words to check.

    Returns:
        bool
    """
    severe_terms = _load_severe_terms()
    words = set(clean_text_value.split())
    return bool(words & severe_terms)


if __name__ == "__main__":
    from src.preprocessing import clean_text

    examples = [
        "Have a wonderful day, thank you for your help!",
        "you are such an idiot honestly and I cannot believe you did "
        "that again after everything we discussed last week",
        "you are such a fucking idiot honestly and I cannot believe "
        "you did that again after everything we discussed last week",
    ]
    for text in examples:
        flagged = contains_severe_term(clean_text(text))
        print(f"{flagged}  {text!r}") 