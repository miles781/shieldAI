"""
build_severe_lexicon.py

WHY THIS EXISTS
----------------
TF-IDF + Logistic Regression is a bag-of-words model: it blends every
word in a document into one vector. That means a single severe word
(slur, extreme profanity) gets diluted as more neutral words surround
it - a long enough "polite" sentence can push a genuinely harmful
comment below the Harmful threshold even though the severe word is
still sitting right there in the text. This is not a bug in the
model or the preprocessing - it's an architectural limitation of
linear bag-of-words classifiers (confirmed experimentally: retraining
with different TF-IDF normalizations/n-grams does not remove the
effect).

This script builds a small, high-precision safety-net lexicon so
moderation.py can force a "Harmful" decision when one of these terms
appears anywhere in the text, regardless of how the trained model
scores the full sentence.

METHOD
------
Rather than hand-typing a wordlist, we mine it from data we already
have. LABEL_COLUMNS in config.py includes severe_toxic, threat, and
identity_hate - categories that are qualitatively more severe than
plain "toxic" or "insult". We compute, for every word, how much more
often it appears in comments labeled severe_toxic/threat/identity_hate
than in comments with zero labels ("safe"), then keep only words that
are both frequent enough and disproportionately severe.

We deliberately exclude words that are statistically correlated with
severity in this dataset but have common legitimate/non-abusive
meanings (identity terms, anatomy terms, mild slang, homographs). The
Jigsaw dataset is well documented as having exactly this bias - see
Jigsaw's own "Unintended Bias in Toxicity Classification" follow-up
challenge - so blindly keeping every high-ratio word would make the
safety net flag ordinary identity-related discussion as Harmful. This
lexicon is meant to catch unambiguous severe language only; anything
borderline is left to the trained model.

USAGE
-----
Run this after run_preprocessing.py has produced clean_comments.csv:

    python build_severe_lexicon.py

Writes models/severe_terms.json (list of terms), which src/severe_terms.py
loads at runtime. Re-run this if the training data changes.
"""

import os
import json
from collections import Counter

import pandas as pd

from config import PROCESSED_DATA_PATH, LABEL_COLUMNS, MODEL_DIR

SEVERE_TERMS_PATH = os.path.join(MODEL_DIR, "severe_terms.json")

# A word must appear in at least this many distinct severe-labeled
# comments to be considered - filters out one-off typos/noise.
MIN_SEVERE_DOC_FREQ = 10

# A word's rate in severe comments must be at least this many times
# its rate in safe comments. High bar on purpose: this lexicon is a
# hard override, so it should only contain terms with almost no
# legitimate/ambiguous use.
MIN_RATIO = 50

STOPWORDS = set(
    "the a an is are was were be been being to of and or but if in on "
    "at for with as by this that it its i you he she they we not no "
    "do does did have has had can will would should could".split()
)

# Words that are statistically severe in THIS dataset but have common
# legitimate meanings (identity terms, anatomy, mild slang, homographs).
# Excluded so the safety net doesn't flag ordinary discussion.
AMBIGUOUS_EXCLUDE = {
    "gay", "queer", "homo", "fat", "die", "loser", "nerd", "penis",
    "twat", "dick", "dicks", "ass", "suck", "sucks", "sucking", "cum",
    "pigs", "smelly", "goddamn", "prick", "sucker", "cock", "cocks",
    "pussy", "fag", "licker", "lick",
}


def _doc_freq(texts):
    """Word -> number of DISTINCT comments containing it (not raw count)."""
    counts = Counter()
    for text in texts:
        counts.update(set(text.split()))
    return counts


def build_lexicon(df):
    df = df.copy()
    df["clean_text"] = df["clean_text"].fillna("")

    severe_mask = (df["severe_toxic"] == 1) | (df["threat"] == 1) | (df["identity_hate"] == 1)
    safe_mask = df[LABEL_COLUMNS].sum(axis=1) == 0

    severe_counts = _doc_freq(df.loc[severe_mask, "clean_text"])
    safe_counts = _doc_freq(df.loc[safe_mask, "clean_text"])
    n_severe = severe_mask.sum()
    n_safe = safe_mask.sum()

    terms = []
    for word, severe_df in severe_counts.items():
        if severe_df < MIN_SEVERE_DOC_FREQ or len(word) < 3:
            continue
        if word in STOPWORDS or word in AMBIGUOUS_EXCLUDE:
            continue

        safe_df = safe_counts.get(word, 0)
        rate_severe = severe_df / n_severe
        rate_safe = safe_df / n_safe
        ratio = rate_severe / (rate_safe + 0.0001)

        if ratio >= MIN_RATIO:
            terms.append(word)

    return sorted(terms)


def main():
    print(f"Loading {PROCESSED_DATA_PATH} ...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    terms = build_lexicon(df)
    print(f"Built lexicon with {len(terms)} severe terms.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SEVERE_TERMS_PATH, "w") as f:
        json.dump(terms, f, indent=2)
    print(f"Saved to {SEVERE_TERMS_PATH}")


if __name__ == "__main__":
    main()