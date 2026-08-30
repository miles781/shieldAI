"""
features.py

Responsible for one job: converting clean text into numeric TF-IDF
features that a machine learning model can actually train on.

We use scikit-learn's TfidfVectorizer, which handles the tokenizing,
counting, and TF-IDF math for us - we just configure it sensibly.
"""

import sys
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VECTORIZER_PATH, MODEL_DIR


def build_vectorizer(max_features=8000):
    """
    Creates a TF-IDF vectorizer.

    max_features=8000 (increased from 5000) means: keep the top 8000
    most informative words across the dataset. This was increased
    after testing showed relatively rare-but-real insult vocabulary
    (e.g. "demon") wasn't appearing in the vocabulary at all with a
    smaller cap and a smaller training sample, so the model had zero
    signal for those words. Combined with a larger training sample
    (60,000 rows instead of 30,000, see run_preprocessing.py), this
    gives the model more chances to see and learn weights for less
    common toxic vocabulary, while still keeping training fast.

    stop_words="english" removes common English function words (e.g.
    "you", "are", "the", "is"). This was added after testing revealed
    the model had learned a spurious correlation between the pronoun
    "you" and toxicity, because insults in the training data are
    disproportionately phrased as direct address ("you are ___"). Bag-
    of-words models have no concept of word order, so without this fix
    the model treated "you are good" and "you are stupid" as similar
    just because both contain "you" and "are". Removing these function
    words forces the model to rely on the actual sentiment-bearing
    words instead - and testing confirmed this improved accuracy,
    precision, and F1 simultaneously, not just fixed the specific bug.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),   # single words only, not word pairs - keeps it simple
        min_df=2,              # ignore words that appear in fewer than 2 comments (likely typos/noise)
        stop_words="english"  # remove common function words (see docstring above)
    )


def fit_transform_text(texts, vectorizer=None, max_features=8000):
    """
    Fits a TF-IDF vectorizer on the given texts and transforms them
    into a numeric feature matrix.

    Returns:
        (feature_matrix, fitted_vectorizer)
    """
    if vectorizer is None:
        vectorizer = build_vectorizer(max_features=max_features)

    feature_matrix = vectorizer.fit_transform(texts)
    return feature_matrix, vectorizer


def save_vectorizer(vectorizer):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved TF-IDF vectorizer to {VECTORIZER_PATH}")


def load_vectorizer():
    return joblib.load(VECTORIZER_PATH)


if __name__ == "__main__":
    # Small demo using the worked example from the explanation above,
    # so you can see real TF-IDF numbers on tiny, understandable input.
    demo_texts = [
        "you are stupid",
        "have a nice day",
        "you are so stupid and rude",
    ]

    matrix, vec = fit_transform_text(demo_texts, max_features=20)
    feature_names = vec.get_feature_names_out()

    print("Vocabulary (words TF-IDF is scoring):", list(feature_names))
    print()
    for i, text in enumerate(demo_texts):
        print(f"Comment: {text!r}")
        row = matrix[i].toarray()[0]
        # Only print non-zero scores for readability
        scores = {feature_names[j]: round(row[j], 3) for j in range(len(row)) if row[j] > 0}
        print("TF-IDF scores:", scores)
        print("-" * 60)
