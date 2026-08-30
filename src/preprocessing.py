"""
preprocessing.py

Responsible for one job: turning messy raw text into clean, normalized
text that's ready to be converted into numbers (TF-IDF, next section).

We deliberately keep this simple - no stemming/lemmatization libraries,
no external NLP toolkits. Just regex-based cleaning that a beginner can
read and explain line by line.
"""

import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clean_text(text):
    """
    Cleans a single piece of text.

    Steps (in order):
    1. Lowercase everything (normalization)
    2. Remove newlines/tabs
    3. Remove IP addresses (common in this Wikipedia-talk-page dataset,
       e.g. "192.168.1.1" left behind by signed comments)
    4. Remove URLs
    5. Remove anything that isn't a letter or basic whitespace
    6. Collapse multiple spaces into one
    7. Strip leading/trailing whitespace
    """
    text = str(text).lower()
    text = re.sub(r"[\n\t\r]", " ", text)
    text = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", text)  # IP addresses
    text = re.sub(r"http\S+|www\S+", " ", text)                      # URLs
    text = re.sub(r"[^a-z\s]", " ", text)                            # keep letters only
    text = re.sub(r"\s+", " ", text).strip()                         # collapse spaces
    return text


def preprocess_dataframe(df, text_column="comment_text"):
    """
    Applies clean_text() to every row in the given column and returns
    a new column called 'clean_text'.
    """
    df = df.copy()
    df["clean_text"] = df[text_column].apply(clean_text)
    return df


if __name__ == "__main__":
    # Quick demo on a few example sentences so you can SEE what
    # preprocessing does before running it on the full dataset.
    examples = [
        "Explanation\nWhy the edits made under my username Hardcore Metallica Fan were reverted?",
        "YOU ARE SO STUPID!!! Check http://example.com or 192.168.1.1",
        "This   has     extra     spaces and Numbers123",
    ]
    for ex in examples:
        print("BEFORE:", repr(ex))
        print("AFTER: ", repr(clean_text(ex)))
        print("-" * 60)
