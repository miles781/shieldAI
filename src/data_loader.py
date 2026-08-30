"""
data_loader.py

Responsible for one job only: loading the raw dataset from disk and
giving us a quick sanity check on what's inside it.

Keeping "loading" separate from "cleaning" and "training" makes the
codebase easier to read - each file answers one question.
"""

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_PATH, TEXT_COLUMN, LABEL_COLUMNS


def load_raw_data():
    """
    Loads the raw Jigsaw dataset from data/raw/train.csv.

    Returns:
        pandas.DataFrame with a comment_text column and 6 label columns.
    """
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Could not find dataset at {RAW_DATA_PATH}.\n"
            "Download 'train.csv' from the Jigsaw Toxic Comment Classification "
            "Challenge on Kaggle and place it in data/raw/train.csv"
        )

    df = pd.read_csv(RAW_DATA_PATH)
    return df


def inspect_data(df):
    """
    Prints a quick summary of the dataset so we understand what we're
    working with before doing anything else to it.
    """
    print("Shape (rows, columns):", df.shape)
    print("\nColumn names:", list(df.columns))
    print("\nFirst 3 rows:")
    print(df[[TEXT_COLUMN] + LABEL_COLUMNS].head(3))

    print("\nClass balance (how many 1s in each label column):")
    print(df[LABEL_COLUMNS].sum())

    print("\nPercentage of comments that are 'safe' (all labels = 0):")
    safe_count = (df[LABEL_COLUMNS].sum(axis=1) == 0).sum()
    print(f"{safe_count} out of {len(df)} ({safe_count / len(df) * 100:.1f}%)")


if __name__ == "__main__":
    data = load_raw_data()
    inspect_data(data)
