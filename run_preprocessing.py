"""
run_preprocessing.py

Loads the raw dataset, cleans the text column, subsamples it to a
manageable size for fast iteration, and saves the result to
data/processed/clean_comments.csv.

Run this once before training.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_DATA_PATH, TEXT_COLUMN, LABEL_COLUMNS, RANDOM_STATE
from src.data_loader import load_raw_data
from src.preprocessing import preprocess_dataframe

SAMPLE_SIZE = 60000  # increased from 30000 for better vocabulary coverage of rarer terms (see chat)


def main():
    print("Loading raw data...")
    df = load_raw_data()
    print(f"Loaded {len(df)} rows.")

    # Subsample for speed, but keep it reproducible with a fixed random_state.
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows for faster training.")

    print("Cleaning text...")
    df = preprocess_dataframe(df, text_column=TEXT_COLUMN)

    # Keep only what we need downstream
    keep_columns = ["clean_text"] + LABEL_COLUMNS
    df = df[keep_columns]

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved cleaned data to {PROCESSED_DATA_PATH}")

    print("\nClass balance in this sample:")
    print(df[LABEL_COLUMNS].sum())


if __name__ == "__main__":
    main()
