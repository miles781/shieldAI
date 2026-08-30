"""
compare_models.py

Trains and evaluates four algorithms on the SAME train/test split and
SAME TF-IDF features, so the comparison is fair. Used to justify the
"Selected Model" decision in Chapter 3 with real numbers rather than
an assumption.

Algorithms compared:
- Logistic Regression (already our primary model)
- Multinomial Naive Bayes
- Linear SVM
- Random Forest
"""

import sys
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_DATA_PATH, PRIMARY_LABEL, TEST_SIZE, RANDOM_STATE
from src.features import fit_transform_text


def main():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df["clean_text"] = df["clean_text"].fillna("")
    X_text = df["clean_text"]
    y = df[PRIMARY_LABEL]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, vectorizer = fit_transform_text(X_train_text, max_features=8000)
    X_test = vectorizer.transform(X_test_text)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = []
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-score": round(f1_score(y_test, y_pred), 4),
            "Train time (s)": round(train_time, 2),
        })
        print(f"Trained {name} in {train_time:.2f}s")

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (same train/test split, same TF-IDF features)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join("models", "model_comparison.csv"), index=False)
    print("\nSaved comparison table to models/model_comparison.csv")


if __name__ == "__main__":
    main()
