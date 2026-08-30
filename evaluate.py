"""
evaluate.py

Loads the trained model and the held-out test set (saved by train.py),
runs predictions, and reports honest evaluation metrics.

No numbers here are hand-typed - everything is computed from the
actual model output on actual unseen test data.
"""

import sys
import os
import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_PATH, MODEL_DIR


def main():
    print("Loading trained model and test split...")
    model = joblib.load(MODEL_PATH)
    X_test, y_test = joblib.load(os.path.join(MODEL_DIR, "test_split.joblib"))

    print(f"Test set size: {X_test.shape[0]} comments")
    print("Running predictions on the test set...\n")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("=" * 50)
    print("EVALUATION RESULTS (Phase A: Safe vs Toxic)")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print()
    print("Full classification report:")
    print(classification_report(y_test, y_pred, target_names=["safe", "toxic"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(cm)

    # Save a visual confusion matrix image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["safe", "toxic"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Safe vs Toxic")
    output_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(output_path, bbox_inches="tight")
    print(f"\nSaved confusion matrix image to {output_path}")


if __name__ == "__main__":
    main()
