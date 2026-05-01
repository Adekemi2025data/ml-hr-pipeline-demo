"""
Evaluation script for HR Attrition Prediction Model.

Loads:
- Config YAML
- Processed dataset
- Trained sklearn model

Computes:
- Accuracy
- Precision
- Recall
- F1 Score

Outputs:
- metrics JSON file
- printed summary
"""

import json
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_config(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(processed_path: str, target_col: str):
    df = pd.read_csv(processed_path)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def load_model(model_path: str):
    return joblib.load(model_path)


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
    }

    return metrics


def save_metrics(metrics: dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)


def print_summary(metrics: dict):
    print("\n=== HR Attrition Model Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("====================================\n")


def main():
    cfg = load_config()

    processed_path = cfg["processed"]["processed_path"]
    target_col = cfg["source"]["target_column"]
    model_path = cfg["artifacts"]["model_path"]
    metrics_path = cfg["artifacts"]["metrics_path"]

    X_test, y_test = load_data(processed_path, target_col)
    model = load_model(model_path)

    metrics = evaluate_model(model, X_test, y_test)

    save_metrics(metrics, metrics_path)
    print_summary(metrics)


if __name__ == "__main__":
    main()
