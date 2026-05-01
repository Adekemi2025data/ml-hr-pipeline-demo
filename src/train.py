import csv
import os
import sys
import random
import pickle
import json
from collections import Counter


def load_data(path):
    """Load CSV data and return rows as dictionaries."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def simple_train_test_split(rows, test_ratio=0.2, seed=42):
    """Split data into train and test sets."""
    random.seed(seed)
    random.shuffle(rows)
    split_idx = int(len(rows) * (1 - test_ratio))
    return rows[:split_idx], rows[split_idx:]


def train_simple_model(train_rows):
    """
    Train a simple rule-based HR attrition model.
    This mirrors the churn example but adapted to HR fields.
    """

    # Attrition rate by OverTime
    overtime_rates = {}
    for ot in ["Yes", "No"]:
        matching = [r for r in train_rows if r["OverTime"] == ot]
        if matching:
            left = sum(1 for r in matching if r["Attrition"] == "Yes")
            overtime_rates[ot] = left / len(matching)
        else:
            overtime_rates[ot] = 0.5

    # Learn average YearsAtCompany for employees who left
    left_years = [
        int(r["YearsAtCompany"]) for r in train_rows if r["Attrition"] == "Yes"
    ]
    avg_left_years = sum(left_years) / len(left_years) if left_years else 3

    # Learn satisfaction threshold
    sat_scores = [int(r["JobSatisfaction"]) for r in train_rows]
    avg_satisfaction = sum(sat_scores) / len(sat_scores) if sat_scores else 3

    model = {
        "attrition_by_overtime": overtime_rates,
        "years_threshold": avg_left_years,
        "satisfaction_threshold": avg_satisfaction,
    }

    return model


def predict(model, row):
    """Predict attrition for a single employee row."""
    score = model["attrition_by_overtime"].get(row["OverTime"], 0.5)

    # Low tenure increases attrition
    if int(row["YearsAtCompany"]) < model["years_threshold"]:
        score += 0.1

    # Low job satisfaction increases attrition
    if int(row["JobSatisfaction"]) < model["satisfaction_threshold"]:
        score += 0.1

    return 1 if score > 0.4 else 0


def evaluate(model, test_rows):
    """Compute accuracy, precision, recall."""
    correct = 0
    true_pos = 0
    false_pos = 0
    false_neg = 0

    for row in test_rows:
        pred = predict(model, row)
        actual = 1 if row["Attrition"] == "Yes" else 0

        if pred == actual:
            correct += 1
        if pred == 1 and actual == 1:
            true_pos += 1
        if pred == 1 and actual == 0:
            false_pos += 1
        if pred == 0 and actual == 1:
            false_neg += 1

    accuracy = correct / len(test_rows)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "test_size": len(test_rows),
    }


if __name__ == "__main__":
    data_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/raw/WAFn-UseC-HR-Employee-Attrition.csv"
    )

    print(f"Loading data from {data_path}...")
    rows = load_data(data_path)
    print(f"Loaded {len(rows)} rows")

    train_rows, test_rows = simple_train_test_split(rows)
    print(f"Train: {len(train_rows)} rows, Test: {len(test_rows)} rows")

    print("Training model...")
    model = train_simple_model(train_rows)

    print("Evaluating...")
    metrics = evaluate(model, test_rows)
    print(f"Accuracy:  {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall:    {metrics['recall']}")

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/hr_attrition_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to models/hr_attrition_model.pkl")

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved to metrics/results.json")
