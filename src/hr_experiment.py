import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import sys
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ─── Configuration ───────────────────────────────────────────────
config = {
    "model_type": "logistic_regression",   # logistic_regression, random_forest, gradient_boosting
    "test_size": 0.2,
    "random_state": 42,
    "handle_missing": "median",            # median, drop
    "scale_features": True,
    "features_to_drop": ["EmployeeNumber"],

    # Model-specific hyperparameters
    "lr_C": 1.0,
    "rf_n_estimators": 150,
    "rf_max_depth": None,
    "gb_n_estimators": 150,
    "gb_learning_rate": 0.1,
    "gb_max_depth": 3,
}


# ────────────────────────────────────────────────────────────────
# DATA LOADING + PREPROCESSING
# ────────────────────────────────────────────────────────────────
def load_and_prepare_data(config):
    path = "data/raw/WAFn-UseC-HR-Employee-Attrition.csv"
    print(f"Loading HR dataset from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Drop user-specified columns
    if config["features_to_drop"]:
        df = df.drop(columns=config["features_to_drop"], errors="ignore")
        print(f"Dropped features: {config['features_to_drop']}")

    # ───────────────────────────────────────────────
    # CLEAN ATTRITION COLUMN
    # ───────────────────────────────────────────────
    df["Attrition"] = df["Attrition"].astype(str).str.strip().str.lower()
    df["Attrition"] = df["Attrition"].replace({
        "yes": 1, "y": 1, "1": 1, "true": 1,
        "no": 0, "n": 0, "0": 0, "false": 0
    })
    df = df[df["Attrition"].isin([0, 1])]
    df["Attrition"] = df["Attrition"].astype(int)

    print("Attrition value counts after cleaning:")
    print(df["Attrition"].value_counts())

    # Debug prints
    print(f"DataFrame shape before X/y split: {df.shape}")
    print(f"Target column 'Attrition' exists: {'Attrition' in df.columns}")
    print(f"DataFrame columns: {df.columns.tolist()}")

    # ───────────────────────────────────────────────
    # DEFINE numeric_cols AND categorical_cols HERE
    # ───────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # ───────────────────────────────────────────────
    # CREATE X AND y
    # ───────────────────────────────────────────────
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    print(f"X shape after creation: {X.shape}")
    print(f"y shape after creation: {y.shape}")

    return X, y, len(df), numeric_cols, categorical_cols



    # Clean target column
    df["Attrition"] = (
        df["Attrition"]
        .astype(str)
        .str.strip()
        .str.title()
        .map({"Yes": 1, "No": 0})
    )

    # Drop rows where mapping failed
    df = df.dropna(subset=["Attrition"])
    df["Attrition"] = df["Attrition"].astype(int)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if config["handle_missing"] == "median":
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        print("Filled missing numeric values with median")
    elif config["handle_missing"] == "drop":
        before = len(df)
        df = df.dropna()
        print(f"Dropped rows with missing values: {before} → {len(df)}")

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Separate features and target
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]

    return X, y, len(df), numeric_cols, categorical_cols


# ────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ────────────────────────────────────────────────────────────────
def build_model(config):
    """Create a model based on the config."""

    if config["model_type"] == "logistic_regression":
        return LogisticRegression(
            C=config["lr_C"],
            random_state=config["random_state"],
            max_iter=1000
        )

    elif config["model_type"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["rf_n_estimators"],
            max_depth=config["rf_max_depth"],
            random_state=config["random_state"]
        )

    elif config["model_type"] == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config["gb_n_estimators"],
            learning_rate=config["gb_learning_rate"],
            max_depth=config["gb_max_depth"],
            random_state=config["random_state"]
        )

    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")


# ────────────────────────────────────────────────────────────────
# EXPERIMENT RUNNER
# ────────────────────────────────────────────────────────────────
def run_experiment(config):
    """Run a single MLflow experiment."""

    mlflow.set_experiment("hr-attrition-prediction")

    with mlflow.start_run():

        # Log config parameters
        for key, value in config.items():
            mlflow.log_param(key, value)

        # Load + prepare data
        X, y, n_rows, numeric_cols, categorical_cols = load_and_prepare_data(config)

        mlflow.log_param("n_rows", n_rows)
        mlflow.log_param("n_features", X.shape[1])

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"X columns: {X.columns.tolist()}")
        print(f"y unique values: {y.unique()}")
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=y
        )

        # Scale numeric features
        if config["scale_features"]:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Train model
        model = build_model(config)
        print(f"\nTraining {config['model_type']}...")
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Log config snapshot
        config_path = "config_snapshot.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        mlflow.log_artifact(config_path)
        os.remove(config_path)

        # Print results
        print("\n" + "=" * 50)
        print(f"Model:     {config['model_type']}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print("=" * 50)

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print("View this run in the UI: mlflow ui")

    return run_id


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_experiment(config)
