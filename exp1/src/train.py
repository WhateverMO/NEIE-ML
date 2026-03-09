"""
Bank Marketing Prediction - Logistic Regression
Author: OpenCode Agent
Date: 2026-03-09
"""

import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
)

# ========== Configuration Constants ==========
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(file_path: str) -> pd.DataFrame:
    """Load the dataset from CSV (semicolon delimited)."""
    df = pd.read_csv(file_path, sep=";")
    return df


def explore_data(df: pd.DataFrame) -> dict:
    """Explore the dataset and return a summary dictionary."""
    summary = {
        "total_samples": len(df),
        "target_distribution": df["y"].value_counts().to_dict(),
        "columns": df.columns.tolist(),
        "missing_values_unknown": (df == "unknown").sum().to_dict(),
    }

    with open(OUTPUT_DIR / "data_summary.txt", "w") as f:
        f.write("=== Dataset Exploration Summary ===\n")
        f.write(f"Total Samples: {summary['total_samples']}\n\n")
        f.write("Target (y) Distribution:\n")
        for k, v in summary["target_distribution"].items():
            percentage = (v / summary["total_samples"]) * 100
            f.write(f"  {k}: {v} ({percentage:.2f}%)\n")
        f.write("\n'unknown' Values per Column:\n")
        for col, count in summary["missing_values_unknown"].items():
            if count > 0:
                f.write(f"  {col}: {count}\n")
    return summary


def preprocess_data(df: pd.DataFrame):
    """
    Handle encoding, feature scaling, and data splitting.
    - education: Manual ordinal mapping.
    - other categorical: One-Hot Encoding.
    - numerical: StandardScaler.
    - target: LabelEncoder (no=0, yes=1).
    """
    # 1. Manual ordinal mapping for 'education'
    education_map = {
        "unknown": 0,
        "illiterate": 1,
        "basic.4y": 2,
        "basic.6y": 3,
        "basic.9y": 4,
        "high.school": 5,
        "professional.course": 6,
        "university.degree": 7,
    }
    df["education"] = df["education"].map(education_map)

    # 2. Separate categorical and numerical features
    categorical_cols = [
        "job",
        "marital",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "poutcome",
    ]
    numerical_cols = [
        "age",
        "education",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]

    # 3. Target Encoding
    le = LabelEncoder()
    y = le.fit_transform(df["y"])

    # 4. Feature Encoding (One-Hot for non-ordinal categoricals)
    df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
    X = pd.concat([df[numerical_cols], df_encoded], axis=1)

    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 6. Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


def train_model(X_train, y_train):
    """Perform GridSearch with Cross-Validation for Logistic Regression."""
    lr = LogisticRegression(
        class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE
    )

    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "max_iter": [1000],
    }

    grid_search = GridSearchCV(
        lr, param_grid, cv=CV_FOLDS, scoring="f1", verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Save best parameters
    with open(OUTPUT_DIR / "best_params.txt", "w") as f:
        f.write("=== Best Parameters from GridSearchCV ===\n")
        for k, v in grid_search.best_params_.items():
            f.write(f"{k}: {v}\n")

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_test, y_test, best_params, data_info):
    """Evaluate model performance and save metrics/charts."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 1. Classification Report
    report = classification_report(y_test, y_pred, target_names=["no", "yes"])
    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write("=== Classification Report ===\n")
        f.write(report)

    # 2. Confusion Matrix PDF
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["no", "yes"],
        yticklabels=["no", "yes"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.pdf")
    plt.close()

    # 3. ROC Curve PDF & AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.pdf")
    plt.close()

    # 4. Comprehensive Evaluation Summary
    with open(OUTPUT_DIR / "model_evaluation.txt", "w") as f:
        f.write("=== Model Evaluation Summary ===\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}\n")
        f.write(f"ROC-AUC Score: {roc_auc:.4f}\n")
        f.write("\nNote: AUC requirement is >= 0.94\n")
        if roc_auc >= 0.94:
            f.write("Requirement Met: YES\n")
        else:
            f.write("Requirement Met: NO (Check duration feature/imbalance handling)\n")

    return roc_auc


def main():
    print("Step 1: Loading data...")
    data_path = "data/bank-additional-full.csv"
    df = load_data(data_path)

    print("Step 2: Exploring data...")
    data_info = explore_data(df)

    print("Step 3: Preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    print(
        f"Step 4: Training Logistic Regression with GridSearch ({CV_FOLDS}-fold CV)..."
    )
    best_model, best_params = train_model(X_train, y_train)

    print("Step 5: Evaluating model...")
    roc_auc = evaluate_model(best_model, X_test, y_test, best_params, data_info)

    print("\n--- Process Completed ---")
    print(f"Final AUC: {roc_auc:.4f}")
    print(f"Results saved in '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()
