import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime


OUTPUT_DIR = "output"
AUC_REQUIREMENT = 0.80


def setup_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def log_message(message, log_file=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    if log_file:
        log_file.write(formatted_msg + "\n")
        log_file.flush()


def load_data(filepath):
    log_message(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    log_message(f"Data loaded successfully. Shape: {df.shape}")
    return df


def explore_data(df):
    log_message("Exploring dataset...")

    info = {
        "total_samples": len(df),
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "target_distribution": df["Churn"].value_counts().to_dict(),
    }

    for col in df.columns:
        if df[col].dtype == "object":
            unique_vals = df[col].unique()
            if "Unknown" in unique_vals or " " in unique_vals:
                log_message(f"Column '{col}' has 'Unknown' or blank values")

    return info


def preprocess_data(df, log_file):
    log_message("Starting data preprocessing...", log_file)
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)
        log_message("Dropped customerID column", log_file)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_total = df["TotalCharges"].isnull().sum()
    if missing_total > 0:
        median_total = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_total)
        log_message(
            f"Filled {missing_total} missing TotalCharges with median: {median_total:.2f}",
            log_file,
        )

    label_encoder = LabelEncoder()
    contract_mapping = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["Contract"] = df["Contract"].map(contract_mapping)
    log_message(f"Label encoded 'Contract': {contract_mapping}", log_file)

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "Churn"]
    log_message(f"One-Hot Encoding columns: {categorical_cols}", log_file)

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    log_message(f"Data shape after encoding: {df.shape}", log_file)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    scaler = StandardScaler()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    log_message("Applied StandardScaler to numeric features", log_file)

    return X, y, scaler


def get_param_grid():
    param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", None],
        "criterion": ["gini", "entropy"],
    }
    return param_grid


def train_model(X_train, y_train, log_file):
    log_message("Initializing Decision Tree with class_weight='balanced'", log_file)

    base_model = DecisionTreeClassifier(class_weight="balanced", random_state=42)

    param_grid = get_param_grid()
    log_message(f"Parameter grid: {param_grid}", log_file)

    log_message("Starting GridSearchCV with 5-fold cross-validation...", log_file)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    log_message(
        f"GridSearchCV completed. Best Score (ROC-AUC): {grid_search.best_score_:.4f}",
        log_file,
    )
    log_message(f"Best Parameters: {grid_search.best_params_}", log_file)

    return grid_search.best_estimator_, grid_search


def evaluate_model(model, X_test, y_test, log_file):
    log_message("Evaluating model on test set...", log_file)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    log_message(f"Accuracy: {accuracy:.4f}", log_file)
    log_message(f"ROC-AUC Score: {roc_auc:.4f}", log_file)

    report = classification_report(y_test, y_pred, target_names=["No", "Yes"])
    log_message("\nClassification Report:\n" + report, log_file)

    cm = confusion_matrix(y_test, y_pred)
    log_message(f"\nConfusion Matrix:\n{cm}", log_file)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc_value = auc(fpr, tpr)

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "classification_report": report,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc_value": roc_auc_value,
    }

    return metrics, y_pred, y_pred_proba


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title("Confusion Matrix - Decision Tree", fontsize=14)
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Decision Tree", fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.close()
    print(f"ROC curve saved to {save_path}")


def save_metrics(metrics, best_params, log_file):
    data_summary_path = os.path.join(OUTPUT_DIR, "data_summary.txt")
    with open(data_summary_path, "w") as f:
        f.write("=== Dataset Exploration Summary ===\n")
        f.write(f"Total Samples: {len(y)}\n")
        f.write(f"\nTarget (y) Distribution:\n")
        f.write(f"  No: {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)\n")
        f.write(f"  Yes: {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)\n")
    print(f"Data summary saved to {data_summary_path}")

    best_params_path = os.path.join(OUTPUT_DIR, "best_params.txt")
    with open(best_params_path, "w") as f:
        f.write("=== Best Parameters from GridSearchCV ===\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
    print(f"Best parameters saved to {best_params_path}")

    classification_report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(classification_report_path, "w") as f:
        f.write("=== Classification Report ===\n")
        f.write(metrics["classification_report"])
    print(f"Classification report saved to {classification_report_path}")

    model_eval_path = os.path.join(OUTPUT_DIR, "model_evaluation.txt")
    with open(model_eval_path, "w") as f:
        f.write("=== Model Evaluation Summary ===\n")
        f.write(f"Best Params: {best_params}\n")
        f.write(f"Accuracy Score: {metrics['accuracy']:.4f}\n")
        f.write(f"ROC-AUC Score: {metrics['roc_auc']:.4f}\n")
        f.write(f"\nNote: AUC requirement is >= {AUC_REQUIREMENT}\n")
        requirement_met = "YES" if metrics["roc_auc"] >= AUC_REQUIREMENT else "NO"
        f.write(f"Requirement Met: {requirement_met}\n")
    print(f"Model evaluation saved to {model_eval_path}")

    return (
        classification_report_path,
        best_params_path,
        model_eval_path,
        data_summary_path,
    )


def main():
    global y

    setup_output_directory()

    log_file_path = os.path.join(OUTPUT_DIR, "log.txt")
    log_file = open(log_file_path, "w")

    log_message("=" * 60, log_file)
    log_message("Telecom Customer Churn Prediction - Decision Tree", log_file)
    log_message("=" * 60, log_file)

    data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = load_data(data_path)

    info = explore_data(df)

    X, y, scaler = preprocess_data(df, log_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log_message(f"Train size: {len(X_train)}, Test size: {len(X_test)}", log_file)
    log_message(
        f"Train target distribution: No={sum(y_train == 0)}, Yes={sum(y_train == 1)}",
        log_file,
    )
    log_message(
        f"Test target distribution: No={sum(y_test == 0)}, Yes={sum(y_test == 1)}",
        log_file,
    )

    model, grid_search = train_model(X_train, y_train, log_file)

    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, log_file)

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.pdf")
    plot_confusion_matrix(metrics["confusion_matrix"], cm_path)

    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.pdf")
    plot_roc_curve(metrics["fpr"], metrics["tpr"], metrics["roc_auc_value"], roc_path)

    save_metrics(metrics, grid_search.best_params_, log_file)

    log_message("=" * 60, log_file)
    log_message("Training and evaluation completed successfully!", log_file)
    log_message("=" * 60, log_file)

    log_file.close()

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
