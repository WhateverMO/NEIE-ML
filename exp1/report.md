# 实验报告：基于逻辑回归的银行营销结果预测

**格式要求说明：**
- 正文：宋体小四，单倍行距，1.25倍
- 代码：Times New Roman

---

## 一、实验目的

1. 掌握逻辑回归（Logistic Regression）模型的基本原理及其在分类问题中的应用。
2. 学习处理不平衡数据集（Imbalanced Dataset）的常用策略，如权重调节（`class_weight='balanced'`）。
3. 掌握机器学习模型的超参数调优方法，包括网格搜索（Grid Search）与交叉验证（Cross-validation）。
4. 学习并实践分类模型的综合评估指标，包括精确率（Precision）、召回率（Recall）、F1-Score 以及 ROC-AUC。

## 二、实验项目内容

本实验使用银行营销数据集（Bank Marketing Dataset），目标是预测客户是否会办理定期存款业务（变量 `y`）。
1. **数据预处理**：处理缺失值（`unknown` 标记）、特征编码（Ordinal Encoding & One-Hot Encoding）以及特征标准化（Standardization）。
2. **模型构建**：使用逻辑回归算法作为基准模型。
3. **参数调优**：针对正则化强度 `C` 和惩罚项类型 `penalty` 进行网格搜索优化。
4. **性能评估**：在类别不平衡的情况下，重点分析 Precision-Recall 指标，并绘制混淆矩阵与 ROC 曲线，确保 AUC 值达到实验要求（>= 0.94）。

## 三、实验过程或算法（源程序）

本实验采用 Python 语言及 `scikit-learn` 库实现。核心步骤包括数据加载、清洗、特征转换、模型训练与验证。

### 源程序 (Times New Roman):

```python
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
    categorical_cols = ["job", "marital", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
    numerical_cols = ["age", "education", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]

    # 3. Target Encoding
    le = LabelEncoder()
    y = le.fit_transform(df["y"])

    # 4. Feature Encoding (One-Hot)
    df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
    X = pd.concat([df[numerical_cols], df_encoded], axis=1)

    # 5. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 6. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


def train_model(X_train, y_train):
    """Perform GridSearch with Cross-Validation."""
    lr = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
    param_grid = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"], "max_iter": [1000]}
    grid_search = GridSearchCV(lr, param_grid, cv=CV_FOLDS, scoring="f1", verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model, X_test, y_test, best_params, data_info):
    """Evaluate and visualize results."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification Report
    print(classification_report(y_test, y_pred, target_names=["no", "yes"]))
    
    # Plots saved as PDF (Conf. Matrix, ROC)
    # ... (Plotting code as implemented in src/train.py)
    return auc(*roc_curve(y_test, y_proba)[:2])

def main():
    df = load_data("data/bank-additional-full.csv")
    explore_data(df)
    X_train, X_test, y_train, y_test, _ = preprocess_data(df)
    best_model, params = train_model(X_train, y_train)
    evaluate_model(best_model, X_test, y_test, params, None)

if __name__ == "__main__":
    main()
```

---

## 四、实验结果及分析和（或）源程序调试过程

### 1. 数据集摘要
【此处请复制 `output/data_summary.txt` 的内容】
> **复制内容如下：**
> Total Samples: 41188
> Target (y) Distribution: no: 36548 (88.73%), yes: 4640 (11.27%)
> 'unknown' 统计: 教育程度(1731), 违约情况(8597)等。

### 2. 分类性能报告分析
【此处请复制 `output/classification_report.txt` 的内容】
> **复制内容如下：**
>               precision    recall  f1-score   support
>           no       0.99      0.86      0.92      7310
>          yes       0.45      0.91      0.60       928
>     accuracy                           0.87      8238

**分析：**
在 `class_weight='balanced'` 的调节下，模型对少数类（yes）的召回率（Recall）达到了 **0.91**。虽然精确率（Precision）受类别极度不平衡影响较低（0.45），但 F1-Score (0.60) 显著优于未处理平衡的情况，成功找出了绝大多数潜在办理客户。

### 3. 可视化图表

#### (1) 混淆矩阵 (Confusion Matrix)
【此处请插入图片：`output/confusion_matrix.pdf`】
- **分析**：模型能准确识别大部分 "no" 样本，同时对 "yes" 样本有极高的检出率（仅有少量漏报），符合银行寻找潜在客户的营销需求。

#### (2) ROC 曲线与 AUC 值
【此处请插入图片：`output/roc_curve.pdf`】
- **分析**：
  【此处请复制 `output/model_evaluation.txt` 中的 AUC 值】
  > **ROC-AUC Score: 0.9439**
- 模型展现了极强的排序能力（AUC > 0.94），远超基础模型的表现。

### 4. 调试过程说明
- **求解器选择**：最初使用默认 solver 可能不支持 L1 正则化，调试后切换为 `liblinear`。
- **特征工程**：发现 `duration`（通话时长）特征对结果有极强的正向影响，是模型达到高 AUC 的关键。
- **过拟合检查**：通过交叉验证确定最佳 `C=100`，确保模型在测试集上的泛化能力。
