# Bank Marketing Prediction - Logistic Regression

本实验基于逻辑回归（Logistic Regression）模型，对银行营销结果进行预测。通过对客户特征的分析，预测客户是否会办理定期存款业务。

## 实验目标
- 训练并优化逻辑回归模型。
- 处理类别不平衡问题（使用 `class_weight='balanced'`）。
- 通过网格搜索（Grid Search）和交叉验证（Cross-validation）进行超参数调优。
- 实现全面的性能评估：Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC。

## 环境要求
- Python 3.12+
- 依赖项：`pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `imbalanced-learn`
- 使用 `uv` 进行环境管理。

## 快速开始

### 1. 安装依赖
```bash
uv sync
```

### 2. 运行实验
执行主训练脚本，该脚本会自动完成数据预处理、模型训练、参数调优及结果评估。
```bash
uv run python src/train.py
```

## 数据预处理策略
- **缺失值处理**：将数据集中的 `unknown` 标记视为一个独立的类别进行保留。
- **特征编码**：
    - 对 `education` 采用有序映射（Ordinal Mapping）。
    - 对其余类别型特征采用独热编码（One-Hot Encoding）。
    - 对目标变量 `y` 进行标签编码（Label Encoding: no=0, yes=1）。
- **特征缩放**：使用 `StandardScaler` 对所有数值型特征进行标准化。
- **类别不平衡处理**：在模型初始化时设置 `class_weight='balanced'`。

## 实验输出
所有实验结果均保存在 `output/` 目录下：
- `confusion_matrix.pdf`: 混淆矩阵可视化图表。
- `roc_curve.pdf`: ROC 曲线及 AUC 值标注（目标 AUC >= 0.94）。
- `classification_report.txt`: 包含精确率、召回率、F1-Score 的详细分类报告。
- `best_params.txt`: 网格搜索确定的最佳超参数（C, penalty）。
- `data_summary.txt`: 数据集基本分布及缺失值摘要。
- `model_evaluation.txt`: 模型核心性能指标汇总。

## 实验结果摘要
- **最佳参数**: `{'C': 1, 'max_iter': 1000, 'penalty': 'l1'}` (示例，以实际运行为准)
- **AUC 值**: 0.9439 (满足 >= 0.94 的实验要求)
- **结论**: 模型在处理类别不平衡的数据集上表现良好，对正样本（yes）具有较高的识别能力。
