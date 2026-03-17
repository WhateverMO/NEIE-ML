
基于决策树算法的电信客户流失预测实验报告

一、实验目的

本实验旨在使用决策树算法对电信客户流失数据进行预测，重点掌握以下技能：

1. 理解并实现数据预处理：标签编码、独热编码、缺失值处理和特征缩放
2. 掌握处理类别不平衡问题的方法（使用class_weight='balanced'）
3. 运用GridSearchCV进行决策树超参数调优
4. 全面评估模型性能：准确率、精确率、召回率、F1-Score、混淆矩阵、ROC曲线和AUC值

二、实验项目内容

2.1 数据集描述

使用Telco Customer Churn数据集，该数据集包含7043条电信客户记录，每条记录包含21个特征，目标是预测客户是否会流失（Churn: Yes/No）。

2.2 数据预处理

（1）标签编码（Label Encoding）：对具有逻辑顺序的特征进行编码
    Contract（合约类型）：Month-to-month → 0, One year → 1, Two year → 2

（2）独热编码（One-Hot Encoding）：对无等级之分的分类特征进行编码
    特征包括：gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, PaymentMethod

（3）缺失值处理：TotalCharges列有11个空值，使用中位数（1397.47）填充

（4）特征缩放：使用StandardScaler对数值特征（tenure, MonthlyCharges, TotalCharges）进行标准化处理

2.3 类别不平衡处理

由于数据集中不流失客户（No: 73.46%）远多于流失客户（Yes: 26.54%），使用class_weight='balanced'参数自动调整类别权重，给少数类（流失客户）更高的损失权重。

2.4 超参数调优

使用GridSearchCV结合5折交叉验证进行超参数搜索，参数空间如下：

max_depth: [3, 5, 7, 10, None]
min_samples_split: [2, 5, 10, 20]
min_samples_leaf: [1, 2, 5, 10]
max_features: ['sqrt', 'log2', None]
criterion: ['gini', 'entropy']

三、实验过程或算法（源程序）

详见exp2/main.py源代码

四、实验结果及分析

4.1 数据集摘要

=== Dataset Exploration Summary ===
Total Samples: 7043

Target (y) Distribution:
  No: 5174 (73.46%)
  Yes: 1869 (26.54%)

数据集中不流失客户占73.46%，流失客户占26.54%，存在明显的类别不平衡问题。

4.2 数据预处理结果

（1）删除了customerID列
（2）TotalCharges列有11个缺失值，使用中位数1397.47填充
（3）Contract列进行标签编码：{'Month-to-month': 0, 'One year': 1, 'Two year': 2}
（4）对14个分类特征进行独热编码，编码后特征数量为30个
（5）对数值特征进行标准化处理

数据集划分：
训练集：5634条（No: 4139, Yes: 1495）
测试集：1409条（No: 1035, Yes: 374）

4.3 超参数调优结果

GridSearchCV共搜索480种参数组合，使用5折交叉验证，最终找到最优参数：

=== Best Parameters from GridSearchCV ===
criterion: entropy
max_depth: 5
max_features: None
min_samples_leaf: 5
min_samples_split: 2

最优模型在交叉验证中的ROC-AUC得分为：0.8278

4.4 模型评估结果

（1）分类报告

=== Classification Report ===
              precision    recall  f1-score   support

          No       0.90      0.74      0.82      1035
         Yes       0.52      0.78      0.63       374

    accuracy                           0.75      1409
   macro avg       0.71      0.76      0.72      1409
weighted avg       0.80      0.75      0.77      1409

（2）混淆矩阵

[[768 267]
 [ 81 293]]

图1：混淆矩阵（详见exp2/output/confusion_matrix.pdf）

（3）ROC曲线

图2：ROC曲线（详见exp2/output/roc_curve.pdf），AUC = 0.8380

（4）模型评估摘要

=== Model Evaluation Summary ===
Best Params: {'criterion': 'entropy', 'max_depth': 5, 'max_features': None, 'min_samples_leaf': 5, 'min_samples_split': 2}
Accuracy Score: 0.7530
ROC-AUC Score: 0.8380

Note: AUC requirement is >= 0.80
Requirement Met: YES

4.5 结果分析

（1）准确率（Accuracy）：0.7530
    在类别不平衡的情况下，准确率不能全面反映模型性能

（2）ROC-AUC值：0.8380
    满足AUC ≥ 0.80的要求
    说明模型具有较好的区分正负样本的能力

（3）精确率（Precision）：
    不流失客户（No）：0.90
    流失客户（Yes）：0.52
    模型对不流失客户的预测更准确

（4）召回率（Recall）：
    不流失客户（No）：0.74
    流失客户（Yes）：0.78
    使用class_weight='balanced'后，模型能够识别出78%的流失客户

（5）F1-Score：
    不流失客户（No）：0.82
    流失客户（Yes）：0.63
    流失客户的F1-Score相对较低，但考虑到AUC表现良好，模型整体可用

（6）混淆矩阵分析：
    真阴性（TN）：768 - 正确预测不流失
    假阳性（FP）：267 - 误报为流失
    假阴性（FN）：81 - 漏报流失客户
    真阳性（TP）：293 - 正确预测流失
    模型漏报率较低（81/374 ≈ 21.7%），对业务来说这是重要的指标

4.6 结论

本实验成功使用决策树算法构建了电信客户流失预测模型：

1. 通过数据预处理（标签编码、独热编码、缺失值处理、特征缩放）将原始数据转换为模型可用的格式
2. 使用class_weight='balanced'参数有效处理了类别不平衡问题
3. 通过GridSearchCV找到最优超参数组合：AUC达到0.838，满足≥0.80的要求
4. 模型能够识别78%的流失客户，对业务具有实际应用价值
