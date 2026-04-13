# AGENTS.md

## 项目概述

本项目为机器学习实验仓库，包含多个实验（exp1, exp2, ...），每个实验对应一个独立的机器学习任务。

## 实验工作流程

### 1. 检查文件夹状态
- 检查目标文件夹 `exp{数字}` 是否已存在
- 检查数据文件是否已准备好
- 确认后再进行下一步

### 2. 初始化项目（如需新建）
```bash
cd exp{数字}
uv init
```

### 3. 添加依赖包
```bash
uv add pandas scikit-learn imbalanced-learn matplotlib seaborn
```

### 4. 编写代码
- 创建 `main.py` 文件
- 按照任务要求实现功能

### 5. 运行脚本
```bash
uv run main.py
```
注意：使用 `uv run main.py` 而非 `uv run python main.py`

### 6. 调试与修改
- 运行后如有错误，修改代码
- 重复运行直到成功

### 7. 生成报告
- 创建 `report.md`
- 从实际输出中完全复制内容
- 图片标注格式：`图1：xxx（详见exp2/output/xxx.pdf）`
- 禁止使用markdown语法（不要 `#`、`##`、`**`、`` ``` ``）
- 纯文本格式，直接复制输出内容
- 第四部分"实验结果及分析和（或）源程序调试过程"中的数据用截图展示，只保留分析性文字

### 8. 整理文档
- 从 `exp1/doc/` 复制实验文档模板
- 重命名为 `实验{数字}.doc`

## 运行命令

### 单个脚本
```bash
uv run main.py
```

### 添加依赖
```bash
uv add <package_name>
```

### 查看依赖树
```bash
uv tree
```

### 代码格式化
```bash
ruff format
```

### 代码检查
```bash
ruff check .
```

## 代码风格指南

### 导入顺序
1. 标准库 (os, sys, datetime, etc.)
2. 第三方库 (pandas, numpy, sklearn, etc.)
3. 本地模块

### 命名约定
- 函数名：snake_case (如 `load_data`, `preprocess_data`)
- 变量名：snake_case (如 `X_train`, `y_test`)
- 常量：UPPER_SNAKE_CASE (如 `OUTPUT_DIR`, `AUC_REQUIREMENT`)
- 类名：PascalCase

### 代码结构
- 每个功能模块使用独立函数
- 函数要有清晰的输入输出
- 使用全局变量存储配置常量

### 错误处理
- 使用try-except捕获关键错误
- 打印清晰的错误信息

### 输出规范
- 所有输出文件放在 `output/` 目录
- 输出格式要求：
  - 文本文件：`.txt` 格式
  - 图片文件：`.pdf` 格式（务必保证高清晰度，保存时设置高分辨率如 `dpi=300`；图像中可视化绘制线条需加粗，文字需放大并带背景底色）
  - 模型训练曲线：必须绘制 loss 或其他指标的变化曲线。若受限于算力仅训练单轮(epoch=1)，则必须按 batch(步) 收集数据以绘制平滑曲线，不可出现没有曲线的情况。
- 指标文件命名：
  - `data_summary.txt` - 数据集摘要
  - `best_params.txt` - 最佳超参数
  - `classification_report.txt` - 分类报告
  - `model_evaluation.txt` - 模型评估
  - `confusion_matrix.pdf` - 混淆矩阵
  - `roc_curve.pdf` - ROC曲线
  - `log.txt` - 训练日志

### Report格式要求
- 纯文本格式，无markdown语法
- 报告必须严格包含以下四个标题段落：
  一、实验目的
  二、实验项目内容
  三、实验过程或算法
  四、实验结果及分析和（或）源程序调试过程
- 禁止使用：`#`、`##`、`**`、`` ``` ``
- 第四部分"实验结果及分析和（或）源程序调试过程"：
  - 数据内容用截图展示（标注详见xxx.pdf）
  - 只保留结论性、分析性文字
- 图片标注格式：`图1：xxx（详见exp2/output/xxx.pdf）`
