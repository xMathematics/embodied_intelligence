# 评估脚本

## 概述

本目录包含常用的评估脚本，用于计算各种评估指标。

## 脚本列表

### 1. 分类评估
- `classification_metrics.py` - 分类任务指标（准确率、精确率、召回率、F1）

### 2. 回归评估
- `regression_metrics.py` - 回归任务指标（MSE、RMSE、MAE、R²）

### 3. 检测评估
- `detection_metrics.py` - 目标检测指标（IoU、mAP）

### 4. 分割评估
- `segmentation_metrics.py` - 分割任务指标（IoU、Dice系数）

### 5. 强化学习评估
- `rl_metrics.py` - RL任务指标（奖励、成功率）

## 使用方法

```python
from evaluation_scripts.classification_metrics import ClassificationMetrics

metrics = ClassificationMetrics()
result = metrics.compute(y_true, y_pred)
print(result)
```

## 指标说明

| 指标 | 描述 | 适用任务 |
|------|------|---------|
| Accuracy | 准确率 | 分类 |
| Precision | 精确率 | 分类 |
| Recall | 召回率 | 分类 |
| F1 Score | F1分数 | 分类 |
| MSE | 均方误差 | 回归 |
| RMSE | 均方根误差 | 回归 |
| IoU | 交并比 | 检测/分割 |
| mAP | 平均精度均值 | 检测 |
