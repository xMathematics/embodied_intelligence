# 1.2 监督学习基础

## 目录

- [1. 监督学习概述](#1-监督学习概述)
- [2. 回归问题](#2-回归问题)
- [3. 分类问题](#3-分类问题)
- [4. 评估指标](#4-评估指标)
- [5. 正则化](#5-正则化)
- [6. 实践练习](#6-实践练习)

---

## 1. 监督学习概述

### 1.1 定义

**监督学习**：从标注数据中学习输入到输出的映射函数。

**目标**：
$$f: X \rightarrow Y$$

其中：
- $X$：输入特征空间
- $Y$：输出标签空间

### 1.2 问题类型

| 类型 | 输出类型 | 示例 |
|------|---------|------|
| 回归 | 连续值 | 房价预测、温度预测 |
| 分类 | 离散标签 | 垃圾邮件检测、图像分类 |
| 排序 | 有序标签 | 推荐排序、搜索结果 |

### 1.3 学习流程

```
输入数据 (X, y) → 特征工程 → 模型选择 → 训练 → 评估 → 预测
```

---

## 2. 回归问题

### 2.1 线性回归

**模型形式**：
$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n$$

**目标函数**（最小二乘法）：
$$\min_{\beta} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

**求解方法**：
- 正规方程：$\beta = (X^T X)^{-1} X^T y$
- 梯度下降

### 2.2 多元线性回归

**矩阵形式**：
$$\hat{y} = X \beta$$

**正则化**：
- L1正则化（Lasso）：$\min \|y - X\beta\|^2 + \lambda \|\beta\|_1$
- L2正则化（Ridge）：$\min \|y - X\beta\|^2 + \lambda \|\beta\|_2^2$

### 2.3 非线性回归

**多项式回归**：
$$\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_n x^n$$

**广义线性模型**：
- Logistic回归（分类）
- Poisson回归（计数数据）
- Gamma回归（正连续数据）

---

## 3. 分类问题

### 3.1 二分类问题

**Logistic回归**：
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
$$P(y=1|x) = \sigma(\beta_0 + \beta_1 x)$$

**决策边界**：
$$\beta_0 + \beta_1 x = 0$$

### 3.2 多分类问题

**Softmax回归**：
$$P(y=k|x) = \frac{e^{\beta_k^T x}}{\sum_{j=1}^{K} e^{\beta_j^T x}}$$

**One-vs-Rest**：
- 为每个类别训练一个二分类器
- 选择概率最大的类别

### 3.3 分类器评估

**混淆矩阵**：

| 真实\预测 | 正类 | 负类 |
|-----------|------|------|
| 正类 | TP | FN |
| 负类 | FP | TN |

**评估指标**：
- 准确率：$\frac{TP + TN}{TP + TN + FP + FN}$
- 精确率：$\frac{TP}{TP + FP}$
- 召回率：$\frac{TP}{TP + FN}$
- F1分数：$2 \times \frac{精确率 \times 召回率}{精确率 + 召回率}$

---

## 4. 评估指标

### 4.1 回归指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 均方误差 | $MSE = \frac{1}{m} \sum (y_i - \hat{y}_i)^2$ | 惩罚大误差 |
| 均方根误差 | $RMSE = \sqrt{MSE}$ | 与目标变量同单位 |
| 平均绝对误差 | $MAE = \frac{1}{m} \sum |y_i - \hat{y}_i|$ | 对异常值鲁棒 |
| $R^2$分数 | $R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | 解释方差比例 |

### 4.2 分类指标

| 指标 | 公式 | 说明 |
|------|------|------|
| 准确率 | $\frac{正确预测}{总样本}$ | 整体性能 |
| 精确率 | $\frac{TP}{TP + FP}$ | 避免误报 |
| 召回率 | $\frac{TP}{TP + FN}$ | 避免漏报 |
| F1分数 | $2 \times \frac{P \times R}{P + R}$ | 平衡P和R |
| AUC-ROC | 曲线下面积 | 整体分类能力 |

---

## 5. 正则化

### 5.1 过拟合问题

**原因**：
- 模型过于复杂
- 训练数据不足
- 噪声过多

**表现**：
- 训练误差低
- 测试误差高

### 5.2 正则化方法

**L1正则化（Lasso）**：
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x_i) - y_i)^2 + \lambda \sum_{j=1}^{n} |\theta_j|$$
- 产生稀疏解
- 特征选择

**L2正则化（Ridge）**：
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x_i) - y_i)^2 + \lambda \sum_{j=1}^{n} \theta_j^2$$
- 权重衰减
- 更稳定的解

**Elastic Net**：
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x_i) - y_i)^2 + \lambda_1 \sum_{j=1}^{n} |\theta_j| + \lambda_2 \sum_{j=1}^{n} \theta_j^2$$

---

## 6. 实践练习

### 练习1：线性回归

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 生成数据
np.random.seed(42)
X = np.random.rand(200, 1) * 10
y = 3 * X + 5 + np.random.randn(200, 1) * 2

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print(f"系数: {model.coef_[0][0]:.2f}")
print(f"截距: {model.intercept_[0]:.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

# 可视化
plt.scatter(X_test, y_test, label='真实值')
plt.plot(X_test, y_pred, 'r-', label='预测线')
plt.legend()
plt.show()
```

### 练习2：Logistic回归分类

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 生成数据
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# 可视化决策边界
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.show()
```

---

**下一步**：[无监督学习基础](03-unsupervised-learning.md)