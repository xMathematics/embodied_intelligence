# 2.2 逻辑回归

## 目录

- [1. 逻辑回归概述](#1-逻辑回归概述)
- [2. Sigmoid函数](#2-sigmoid函数)
- [3. 模型形式](#3-模型形式)
- [4. 损失函数](#4-损失函数)
- [5. 参数估计](#5-参数估计)
- [6. 多分类扩展](#6-多分类扩展)
- [7. 实践练习](#7-实践练习)

---

## 1. 逻辑回归概述

### 1.1 定义

**逻辑回归**（Logistic Regression）是一种用于分类任务的监督学习算法，尽管名字中有"回归"，但它实际上是分类算法。

### 1.2 与线性回归的区别

| 维度 | 线性回归 | 逻辑回归 |
|------|---------|---------|
| **任务类型** | 回归（连续输出） | 分类（离散输出） |
| **输出范围** | (-∞, +∞) | [0, 1] |
| **激活函数** | 无 | Sigmoid |
| **损失函数** | MSE | 交叉熵 |

### 1.3 应用场景

| 场景 | 示例 |
|------|------|
| 垃圾邮件检测 | 判断邮件是否为垃圾邮件 |
| 疾病诊断 | 判断患者是否患病 |
| 信用评估 | 判断贷款是否会违约 |
| 情感分析 | 判断文本情感倾向 |

---

## 2. Sigmoid函数

### 2.1 定义

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### 2.2 特性

| 特性 | 说明 |
|------|------|
| **输出范围** | [0, 1] |
| **中心点** | σ(0) = 0.5 |
| **导数** | σ(z)(1 - σ(z)) |
| **对称性** | σ(-z) = 1 - σ(z) |

### 2.3 函数图像

```
σ(z)
 ^
1|        __
 |      _/
 |    _/
0.5|  _/
 | _/
0|/
 -----------> z
   -5   0   5
```

---

## 3. 模型形式

### 3.1 二分类模型

$$\hat{y} = \sigma(\mathbf{X}\theta) = \frac{1}{1 + e^{-\mathbf{X}\theta}}$$

其中 $\hat{y}$ 表示样本属于正类的概率。

### 3.2 决策边界

当 $\hat{y} \geq 0.5$ 时，预测为正类；否则预测为负类。

决策边界满足：
$$\mathbf{X}\theta = 0$$

### 3.3 概率解释

- $P(y=1|x; \theta) = \sigma(\mathbf{X}\theta)$
- $P(y=0|x; \theta) = 1 - \sigma(\mathbf{X}\theta)$

---

## 4. 损失函数

### 4.1 对数损失（交叉熵）

$$L(\theta) = -\frac{1}{m} \sum_{i=1}^m [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

### 4.2 损失函数的直观理解

| y | ŷ | 损失 | 含义 |
|---|---|------|------|
| 1 | 1 | 0 | 预测正确 |
| 1 | 0 | +∞ | 预测错误 |
| 0 | 0 | 0 | 预测正确 |
| 0 | 1 | +∞ | 预测错误 |

### 4.3 梯度计算

$$\frac{\partial L}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i) x_{ij}$$

**重要发现**：逻辑回归的梯度形式与线性回归相同！

---

## 5. 参数估计

### 5.1 极大似然估计

目标是最大化似然函数：

$$L(\theta) = \prod_{i=1}^m P(y_i | x_i; \theta) = \prod_{i=1}^m \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}$$

取对数后：

$$\log L(\theta) = \sum_{i=1}^m [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

最大化对数似然等价于最小化交叉熵损失。

### 5.2 梯度下降优化

由于没有解析解，需要使用迭代优化方法：

$$\theta_{t+1} = \theta_t - \alpha \frac{1}{m} \mathbf{X}^T (\sigma(\mathbf{X}\theta_t) - \mathbf{y})$$

---

## 6. 多分类扩展

### 6.1 一对多（One-vs-Rest）

- 为每个类别训练一个二分类器
- 预测时选择概率最大的类别

### 6.2 Softmax回归

$$\hat{y}_k = \frac{e^{\mathbf{X}\theta_k}}{\sum_{j=1}^K e^{\mathbf{X}\theta_j}}$$

**损失函数**（多类交叉熵）：

$$L(\theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_{ik} \log(\hat{y}_{ik})$$

---

## 7. 实践练习

### 练习1：逻辑回归实现

```python
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 生成二分类数据
np.random.seed(42)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

# 添加截距项
X_with_bias = np.column_stack((np.ones(len(X)), X))

# 梯度下降训练
def train_logistic_regression(X, y, alpha=0.1, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(iterations):
        z = X @ theta
        y_pred = sigmoid(z)
        gradients = (1/m) * X.T @ (y_pred - y)
        theta -= alpha * gradients
    
    return theta

theta = train_logistic_regression(X_with_bias, y)
print(f"参数: θ={theta}")

# 可视化决策边界
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.6)

# 绘制决策边界
x_boundary = np.linspace(-3, 3, 100)
y_boundary = -(theta[0] + theta[1] * x_boundary) / theta[2]
plt.plot(x_boundary, y_boundary, 'k-', label='决策边界')

plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()
plt.title('逻辑回归决策边界')
plt.show()
```

### 练习2：使用Scikit-learn进行逻辑回归

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 生成分类数据
X, y = make_classification(
    n_samples=200, n_features=4, n_informative=2,
    n_redundant=0, random_state=42
)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建并训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 评估
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("逻辑回归结果:")
print(f"准确率: {accuracy:.4f}")
print("\n混淆矩阵:")
print(cm)
print("\n分类报告:")
print(report)
```

### 练习3：多分类逻辑回归

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载手写数字数据集
digits = load_digits()
X, y = digits.data, digits.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建多分类逻辑回归模型
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"多分类逻辑回归准确率: {accuracy:.4f}")

# 可视化部分预测结果
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f"预测: {y_pred[i]}, 真实: {y_test[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

---

**上一节**：[线性回归](01-linear-regression.md)
**下一节**：[线性模型正则化](03-regularization.md)
