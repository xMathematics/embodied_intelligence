# 1.2 监督学习基础

## 目录

- [1. 监督学习概述](#1-监督学习概述)
- [2. 分类问题](#2-分类问题)
- [3. 回归问题](#3-回归问题)
- [4. 线性模型基础](#4-线性模型基础)
- [5. 损失函数](#5-损失函数)
- [6. 模型评估指标](#6-模型评估指标)
- [7. 实践练习](#7-实践练习)

---

## 1. 监督学习概述

### 1.1 定义

**监督学习**（Supervised Learning）是机器学习中最常见的范式，其核心特点是：

- **训练数据包含标签**：每个样本都有对应的输出（标签）
- **学习映射函数**：目标是学习从输入特征到输出标签的映射关系
- **预测未知数据**：利用学到的函数对未见过的数据进行预测

### 1.2 监督学习的基本框架

```
输入 X → 模型 f(X; θ) → 预测 ŷ
        ↓
    损失 L(y, ŷ) → 优化 θ
```

### 1.3 关键要素

| 要素 | 说明 | 示例 |
|------|------|------|
| **输入特征 X** | 数据的属性描述 | 图像像素、文本特征、数值特征 |
| **标签 y** | 期望的输出 | 类别标签（猫/狗）、连续值（房价） |
| **模型 f** | 学习到的函数 | 线性回归、决策树、神经网络 |
| **参数 θ** | 模型的可学习参数 | 权重、偏置 |
| **损失函数 L** | 衡量预测误差 | MSE、交叉熵 |
| **优化器** | 更新参数的算法 | 梯度下降、SGD |

### 1.4 监督学习的流程

1. **数据收集**：获取带标签的数据集
2. **数据预处理**：清洗、标准化、特征工程
3. **数据集划分**：训练集、验证集、测试集
4. **模型选择**：选择合适的模型架构
5. **模型训练**：通过优化损失函数学习参数
6. **模型评估**：在测试集上评估性能
7. **模型部署**：应用到实际场景

---

## 2. 分类问题

### 2.1 定义

**分类**（Classification）是监督学习的核心任务之一，目标是将输入数据分配到预定义的类别中。

### 2.2 分类类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **二分类** | 只有两个类别 | 垃圾邮件识别（是/否）、疾病诊断（有病/无病） |
| **多分类** | 两个以上类别 | 图像分类（猫/狗/鸟）、手写数字识别（0-9） |
| **多标签分类** | 一个样本可属于多个类别 | 图像标注（一张图可能包含猫和狗） |

### 2.3 常见分类算法

| 算法 | 原理 | 特点 |
|------|------|------|
| **逻辑回归** | Sigmoid函数输出概率 | 简单、可解释性强 |
| **K近邻** | 基于距离的投票 | 非参数、无需训练 |
| **决策树** | 基于特征的递归划分 | 直观、可解释 |
| **支持向量机** | 寻找最大间隔超平面 | 高维空间表现好 |
| **朴素贝叶斯** | 基于贝叶斯定理 | 计算效率高 |

### 2.4 分类模型的输出

**硬预测**：直接输出类别标签
```python
# 输出：类别 1 或 0
y_pred = model.predict(X)
```

**软预测**：输出概率分布
```python
# 输出：属于每个类别的概率
y_proba = model.predict_proba(X)
```

---

## 3. 回归问题

### 3.1 定义

**回归**（Regression）是预测连续数值输出的任务。

### 3.2 回归类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **线性回归** | 输出与特征线性相关 | 房价预测、销售额预测 |
| **非线性回归** | 输出与特征非线性相关 | 时间序列预测、复杂函数拟合 |
| **多元回归** | 多个输入特征 | 基于面积、位置、房间数预测房价 |

### 3.3 常见回归算法

| 算法 | 原理 | 适用场景 |
|------|------|---------|
| **线性回归** | 最小二乘法拟合直线 | 简单线性关系 |
| **岭回归** | L2正则化的线性回归 | 防止过拟合 |
| **Lasso回归** | L1正则化的线性回归 | 特征选择 |
| **多项式回归** | 多项式特征的线性回归 | 非线性关系 |
| **决策树回归** | 基于树的分段拟合 | 复杂非线性关系 |

### 3.4 回归模型的评估

回归任务关注预测值与真实值之间的误差，常用指标包括：
- 均方误差（MSE）
- 均方根误差（RMSE）
- 平均绝对误差（MAE）
- R²分数

---

## 4. 线性模型基础

### 4.1 线性回归

**模型形式**：
$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$$

**向量形式**：
$$\hat{y} = \mathbf{X}\theta$$

其中：
- $\mathbf{X}$ 是特征矩阵（$m \times n$）
- $\theta$ 是参数向量（$n \times 1$）

### 4.2 最小二乘法

**目标**：最小化预测值与真实值之间的平方误差

$$\min_\theta \sum_{i=1}^m (y_i - \hat{y}_i)^2 = \min_\theta \|\mathbf{y} - \mathbf{X}\theta\|^2$$

**解析解**：
$$\theta = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

### 4.3 逻辑回归

**模型形式**：
$$\hat{y} = \sigma(\mathbf{X}\theta)$$

其中 $\sigma$ 是Sigmoid函数：
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**输出解释**：$\hat{y}$ 表示样本属于正类的概率

### 4.4 线性模型的优缺点

| 优点 | 缺点 |
|------|------|
| 训练速度快 | 假设特征与输出线性相关 |
| 易于解释 | 无法处理非线性关系 |
| 参数数量少 | 对异常值敏感 |
| 泛化能力强 | 特征工程要求高 |

---

## 5. 损失函数

### 5.1 回归任务的损失函数

#### 均方误差（MSE）
$$L(y, \hat{y}) = \frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2$$

**特点**：对大误差惩罚更重，对异常值敏感

#### 平均绝对误差（MAE）
$$L(y, \hat{y}) = \frac{1}{m} \sum_{i=1}^m |y_i - \hat{y}_i|$$

**特点**：对异常值更鲁棒

#### Huber损失
$$L(y, \hat{y}) = \begin{cases} 
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}$$

**特点**：结合MSE和MAE的优点

### 5.2 分类任务的损失函数

#### 二元交叉熵（Binary Cross-Entropy）
$$L(y, \hat{y}) = -\frac{1}{m} \sum_{i=1}^m [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

#### 多类交叉熵（Categorical Cross-Entropy）
$$L(y, \hat{y}) = -\frac{1}{m} \sum_{i=1}^m \sum_{j=1}^c y_{ij} \log(\hat{y}_{ij})$$

其中 $c$ 是类别数量

### 5.3 损失函数选择指南

| 任务类型 | 推荐损失函数 | 适用场景 |
|---------|------------|---------|
| 回归 | MSE | 一般回归问题 |
| 回归 | MAE | 存在异常值 |
| 二分类 | 二元交叉熵 | 概率输出 |
| 多分类 | 多类交叉熵 | 多类别预测 |

---

## 6. 模型评估指标

### 6.1 分类任务指标

#### 混淆矩阵

| | 预测正类 | 预测负类 |
|---|---------|---------|
| **实际正类** | TP（真正例） | FN（假负例） |
| **实际负类** | FP（假正例） | TN（真负例） |

#### 常用指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **准确率** | $(TP + TN) / (TP + TN + FP + FN)$ | 整体预测正确率 |
| **精确率** | $TP / (TP + FP)$ | 预测为正类的样本中真正为正类的比例 |
| **召回率** | $TP / (TP + FN)$ | 真正为正类的样本中被正确预测的比例 |
| **F1分数** | $2 \times \text{精确率} \times \text{召回率} / (\text{精确率} + \text{召回率})$ | 精确率和召回率的调和平均 |

#### ROC曲线与AUC

- **ROC曲线**：以假正率（FPR）为横轴，真正率（TPR）为纵轴的曲线
- **AUC**：ROC曲线下的面积，衡量模型的整体性能

### 6.2 回归任务指标

| 指标 | 公式 | 含义 |
|------|------|------|
| **MSE** | $\frac{1}{m} \sum (y - \hat{y})^2$ | 均方误差 |
| **RMSE** | $\sqrt{\frac{1}{m} \sum (y - \hat{y})^2}$ | 均方根误差（与y同单位） |
| **MAE** | $\frac{1}{m} \sum |y - \hat{y}|$ | 平均绝对误差 |
| **R²** | $1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$ | 决定系数（解释方差比例） |

### 6.3 指标选择指南

| 场景 | 推荐指标 |
|------|---------|
| 类别平衡 | 准确率 |
| 类别不平衡 | F1分数、AUC |
| 关注预测概率 | 对数损失 |
| 回归任务 | RMSE、R² |

---

## 7. 实践练习

### 练习1：线性回归实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 生成模拟数据
np.random.seed(42)
X = np.random.rand(100, 2)  # 两个特征：面积、房间数
y = 50000 * X[:, 0] + 10000 * X[:, 1] + 30000 + np.random.randn(100) * 5000

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("线性回归结果:")
print(f"  系数: {model.coef_}")
print(f"  截距: {model.intercept_:.2f}")
print(f"  MSE: {mse:.2f}")
print(f"  R²: {r2:.4f}")

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('真实价格')
plt.ylabel('预测价格')
plt.title('真实价格 vs 预测价格')
plt.show()
```

### 练习2：逻辑回归分类

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 生成分类数据
X, y = make_classification(
    n_samples=200, n_features=2, n_informative=2, 
    n_redundant=0, n_classes=2, random_state=42
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
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("逻辑回归分类结果:")
print(f"  准确率: {accuracy:.4f}")
print(f"  精确率: {precision:.4f}")
print(f"  召回率: {recall:.4f}")
print(f"  F1分数: {f1:.4f}")
print("\n混淆矩阵:")
print(cm)

# 可视化决策边界
plt.figure(figsize=(10, 6))
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('逻辑回归决策边界')
plt.show()
```

### 练习3：模型评估对比

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# 生成回归数据
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练不同模型
models = {
    '线性回归': LinearRegression(),
    'Ridge回归': Ridge(alpha=1.0),
    'Lasso回归': Lasso(alpha=0.1)
}

# 评估对比
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R²': r2}

# 打印结果
print("模型评估对比:")
print(f"{'模型':<12} {'MSE':<10} {'R²':<10}")
print("-" * 35)
for name, metrics in results.items():
    print(f"{name:<12} {metrics['MSE']:<10.4f} {metrics['R²']:<10.4f}")
```

---

**下一步**：[无监督学习基础](03-unsupervised-learning.md)
