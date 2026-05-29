# 2.4 支持向量机

## 目录

- [1. 支持向量机概述](#1-支持向量机概述)
- [2. 最大间隔分类器](#2-最大间隔分类器)
- [3. 软间隔与正则化](#3-软间隔与正则化)
- [4. 核方法](#4-核方法)
- [5. SVM回归](#5-svm回归)
- [6. 实践练习](#6-实践练习)

---

## 1. 支持向量机概述

### 1.1 定义

**支持向量机**（Support Vector Machine, SVM）是一种强大的监督学习算法，用于分类和回归任务。

### 1.2 SVM的核心思想

- **最大间隔**：找到能够分离两类数据的最优超平面，使得两类数据到超平面的距离最大
- **支持向量**：离超平面最近的样本点，决定了超平面的位置

### 1.3 应用场景

| 场景 | 示例 |
|------|------|
| 文本分类 | 垃圾邮件检测、情感分析 |
| 图像识别 | 手写数字识别、人脸识别 |
| 生物信息学 | 蛋白质结构预测、基因表达分析 |
| 金融风控 | 信用评分、欺诈检测 |

---

## 2. 最大间隔分类器

### 2.1 线性可分情况

假设数据线性可分，存在超平面：
$$\mathbf{w}^T \mathbf{x} + b = 0$$

**分类规则**：
- 如果 $\mathbf{w}^T \mathbf{x} + b \geq 0$，预测为正类
- 如果 $\mathbf{w}^T \mathbf{x} + b < 0$，预测为负类

### 2.2 间隔的定义

**几何间隔**：点到超平面的距离
$$\gamma_i = \frac{|\mathbf{w}^T \mathbf{x}_i + b|}{\|\mathbf{w}\|}$$

**目标**：最大化最小间隔
$$\max_{\mathbf{w}, b} \min_i \frac{y_i (\mathbf{w}^T \mathbf{x}_i + b)}{\|\mathbf{w}\|}$$

### 2.3 支持向量

支持向量是满足 $\gamma_i = \gamma_{\text{min}}$ 的样本点，即离超平面最近的点。

### 2.4 优化问题

等价于：
$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$$
$$\text{s.t. } y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \forall i$$

---

## 3. 软间隔与正则化

### 3.1 线性不可分情况

现实中数据往往不是完全线性可分的，需要允许一些样本违反间隔约束。

### 3.2 引入松弛变量

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^m \xi_i$$
$$\text{s.t. } y_i (\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \forall i$$
$$\xi_i \geq 0, \forall i$$

其中：
- $\xi_i$：松弛变量，表示第i个样本的违反程度
- $C$：正则化参数，控制分类错误和间隔之间的权衡

### 3.3 参数C的影响

| C值 | 效果 |
|-----|------|
| 很小 | 允许更多误分类，间隔较大 |
| 很大 | 严格要求正确分类，间隔较小 |

---

## 4. 核方法

### 4.1 核函数的作用

将低维空间中的非线性问题映射到高维空间中变为线性问题。

### 4.2 常用核函数

| 核函数 | 公式 | 特点 |
|--------|------|------|
| **线性核** | $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$ | 线性分类 |
| **多项式核** | $K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^T \mathbf{x}' + c)^d$ | 多项式分类 |
| **高斯核（RBF）** | $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$ | 通用核，适合非线性 |
| **Sigmoid核** | $K(\mathbf{x}, \mathbf{x}') = \tanh(\alpha \mathbf{x}^T \mathbf{x}' + c)$ | 神经网络相关 |

### 4.3 核技巧

不需要显式计算高维映射，直接计算核函数：
$$K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T \phi(\mathbf{x}')$$

### 4.4 对偶问题

通过拉格朗日对偶性，原问题转化为：
$$\max_\alpha \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$
$$\text{s.t. } 0 \leq \alpha_i \leq C, \forall i$$
$$\sum_{i=1}^m \alpha_i y_i = 0$$

---

## 5. SVM回归

### 5.1 ε-不敏感损失

$$L_\epsilon(y, \hat{y}) = \max(0, |y - \hat{y}| - \epsilon)$$

### 5.2 SVM回归优化问题

$$\min_{\mathbf{w}, b, \xi, \xi^*} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^m (\xi_i + \xi_i^*)$$
$$\text{s.t. } \hat{y}_i - y_i \leq \epsilon + \xi_i$$
$$y_i - \hat{y}_i \leq \epsilon + \xi_i^*$$
$$\xi_i, \xi_i^* \geq 0$$

---

## 6. 实践练习

### 练习1：线性SVM分类

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 生成线性可分数据
X, y = make_classification(
    n_samples=100, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建线性SVM模型
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# 预测
y_pred = svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"线性SVM准确率: {accuracy:.4f}")

# 可视化决策边界
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.6)

# 绘制超平面
w = svm.coef_[0]
b = svm.intercept_[0]
x_boundary = np.linspace(-3, 3, 100)
y_boundary = -(w[0] * x_boundary + b) / w[1]
plt.plot(x_boundary, y_boundary, 'k-', label='决策边界')

# 绘制支持向量
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], 
            s=100, facecolors='none', edgecolors='red', label='支持向量')

plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()
plt.title('线性SVM决策边界与支持向量')
plt.show()
```

### 练习2：核SVM分类

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# 生成非线性数据
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# 创建不同核函数的SVM模型
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
plt.figure(figsize=(16, 4))

for i, kernel in enumerate(kernels):
    svm = SVC(kernel=kernel, C=1.0)
    svm.fit(X, y)
    
    # 绘制决策边界
    plt.subplot(1, 4, i+1)
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm')
    plt.title(f'核函数: {kernel}')

plt.tight_layout()
plt.show()
```

### 练习3：SVM参数调优

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# 生成数据
X, y = make_classification(
    n_samples=200, n_features=10, n_informative=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 网格搜索超参数
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("最优参数:")
print(grid_search.best_params_)
print(f"\n交叉验证准确率: {grid_search.best_score_:.4f}")

# 测试集评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"测试集准确率: {accuracy:.4f}")
```

---

**上一节**：[线性模型正则化](03-regularization.md)
**下一节**：[决策树](05-decision-tree.md)
