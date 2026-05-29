# 2.3 支持向量机

## 目录

- [1. SVM概述](#1-svm概述)
- [2. 线性SVM](#2-线性svm)
- [3. 核方法](#3-核方法)
- [4. SVM用于回归](#4-svm用于回归)
- [5. 实践练习](#5-实践练习)

---

## 1. SVM概述

### 1.1 什么是支持向量机？

**支持向量机（SVM）**：一种强大的分类算法，寻找最优超平面来分隔不同类别的数据。

**核心思想**：
- 最大化分类间隔
- 只关注支持向量

### 1.2 SVM的特点

| 优点 | 缺点 |
|------|------|
| 高维空间有效 | 计算复杂度高 |
| 正则化内置 | 对参数敏感 |
| 核技巧强大 | 大规模数据困难 |

---

## 2. 线性SVM

### 2.1 最大间隔分类器

**目标**：找到最大间隔的超平面

**超平面方程**：
$$w^T x + b = 0$$

**间隔**：
$$\text{Margin} = \frac{2}{\|w\|}$$

**约束条件**：
$$y_i (w^T x_i + b) \geq 1$$

### 2.2 软间隔SVM

**引入松弛变量**：
$$y_i (w^T x_i + b) \geq 1 - \xi_i$$

**目标函数**：
$$\min_{w,b,\xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{m} \xi_i$$

其中 $C$ 是正则化参数。

### 2.3 对偶问题

**拉格朗日函数**：
$$\mathcal{L}(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^{m} \alpha_i (y_i (w^T x_i + b) - 1)$$

**对偶形式**：
$$\max_{\alpha} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i,j=1}^{m} \alpha_i \alpha_j y_i y_j x_i^T x_j$$

**KKT条件**：
$$\alpha_i (y_i (w^T x_i + b) - 1) = 0$$

---

## 3. 核方法

### 3.1 核函数

**核技巧**：将低维数据映射到高维空间

**常见核函数**：

| 核函数 | 公式 | 适用场景 |
|--------|------|---------|
| 线性核 | $K(x_i, x_j) = x_i^T x_j$ | 线性可分数据 |
| 多项式核 | $K(x_i, x_j) = (x_i^T x_j + 1)^d$ | 多项式关系 |
| RBF核 | $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ | 非线性数据 |
| Sigmoid核 | $K(x_i, x_j) = \tanh(\alpha x_i^T x_j + c)$ | 神经网络风格 |

### 3.2 核函数选择

**选择策略**：
- 线性数据：线性核
- 文本数据：线性核或RBF核
- 图像数据：RBF核
- 不确定时：先尝试RBF核

### 3.3 实践示例

```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成非线性数据
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型（使用RBF核）
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")

# 可视化决策边界
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title('SVM with RBF Kernel')
plt.show()
```

---

## 4. SVM用于回归

### 4.1 支持向量回归（SVR）

**目标**：找到一个函数，使得大多数样本在 $\epsilon$-带内

**损失函数**：
$$\text{Loss}(y, \hat{y}) = \max(0, |y - \hat{y}| - \epsilon)$$

**目标函数**：
$$\min_{w,b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{m} \xi_i$$

### 4.2 实践示例

```python
from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

# 生成数据
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

# 可视化
plt.scatter(X, y, label='数据')
plt.plot(X, model.predict(X), 'r-', label='SVR')
plt.legend()
plt.show()
```

---

## 5. 实践练习

### 练习1：手写数字识别

```python
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score

# 加载数据
digits = load_digits()
X, y = digits.data, digits.target

# 创建模型
model = SVC(kernel='rbf', C=1.0, gamma=0.001)

# 交叉验证
scores = cross_val_score(model, X, y, cv=5)
print(f"准确率: {scores.mean():.2f} (+/- {scores.std():.2f})")
```

### 练习2：参数调优

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf']
}

# 创建模型
model = SVC()

# 网格搜索
grid = GridSearchCV(model, param_grid, refit=True, verbose=2, cv=5)
grid.fit(X, y)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳得分: {grid.best_score_:.2f}")
```

---

**下一步**：[聚类算法](04-clustering.md)