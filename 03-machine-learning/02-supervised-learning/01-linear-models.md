# 2.1 线性回归与逻辑回归

## 目录

- [1. 线性回归](#1-线性回归)
- [2. 逻辑回归](#2-逻辑回归)
- [3. 正则化线性模型](#3-正则化线性模型)
- [4. 实践练习](#4-实践练习)

---

## 1. 线性回归

### 1.1 模型定义

**线性回归**：假设目标变量与特征之间存在线性关系。

**单变量线性回归**：
$$y = \beta_0 + \beta_1 x + \epsilon$$

**多元线性回归**：
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon$$

**矩阵形式**：
$$y = X\beta + \epsilon$$

其中：
- $y$：目标变量
- $X$：特征矩阵
- $\beta$：系数向量
- $\epsilon$：误差项

### 1.2 最小二乘法

**目标函数**：
$$\min_{\beta} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 = \min_{\beta} \|y - X\beta\|^2$$

**正规方程解**：
$$\beta = (X^T X)^{-1} X^T y$$

**梯度下降**：
$$\beta_j := \beta_j - \alpha \frac{\partial}{\partial \beta_j} J(\beta)$$

其中 $\alpha$ 是学习率。

### 1.3 假设检验

**t检验**：检验单个系数的显著性
$$t = \frac{\hat{\beta}_j - 0}{SE(\hat{\beta}_j)}$$

**F检验**：检验整体模型的显著性
$$F = \frac{SSR/k}{SSE/(n-k-1)}$$

### 1.4 实践示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 生成数据
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + 3 + np.random.randn(100, 1) * 1.5

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
plt.plot(X_test, y_pred, 'r-', label='拟合线')
plt.legend()
plt.show()
```

---

## 2. 逻辑回归

### 2.1 模型定义

**逻辑回归**：用于二分类问题，输出概率值。

**Sigmoid函数**：
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**预测概率**：
$$P(y=1|x) = \sigma(\beta_0 + \beta_1 x)$$

**决策边界**：
$$\beta_0 + \beta_1 x = 0$$

### 2.2 损失函数

**对数似然损失**：
$$L(\beta) = \sum_{i=1}^{m} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

**梯度下降更新**：
$$\beta_j := \beta_j + \alpha \sum_{i=1}^{m} (y_i - \hat{y}_i) x_{ij}$$

### 2.3 多分类逻辑回归

**Softmax函数**：
$$P(y=k|x) = \frac{e^{\beta_k^T x}}{\sum_{j=1}^{K} e^{\beta_j^T x}}$$

**交叉熵损失**：
$$L(\beta) = -\sum_{i=1}^{m} \sum_{k=1}^{K} y_{ik} \log(P(y=k|x_i))$$

### 2.4 实践示例

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
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

## 3. 正则化线性模型

### 3.1 Ridge回归（L2正则化）

**目标函数**：
$$\min_{\beta} \|y - X\beta\|^2 + \lambda \|\beta\|^2$$

**优点**：
- 防止过拟合
- 提高模型稳定性
- 适用于多重共线性数据

### 3.2 Lasso回归（L1正则化）

**目标函数**：
$$\min_{\beta} \|y - X\beta\|^2 + \lambda \|\beta\|_1$$

**优点**：
- 特征选择（产生稀疏解）
- 可解释性强

### 3.3 Elastic Net

**目标函数**：
$$\min_{\beta} \|y - X\beta\|^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|^2$$

**优点**：
- 结合L1和L2的优点
- 适用于高维数据

### 3.4 实践示例

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score

# 加载数据
data = load_diabetes()
X, y = data.data, data.target

# 定义模型
models = [
    ('Ridge', Ridge(alpha=1.0)),
    ('Lasso', Lasso(alpha=0.1)),
    ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5))
]

# 交叉验证
for name, model in models:
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"{name}: MSE = {-scores.mean():.2f} (+/- {scores.std():.2f})")
```

---

## 4. 实践练习

### 练习1：波士顿房价预测

```python
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据
housing = fetch_openml(name='boston', version=1, parser='auto')
X, y = housing.data, housing.target

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# 预测
y_pred_linear = linear_model.predict(X_test)
y_pred_ridge = ridge_model.predict(X_test)

# 评估
print(f"线性回归 MSE: {mean_squared_error(y_test, y_pred_linear):.2f}")
print(f"Ridge回归 MSE: {mean_squared_error(y_test, y_pred_ridge):.2f}")
```

### 练习2：鸢尾花分类

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 创建模型
model = LogisticRegression(max_iter=200)

# 交叉验证
scores = cross_val_score(model, X, y, cv=5)
print(f"准确率: {scores.mean():.2f} (+/- {scores.std():.2f})")
```

---

**下一步**：[决策树与集成学习](02-decision-trees.md)