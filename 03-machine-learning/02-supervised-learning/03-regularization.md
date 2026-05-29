# 2.3 线性模型正则化

## 目录

- [1. 正则化概述](#1-正则化概述)
- [2. L2正则化（Ridge回归）](#2-l2正则化ridge回归)
- [3. L1正则化（Lasso回归）](#3-l1正则化lasso回归)
- [4. Elastic Net](#4-elastic-net)
- [5. 正则化的效果](#5-正则化的效果)
- [6. 实践练习](#6-实践练习)

---

## 1. 正则化概述

### 1.1 定义

**正则化**（Regularization）是通过在损失函数中添加惩罚项来防止过拟合的技术。

### 1.2 正则化的目的

| 问题 | 原因 | 正则化的作用 |
|------|------|-------------|
| 过拟合 | 模型过于复杂 | 限制模型复杂度 |
| 参数过大 | 某些特征影响过大 | 约束参数值 |
| 多重共线性 | 特征之间高度相关 | 稳定参数估计 |

### 1.3 正则化的一般形式

$$\text{Loss} = \text{经验损失} + \lambda \times \text{正则化项}$$

其中 $\lambda$ 是正则化强度超参数：
- $\lambda = 0$：无正则化
- $\lambda$ 越大：正则化越强

---

## 2. L2正则化（Ridge回归）

### 2.1 定义

L2正则化在损失函数中添加参数的平方和：

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}_i - y_i)^2 + \frac{\lambda}{2} \sum_{j=1}^n \theta_j^2$$

### 2.2 几何解释

L2正则化相当于在参数空间中添加一个圆形约束：

```
        θ₂
        ^
        |
   +----+----+
   |    |    |
   |  θ*|    |
   |    |    |
   +----+----+---> θ₁
        |
```

最优参数在约束区域与损失函数等高线的切点处。

### 2.3 解析解

$$\theta = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

其中 $\mathbf{I}$ 是单位矩阵。

### 2.4 特点

| 特点 | 说明 |
|------|------|
| **参数收缩** | 参数值变小但不会为零 |
| **多重共线性** | 有效处理特征相关问题 |
| **计算效率** | 仍有解析解，计算稳定 |

---

## 3. L1正则化（Lasso回归）

### 3.1 定义

L1正则化在损失函数中添加参数的绝对值之和：

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}_i - y_i)^2 + \lambda \sum_{j=1}^n |\theta_j|$$

### 3.2 几何解释

L1正则化相当于在参数空间中添加一个菱形约束：

```
        θ₂
        ^
      / | \
     /  |  \
    / θ*|   \
   /    |    \
  +-----+-----+---> θ₁
```

### 3.3 稀疏性

L1正则化会产生稀疏解，即部分参数变为零，实现特征选择。

### 3.4 求解方法

L1正则化没有解析解，需要使用：
- 坐标下降法（Coordinate Descent）
- 近端梯度下降（Proximal Gradient Descent）
- LARS算法

---

## 4. Elastic Net

### 4.1 定义

Elastic Net结合了L1和L2正则化：

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}_i - y_i)^2 + \lambda_1 \sum_{j=1}^n |\theta_j| + \frac{\lambda_2}{2} \sum_{j=1}^n \theta_j^2$$

### 4.2 参数设置

| 参数 | 作用 |
|------|------|
| $\lambda_1$ | L1正则化强度 |
| $\lambda_2$ | L2正则化强度 |
| $\alpha$ | 混合比例（$\alpha=1$为Lasso，$\alpha=0$为Ridge） |

### 4.3 优势

| 优势 | 说明 |
|------|------|
| **特征选择** | 继承L1的稀疏性 |
| **稳定性** | 继承L2处理多重共线性的能力 |
| **灵活性** | 通过参数调整平衡两种正则化 |

---

## 5. 正则化的效果

### 5.1 正则化强度与模型复杂度

| $\lambda$ | 模型复杂度 | 偏差 | 方差 |
|-----------|-----------|------|------|
| 很小 | 高 | 低 | 高（过拟合） |
| 适中 | 适中 | 适中 | 适中（最优） |
| 很大 | 低 | 高 | 低（欠拟合） |

### 5.2 正则化路径

随着正则化强度增加，参数逐渐收缩：

```
参数值
  ^
  |  θ₁\    θ₂\
  |     \      \
  |      \      \
  |       \______\____ λ
  |
  +-------------------> λ
```

### 5.3 超参数选择

使用交叉验证选择最优的正则化参数：

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

alphas = [0.1, 1, 10, 100]
best_alpha = None
best_score = float('-inf')

for alpha in alphas:
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X, y, cv=5)
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_alpha = alpha

print(f"最优alpha: {best_alpha}")
```

---

## 6. 实践练习

### 练习1：Ridge回归 vs Lasso回归

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

# 生成高维数据（特征数大于样本数）
np.random.seed(42)
X, y = make_regression(n_samples=50, n_features=20, noise=0.1, random_state=42)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 测试不同正则化强度
alphas = np.logspace(-3, 2, 20)
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_scores.append(mean_squared_error(y_test, ridge.predict(X_test)))
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_scores.append(mean_squared_error(y_test, lasso.predict(X_test)))

# 可视化结果
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, ridge_scores, label='Ridge回归', marker='o')
plt.semilogx(alphas, lasso_scores, label='Lasso回归', marker='s')
plt.xlabel('正则化参数 α')
plt.ylabel('测试MSE')
plt.title('正则化强度对模型性能的影响')
plt.legend()
plt.grid(True)
plt.show()
```

### 练习2：Lasso特征选择

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso

# 生成数据（只有前5个特征有真实信号）
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=15, n_informative=5, noise=0.1, random_state=42)

# 使用Lasso进行特征选择
lasso = Lasso(alpha=0.5)
lasso.fit(X, y)

# 获取非零系数的特征索引
selected_features = np.where(lasso.coef_ != 0)[0]
print(f"原始特征数: {X.shape[1]}")
print(f"选中的特征数: {len(selected_features)}")
print(f"选中的特征索引: {selected_features}")

# 可视化系数
plt.figure(figsize=(10, 6))
plt.stem(lasso.coef_, use_line_collection=True)
plt.xlabel('特征索引')
plt.ylabel('系数值')
plt.title('Lasso系数（非零系数表示选中的特征）')
plt.show()
```

### 练习3：Elastic Net实践

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score

# 生成数据
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用网格搜索选择超参数
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

elastic_net = ElasticNet(max_iter=1000)
grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print("最优参数:")
print(f"  alpha: {grid_search.best_params_['alpha']}")
print(f"  l1_ratio: {grid_search.best_params_['l1_ratio']}")

# 使用最优模型预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\n测试集R²: {r2:.4f}")
```

---

**上一节**：[逻辑回归](02-logistic-regression.md)
**下一节**：[支持向量机](04-svm.md)
