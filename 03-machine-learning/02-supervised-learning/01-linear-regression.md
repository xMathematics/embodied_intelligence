# 2.1 线性回归

## 目录

- [1. 线性回归概述](#1-线性回归概述)
- [2. 简单线性回归](#2-简单线性回归)
- [3. 多元线性回归](#3-多元线性回归)
- [4. 最小二乘法](#4-最小二乘法)
- [5. 梯度下降法](#5-梯度下降法)
- [6. 模型评估](#6-模型评估)
- [7. 实践练习](#7-实践练习)

---

## 1. 线性回归概述

### 1.1 定义

**线性回归**（Linear Regression）是一种用于预测连续数值输出的监督学习算法，假设输入特征与输出之间存在线性关系。

### 1.2 应用场景

| 场景 | 示例 |
|------|------|
| 房价预测 | 根据面积、房间数预测房价 |
| 销售预测 | 根据广告投入预测销售额 |
| 需求预测 | 根据历史数据预测产品需求 |
| 风险评估 | 根据个人信息评估信用风险 |

### 1.3 线性回归的假设

1. **线性关系**：输出与特征之间存在线性关系
2. **误差独立性**：误差项相互独立
3. **误差正态性**：误差项服从正态分布
4. **同方差性**：误差项的方差恒定

---

## 2. 简单线性回归

### 2.1 模型形式

**简单线性回归**只有一个输入特征：

$$y = \theta_0 + \theta_1 x + \epsilon$$

其中：
- $y$：输出变量
- $x$：输入特征
- $\theta_0$：截距（偏置项）
- $\theta_1$：斜率（权重）
- $\epsilon$：误差项

### 2.2 预测函数

$$\hat{y} = \theta_0 + \theta_1 x$$

### 2.3 几何解释

简单线性回归在二维平面上表现为一条直线，目标是找到最佳拟合直线。

```
y
^
|     *
|   *     *
| *         *
|_________________> x
      拟合直线
```

---

## 3. 多元线性回归

### 3.1 模型形式

**多元线性回归**有多个输入特征：

$$y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n + \epsilon$$

### 3.2 向量形式

$$y = \mathbf{X}\theta + \epsilon$$

其中：
- $\mathbf{X}$：特征矩阵（$m \times (n+1)$），第一列全为1（截距项）
- $\theta$：参数向量（$(n+1) \times 1$）
- $\epsilon$：误差向量（$m \times 1$）

### 3.3 矩阵表示示例

```
特征矩阵 X (m=3个样本, n=2个特征):
[1, x₁₁, x₁₂]
[1, x₂₁, x₂₂]
[1, x₃₁, x₃₂]

参数向量 θ:
[θ₀]
[θ₁]
[θ₂]

预测值 ŷ = Xθ:
[θ₀ + θ₁x₁₁ + θ₂x₁₂]
[θ₀ + θ₁x₂₁ + θ₂x₂₂]
[θ₀ + θ₁x₃₁ + θ₂x₃₂]
```

---

## 4. 最小二乘法

### 4.1 目标函数

最小化预测值与真实值之间的平方误差：

$$\min_\theta \sum_{i=1}^m (y_i - \hat{y}_i)^2 = \min_\theta \|\mathbf{y} - \mathbf{X}\theta\|^2$$

### 4.2 解析解（正规方程）

对目标函数求导并令导数为零：

$$\frac{\partial}{\partial \theta} \|\mathbf{y} - \mathbf{X}\theta\|^2 = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\theta) = 0$$

解得：

$$\theta = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

### 4.3 正规方程的优缺点

| 优点 | 缺点 |
|------|------|
| 直接得到最优解 | 计算复杂度高（O(n³)） |
| 无需迭代 | 不适用于大规模数据 |
| 无需调参 | 需要计算矩阵逆 |

### 4.4 适用条件

- 特征数量不太大（n < 10,000）
- 不需要在线学习
- 数据集可以一次性加载到内存

---

## 5. 梯度下降法

### 5.1 基本思想

梯度下降是一种迭代优化算法，通过沿着损失函数的负梯度方向更新参数：

$$\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)$$

其中：
- $\alpha$：学习率
- $\nabla J(\theta)$：损失函数的梯度

### 5.2 梯度计算

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i) x_{ij}$$

向量形式：

$$\nabla J(\theta) = \frac{1}{m} \mathbf{X}^T (\mathbf{X}\theta - \mathbf{y})$$

### 5.3 梯度下降的变种

| 方法 | 描述 | 特点 |
|------|------|------|
| **批量梯度下降** | 使用全部样本计算梯度 | 收敛稳定，速度慢 |
| **随机梯度下降** | 使用单个样本计算梯度 | 速度快，波动大 |
| **小批量梯度下降** | 使用部分样本计算梯度 | 平衡速度和稳定性 |

### 5.4 学习率的选择

| 学习率 | 效果 |
|--------|------|
| 太小 | 收敛速度慢 |
| 太大 | 可能发散 |
| 合适 | 快速收敛到最优解 |

**自适应学习率策略**：
- AdaGrad
- RMSprop
- Adam

---

## 6. 模型评估

### 6.1 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **MSE** | $\frac{1}{m} \sum (y - \hat{y})^2$ | 均方误差 |
| **RMSE** | $\sqrt{\frac{1}{m} \sum (y - \hat{y})^2}$ | 均方根误差 |
| **MAE** | $\frac{1}{m} \sum |y - \hat{y}|$ | 平均绝对误差 |
| **R²** | $1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$ | 决定系数 |

### 6.2 R²的含义

- R² = 1：完美拟合
- R² = 0：模型效果等于均值预测
- R² < 0：模型效果差于均值预测

### 6.3 假设检验

**t检验**：检验单个系数的显著性
**F检验**：检验整个模型的显著性

---

## 7. 实践练习

### 练习1：简单线性回归实现

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 2 * x + 1 + np.random.randn(50) * 1.5

# 添加截距项
X = np.column_stack((np.ones(len(x)), x))

# 方法1：使用正规方程求解
theta = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"正规方程结果: θ₀={theta[0]:.4f}, θ₁={theta[1]:.4f}")

# 方法2：使用梯度下降求解
def gradient_descent(X, y, alpha=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    
    for _ in range(iterations):
        gradients = (1/m) * X.T @ (X @ theta - y)
        theta -= alpha * gradients
    
    return theta

theta_gd = gradient_descent(X, y)
print(f"梯度下降结果: θ₀={theta_gd[0]:.4f}, θ₁={theta_gd[1]:.4f}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='数据点', alpha=0.6)
plt.plot(x, X @ theta, 'r-', label='拟合直线')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('简单线性回归')
plt.show()
```

### 练习2：多元线性回归

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 生成多元回归数据
np.random.seed(42)
X = np.random.rand(100, 3)  # 3个特征
y = 5 * X[:, 0] + 3 * X[:, 1] + 2 * X[:, 2] + 1 + np.random.randn(100) * 0.3

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用Scikit-learn进行线性回归
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("多元线性回归结果:")
print(f"  系数: {model.coef_}")
print(f"  截距: {model.intercept_:.4f}")
print(f"  MSE: {mse:.4f}")
print(f"  R²: {r2:.4f}")

# 验证理论系数
print("\n理论系数: [5, 3, 2], 截距: 1")
print(f"实际系数: [{model.coef_[0]:.2f}, {model.coef_[1]:.2f}, {model.coef_[2]:.2f}], 截距: {model.intercept_:.2f}")
```

### 练习3：梯度下降与学习率

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(42)
x = np.linspace(0, 10, 100)
y = 3 * x + 2 + np.random.randn(100) * 2
X = np.column_stack((np.ones(len(x)), x))

# 测试不同学习率
learning_rates = [0.001, 0.01, 0.05, 0.1]
iterations = 100
results = {}

for alpha in learning_rates:
    theta = np.zeros(2)
    history = [theta.copy()]
    
    for _ in range(iterations):
        gradients = (1/len(x)) * X.T @ (X @ theta - y)
        theta -= alpha * gradients
        history.append(theta.copy())
    
    results[alpha] = np.array(history)

# 可视化参数收敛过程
plt.figure(figsize=(12, 8))
for i, alpha in enumerate(learning_rates):
    history = results[alpha]
    plt.subplot(2, 2, i+1)
    plt.plot(history[:, 0], history[:, 1], 'b-', alpha=0.7)
    plt.scatter(2, 3, c='red', marker='x', s=100, label='真实值')
    plt.xlabel('θ₀')
    plt.ylabel('θ₁')
    plt.title(f'学习率 α={alpha}')
    plt.legend()

plt.tight_layout()
plt.show()
```

---

**上一节**：[机器学习基础](../01-fundamentals/04-evaluation.md)
**下一节**：[逻辑回归](02-logistic-regression.md)
