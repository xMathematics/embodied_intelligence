# 1.4 学习理论

## 目录

- [1. 学习理论概述](#1-学习理论概述)
- [2. 偏差-方差权衡](#2-偏差-方差权衡)
- [3. 泛化能力与过拟合](#3-泛化能力与过拟合)
- [4. VC维理论](#4-vc维理论)
- [5. 正则化理论](#5-正则化理论)
- [6. 模型选择与评估](#6-模型选择与评估)
- [7. 实践练习](#7-实践练习)

---

## 1. 学习理论概述

### 1.1 定义

**学习理论**（Learning Theory）是机器学习的理论基础，研究机器学习算法的学习能力、泛化能力和样本复杂度。

### 1.2 核心问题

学习理论试图回答以下问题：
- **学习是否可能**：在什么条件下可以从有限样本中学习到泛化规则？
- **需要多少样本**：学习一个好的模型需要多少训练数据？
- **如何选择模型**：如何在模型复杂度和泛化能力之间权衡？

### 1.3 学习理论的分类

| 理论类型 | 研究内容 | 核心方法 |
|---------|---------|---------|
| **计算学习理论** | 学习的可行性和样本复杂度 | PAC学习、VC维 |
| **统计学习理论** | 泛化误差的界 | 经验风险最小化 |
| **在线学习理论** | 序列预测的错误界 | 后悔分析 |

---

## 2. 偏差-方差权衡

### 2.1 误差分解

**期望预测误差**可以分解为三个部分：

$$\text{EPE}(f) = \text{Bias}^2(f) + \text{Variance}(f) + \text{Noise}$$

其中：
- **Bias**：模型的系统性误差
- **Variance**：模型的随机性误差
- **Noise**：数据本身的噪声（不可约误差）

### 2.2 偏差与方差的定义

**偏差**（Bias）：
$$\text{Bias}(f) = \mathbb{E}[f(x)] - y$$
衡量模型的平均预测与真实值之间的差距。

**方差**（Variance）：
$$\text{Variance}(f) = \mathbb{E}[(f(x) - \mathbb{E}[f(x)])^2]$$
衡量模型预测的波动性。

### 2.3 模型复杂度与偏差-方差的关系

| 模型复杂度 | 偏差 | 方差 | 泛化能力 |
|-----------|------|------|---------|
| **低**（简单模型） | 高 | 低 | 欠拟合 |
| **适中** | 适中 | 适中 | 最优 |
| **高**（复杂模型） | 低 | 高 | 过拟合 |

### 2.4 可视化理解

```
误差
  |
高|        *         *
  |      *   *     *   *
  |    *       * *       *
  |  *           *           *
低|-------------------------------
    低      适中      高
      模型复杂度
```

---

## 3. 泛化能力与过拟合

### 3.1 泛化能力

**泛化能力**（Generalization Ability）指模型对未见过的数据的预测能力。

**泛化误差**：
$$\text{Generalization Error} = \mathbb{E}_{(X,Y) \sim P} [L(Y, f(X))]$$

### 3.2 过拟合与欠拟合

**过拟合**（Overfitting）：
- 模型在训练集上表现很好，但在测试集上表现很差
- 原因：模型过于复杂，学习了训练数据中的噪声

**欠拟合**（Underfitting）：
- 模型在训练集和测试集上表现都很差
- 原因：模型过于简单，无法捕捉数据的真实模式

### 3.3 过拟合的检测

| 检测方法 | 描述 |
|---------|------|
| **训练误差 vs 测试误差** | 训练误差远小于测试误差 |
| **交叉验证** | 验证集误差明显上升 |
| **学习曲线** | 训练误差继续下降，验证误差上升 |

---

## 4. VC维理论

### 4.1 定义

**VC维**（Vapnik-Chervonenkis Dimension）是衡量模型表达能力的指标。

**定义**：一个假设空间H的VC维是能够被H打散（shatter）的最大样本集大小。

**打散**：对于一个样本集S，如果假设空间H能够实现S的所有可能的二分类，则称H打散S。

### 4.2 常见模型的VC维

| 模型 | VC维 |
|------|------|
| 线性分类器（d维） | d+1 |
| 多项式分类器（d维，k次） | $\binom{d+k}{k}$ |
| 决策树（深度h） | $O(h \cdot 2^h)$ |
| 神经网络（L层，每层n个神经元） | $O(L \cdot n^2)$ |

### 4.3 VC维与泛化误差界

**VC界**：
$$\mathbb{P}(|\hat{R}(f) - R(f)| > \epsilon) \leq 4 \cdot \exp\left(-\frac{\epsilon^2 m}{8 \cdot \text{VC}(H)}\right)$$

其中：
- $\hat{R}(f)$：经验风险
- $R(f)$：真实风险
- $m$：样本数量
- $\text{VC}(H)$：假设空间的VC维

**结论**：模型的VC维越大，需要的样本数越多才能保证泛化能力。

---

## 5. 正则化理论

### 5.1 正则化的作用

**正则化**（Regularization）是通过引入额外约束来防止过拟合的技术。

### 5.2 常见正则化方法

#### L2正则化（Ridge）
$$\text{Loss} = \sum_{i=1}^m L(y_i, \hat{y}_i) + \lambda \sum_{j=1}^d \theta_j^2$$

**效果**：使参数值变小，防止某些特征对预测产生过大影响。

#### L1正则化（Lasso）
$$\text{Loss} = \sum_{i=1}^m L(y_i, \hat{y}_i) + \lambda \sum_{j=1}^d |\theta_j|$$

**效果**：产生稀疏解，自动进行特征选择。

#### Elastic Net
$$\text{Loss} = \sum_{i=1}^m L(y_i, \hat{y}_i) + \lambda_1 \sum_{j=1}^d |\theta_j| + \lambda_2 \sum_{j=1}^d \theta_j^2$$

**效果**：结合L1和L2的优点。

#### Dropout
**思想**：训练时随机丢弃部分神经元。
**效果**：防止神经元之间的共适应，增强泛化能力。

### 5.3 正则化与偏差-方差的关系

| 正则化强度 | 偏差 | 方差 | 效果 |
|-----------|------|------|------|
| **弱** | 低 | 高 | 过拟合 |
| **适中** | 适中 | 适中 | 最优 |
| **强** | 高 | 低 | 欠拟合 |

---

## 6. 模型选择与评估

### 6.1 数据集划分

| 数据集 | 用途 | 占比 |
|--------|------|------|
| **训练集** | 训练模型参数 | 60%-80% |
| **验证集** | 调整超参数 | 10%-20% |
| **测试集** | 评估最终模型 | 10%-20% |

### 6.2 交叉验证

**K折交叉验证**：
1. 将数据分成K个折叠
2. 轮流用K-1个折叠训练，1个折叠验证
3. 最终性能取K次验证的平均值

**优点**：
- 充分利用数据
- 减少随机性
- 更可靠的评估

### 6.3 超参数调优

| 方法 | 描述 | 复杂度 |
|------|------|--------|
| **网格搜索** | 穷举所有参数组合 | O(n^k) |
| **随机搜索** | 随机采样参数组合 | O(n) |
| **贝叶斯优化** | 基于概率模型选择参数 | 自适应 |

### 6.4 模型评估指标

#### 分类任务
| 指标 | 公式 | 适用场景 |
|------|------|---------|
| **准确率** | $(TP+TN)/(TP+TN+FP+FN)$ | 类别平衡 |
| **F1分数** | $2 \cdot P \cdot R / (P + R)$ | 类别不平衡 |
| **AUC** | ROC曲线下面积 | 概率输出 |

#### 回归任务
| 指标 | 公式 | 适用场景 |
|------|------|---------|
| **RMSE** | $\sqrt{\text{MSE}}$ | 一般回归 |
| **MAE** | 平均绝对误差 | 存在异常值 |
| **R²** | 决定系数 | 解释方差 |

---

## 7. 实践练习

### 练习1：偏差-方差分解

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 生成真实数据
np.random.seed(42)
x = np.linspace(0, 1, 100)
y_true = np.sin(2 * np.pi * x)
y_noisy = y_true + np.random.normal(0, 0.2, size=len(x))

# 转换为二维数组
X = x.reshape(-1, 1)

# 定义模型复杂度
degrees = [1, 3, 10, 20]

plt.figure(figsize=(12, 8))

for i, degree in enumerate(degrees):
    # 创建多项式回归模型
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    # 多次训练（模拟不同训练集）
    predictions = []
    for _ in range(50):
        # 随机选择50个样本
        idx = np.random.choice(len(X), 50, replace=False)
        X_train, y_train = X[idx], y_noisy[idx]
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X)
        predictions.append(y_pred)
    
    # 计算偏差和方差
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    variance = predictions.var(axis=0)
    
    # 可视化
    plt.subplot(2, 2, i+1)
    plt.scatter(X, y_noisy, s=10, label='训练数据', alpha=0.5)
    plt.plot(x, y_true, 'g--', label='真实函数')
    plt.plot(x, mean_pred, 'r-', label='平均预测')
    plt.fill_between(x, mean_pred - np.sqrt(variance), 
                     mean_pred + np.sqrt(variance), 
                     alpha=0.2, color='orange', label='方差范围')
    plt.title(f'多项式次数 = {degree}')
    plt.legend()
    plt.ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()

# 计算整体偏差和方差
print("模型复杂度分析:")
print(f"{'次数':<6} {'偏差²':<10} {'方差':<10} {'总误差':<10}")
print("-" * 40)

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    
    predictions = []
    for _ in range(100):
        idx = np.random.choice(len(X), 50, replace=False)
        model.fit(X[idx], y_noisy[idx])
        predictions.append(model.predict(X))
    
    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    
    bias_squared = np.mean((mean_pred - y_true) ** 2)
    variance = np.mean(predictions.var(axis=0))
    total_error = bias_squared + variance + np.var(y_noisy - y_true)
    
    print(f"{degree:<6} {bias_squared:<10.4f} {variance:<10.4f} {total_error:<10.4f}")
```

### 练习2：交叉验证与模型选择

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 加载数据集
data = load_diabetes()
X, y = data.data, data.target

# 定义模型
models = {
    '线性回归': LinearRegression(),
    'Ridge回归': Ridge(alpha=1.0),
    'Lasso回归': Lasso(alpha=0.1),
    '决策树': DecisionTreeRegressor(max_depth=3),
    '随机森林': RandomForestRegressor(n_estimators=100)
}

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("模型交叉验证结果:")
print(f"{'模型':<12} {'平均R²':<10} {'标准差':<10}")
print("-" * 35)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    mean_score = scores.mean()
    std_score = scores.std()
    print(f"{name:<12} {mean_score:<10.4f} {std_score:<10.4f}")
```

### 练习3：正则化效果对比

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# 生成高维数据（特征数大于样本数）
X, y = make_regression(n_samples=50, n_features=100, noise=0.1, random_state=42)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 测试不同正则化强度
alphas = np.logspace(-4, 2, 20)
ridge_mse = []
lasso_mse = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_mse.append(mean_squared_error(y_test, ridge.predict(X_test)))
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    lasso_mse.append(mean_squared_error(y_test, lasso.predict(X_test)))

# 可视化结果
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, ridge_mse, label='Ridge回归', marker='o')
plt.semilogx(alphas, lasso_mse, label='Lasso回归', marker='s')
plt.xlabel('正则化参数 α')
plt.ylabel('测试MSE')
plt.title('正则化强度对模型性能的影响')
plt.legend()
plt.grid(True)
plt.show()

# 查看Lasso的特征选择效果
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
non_zero_coefs = np.sum(lasso.coef_ != 0)
print(f"Lasso回归（α=0.1）保留的特征数: {non_zero_coefs}/{len(lasso.coef_)}")
```

---

**下一步**：[机器学习进阶](02-advanced/README.md)
