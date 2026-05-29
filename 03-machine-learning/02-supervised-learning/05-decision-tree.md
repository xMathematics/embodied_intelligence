# 2.5 决策树

## 目录

- [1. 决策树概述](#1-决策树概述)
- [2. 决策树构建](#2-决策树构建)
- [3. 分裂准则](#3-分裂准则)
- [4. 剪枝技术](#4-剪枝技术)
- [5. 决策树回归](#5-决策树回归)
- [6. 实践练习](#6-实践练习)

---

## 1. 决策树概述

### 1.1 定义

**决策树**（Decision Tree）是一种基于树结构进行决策的监督学习算法，通过递归划分特征空间来进行分类或回归。

### 1.2 决策树的组成

| 组成部分 | 说明 |
|---------|------|
| **根节点** | 整个树的起点，包含所有样本 |
| **内部节点** | 进行特征判断的节点 |
| **分支** | 特征判断的结果（是/否） |
| **叶节点** | 最终的分类或回归结果 |

### 1.3 决策树的优点

| 优点 | 说明 |
|------|------|
| **直观易懂** | 决策过程可视化，易于解释 |
| **无需预处理** | 不需要特征标准化 |
| **处理混合数据** | 同时处理数值和类别特征 |
| **可扩展性** | 容易集成到集成学习中 |

---

## 2. 决策树构建

### 2.1 构建流程

1. **选择最优特征**：选择能够最好区分样本的特征
2. **划分数据集**：根据特征值将数据集分成子集
3. **递归构建**：对每个子集重复上述过程
4. **停止条件**：达到停止条件时创建叶节点

### 2.2 停止条件

| 条件 | 说明 |
|------|------|
| **所有样本属于同一类别** | 纯度达到100% |
| **没有更多特征可选择** | 特征用完 |
| **样本数小于阈值** | 避免过拟合 |
| **树深度达到阈值** | 控制树的复杂度 |

---

## 3. 分裂准则

### 3.1 信息增益

**熵**（Entropy）：衡量数据的不确定性
$$H(S) = -\sum_{i=1}^c p_i \log_2(p_i)$$

**条件熵**：给定特征后的熵
$$H(S|A) = \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

**信息增益**：
$$IG(S, A) = H(S) - H(S|A)$$

### 3.2 信息增益比

$$IGR(S, A) = \frac{IG(S, A)}{H_A(S)}$$

其中 $H_A(S) = -\sum_{v \in Values(A)} \frac{|S_v|}{|S|} \log_2(\frac{|S_v|}{|S|})$

### 3.3 Gini不纯度

$$Gini(S) = 1 - \sum_{i=1}^c p_i^2$$

**Gini增益**：
$$Gini\_gain(S, A) = Gini(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Gini(S_v)$$

### 3.4 分裂准则对比

| 准则 | 特点 | 适用场景 |
|------|------|---------|
| **信息增益** | 倾向于选择取值多的特征 | 一般情况 |
| **信息增益比** | 对取值多的特征进行惩罚 | 特征取值不均衡 |
| **Gini不纯度** | 计算效率高 | 大规模数据 |

---

## 4. 剪枝技术

### 4.1 预剪枝（Pre-pruning）

在构建树的过程中提前停止，防止过拟合。

| 方法 | 说明 |
|------|------|
| **设置最大深度** | 限制树的深度 |
| **设置最小样本数** | 叶子节点最少样本数 |
| **设置最小信息增益** | 增益小于阈值则不分裂 |

### 4.2 后剪枝（Post-pruning）

先构建完整的树，然后从底部向上剪枝。

**代价复杂度剪枝**：
$$\alpha(T) = \frac{R(T) + \alpha |T|}{|T|}$$

其中：
- $R(T)$：树的错误率
- $|T|$：叶子节点数
- $\alpha$：正则化参数

### 4.3 剪枝效果

```
原始树（过拟合）:              剪枝后（泛化更好）:
    根节点                         根节点
   /    \                        /    \
  A      B                      A      B
 /|\    /|\                    /|\     |
C D E  F G H                  C D E   F
```

---

## 5. 决策树回归

### 5.1 回归树的构建

使用**均方误差**作为分裂准则：

$$MSE(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y}_S)^2$$

其中 $\bar{y}_S$ 是子集S的均值。

### 5.2 回归树的预测

叶节点的预测值为该节点所有样本的均值：
$$\hat{y} = \frac{1}{|S|} \sum_{i \in S} y_i$$

---

## 6. 实践练习

### 练习1：决策树分类

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据集
iris = load_iris()
X, y = iris.data[:, :2], iris.target  # 使用前两个特征

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建决策树模型
dt = DecisionTreeClassifier(max_depth=3, criterion='gini')
dt.fit(X_train, y_train)

# 预测
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"决策树准确率: {accuracy:.4f}")

# 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=iris.feature_names[:2], 
          class_names=iris.target_names, filled=True)
plt.title('决策树结构')
plt.show()

# 可视化决策边界
plt.figure(figsize=(10, 6))
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('决策树决策边界')
plt.show()
```

### 练习2：决策树回归

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成回归数据
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X) + np.random.randn(100, 1) * 0.2

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建决策树回归模型
dtr = DecisionTreeRegressor(max_depth=3)
dtr.fit(X_train, y_train)

# 预测
y_pred = dtr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"决策树回归MSE: {mse:.4f}")

# 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='数据点', alpha=0.6)
plt.plot(X, dtr.predict(X), 'r-', label='回归曲线')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('决策树回归')
plt.show()
```

### 练习3：剪枝效果对比

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 生成数据
X, y = make_classification(n_samples=200, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 测试不同深度的决策树
depths = [1, 2, 3, 5, 10, None]  # None表示不限制深度

print("决策树深度与准确率:")
print(f"{'深度':<6} {'训练准确率':<12} {'测试准确率':<12}")
print("-" * 35)

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    
    print(f"{str(depth):<6} {train_acc:<12.4f} {test_acc:<12.4f}")
```

---

**上一节**：[支持向量机](04-svm.md)
**下一节**：[无监督学习](../03-unsupervised-learning/01-clustering.md)
