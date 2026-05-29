# 1.4 评估指标与模型选择

## 目录

- [1. 模型评估概述](#1-模型评估概述)
- [2. 交叉验证](#2-交叉验证)
- [3. 模型选择](#3-模型选择)
- [4. 超参数调优](#4-超参数调优)
- [5. 模型评估实战](#5-模型评估实战)
- [6. 实践练习](#6-实践练习)

---

## 1. 模型评估概述

### 1.1 为什么需要评估？

- 评估模型的泛化能力
- 比较不同模型的性能
- 选择最佳模型
- 检测过拟合和欠拟合

### 1.2 评估流程

```
数据划分 → 训练模型 → 预测 → 计算指标 → 分析结果
```

### 1.3 数据划分策略

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| Holdout | 划分训练集和测试集 | 数据量大 |
| K折交叉验证 | 分成K份，轮流作为测试集 | 数据量适中 |
| 留一法 | 每次留一个样本作为测试集 | 数据量小 |
| 分层采样 | 保持类别比例 | 不平衡数据 |

---

## 2. 交叉验证

### 2.1 K折交叉验证

**步骤**：
1. 将数据分成K个相等的部分
2. 对于每个部分i：
   - 使用其他K-1部分训练模型
   - 用第i部分测试模型
3. 计算K次测试结果的平均值

**优点**：
- 充分利用数据
- 结果更可靠
- 减少随机性

### 2.2 分层K折交叉验证

**适用场景**：不平衡数据集

**特点**：
- 保持每个折中的类别比例
- 避免某些折缺少特定类别

### 2.3 时间序列交叉验证

**适用场景**：时间序列数据

**特点**：
- 测试集在训练集之后
- 模拟真实预测场景

---

## 3. 模型选择

### 3.1 偏差-方差权衡

**偏差**：模型预测与真实值的平均偏差
**方差**：模型预测的波动程度

| 模型复杂度 | 偏差 | 方差 | 泛化误差 |
|-----------|------|------|---------|
| 简单模型 | 高 | 低 | 高（欠拟合） |
| 复杂模型 | 低 | 高 | 高（过拟合） |
| 适中模型 | 适中 | 适中 | 低 |

### 3.2 模型选择准则

**AIC（Akaike信息准则）**：
$$AIC = 2k - 2\ln(L)$$

**BIC（Bayesian信息准则）**：
$$BIC = k\ln(n) - 2\ln(L)$$

其中：
- k：模型参数数量
- L：似然函数值
- n：样本数量

### 3.3 正则化与模型选择

**正则化的作用**：
- 控制模型复杂度
- 防止过拟合
- 提高泛化能力

**常见正则化方法**：
- L1正则化（Lasso）
- L2正则化（Ridge）
- Dropout（深度学习）

---

## 4. 超参数调优

### 4.1 网格搜索

**思想**：遍历所有可能的超参数组合

**步骤**：
1. 定义超参数网格
2. 对于每个组合，训练模型并评估
3. 选择性能最好的组合

**缺点**：计算量大

### 4.2 随机搜索

**思想**：随机采样超参数组合

**优点**：
- 比网格搜索更快
- 可能找到更好的组合

### 4.3 贝叶斯优化

**思想**：使用贝叶斯方法建模超参数与性能的关系

**步骤**：
1. 定义目标函数
2. 选择初始超参数
3. 训练模型并记录性能
4. 更新代理模型
5. 选择下一组超参数

**优点**：
- 高效利用历史信息
- 找到最优解的概率更高

---

## 5. 模型评估实战

### 5.1 回归模型评估

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

# 加载数据
data = load_diabetes()
X, y = data.data, data.target

# 创建模型
model = LinearRegression()

# 交叉验证
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores

print(f"交叉验证MSE: {mse_scores}")
print(f"平均MSE: {mse_scores.mean():.2f}")
print(f"标准差: {mse_scores.std():.2f}")
```

### 5.2 分类模型评估

```python
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# 创建模型
model = LogisticRegression(max_iter=200)

# 交叉验证预测
y_pred = cross_val_predict(model, X, y, cv=5)

# 评估
print("混淆矩阵:")
print(confusion_matrix(y, y_pred))
print("\n分类报告:")
print(classification_report(y, y_pred))
```

### 5.3 超参数调优示例

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_digits

# 加载数据
data = load_digits()
X, y = data.data, data.target

# 定义超参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# 创建模型
svc = SVC()

# 网格搜索
grid = GridSearchCV(svc, param_grid, refit=True, verbose=2, cv=5)
grid.fit(X, y)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳得分: {grid.best_score_:.2f}")
```

---

## 6. 实践练习

### 练习1：交叉验证比较模型

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes

# 加载数据
data = load_diabetes()
X, y = data.data, data.target

# 定义模型列表
models = [
    ('Ridge', Ridge()),
    ('Lasso', Lasso()),
    ('Decision Tree', DecisionTreeRegressor()),
    ('Random Forest', RandomForestRegressor())
]

# 比较模型
results = []
names = []
for name, model in models:
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    results.append(-cv_scores)
    names.append(name)
    print(f"{name}: {(-cv_scores.mean()):.2f} (+/- {cv_scores.std():.2f})")

# 可视化结果
plt.boxplot(results, labels=names)
plt.title('Model Comparison')
plt.ylabel('MSE')
plt.show()
```

### 练习2：贝叶斯优化

```python
from bayes_opt import BayesianOptimization
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# 定义目标函数
def svc_cv(C, gamma, data, targets):
    model = SVC(C=C, gamma=gamma, kernel='rbf')
    scores = cross_val_score(model, data, targets, cv=5)
    return scores.mean()

# 加载数据
data = load_digits()
X, y = data.data, data.target

# 定义参数边界
pbounds = {'C': (0.1, 100), 'gamma': (0.001, 1)}

# 创建优化器
optimizer = BayesianOptimization(
    f=lambda C, gamma: svc_cv(C, gamma, X, y),
    pbounds=pbounds,
    random_state=42
)

# 运行优化
optimizer.maximize(init_points=5, n_iter=25)

print(f"最佳参数: {optimizer.max['params']}")
print(f"最佳得分: {optimizer.max['target']:.2f}")
```

---

**下一步**：[经典机器学习算法](../02-classical-algorithms/01-linear-models.md)