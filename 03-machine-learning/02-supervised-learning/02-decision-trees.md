# 2.2 决策树与集成学习

## 目录

- [1. 决策树](#1-决策树)
- [2. 集成学习概述](#2-集成学习概述)
- [3. Bagging与随机森林](#3-bagging与随机森林)
- [4. Boosting](#4-boosting)
- [5. XGBoost与LightGBM](#5-xgboost与lightgbm)
- [6. 实践练习](#6-实践练习)

---

## 1. 决策树

### 1.1 模型定义

**决策树**：通过递归划分特征空间进行预测的树形结构。

**组成部分**：
- **根节点**：整个数据集
- **内部节点**：特征测试
- **分支**：测试结果
- **叶节点**：预测结果

### 1.2 决策树构建

**分裂准则**：

| 准则 | 公式 | 适用场景 |
|------|------|---------|
| 信息增益 | $IG = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$ | 分类 |
| 信息增益比 | $IGR = \frac{IG}{H(S_v)}$ | 分类 |
| Gini指数 | $Gini = 1 - \sum p_i^2$ | 分类 |
| 方差减少 | $\text{Reduction} = Var(S) - \sum \frac{|S_v|}{|S|} Var(S_v)$ | 回归 |

**信息熵**：
$$H(S) = -\sum_{i=1}^{k} p_i \log_2(p_i)$$

### 1.3 剪枝策略

**预剪枝**：在构建过程中提前停止
- 限制树深度
- 限制叶节点数量
- 设置最小样本数

**后剪枝**：构建完整树后剪枝
- 代价复杂度剪枝
- 最小描述长度剪枝

### 1.4 实践示例

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print(f"准确率: {accuracy_score(y_test, y_pred):.2f}")

# 可视化
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
```

---

## 2. 集成学习概述

### 2.1 什么是集成学习？

**集成学习**：将多个模型组合起来，提高整体性能。

**核心思想**：
- 三个臭皮匠，顶个诸葛亮
- 组合多个弱学习器形成强学习器

### 2.2 集成方法分类

| 方法 | 特点 | 示例 |
|------|------|------|
| Bagging | 并行训练，降低方差 | 随机森林 |
| Boosting | 串行训练，降低偏差 | AdaBoost, XGBoost |
| Stacking | 元学习器组合 | StackingClassifier |

### 2.3 Bias-Variance分解

**集成效果**：
$$E[(y - \hat{f}(x))^2] = Bias^2 + Variance + Noise$$

- Bagging降低方差
- Boosting降低偏差

---

## 3. Bagging与随机森林

### 3.1 Bagging

**Bootstrap Aggregating**：
1. 从训练集随机采样（有放回）
2. 训练多个模型
3. 多数投票（分类）或平均（回归）

**优点**：
- 降低方差
- 减少过拟合
- 并行训练

### 3.2 随机森林

**扩展的Bagging**：
- 随机选择特征子集
- 增加多样性
- 进一步降低方差

**特征重要性**：
$$\text{Importance}(f) = \frac{\text{总Gini减少}}{使用f的树数量}$$

### 3.3 实践示例

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score

# 加载数据
digits = load_digits()
X, y = digits.data, digits.target

# 创建模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 交叉验证
scores = cross_val_score(model, X, y, cv=5)
print(f"准确率: {scores.mean():.2f} (+/- {scores.std():.2f})")

# 特征重要性
importances = model.fit(X, y).feature_importances_
plt.bar(range(len(importances)), importances)
plt.title('Feature Importances')
plt.show()
```

---

## 4. Boosting

### 4.1 AdaBoost

**Adaptive Boosting**：
1. 初始化样本权重
2. 训练弱分类器
3. 更新权重（错分样本权重增加）
4. 组合分类器

**权重更新**：
$$w_{i+1} = w_i \exp(\alpha_i I(y_i \neq \hat{y}_i))$$

**分类器权重**：
$$\alpha_i = \frac{1}{2} \log\left(\frac{1 - \epsilon_i}{\epsilon_i}\right)$$

### 4.2 Gradient Boosting

**梯度提升**：
1. 拟合初始模型
2. 计算残差
3. 拟合残差的模型
4. 更新模型

**损失函数梯度**：
$$r_{im} = -\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}$$

### 4.3 实践示例

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成数据
X, y = make_classification(n_samples=500, n_features=20, random_state=42)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建模型
ada = AdaBoostClassifier(n_estimators=50, random_state=42)
gb = GradientBoostingClassifier(n_estimators=50, random_state=42)

# 训练和评估
ada.fit(X_train, y_train)
gb.fit(X_train, y_train)

print(f"AdaBoost准确率: {ada.score(X_test, y_test):.2f}")
print(f"Gradient Boosting准确率: {gb.score(X_test, y_test):.2f}")
```

---

## 5. XGBoost与LightGBM

### 5.1 XGBoost

**Extreme Gradient Boosting**：
- 正则化项
- 并行计算
- 缺失值处理
- 自定义损失函数

**目标函数**：
$$\mathcal{L}(\phi) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

**正则化项**：
$$\Omega(f) = \gamma T + \frac{1}{2} \lambda \|w\|^2$$

### 5.2 LightGBM

**Light Gradient Boosting**：
- 直方图优化
- 按叶子生长
- 并行优化

**优点**：
- 训练速度快
- 内存占用小
- 准确率高

### 5.3 实践示例

```python
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 加载数据
data = load_diabetes()
X, y = data.data, data.target

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
xgb_model = xgb.train(params, dtrain, num_boost_round=100)
y_pred_xgb = xgb_model.predict(dtest)

# LightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {'objective': 'regression', 'metric': 'rmse'}
lgb_model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_eval)
y_pred_lgb = lgb_model.predict(X_test)

# 评估
print(f"XGBoost RMSE: {mean_squared_error(y_test, y_pred_xgb, squared=False):.2f}")
print(f"LightGBM RMSE: {mean_squared_error(y_test, y_pred_lgb, squared=False):.2f}")
```

---

## 6. 实践练习

### 练习1：泰坦尼克号生存预测

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# 加载数据
train = pd.read_csv('train.csv')

# 数据预处理
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna('S', inplace=True)

le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
train['Embarked'] = le.fit_transform(train['Embarked'])

# 选择特征
X = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train['Survived']

# 创建模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 交叉验证
scores = cross_val_score(model, X, y, cv=5)
print(f"准确率: {scores.mean():.2f} (+/- {scores.std():.2f})")
```

### 练习2：模型对比

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.datasets import load_breast_cancer

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# 定义模型
models = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('AdaBoost', AdaBoostClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier())
]

# 比较模型
for name, model in models:
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.2f} (+/- {scores.std():.2f})")
```

---

**下一步**：[支持向量机](03-svm.md)