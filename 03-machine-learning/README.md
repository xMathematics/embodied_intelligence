# 机器学习模块

## 模块概述

机器学习是具身智能的核心技术，本模块专注于经典机器学习算法，涵盖监督学习、无监督学习、特征工程、模型评估与选择等核心内容。

深度学习和强化学习已独立为单独模块：
- [深度学习模块](../04-deep-learning/README.md)
- [强化学习模块](../05-reinforcement-learning/README.md)

## 学习目标

完成本模块学习后，您将能够：
- 理解机器学习的基本概念和术语
- 掌握经典机器学习算法
- 理解模型评估与选择方法
- 能够使用Scikit-learn实现常见ML算法
- 掌握特征工程和数据预处理技巧
- 理解贝叶斯方法和概率建模

---

## 模块结构

```
03-machine-learning/
├── 01-fundamentals/              # 机器学习基础
├── 02-supervised-learning/       # 监督学习
├── 03-unsupervised-learning/     # 无监督学习
├── 04-feature-engineering/       # 特征工程
├── 05-model-evaluation/          # 模型评估与选择
├── 06-dimensionality-reduction/  # 降维方法
├── 07-bayesian-methods/          # 贝叶斯方法
├── 08-ensemble-learning/         # 集成学习
├── 09-applications/              # 应用领域
└── 10-paper-surveys/             # 论文详解
```

---

## 学习路径

### 第一部分：机器学习基础（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 1.1 | 机器学习概述 | ⭐⭐ |
| 1.2 | 监督学习基础 | ⭐⭐⭐ |
| 1.3 | 无监督学习基础 | ⭐⭐⭐ |
| 1.4 | 学习理论 | ⭐⭐⭐⭐ |

**章节链接**：
- [1.1 机器学习概述](01-fundamentals/01-introduction.md)
- [1.2 监督学习基础](01-fundamentals/02-supervised-learning.md)
- [1.3 无监督学习基础](01-fundamentals/03-unsupervised-learning.md)
- [1.4 学习理论](01-fundamentals/04-evaluation.md)

**核心内容**：
- 机器学习定义和分类
- 监督学习 vs 无监督学习 vs 强化学习
- 回归与分类问题
- 泛化能力与过拟合
- 偏差-方差权衡
- 计算学习理论

---

### 第二部分：监督学习（2周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 2.1 | 线性回归 | ⭐⭐⭐ |
| 2.2 | 逻辑回归 | ⭐⭐⭐ |
| 2.3 | 线性模型正则化 | ⭐⭐⭐⭐ |
| 2.4 | 支持向量机 | ⭐⭐⭐⭐ |
| 2.5 | 决策树 | ⭐⭐⭐ |

**核心内容**：
- 线性回归、最小二乘法
- 逻辑回归、Sigmoid函数
- Ridge、Lasso、Elastic Net
- SVM分类与回归
- 核方法、核函数
- 决策树构建与剪枝

---

### 第三部分：无监督学习（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 3.1 | 聚类算法 | ⭐⭐⭐ |
| 3.2 | 密度估计 | ⭐⭐⭐⭐ |
| 3.3 | 异常检测 | ⭐⭐⭐ |
| 3.4 | 关联规则 | ⭐⭐⭐ |

**核心内容**：
- K-Means、K-Medoids
- DBSCAN、OPTICS
- 层次聚类
- Gaussian Mixture Models
- 孤立森林、LOF
- Apriori算法

---

### 第四部分：特征工程（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 4.1 | 数据预处理 | ⭐⭐⭐ |
| 4.2 | 特征提取 | ⭐⭐⭐⭐ |
| 4.3 | 特征选择 | ⭐⭐⭐⭐ |
| 4.4 | 特征变换 | ⭐⭐⭐ |

**核心内容**：
- 缺失值处理
- 数据标准化/归一化
- 类别特征编码
- 文本特征提取
- 过滤式、包裹式、嵌入式选择
- PCA、ICA特征变换

---

### 第五部分：模型评估与选择（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 5.1 | 评估指标 | ⭐⭐⭐ |
| 5.2 | 交叉验证 | ⭐⭐⭐ |
| 5.3 | 超参数调优 | ⭐⭐⭐⭐ |
| 5.4 | 模型选择策略 | ⭐⭐⭐⭐ |

**核心内容**：
- 准确率、精确率、召回率、F1
- ROC曲线、AUC
- K-fold交叉验证
- Grid Search、Random Search
- Bayesian Optimization
- 模型集成策略

---

### 第六部分：降维方法（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 6.1 | PCA | ⭐⭐⭐ |
| 6.2 | 流形学习 | ⭐⭐⭐⭐ |
| 6.3 | 非线性降维 | ⭐⭐⭐⭐ |
| 6.4 | 可视化方法 | ⭐⭐⭐ |

**核心内容**：
- 主成分分析
- 奇异值分解
- t-SNE、UMAP
- Isomap、LLE
- 多维缩放
- 数据可视化

---

### 第七部分：贝叶斯方法（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 7.1 | 贝叶斯定理 | ⭐⭐⭐ |
| 7.2 | 贝叶斯分类器 | ⭐⭐⭐⭐ |
| 7.3 | 贝叶斯网络 | ⭐⭐⭐⭐⭐ |
| 7.4 | MCMC方法 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 贝叶斯推断
- Naive Bayes
- Bayesian Networks
- Markov Chain Monte Carlo
- Gibbs Sampling
- Variational Inference

---

### 第八部分：集成学习（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 8.1 | Bagging | ⭐⭐⭐ |
| 8.2 | Boosting | ⭐⭐⭐⭐ |
| 8.3 | Stacking | ⭐⭐⭐⭐ |
| 8.4 | 梯度提升树 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 随机森林
- AdaBoost
- Gradient Boosting
- XGBoost、LightGBM、CatBoost
- Stacking、Blending
- Ensemble策略

---

### 第九部分：应用领域（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 9.1 | 分类应用 | ⭐⭐⭐ |
| 9.2 | 回归应用 | ⭐⭐⭐ |
| 9.3 | 聚类应用 | ⭐⭐⭐ |
| 9.4 | 推荐系统 | ⭐⭐⭐⭐ |
| 9.5 | 具身智能应用 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 垃圾邮件分类
- 房价预测
- 客户分群
- 协同过滤
- 机器人感知

---

### 第十部分：论文详解（2周）

| 章节 | 论文 | 发表年份 |
|------|------|----------|
| 10.1 | "Random Forests" | 2001 |
| 10.2 | "Support Vector Machines" | 1995 |
| 10.3 | "XGBoost: A Scalable Tree Boosting System" | 2016 |
| 10.4 | "Gradient Boosting Machines" | 1999 |
| 10.5 | "Principal Component Analysis" | 1901 |
| 10.6 | "t-Distributed Stochastic Neighbor Embedding" | 2008 |
| 10.7 | "DBSCAN: Density-Based Spatial Clustering" | 1996 |
| 10.8 | "AdaBoost" | 1997 |
| 10.9 | "Naive Bayes Classifiers" | 1998 |
| 10.10 | "Bayesian Networks" | 1988 |

---

## 推荐学习资源

### 书籍
1. 《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》- Aurélien Géron
2. 《Pattern Recognition and Machine Learning》- Bishop
3. 《统计学习方法》- 李航
4. 《The Elements of Statistical Learning》- Hastie, Tibshirani, Friedman
5. 《Machine Learning: A Probabilistic Perspective》- Kevin Murphy

### 在线课程
1. **Coursera ML** - Andrew Ng
2. **CS229** - Stanford
3. **CS4780** - Cornell
4. **Machine Learning Specialization** - DeepLearning.AI

### 工具与框架
1. **Scikit-learn** - 经典ML算法库
2. **XGBoost** - 梯度提升树
3. **LightGBM** - 轻量级GBM
4. **CatBoost** - 类别特征处理
5. **Statsmodels** - 统计建模

---

## 学习评估

### 自测题
1. 解释监督学习、无监督学习和强化学习的区别
2. 什么是过拟合？如何防止过拟合？
3. 比较线性回归和逻辑回归的区别
4. 解释随机森林的原理
5. 什么是核方法？SVM如何使用核方法？
6. 解释PCA的原理和应用场景
7. 什么是偏差-方差权衡？
8. 比较Bagging和Boosting的区别

### 实践项目
1. 使用Scikit-learn实现经典ML算法
2. 使用XGBoost解决Kaggle竞赛
3. 实现一个完整的机器学习项目流程
4. 特征工程实践与模型调优
5. 使用贝叶斯方法进行概率预测

---

## 模块关系

```
具身智能
├── 03-machine-learning (机器学习) ← 当前模块
├── 04-deep-learning (深度学习) ← 后续模块
└── 05-reinforcement-learning (强化学习) ← 后续模块
```

**学习顺序建议**：
1. 先学习本模块（经典机器学习）
2. 再学习深度学习模块（04-deep-learning）
3. 最后学习强化学习模块（05-reinforcement-learning）

**前置知识**：
- 线性代数
- 概率论与统计学
- Python编程基础

---

## 重要论文列表

| 论文标题 | 作者 | 发表期刊/会议 | 年份 | 核心贡献 |
|----------|------|---------------|------|----------|
| Random Forests | Breiman | Machine Learning | 2001 | 集成学习 |
| Support Vector Machines | Cortes & Vapnik | Machine Learning | 1995 | 核方法 |
| XGBoost | Chen & Guestrin | KDD | 2016 | 梯度提升 |
| AdaBoost | Freund & Schapire | JCSS | 1997 | Boosting |
| t-SNE | van der Maaten & Hinton | JMLR | 2008 | 降维可视化 |
| DBSCAN | Ester et al. | KDD | 1996 | 密度聚类 |
| Principal Component Analysis | Pearson | Philosophical Magazine | 1901 | 降维 |
| Gradient Boosting | Friedman | Annals of Statistics | 2001 | 梯度提升 |
| Naive Bayes | Mitchell | Machine Learning | 1997 | 贝叶斯分类 |
| Bayesian Networks | Pearl | UAI | 1988 | 概率图模型 |

---

**上一个模块**：[SLAM与三维重建](../02-slam-3dreconstruction/README.md)
**下一个模块**：[深度学习](../04-deep-learning/README.md)