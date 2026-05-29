
# 机器学习基础模块

## 模块概述

机器学习是人工智能的核心，本模块涵盖监督学习、无监督学习、评估方法等基础内容。

## 学习目标

完成本模块学习后，您将能够：
- 理解机器学习的基本概念
- 掌握常用机器学习算法
- 能够评估和选择合适的模型

## 学习路径

### 第一部分：监督学习（2周）

#### 1.1 回归问题

**核心概念：**
- 线性回归
- 多项式回归
- 正则化（L1、L2）

**推荐资源：**
- 《Machine Learning》- Mitchell
- 《Pattern Recognition and Machine Learning》- Bishop

**实践练习：**
- 实现线性回归
- 使用正则化防止过拟合

#### 1.2 分类问题

**核心概念：**
- 逻辑回归
- 支持向量机（SVM）
- 决策树与随机森林

**推荐论文：**
- "Support-Vector Networks" - Cortes & Vapnik
- "Random Forests" - Breiman

**实践练习：**
- 实现逻辑回归分类器
- 使用SVM进行分类

---

### 第二部分：无监督学习（1周）

#### 2.1 聚类算法

**核心概念：**
- K-Means
- 层次聚类
- DBSCAN

**推荐论文：**
- "DBSCAN: A Density-Based Algorithm for Discovering Clusters"

**实践练习：**
- 实现K-Means聚类
- 使用DBSCAN处理空间数据

#### 2.2 降维方法

**核心概念：**
- 主成分分析（PCA）
- 线性判别分析（LDA）
- t-SNE

**实践练习：**
- 实现PCA降维
- 使用t-SNE进行可视化

---

### 第三部分：模型评估与选择（1周）

#### 3.1 评估指标

**核心概念：**
- 准确率、精确率、召回率
- F1分数
- ROC曲线与AUC

**实践练习：**
- 计算分类模型的各项指标
- 绘制ROC曲线

#### 3.2 模型选择

**核心概念：**
- 交叉验证
- 网格搜索
- 偏差-方差权衡

**实践练习：**
- 使用交叉验证选择模型参数

---

## 推荐学习资源

### 书籍
1. 《Machine Learning: A Probabilistic Perspective》- Murphy
2. 《Pattern Recognition and Machine Learning》- Bishop
3. 《Introduction to Machine Learning with Python》- Muller & Guido

### 在线课程
1. Coursera - Machine Learning (Andrew Ng)
2. edX - Machine Learning Fundamentals

### 工具
- scikit-learn - 机器学习库
- pandas - 数据处理
- matplotlib - 数据可视化

## 学习评估

### 自测题
1. 解释偏差-方差权衡
2. 比较SVM与逻辑回归的优缺点
3. 解释交叉验证的作用

### 实践项目
1. 使用scikit-learn完成分类任务
2. 对MNIST数据集进行降维和可视化
3. 使用网格搜索优化模型参数

---

**下一个模块：** [深度学习](../04-deep-learning/README.md)
