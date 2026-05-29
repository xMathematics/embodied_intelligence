# 1.1 机器学习概述

## 目录

- [1. 什么是机器学习？](#1-什么是机器学习)
- [2. 机器学习的分类](#2-机器学习的分类)
- [3. 机器学习的发展历程](#3-机器学习的发展历程)
- [4. 机器学习的应用场景](#4-机器学习的应用场景)
- [5. 关键概念](#5-关键概念)
- [6. 实践练习](#6-实践练习)

---

## 1. 什么是机器学习？

### 1.1 定义

**机器学习**：使计算机系统能够从数据中自动学习和改进，而无需进行显式编程。

**核心思想**：
- 从数据中学习模式
- 通过经验自动改进
- 无需人工编写规则

### 1.2 与传统编程的区别

| 传统编程 | 机器学习 |
|---------|---------|
| 人工编写规则 | 从数据中学习规则 |
| 适用于简单问题 | 适用于复杂问题 |
| 需要领域专家 | 需要数据专家 |
| 难以处理不确定性 | 擅长处理不确定性 |

### 1.3 学习过程

```
数据收集 → 特征提取 → 模型训练 → 模型评估 → 部署应用
```

---

## 2. 机器学习的分类

### 2.1 按学习方式分类

**监督学习**：
- 训练数据包含标签
- 学习输入到输出的映射
- 任务：分类、回归

**无监督学习**：
- 训练数据没有标签
- 发现数据中的结构
- 任务：聚类、降维

**强化学习**：
- 通过与环境交互学习
- 最大化累积奖励
- 任务：决策、控制

### 2.2 按模型类型分类

**参数模型**：
- 模型参数数量固定
- 训练速度快
- 例如：线性回归、逻辑回归

**非参数模型**：
- 模型参数随数据增长
- 表达能力强
- 例如：决策树、K近邻

### 2.3 按学习目标分类

**分类**：预测离散标签
**回归**：预测连续值
**聚类**：将相似数据分组
**降维**：减少特征维度

---

## 3. 机器学习的发展历程

### 3.1 早期阶段（1950-1980）

- **1950**：图灵测试
- **1956**：达特茅斯会议，AI诞生
- **1957**：Rosenblatt提出感知机
- **1969**：Minsky证明感知机局限性

### 3.2 机器学习兴起（1980-2010）

- **1986**：反向传播算法
- **1995**：支持向量机
- **1997**：决策树、随机森林
- **2006**：深度学习复兴（Hinton）

### 3.3 深度学习时代（2010-至今）

- **2012**：AlexNet赢得ImageNet
- **2014**：GAN、ResNet
- **2017**：Transformer
- **2018**：BERT、GPT
- **2020**：GPT-3、扩散模型
- **2023**：GPT-4、大语言模型

---

## 4. 机器学习的应用场景

### 4.1 计算机视觉

- 图像分类
- 目标检测
- 图像分割
- 人脸识别

### 4.2 自然语言处理

- 文本分类
- 机器翻译
- 问答系统
- 情感分析

### 4.3 推荐系统

- 商品推荐
- 内容推荐
- 个性化推荐

### 4.4 语音识别

- 语音转文字
- 语音合成
- 声纹识别

### 4.5 医疗诊断

- 疾病预测
- 医学影像分析
- 药物发现

---

## 5. 关键概念

### 5.1 特征

**特征**：数据的可测量属性

**特征工程**：
- 选择有用特征
- 转换特征
- 创建新特征

### 5.2 模型

**模型**：学习到的函数或规则

**模型选择**：
- 根据任务选择合适模型
- 考虑复杂度和性能

### 5.3 训练与测试

**训练集**：用于训练模型
**验证集**：用于调整超参数
**测试集**：用于评估最终性能

**数据划分**：
- 常见比例：70%训练，20%验证，10%测试

### 5.4 过拟合与欠拟合

**过拟合**：模型在训练集上表现好，但泛化能力差
**欠拟合**：模型太简单，无法捕捉数据模式

**解决方法**：
- 过拟合：增加数据、正则化、减少模型复杂度
- 欠拟合：增加模型复杂度、添加特征

---

## 6. 实践练习

### 练习1：理解机器学习流程

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 生成模拟数据
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 3 + np.random.randn(100, 1) * 0.5

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
print(f"系数: {model.coef_[0][0]:.2f}")
print(f"截距: {model.intercept_[0]:.2f}")
print(f"均方误差: {mse:.4f}")
```

### 练习2：可视化过拟合与欠拟合

```python
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# 生成非线性数据
np.random.seed(0)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X) + np.random.randn(100, 1) * 0.1

# 创建不同复杂度的模型
models = [
    Pipeline([('poly', PolynomialFeatures(degree=1)), ('linear', LinearRegression())]),
    Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression())]),
    Pipeline([('poly', PolynomialFeatures(degree=15)), ('linear', LinearRegression())])
]

labels = ['欠拟合 (degree=1)', '合适 (degree=3)', '过拟合 (degree=15)']

plt.figure(figsize=(12, 4))
for i, (model, label) in enumerate(zip(models, labels)):
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.subplot(1, 3, i+1)
    plt.scatter(X, y, s=10, label='数据')
    plt.plot(X, y_pred, 'r-', label=label)
    plt.legend()
    plt.title(label)

plt.tight_layout()
plt.show()
```

---

**下一步**：[监督学习基础](02-supervised-learning.md)