# 1.3 无监督学习基础

## 目录

- [1. 无监督学习概述](#1-无监督学习概述)
- [2. 聚类算法](#2-聚类算法)
- [3. 降维算法](#3-降维算法)
- [4. 密度估计](#4-密度估计)
- [5. 实践练习](#5-实践练习)

---

## 1. 无监督学习概述

### 1.1 定义

**无监督学习**：从未标注数据中发现结构和模式。

**目标**：
- 发现数据的内在结构
- 识别数据中的模式
- 提取有用特征

### 1.2 与监督学习的区别

| 方面 | 监督学习 | 无监督学习 |
|------|---------|-----------|
| 数据标注 | 有标签 | 无标签 |
| 学习目标 | 预测映射 | 发现结构 |
| 评估方式 | 预测准确率 | 聚类质量 |
| 应用场景 | 分类、回归 | 聚类、降维 |

### 1.3 主要任务

1. **聚类**：将相似数据分组
2. **降维**：减少特征维度
3. **密度估计**：估计数据分布
4. **异常检测**：识别异常数据

---

## 2. 聚类算法

### 2.1 K-Means聚类

**核心思想**：将数据分为K个簇，使簇内相似度最大，簇间相似度最小。

**算法流程**：
1. 随机选择K个质心
2. 将每个数据点分配到最近的质心
3. 更新质心为簇内所有点的均值
4. 重复步骤2-3直到收敛

**目标函数**：
$$J = \sum_{i=1}^{n} \sum_{k=1}^{K} w_{ik} \|x_i - \mu_k\|^2$$

其中 $w_{ik}=1$ 如果 $x_i$ 属于簇k，否则为0。

**优缺点**：
| 优点 | 缺点 |
|------|------|
| 简单高效 | 需要预先指定K |
| 易于实现 | 对初始质心敏感 |
| 适用于大型数据 | 对异常值敏感 |

### 2.2 DBSCAN

**核心思想**：基于密度的聚类，发现任意形状的簇。

**关键概念**：
- $\epsilon$：邻域半径
- MinPts：邻域内最小点数
- 核心点：邻域内点数≥MinPts的点
- 边界点：邻域内点数<MinPts但在核心点邻域内的点
- 噪声点：既不是核心点也不是边界点的点

**算法流程**：
1. 随机选择一个未访问的点
2. 如果是核心点，开始一个新簇
3. 递归添加所有密度可达的点
4. 重复直到所有点都被访问

**优缺点**：
| 优点 | 缺点 |
|------|------|
| 发现任意形状簇 | 对参数敏感 |
| 自动发现簇数量 | 高维数据效果差 |
| 识别噪声点 | 密度不均匀时效果差 |

### 2.3 层次聚类

**核心思想**：构建层次化的簇结构。

**凝聚式聚类**：
1. 每个点作为一个簇
2. 合并最相似的两个簇
3. 重复直到所有点合并为一个簇

**分裂式聚类**：
1. 所有点作为一个簇
2. 分裂最不相似的簇
3. 重复直到每个点都是一个簇

**相似度度量**：
- 单链接：最小距离
- 完全链接：最大距离
- 平均链接：平均距离

**优点**：
- 不需要指定簇数量
- 提供层次结构

---

## 3. 降维算法

### 3.1 主成分分析（PCA）

**核心思想**：找到数据的主成分，将数据投影到低维空间。

**算法流程**：
1. 标准化数据
2. 计算协方差矩阵
3. 计算特征值和特征向量
4. 选择前k个特征向量
5. 将数据投影到新空间

**数学原理**：
$$PCA(X) = X W$$

其中W是包含前k个特征向量的矩阵。

**应用场景**：
- 数据可视化
- 特征提取
- 去噪

### 3.2 t-SNE

**核心思想**：在低维空间中保持数据点之间的相似性。

**与PCA的区别**：
| 方面 | PCA | t-SNE |
|------|------|-------|
| 线性/非线性 | 线性 | 非线性 |
| 保留结构 | 全局结构 | 局部结构 |
| 计算复杂度 | O(n²) | O(n²) |
| 适用维度 | 任意 | 主要用于可视化（2-3维） |

**算法流程**：
1. 计算高维空间中的相似度
2. 初始化低维嵌入
3. 优化KL散度

### 3.3 自动编码器

**核心思想**：使用神经网络学习数据的低维表示。

**架构**：
- 编码器：将高维数据编码为低维表示
- 解码器：将低维表示解码为原始数据

**损失函数**：
$$L = \|x - \hat{x}\|^2$$

**优点**：
- 可以学习非线性特征
- 自动学习特征表示

---

## 4. 密度估计

### 4.1 核密度估计（KDE）

**核心思想**：使用核函数估计概率密度。

**公式**：
$$\hat{f}(x) = \frac{1}{n h} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

其中：
- $K$：核函数
- $h$：带宽

**核函数**：
- 高斯核
- 均匀核
- 三角核

### 4.2 高斯混合模型（GMM）

**核心思想**：用多个高斯分布的混合来拟合数据。

**概率模型**：
$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)$$

其中：
- $\pi_k$：权重
- $\mu_k$：均值
- $\Sigma_k$：协方差矩阵

**EM算法**：
1. E-step：计算每个点属于每个分量的概率
2. M-step：更新参数

---

## 5. 实践练习

### 练习1：K-Means聚类

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 应用K-Means
kmeans = KMeans(n_clusters=4, random_state=0)
y_pred = kmeans.fit_predict(X)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('K-Means Clustering')
plt.show()
```

### 练习2：PCA降维

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# 加载数据
digits = load_digits()
X = digits.data
y = digits.target

# 应用PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 可视化
plt.figure(figsize=(8, 6))
for digit in range(10):
    plt.scatter(X_pca[y == digit, 0], X_pca[y == digit, 1], 
                label=str(digit), alpha=0.6)
plt.legend()
plt.title('PCA: Digits Dataset')
plt.show()

# 解释方差
print(f"解释方差比例: {pca.explained_variance_ratio_.sum():.2f}")
```

### 练习3：t-SNE可视化

```python
from sklearn.manifold import TSNE

# 应用t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 可视化
plt.figure(figsize=(8, 6))
for digit in range(10):
    plt.scatter(X_tsne[y == digit, 0], X_tsne[y == digit, 1], 
                label=str(digit), alpha=0.6)
plt.legend()
plt.title('t-SNE: Digits Dataset')
plt.show()
```

---

**下一步**：[评估指标与模型选择](04-evaluation.md)