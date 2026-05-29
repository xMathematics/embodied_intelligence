# 1.3 无监督学习基础

## 目录

- [1. 无监督学习概述](#1-无监督学习概述)
- [2. 聚类分析](#2-聚类分析)
- [3. 降维方法](#3-降维方法)
- [4. 关联规则挖掘](#4-关联规则挖掘)
- [5. 密度估计](#5-密度估计)
- [6. 实践练习](#6-实践练习)

---

## 1. 无监督学习概述

### 1.1 定义

**无监督学习**（Unsupervised Learning）是机器学习的重要范式，其核心特点是：

- **训练数据没有标签**：只有输入特征，没有对应的输出
- **发现数据结构**：目标是发现数据中的内在模式和结构
- **探索性分析**：用于数据探索和理解

### 1.2 无监督学习与监督学习的对比

| 维度 | 监督学习 | 无监督学习 |
|------|---------|-----------|
| **数据标签** | 有标签 | 无标签 |
| **目标** | 预测输出 | 发现结构 |
| **学习方式** | 学习映射函数 | 学习数据分布 |
| **典型任务** | 分类、回归 | 聚类、降维 |
| **评估方式** | 预测准确率 | 内部指标（如簇内相似度） |

### 1.3 无监督学习的应用场景

| 场景 | 任务 | 应用举例 |
|------|------|---------|
| **数据探索** | 发现数据结构 | 客户细分、图像分割 |
| **数据压缩** | 降维 | 特征提取、可视化 |
| **异常检测** | 识别异常模式 | 欺诈检测、故障诊断 |
| **数据生成** | 学习数据分布 | 图像生成、文本生成 |

---

## 2. 聚类分析

### 2.1 定义

**聚类**（Clustering）是将数据样本分组的过程，使得同一组内的样本相似度高，不同组间的样本相似度低。

### 2.2 聚类类型

| 类型 | 描述 | 算法 |
|------|------|------|
| **划分式聚类** | 将数据划分为不重叠的簇 | K-Means、K-Medoids |
| **层次聚类** | 构建层次化的簇结构 | 凝聚式、分裂式 |
| **密度聚类** | 基于密度发现簇 | DBSCAN、OPTICS |
| **基于模型的聚类** | 假设数据服从某种分布 | GMM、EM算法 |

### 2.3 K-Means聚类

**算法步骤**：
1. 随机选择K个初始质心
2. 计算每个样本到各质心的距离，分配到最近的簇
3. 更新每个簇的质心（取簇内样本的均值）
4. 重复步骤2-3，直到质心不再变化

**目标函数**：最小化簇内平方和
$$J = \sum_{i=1}^n \sum_{j=1}^K z_{ij} \|x_i - \mu_j\|^2$$

其中 $z_{ij}=1$ 表示样本i属于簇j，$\mu_j$ 是簇j的质心。

### 2.4 DBSCAN聚类

**算法步骤**：
1. 随机选择一个未访问的样本
2. 找到所有密度可达的样本，形成一个簇
3. 重复步骤1-2，直到所有样本都被访问

**关键参数**：
- **ε（epsilon）**：邻域半径
- **min_samples**：形成密集区域所需的最小样本数

**优点**：
- 能发现任意形状的簇
- 能识别噪声点
- 不需要预先指定簇的数量

### 2.5 聚类评估指标

#### 内部指标（无需标签）
| 指标 | 描述 |
|------|------|
| **SSE** | 簇内平方和，越小越好 |
| **轮廓系数** | 衡量样本与其所在簇的相似度，范围[-1,1]，越大越好 |
| **Calinski-Harabasz指数** | 簇间方差与簇内方差的比值，越大越好 |

#### 外部指标（需要标签）
| 指标 | 描述 |
|------|------|
| **兰德指数** | 衡量聚类结果与真实标签的一致性，范围[0,1] |
| **互信息** | 衡量两个聚类结果的相似度 |

---

## 3. 降维方法

### 3.1 定义

**降维**（Dimensionality Reduction）是将高维数据映射到低维空间的过程，目的是：
- 减少计算复杂度
- 去除冗余信息
- 便于可视化
- 缓解维度灾难

### 3.2 降维类型

| 类型 | 描述 | 算法 |
|------|------|------|
| **线性降维** | 线性映射 | PCA、LDA |
| **非线性降维** | 非线性映射 | t-SNE、UMAP、Autoencoder |

### 3.3 主成分分析（PCA）

**核心思想**：找到数据方差最大的方向作为主成分

**算法步骤**：
1. 数据标准化（零均值、单位方差）
2. 计算协方差矩阵
3. 计算协方差矩阵的特征值和特征向量
4. 选择前k个最大特征值对应的特征向量
5. 将数据投影到这些特征向量上

**数学原理**：
$$\text{最大化方差：} \max_w \frac{1}{n} \sum_{i=1}^n (w^T x_i)^2$$

**性质**：
- 主成分之间正交
- 保留最大方差
- 无监督方法（不使用标签）

### 3.4 t-SNE

**核心思想**：在低维空间中保持数据点之间的局部相似度

**算法步骤**：
1. 计算高维空间中数据点的相似度（高斯核）
2. 在低维空间中初始化数据点位置
3. 使用梯度下降优化，使低维相似度接近高维相似度

**目标函数**（KL散度）：
$$KL(P||Q) = \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

**特点**：
- 擅长可视化高维数据
- 计算复杂度较高（O(n²)）
- 主要用于可视化，不适合大规模数据

### 3.5 UMAP

**核心思想**：结合流形学习和图论的降维方法

**特点**：
- 比t-SNE更快（O(n log n)）
- 更好地保留全局结构
- 参数更直观
- 可用于大规模数据

---

## 4. 关联规则挖掘

### 4.1 定义

**关联规则挖掘**（Association Rule Mining）用于发现数据集中项之间的关联关系。

### 4.2 核心概念

| 概念 | 定义 | 示例 |
|------|------|------|
| **项集** | 一组项的集合 | {牛奶, 面包} |
| **支持度** | 项集出现的频率 | support({牛奶,面包}) = 0.3 |
| **置信度** | 规则的可靠性 | confidence(牛奶→面包) = 0.7 |
| **提升度** | 规则的有效性 | lift(牛奶→面包) = 1.5 |

### 4.3 Apriori算法

**核心思想**：频繁项集的子集也是频繁的

**算法步骤**：
1. 生成所有1-项集，计算支持度
2. 筛选出支持度大于阈值的频繁项集
3. 组合生成候选k-项集
4. 重复步骤2-3，直到没有新的频繁项集

### 4.4 应用场景

- **购物篮分析**：发现顾客购买模式
- **网页推荐**：关联页面浏览
- **医疗诊断**：发现症状与疾病的关联
- **金融风控**：发现欺诈模式

---

## 5. 密度估计

### 5.1 定义

**密度估计**（Density Estimation）是从样本数据中估计概率密度函数的过程。

### 5.2 参数方法

| 方法 | 假设 | 适用场景 |
|------|------|---------|
| **高斯混合模型** | 数据服从多个高斯分布的混合 | 连续数据、聚类 |
| **核密度估计** | 使用核函数平滑数据点 | 非参数、灵活 |

### 5.3 高斯混合模型（GMM）

**模型形式**：
$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)$$

其中 $\pi_k$ 是混合权重，$\mathcal{N}$ 是高斯分布。

**EM算法求解**：
1. E步：计算每个样本属于各成分的后验概率
2. M步：更新均值、协方差和混合权重
3. 重复直到收敛

---

## 6. 实践练习

### 练习1：K-Means聚类

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 生成模拟数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# 尝试不同的K值
silhouette_scores = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, labels))

# 绘制轮廓系数
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_values, silhouette_scores, 'bo-')
plt.xlabel('K值')
plt.ylabel('轮廓系数')
plt.title('K值选择')

# 使用最佳K值（K=4）
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='red')
plt.title('K-Means聚类结果')
plt.show()

print(f"最佳K值: {k_values[np.argmax(silhouette_scores)]}")
print(f"对应的轮廓系数: {max(silhouette_scores):.4f}")
```

### 练习2：PCA降维与可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 加载手写数字数据集
digits = load_digits()
X = digits.data
y = digits.target

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 执行PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 计算解释方差比
print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累计解释方差: {sum(pca.explained_variance_ratio_):.4f}")

# 可视化结果
plt.figure(figsize=(10, 8))
for digit in range(10):
    mask = y == digit
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=str(digit), alpha=0.7)

plt.legend()
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('手写数字PCA降维可视化')
plt.show()

# 查看主成分的方差解释
plt.figure(figsize=(10, 5))
pca_full = PCA()
pca_full.fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'b-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95%方差')
plt.xlabel('主成分数量')
plt.ylabel('累计解释方差')
plt.title('PCA方差解释曲线')
plt.legend()
plt.show()

# 找到解释95%方差所需的主成分数
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"解释95%方差需要 {n_components_95} 个主成分")
```

### 练习3：t-SNE可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 加载数据
digits = load_digits()
X = digits.data[:500]  # 使用部分数据加速
y = digits.target[:500]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 执行t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# 可视化结果
plt.figure(figsize=(10, 8))
for digit in range(10):
    mask = y == digit
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=str(digit), alpha=0.7)

plt.legend()
plt.xlabel('t-SNE维度1')
plt.ylabel('t-SNE维度2')
plt.title('手写数字t-SNE可视化')
plt.show()

# 尝试不同的perplexity参数
perplexities = [5, 30, 50]
plt.figure(figsize=(15, 5))

for i, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.subplot(1, 3, i+1)
    for digit in range(10):
        mask = y == digit
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=str(digit), alpha=0.6)
    plt.title(f'perplexity={perplexity}')
    plt.legend()

plt.tight_layout()
plt.show()
```

---

**下一步**：[学习理论](04-evaluation.md)
