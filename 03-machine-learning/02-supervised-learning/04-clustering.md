# 2.4 聚类算法

## 目录

- [1. 聚类概述](#1-聚类概述)
- [2. K-Means聚类](#2-k-means聚类)
- [3. DBSCAN](#3-dbscan)
- [4. 层次聚类](#4-层次聚类)
- [5. 高斯混合模型](#5-高斯混合模型)
- [6. 聚类评估](#6-聚类评估)
- [7. 实践练习](#7-实践练习)

---

## 1. 聚类概述

### 1.1 什么是聚类？

**聚类**：将相似数据点分组的无监督学习方法。

**目标**：
- 簇内相似度最大
- 簇间相似度最小

### 1.2 聚类应用

| 应用场景 | 说明 |
|---------|------|
| 客户分群 | 市场细分、个性化推荐 |
| 图像分割 | 识别图像中的对象 |
| 文档聚类 | 主题提取、信息检索 |
| 异常检测 | 识别欺诈、网络入侵 |

### 1.3 相似度度量

| 度量方法 | 公式 | 适用场景 |
|---------|------|---------|
| 欧氏距离 | $\|x_i - x_j\|_2$ | 连续数据 |
| 曼哈顿距离 | $\|x_i - x_j\|_1$ | 稀疏数据 |
| 余弦相似度 | $\frac{x_i^T x_j}{\|x_i\| \|x_j\|}$ | 文本数据 |
| Jaccard系数 | $\frac{\|A \cap B\|}{\|A \cup B\|}$ | 二进制数据 |

---

## 2. K-Means聚类

### 2.1 算法流程

**步骤**：
1. 随机选择K个质心
2. 分配每个点到最近的质心
3. 更新质心为簇内点的均值
4. 重复直到收敛

**目标函数**：
$$J = \sum_{i=1}^{n} \sum_{k=1}^{K} w_{ik} \|x_i - \mu_k\|^2$$

### 2.2 K值选择

**肘部法则**：
- 计算不同K的SSE
- 找到SSE下降变慢的点

**轮廓系数**：
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

其中：
- $a(i)$：点i到簇内其他点的平均距离
- $b(i)$：点i到最近簇的平均距离

### 2.3 实践示例

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 肘部法则选择K
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()

# 使用最佳K=4
kmeans = KMeans(n_clusters=4, random_state=0)
y_pred = kmeans.fit_predict(X)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('K-Means Clustering')
plt.show()
```

---

## 3. DBSCAN

### 3.1 算法原理

**基于密度的聚类**：
- $\epsilon$：邻域半径
- MinPts：最小点数

**点的类型**：
- 核心点：邻域内点数≥MinPts
- 边界点：在核心点邻域内但非核心点
- 噪声点：既非核心点也非边界点

**算法流程**：
1. 随机选择未访问点
2. 如果是核心点，扩展簇
3. 重复直到所有点被访问

### 3.2 实践示例

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# 生成数据
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# 创建模型
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_pred = dbscan.fit_predict(X)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.show()
```

---

## 4. 层次聚类

### 4.1 凝聚式聚类

**步骤**：
1. 每个点作为单独簇
2. 合并最相似的簇
3. 重复直到所有点合并

**相似度度量**：
- 单链接：最小距离
- 完全链接：最大距离
- 平均链接：平均距离

### 4.2 树状图

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 生成数据
X, _ = make_blobs(n_samples=20, centers=3, cluster_std=0.5, random_state=42)

# 层次聚类
linked = linkage(X, 'ward')

# 绘制树状图
plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=range(20),
            distance_sort='descending',
            show_leaf_counts=True)
plt.title('Dendrogram')
plt.show()

# 应用聚类
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_pred = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.title('Hierarchical Clustering')
plt.show()
```

---

## 5. 高斯混合模型

### 5.1 模型定义

**混合模型**：
$$p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)$$

**参数**：
- $\pi_k$：权重
- $\mu_k$：均值
- $\Sigma_k$：协方差矩阵

### 5.2 EM算法

**E-step**：
$$\gamma(z_{ik}) = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}$$

**M-step**：
$$\mu_k = \frac{\sum_{i=1}^{n} \gamma(z_{ik}) x_i}{\sum_{i=1}^{n} \gamma(z_{ik})}$$

### 5.3 实践示例

```python
from sklearn.mixture import GaussianMixture

# 生成数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 创建模型
gmm = GaussianMixture(n_components=4, random_state=0)
y_pred = gmm.fit_predict(X)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.title('Gaussian Mixture Model')
plt.show()

# 概率预测
probs = gmm.predict_proba(X)
print(f"概率形状: {probs.shape}")
print(f"第一个样本的概率: {probs[0]}")
```

---

## 6. 聚类评估

### 6.1 内部评估

**轮廓系数**：
$$\text{Silhouette Score} = \frac{1}{n} \sum_{i=1}^{n} s(i)$$

**Calinski-Harabasz指数**：
$$CH = \frac{\text{Between-Cluster Variance}}{\text{Within-Cluster Variance}}$$

### 6.2 外部评估（有标签时）

**调整兰德指数**：
$$ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}$$

**互信息**：
$$MI = \sum_{i,j} p(i,j) \log \frac{p(i,j)}{p(i)p(j)}$$

### 6.3 实践示例

```python
from sklearn.metrics import silhouette_score, adjusted_rand_score

# 生成数据（带标签）
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# K-Means聚类
kmeans = KMeans(n_clusters=4, random_state=0)
y_pred = kmeans.fit_predict(X)

# 评估
sil_score = silhouette_score(X, y_pred)
ari_score = adjusted_rand_score(y_true, y_pred)

print(f"轮廓系数: {sil_score:.2f}")
print(f"调整兰德指数: {ari_score:.2f}")
```

---

## 7. 实践练习

### 练习1：客户分群

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 加载数据
data = pd.read_csv('Mall_Customers.csv')
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

# 可视化
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=y_pred, s=50, cmap='viridis')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()
```

### 练习2：图像分割

```python
from sklearn.cluster import KMeans
import cv2

# 加载图像
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 转换为二维数组
pixels = image.reshape(-1, 3)

# K-Means聚类
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(pixels)

# 重建图像
segmented_image = kmeans.cluster_centers_[labels].reshape(image.shape)
segmented_image = segmented_image.astype('uint8')

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title('Segmented')
plt.show()
```

---

**下一步**：[深度学习基础](../03-deep-learning/01-neural-networks.md)