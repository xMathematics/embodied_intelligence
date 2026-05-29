
# 1.2 矩阵分解

## 目录

- [1. LU分解](#1-lu分解)
- [2. QR分解](#2-qr分解)
- [3. Cholesky分解](#3-cholesky分解)
- [4. 奇异值分解(SVD)](#4-奇异值分解svd)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. LU分解

### 1.1 定义

**LU分解**将矩阵 $\mathbf{A}$ 分解为下三角矩阵 $\mathbf{L}$ 和上三角矩阵 $\mathbf{U}$ 的乘积：

$$\mathbf{A} = \mathbf{L}\mathbf{U}$$

其中：
- $\mathbf{L}$：下三角矩阵，对角线元素为1
- $\mathbf{U}$：上三角矩阵

### 1.2 分解过程

**高斯消元法**：

通过初等行变换将矩阵化为上三角形式，记录变换过程得到下三角矩阵。

**示例**：

$$\mathbf{A} = \begin{bmatrix} 2 & 1 & 1 \\ 4 & 3 & 3 \\ 8 & 7 & 9 \end{bmatrix}$$

分解步骤：
1. 第1行乘以2，从第2行减去
2. 第1行乘以4，从第3行减去
3. 第2行乘以3，从第3行减去

得到：

$$\mathbf{L} = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 4 & 3 & 1 \end{bmatrix}, \quad \mathbf{U} = \begin{bmatrix} 2 & 1 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 2 \end{bmatrix}$$

### 1.3 带置换的LU分解（PLU）

当主元为0或很小时，需要进行行交换：

$$\mathbf{P}\mathbf{A} = \mathbf{L}\mathbf{U}$$

其中 $\mathbf{P}$ 是置换矩阵。

### 1.4 应用

**求解线性方程组**：

$$\mathbf{A}\mathbf{x} = \mathbf{b} \Rightarrow \mathbf{L}\mathbf{U}\mathbf{x} = \mathbf{b}$$

分解为两步：
1. 前向替换：$\mathbf{L}\mathbf{y} = \mathbf{b}$
2. 后向替换：$\mathbf{U}\mathbf{x} = \mathbf{y}$

**计算复杂度**：$O(n^3)$ 分解，$O(n^2)$ 求解

---

## 2. QR分解

### 2.1 定义

**QR分解**将矩阵 $\mathbf{A}$ 分解为正交矩阵 $\mathbf{Q}$ 和上三角矩阵 $\mathbf{R}$：

$$\mathbf{A} = \mathbf{Q}\mathbf{R}$$

其中：
- $\mathbf{Q}$：正交矩阵，$\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$
- $\mathbf{R}$：上三角矩阵

### 2.2 计算方法

#### Gram-Schmidt正交化

将列向量组正交化：

$$\mathbf{q}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|}$$

$$\mathbf{q}_k = \frac{\mathbf{a}_k - \sum_{j=1}^{k-1}(\mathbf{a}_k \cdot \mathbf{q}_j)\mathbf{q}_j}{\|\mathbf{a}_k - \sum_{j=1}^{k-1}(\mathbf{a}_k \cdot \mathbf{q}_j)\mathbf{q}_j\|}$$

#### Householder变换

使用Householder矩阵进行正交变换：

$$\mathbf{H} = \mathbf{I} - 2\frac{\mathbf{v}\mathbf{v}^T}{\mathbf{v}^T\mathbf{v}}$$

**优点**：数值稳定性更好

### 2.3 应用

**最小二乘问题**：

$$\min_\mathbf{x} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|^2$$

使用QR分解：

$$\|\mathbf{Q}\mathbf{R}\mathbf{x} - \mathbf{b}\|^2 = \|\mathbf{R}\mathbf{x} - \mathbf{Q}^T\mathbf{b}\|^2$$

由于 $\mathbf{R}$ 是上三角，易于求解。

**特征值计算**：QR迭代算法的基础

---

## 3. Cholesky分解

### 3.1 定义

对于**对称正定矩阵** $\mathbf{A}$，Cholesky分解：

$$\mathbf{A} = \mathbf{L}\mathbf{L}^T$$

其中 $\mathbf{L}$ 是下三角矩阵，对角线元素为正。

### 3.2 计算过程

$$L_{ii} = \sqrt{A_{ii} - \sum_{k=1}^{i-1}L_{ik}^2}$$

$$L_{ji} = \frac{1}{L_{ii}}\left(A_{ji} - \sum_{k=1}^{i-1}L_{jk}L_{ik}\right), \quad j > i$$

### 3.3 性质

- 计算复杂度：$O(n^3/3)$，约为LU分解的一半
- 数值稳定性好
- 仅需存储下三角部分

### 3.4 应用

**多元高斯分布**：

协方差矩阵 $\mathbf{\Sigma}$ 的Cholesky分解用于采样：

$$\mathbf{x} = \mathbf{\mu} + \mathbf{L}\mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

**卡尔曼滤波**：

用于协方差矩阵的平方根滤波，提高数值稳定性。

---

## 4. 奇异值分解(SVD)

### 4.1 定义

**SVD**是矩阵分解中最重要的工具之一：

$$\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

其中：
- $\mathbf{U} \in \mathbb{R}^{m \times m}$：左奇异向量，正交矩阵
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$：对角矩阵，对角线元素为奇异值 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$
- $\mathbf{V} \in \mathbb{R}^{n \times n}$：右奇异向量，正交矩阵

### 4.2 几何解释

SVD将线性变换分解为三个步骤：
1. 旋转（$\mathbf{V}^T$）
2. 缩放（$\mathbf{\Sigma}$）
3. 旋转（$\mathbf{U}$）

### 4.3 与特征值的关系

$$\mathbf{A}^T\mathbf{A} = \mathbf{V}\mathbf{\Sigma}^T\mathbf{\Sigma}\mathbf{V}^T$$

$$\mathbf{A}\mathbf{A}^T = \mathbf{U}\mathbf{\Sigma}\mathbf{\Sigma}^T\mathbf{U}^T$$

**右奇异向量**是 $\mathbf{A}^T\mathbf{A}$ 的特征向量

**左奇异向量**是 $\mathbf{A}\mathbf{A}^T$ 的特征向量

**奇异值**是 $\mathbf{A}^T\mathbf{A}$ 特征值的平方根

### 4.4 低秩近似

**Eckart-Young定理**：

矩阵 $\mathbf{A}$ 的最佳 $k$ 秩近似（Frobenius范数意义下）：

$$\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

**误差**：

$$\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$$

### 4.5 伪逆的计算

$$\mathbf{A}^+ = \mathbf{V}\mathbf{\Sigma}^+\mathbf{U}^T$$

其中 $\mathbf{\Sigma}^+$ 将非零奇异值取倒数后转置。

### 4.6 应用

#### 主成分分析(PCA)

数据矩阵 $\mathbf{X}$ 的SVD：

$$\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$$

主成分方向由 $\mathbf{V}$ 的列给出，方差由奇异值的平方给出。

#### 图像压缩

保留前 $k$ 个奇异值进行图像重建：

$$\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

压缩比：$\frac{mn}{k(m+n+1)}$

#### 推荐系统

用户-物品评分矩阵的低秩近似用于预测缺失评分。

#### 求解线性方程组

对于病态问题，截断SVD提供正则化解：

$$\mathbf{x} = \sum_{i=1}^{k} \frac{\mathbf{u}_i^T\mathbf{b}}{\sigma_i}\mathbf{v}_i$$

---

## 5. 在具身智能中的应用

### 5.1 SLAM中的SVD

**本质矩阵分解**：

从本质矩阵 $\mathbf{E}$ 恢复相机位姿：

$$\mathbf{E} = \mathbf{U}\text{diag}(1,1,0)\mathbf{V}^T$$

**ICP算法**：

点云配准中的SVD用于计算最优刚性变换。

### 5.2 降维与特征提取

**PCA降维**：

高维传感器数据的降维处理。

### 5.3 数值稳定性

**协方差矩阵的Cholesky分解**：

在卡尔曼滤波中提高数值稳定性。

### 5.4 优化问题

**最小二乘的QR分解**：

求解超定线性方程组。

---

## 6. 相关论文

### SVD相关

1. **"The Singular Value Decomposition: Anatomy of Optimizing an Algorithm for Extreme Scale"** - Dongarra et al. (2018)
   - SVD算法优化综述

2. **"Randomized Algorithms for Matrices and Data"** - Mahoney (2011)
   - 大规模矩阵随机算法

3. **"Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions"** - Halko et al. (2011)
   - 随机SVD算法

### 数值线性代数

4. **"Numerical Linear Algebra"** - Trefethen & Bau (1997)
   - 数值线性代数经典教材

5. **"Matrix Computations"** - Golub & Van Loan (2013)
   - 矩阵计算权威参考书

### 应用相关

6. **"A Direct Solution to the Perspectives-n-Point Problem"** - Lepetit et al. (2009)
   - PnP问题中的EPnP算法使用SVD

7. **"Least-Squares Rigid Motion Using SVD"** - Umeyama (1991)
   - SVD在刚性配准中的应用

---

## 7. 实践练习

### 练习1：LU分解实现

```python
import numpy as np
from scipy.linalg import lu

def lu_decomposition(A):
    """
    实现LU分解（无置换）
    """
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    
    for i in range(n):
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]
    
    return L, U

# 验证
A = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]], dtype=float)
L, U = lu_decomposition(A)
print("验证 L @ U = A:", np.allclose(L @ U, A))
```

### 练习2：SVD与PCA

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 加载数据
data = load_iris()
X = data.data

# 方法1：使用sklearn PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 方法2：使用SVD
X_centered = X - np.mean(X, axis=0)
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
X_svd = U[:, :2] * S[:2]

# 可视化比较
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=data.target)
plt.title('PCA (sklearn)')

plt.subplot(1, 2, 2)
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=data.target)
plt.title('PCA (SVD)')
plt.show()
```

### 练习3：图像压缩

```python
from PIL import Image
import numpy as np

def compress_image_svd(image_path, k):
    """
    使用SVD压缩图像
    """
    # 读取图像并转为灰度
    img = Image.open(image_path).convert('L')
    A = np.array(img, dtype=float)
    
    # SVD分解
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # 低秩近似
    A_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    
    # 计算压缩比
    original_size = A.shape[0] * A.shape[1]
    compressed_size = k * (A.shape[0] + A.shape[1] + 1)
    compression_ratio = original_size / compressed_size
    
    return A_compressed, compression_ratio

# 使用示例
# compressed_img, ratio = compress_image_svd('image.jpg', k=50)
# print(f"压缩比: {ratio:.2f}")
```

### 练习4：Cholesky分解与采样

```python
def multivariate_normal_cholesky(mean, cov, n_samples=1):
    """
    使用Cholesky分解生成多元正态分布样本
    """
    # Cholesky分解
    L = np.linalg.cholesky(cov)
    
    # 生成标准正态样本
    z = np.random.standard_normal((len(mean), n_samples))
    
    # 变换
    samples = mean[:, np.newaxis] + L @ z
    
    return samples.T

# 测试
mean = np.array([0, 0])
cov = np.array([[1, 0.5], [0.5, 1]])
samples = multivariate_normal_cholesky(mean, cov, n_samples=1000)

# 验证
print("样本均值:", np.mean(samples, axis=0))
print("样本协方差:", np.cov(samples.T))
```

---

**下一步**：[特征值与特征向量](03-eigenvalues.md)
