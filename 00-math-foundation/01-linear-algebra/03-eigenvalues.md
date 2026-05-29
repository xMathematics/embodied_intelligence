
# 1.3 特征值与特征向量

## 目录

- [1. 特征值与特征向量的定义](#1-特征值与特征向量的定义)
- [2. 特征值的计算](#2-特征值的计算)
- [3. 特征分解](#3-特征分解)
- [4. 特殊矩阵的特征值](#4-特殊矩阵的特征值)
- [5. 广义特征值问题](#5-广义特征值问题)
- [6. 在具身智能中的应用](#6-在具身智能中的应用)
- [7. 相关论文](#7-相关论文)
- [8. 实践练习](#8-实践练习)

---

## 1. 特征值与特征向量的定义

### 1.1 定义

对于方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$，如果存在非零向量 $\mathbf{v}$ 和标量 $\lambda$ 使得：

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

则称：
- $\lambda$ 为 **特征值**（eigenvalue）
- $\mathbf{v}$ 为 **特征向量**（eigenvector）

### 1.2 几何意义

特征向量表示在矩阵变换下**方向不变**（或反向）的向量，特征值表示**缩放因子**。

- $\lambda > 0$：同向拉伸
- $\lambda < 0$：反向拉伸
- $|\lambda| > 1$：放大
- $|\lambda| < 1$：缩小
- $\lambda = 1$：不变

### 1.3 特征方程

$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$

**特征多项式**：

$$p(\lambda) = \det(\mathbf{A} - \lambda\mathbf{I}) = (-\lambda)^n + c_{n-1}(-\lambda)^{n-1} + \cdots + c_0$$

**性质**：
- $n \times n$ 矩阵有 $n$ 个特征值（考虑重数）
- 特征值的和等于迹：$\sum_{i=1}^{n}\lambda_i = \text{tr}(\mathbf{A})$
- 特征值的积等于行列式：$\prod_{i=1}^{n}\lambda_i = \det(\mathbf{A})$

---

## 2. 特征值的计算

### 2.1 2×2矩阵

对于矩阵 $\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$：

$$\det(\mathbf{A} - \lambda\mathbf{I}) = (a-\lambda)(d-\lambda) - bc = 0$$

$$\lambda^2 - (a+d)\lambda + (ad-bc) = 0$$

$$\lambda = \frac{(a+d) \pm \sqrt{(a+d)^2 - 4(ad-bc)}}{2}$$

### 2.2 3×3矩阵

使用**卡尔丹公式**或数值方法求解三次方程。

### 2.3 数值计算方法

#### 幂迭代法

用于求最大特征值：

$$\mathbf{v}_{k+1} = \frac{\mathbf{A}\mathbf{v}_k}{\|\mathbf{A}\mathbf{v}_k\|}$$

$$\lambda_{k+1} = \mathbf{v}_{k+1}^T\mathbf{A}\mathbf{v}_{k+1}$$

收敛到绝对值最大的特征值对应的特征向量。

#### QR算法

通过迭代QR分解计算所有特征值：

$$\mathbf{A}_k = \mathbf{Q}_k\mathbf{R}_k$$

$$\mathbf{A}_{k+1} = \mathbf{R}_k\mathbf{Q}_k$$

序列收敛到上三角矩阵，对角线元素为特征值。

#### Jacobi方法

用于对称矩阵，通过正交变换对角化。

---

## 3. 特征分解

### 3.1 对角化

如果矩阵 $\mathbf{A}$ 有 $n$ 个线性无关的特征向量，则：

$$\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{-1}$$

其中：
- $\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n]$：特征向量矩阵
- $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$：特征值对角矩阵

**条件**：矩阵可对角化当且仅当有 $n$ 个线性无关的特征向量。

### 3.2 对称矩阵的谱分解

对于对称矩阵 $\mathbf{A} = \mathbf{A}^T$：

$$\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$$

**性质**：
- 所有特征值为实数
- 特征向量两两正交
- $\mathbf{V}$ 是正交矩阵

### 3.3 矩阵幂的计算

利用特征分解：

$$\mathbf{A}^k = \mathbf{V}\mathbf{\Lambda}^k\mathbf{V}^{-1}$$

### 3.4 矩阵指数

$$e^{\mathbf{A}} = \mathbf{V}e^{\mathbf{\Lambda}}\mathbf{V}^{-1} = \mathbf{V}\text{diag}(e^{\lambda_1}, e^{\lambda_2}, \ldots, e^{\lambda_n})\mathbf{V}^{-1}$$

**应用**：线性微分方程组的解

---

## 4. 特殊矩阵的特征值

### 4.1 正交矩阵

对于正交矩阵 $\mathbf{Q}$（$\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$）：

- 特征值的模为1：$|\lambda| = 1$
- 实特征值为 $\pm 1$
- 复特征值为 $e^{i\theta}$

### 4.2 对称正定矩阵

- 所有特征值为正实数
- 可用于Cholesky分解

### 4.3 马尔可夫矩阵

- 最大特征值为1
- 对应特征向量为稳态分布

### 4.4 旋转矩阵

2D旋转矩阵：

$$\mathbf{R} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

特征值：$\lambda = e^{\pm i\theta}$

---

## 5. 广义特征值问题

### 5.1 定义

对于矩阵对 $(\mathbf{A}, \mathbf{B})$，求解：

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{B}\mathbf{v}$$

### 5.2 应用

**振动分析**：

$$\mathbf{K}\mathbf{u} = \omega^2\mathbf{M}\mathbf{u}$$

其中 $\mathbf{K}$ 是刚度矩阵，$\mathbf{M}$ 是质量矩阵。

**Fisher线性判别**：

$$\mathbf{S}_B\mathbf{w} = \lambda\mathbf{S}_W\mathbf{w}$$

其中 $\mathbf{S}_B$ 是类间散度矩阵，$\mathbf{S}_W$ 是类内散度矩阵。

---

## 6. 在具身智能中的应用

### 6.1 主成分分析(PCA)

**协方差矩阵的特征分解**：

$$\mathbf{C} = \frac{1}{n}\mathbf{X}^T\mathbf{X} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$$

主成分是协方差矩阵的特征向量，方差是特征值。

### 6.2 刚性变换估计

**Kabsch算法**：

使用SVD（与特征值相关）计算最优刚性变换。

### 6.3 系统稳定性分析

**线性系统**：$\dot{\mathbf{x}} = \mathbf{A}\mathbf{x}$

- 所有特征值实部为负：系统稳定
- 存在正实部特征值：系统不稳定

### 6.4 图的谱分析

**拉普拉斯矩阵**：

$$\mathbf{L} = \mathbf{D} - \mathbf{A}$$

特征值用于图分割、聚类等。

### 6.5 姿态估计

**本质矩阵分解**：

从本质矩阵恢复相机位姿涉及特征值分解。

---

## 7. 相关论文

### 特征值计算

1. **"The QR Algorithm"** - Golub & Van Loan
   - QR算法详解

2. **"Numerical Methods for Large Eigenvalue Problems"** - Saad
   - 大规模特征值问题数值方法

### 应用相关

3. **"Principal Component Analysis"** - Jolliffe (2002)
   - PCA理论与应用

4. **"A Method for Registration of 3-D Shapes"** - Besl & McKay (1992)
   - ICP算法，涉及特征值

5. **"Spectral Clustering"** - von Luxburg (2007)
   - 谱聚类综述

### 理论相关

6. **"Matrix Analysis"** - Horn & Johnson
   - 矩阵理论权威参考书

7. **"The Algebraic Eigenvalue Problem"** - Wilkinson
   - 特征值问题经典著作

---

## 8. 实践练习

### 练习1：手动计算特征值

```python
import numpy as np

def compute_eigenvalues_2x2(A):
    """
    手动计算2x2矩阵的特征值
    """
    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
    
    # 特征方程: lambda^2 - (a+d)*lambda + (ad-bc) = 0
    trace = a + d
    det = a*d - b*c
    
    discriminant = trace**2 - 4*det
    
    if discriminant >= 0:
        lambda1 = (trace + np.sqrt(discriminant)) / 2
        lambda2 = (trace - np.sqrt(discriminant)) / 2
    else:
        lambda1 = (trace + 1j*np.sqrt(-discriminant)) / 2
        lambda2 = (trace - 1j*np.sqrt(-discriminant)) / 2
    
    return lambda1, lambda2

# 验证
A = np.array([[4, 2], [1, 3]])
lambda1, lambda2 = compute_eigenvalues_2x2(A)
print(f"手动计算: {lambda1:.4f}, {lambda2:.4f}")

# 与numpy对比
eigenvalues, _ = np.linalg.eig(A)
print(f"NumPy计算: {eigenvalues}")
```

### 练习2：幂迭代法

```python
def power_iteration(A, num_iterations=100, tol=1e-10):
    """
    幂迭代法求最大特征值
    """
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    
    for i in range(num_iterations):
        Av = A @ v
        v_new = Av / np.linalg.norm(Av)
        
        # 计算Rayleigh商
        lambda_max = v_new.T @ A @ v_new
        
        if np.linalg.norm(v_new - v) < tol:
            print(f"收敛于第{i}次迭代")
            break
        
        v = v_new
    
    return lambda_max, v

# 测试
A = np.array([[4, 1], [2, 3]])
lambda_max, v = power_iteration(A)
print(f"最大特征值: {lambda_max:.6f}")
print(f"特征向量: {v}")

# 验证
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"NumPy结果: {eigenvalues}")
```

### 练习3：PCA实现

```python
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
    
    def fit(self, X):
        # 中心化
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)
        
        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 保存主成分
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        return self
    
    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# 测试
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"主成分:\n{pca.components_}")
print(f"解释方差: {pca.explained_variance_}")
```

### 练习4：系统稳定性分析

```python
def analyze_stability(A):
    """
    分析线性系统的稳定性
    """
    eigenvalues = np.linalg.eigvals(A)
    
    print("特征值:", eigenvalues)
    print("实部:", np.real(eigenvalues))
    
    if all(np.real(eigenvalues) < 0):
        print("系统稳定")
    elif any(np.real(eigenvalues) > 0):
        print("系统不稳定")
    else:
        print("系统临界稳定")
    
    return eigenvalues

# 测试不同系统
A_stable = np.array([[-2, 1], [0, -3]])
A_unstable = np.array([[1, 2], [0, -1]])

print("稳定系统:")
analyze_stability(A_stable)

print("\n不稳定系统:")
analyze_stability(A_unstable)
```

---

**下一步**：[向量空间与变换](04-vector-spaces.md)
