# 4.1 数值线性代数

## 目录

- [1. 数值稳定性](#1-数值稳定性)
- [2. 线性方程组求解](#2-线性方程组求解)
- [3. 矩阵分解](#3-矩阵分解)
- [4. 特征值问题](#4-特征值问题)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 数值稳定性

### 1.1 误差分析

**绝对误差**：

$$e_a = |x_{true} - x_{approx}|$$

**相对误差**：

$$e_r = \frac{|x_{true} - x_{approx}|}{|x_{true}|}$$

**条件数**：

$$\text{cond}(A) = \|A\| \|A^{-1}\|$$

**意义**：条件数衡量问题的敏感性。

### 1.2 数值稳定性

**稳定算法**：输入的微小扰动不会导致输出的巨大变化。

**不稳定算法**：输入的微小扰动会导致输出的巨大变化。

### 1.3 浮点运算

**IEEE 754标准**：
- 单精度（32位）
- 双精度（64位）

**浮点误差**：
- 舍入误差
- 截断误差

---

## 2. 线性方程组求解

### 2.1 直接法

#### LU分解

**分解**：$A = LU$

其中 $L$ 是下三角矩阵，$U$ 是上三角矩阵。

**求解步骤**：
1. 前向替换：$Ly = b$
2. 后向替换：$Ux = y$

#### Cholesky分解

**分解**：$A = LL^T$

其中 $L$ 是下三角矩阵，且对角线元素为正。

**适用条件**：$A$ 是对称正定矩阵。

#### QR分解

**分解**：$A = QR$

其中 $Q$ 是正交矩阵，$R$ 是上三角矩阵。

**优势**：数值稳定性好。

### 2.2 迭代法

#### 雅可比迭代

**迭代公式**：

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left(b_i - \sum_{j \neq i} a_{ij} x_j^{(k)}\right)$$

**收敛条件**：$A$ 严格对角占优。

#### 高斯-赛德尔迭代

**迭代公式**：

$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left(b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)}\right)$$

**优势**：收敛速度通常比雅可比迭代快。

#### 逐次超松弛（SOR）

**迭代公式**：

$$x_i^{(k+1)} = (1-\omega)x_i^{(k)} + \frac{\omega}{a_{ii}} \left(b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)}\right)$$

其中 $\omega$ 是松弛因子。

**最优松弛因子**：$\omega \in (1, 2)$

### 2.3 共轭梯度法

**适用条件**：$A$ 是对称正定矩阵。

**算法步骤**：
1. 初始化：$r_0 = b - Ax_0$，$p_0 = r_0$
2. 迭代：
   $$\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}$$
   $$x_{k+1} = x_k + \alpha_k p_k$$
   $$r_{k+1} = r_k - \alpha_k A p_k$$
   $$\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}$$
   $$p_{k+1} = r_{k+1} + \beta_k p_k$$

**收敛性**：最多 $n$ 次迭代收敛。

---

## 3. 矩阵分解

### 3.1 奇异值分解（SVD）

**分解**：$A = U \Sigma V^T$

其中：
- $U$ 是左奇异向量矩阵（正交矩阵）
- $\Sigma$ 是奇异值矩阵（对角矩阵）
- $V$ 是右奇异向量矩阵（正交矩阵）

**性质**：
- 奇异值按降序排列
- $\sigma_i \geq 0$

**应用**：
- 伪逆计算
- 降维
- 数据压缩

### 3.2 特征值分解

**分解**：$A = P \Lambda P^{-1}$

其中：
- $\Lambda$ 是对角矩阵，对角线元素是特征值
- $P$ 是特征向量矩阵

**适用条件**：$A$ 是方阵。

### 3.3 广义特征值问题

**问题**：$Ax = \lambda Bx$

**求解**：通常转换为标准特征值问题。

---

## 4. 特征值问题

### 4.1 幂法

**目标**：求最大特征值和对应的特征向量。

**迭代公式**：

$$v_{k+1} = \frac{A v_k}{\|A v_k\|}$$

**收敛**：$v_k \to v_{max}$，$\|A v_k\| \to \lambda_{max}$

### 4.2 反幂法

**目标**：求最小特征值和对应的特征向量。

**迭代公式**：

$$v_{k+1} = \frac{A^{-1} v_k}{\|A^{-1} v_k\|}$$

**收敛**：$v_k \to v_{min}$，$\|A^{-1} v_k\| \to 1/\lambda_{min}$

### 4.3 QR算法

**目标**：求所有特征值。

**算法步骤**：
1. 初始化：$A_0 = A$
2. 迭代：$A_k = Q_k R_k$，$A_{k+1} = R_k Q_k$

**收敛**：$A_k$ 收敛到上三角矩阵，对角线元素是特征值。

---

## 5. 在具身智能中的应用

### 5.1 SLAM

**稀疏矩阵求解**：使用稀疏LU分解或共轭梯度法。

**大规模优化**：使用增量方法或分块方法。

### 5.2 机器人运动学

**逆运动学求解**：使用数值方法求解非线性方程组。

**雅可比矩阵**：用于计算关节速度与末端执行器速度的关系。

### 5.3 控制系统

**状态估计**：卡尔曼滤波中的矩阵运算。

**控制器设计**：求解Riccati方程。

### 5.4 计算机视觉

**特征提取**：使用SVD进行降维。

**图像恢复**：使用伪逆求解超定方程组。

---

## 6. 相关论文

### 数值线性代数

1. **"Numerical Linear Algebra"** - Trefethen & Bau (1997)
   - 数值线性代数经典教材

2. **"Matrix Computations"** - Golub & Van Loan (2013)
   - 矩阵计算权威教材

### 应用相关

3. **"Sparse Matrix Computations"** - George & Liu (1981)
   - 稀疏矩阵计算

4. **"Conjugate Gradient Methods for Linear Equations"** - Hestenes & Stiefel (1952)
   - 共轭梯度法奠基论文

---

## 7. 实践练习

### 练习1：LU分解

```python
import numpy as np
from scipy.linalg import lu, solve

# 创建矩阵
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

# LU分解
P, L, U = lu(A)

print("LU分解:")
print(f"P =\n{P}")
print(f"L =\n{L}")
print(f"U =\n{U}")

# 验证分解
print(f"\n验证 A = PLU:")
print(f"PLU =\n{P @ L @ U}")
print(f"A =\n{A}")

# 求解方程组
x = solve(A, b)
print(f"\n方程组的解: x = {x}")
print(f"验证 Ax = b: {A @ x}")
```

### 练习2：共轭梯度法

```python
import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=1000):
    """
    共轭梯度法求解 Ax = b
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    r = b - A @ x
    p = r.copy()
    r_norm_sq = r @ r
    
    for k in range(max_iter):
        if r_norm_sq < tol:
            break
        
        Ap = A @ p
        alpha = r_norm_sq / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        
        r_norm_sq_new = r @ r
        beta = r_norm_sq_new / r_norm_sq
        p = r + beta * p
        
        r_norm_sq = r_norm_sq_new
    
    return x, k

# 创建对称正定矩阵
n = 10
A = np.random.randn(n, n)
A = A @ A.T + np.eye(n)  # 确保对称正定
b = np.random.randn(n)

# 使用共轭梯度法求解
x_cg, iter_cg = conjugate_gradient(A, b)
print(f"共轭梯度法迭代次数: {iter_cg}")

# 使用numpy验证
x_np = np.linalg.solve(A, b)
print(f"解的误差: {np.linalg.norm(x_cg - x_np):.6e}")
```

### 练习3：奇异值分解

```python
import numpy as np
from scipy.linalg import svd

# 创建矩阵
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

# SVD分解
U, S, VT = svd(A)

print("SVD分解:")
print(f"U =\n{U}")
print(f"S = {S}")
print(f"V^T =\n{VT}")

# 验证分解
print(f"\n验证 A = U S V^T:")
Sigma = np.zeros(A.shape)
np.fill_diagonal(Sigma, S)
print(f"U S V^T =\n{U @ Sigma @ VT}")

# 低秩近似
k = 2
Sigma_k = np.zeros(A.shape)
np.fill_diagonal(Sigma_k[:k, :k], S[:k])
A_approx = U @ Sigma_k @ VT

print(f"\n原始矩阵的秩: {np.linalg.matrix_rank(A)}")
print(f"秩{k}近似的误差: {np.linalg.norm(A - A_approx):.6e}")

# 伪逆
A_pinv = np.linalg.pinv(A)
print(f"\n伪逆 A^+:")
print(A_pinv)
```

### 练习4：特征值问题

```python
import numpy as np
from scipy.linalg import eig

# 创建矩阵
A = np.array([[4, 2],
              [1, 3]], dtype=float)

# 特征值分解
eigenvalues, eigenvectors = eig(A)

print("特征值分解:")
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 验证
for i in range(len(eigenvalues)):
    Av = A @ eigenvectors[:, i]
    lambdav = eigenvalues[i] * eigenvectors[:, i]
    print(f"\n验证第{i+1}个特征值:")
    print(f"Av = {Av}")
    print(f"λv = {lambdav}")

# 幂法求最大特征值
def power_method(A, max_iter=100, tol=1e-10):
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    
    for _ in range(max_iter):
        v_new = A @ v
        lambda_estimate = np.linalg.norm(v_new)
        v_new = v_new / lambda_estimate
        
        if np.linalg.norm(v_new - v) < tol:
            break
        
        v = v_new
    
    return lambda_estimate, v

lambda_max, v_max = power_method(A)
print(f"\n幂法结果:")
print(f"最大特征值: {lambda_max:.6f}")
print(f"对应的特征向量: {v_max}")
```

### 练习5：稀疏线性方程组

```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# 创建稀疏矩阵
n = 1000
# 三对角矩阵
data = np.ones(3 * n - 2)
row = []
col = []

for i in range(n):
    row.append(i)
    col.append(i)
    
    if i > 0:
        row.append(i)
        col.append(i - 1)
    
    if i < n - 1:
        row.append(i)
        col.append(i + 1)

A_sparse = csr_matrix((data, (row, col)), shape=(n, n))
b = np.random.randn(n)

# 求解稀疏方程组
x_sparse = spsolve(A_sparse, b)

# 验证
error = np.linalg.norm(A_sparse @ x_sparse - b)
print(f"稀疏方程组求解误差: {error:.6e}")

# 与稠密方法比较
A_dense = A_sparse.toarray()
x_dense = np.linalg.solve(A_dense, b)
diff = np.linalg.norm(x_sparse - x_dense)
print(f"稀疏解与稠密解的差异: {diff:.6e}")
```

---

**下一步**：[数值优化](02-numerical-optimization.md)