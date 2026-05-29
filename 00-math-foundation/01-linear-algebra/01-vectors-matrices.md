
# 1.1 向量与矩阵基础

## 目录

- [1. 向量基础](#1-向量基础)
- [2. 矩阵基础](#2-矩阵基础)
- [3. 矩阵运算](#3-矩阵运算)
- [4. 特殊矩阵](#4-特殊矩阵)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 向量基础

### 1.1 向量的定义

**向量**是既有大小又有方向的量，是线性代数的基本元素。

在 $n$ 维空间中，向量 $\mathbf{v}$ 表示为：

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n$$

### 1.2 向量的基本运算

#### 向量加法

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}$$

**几何意义**：平行四边形法则

#### 标量乘法

$$c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \\ cv_n \end{bmatrix}$$

**几何意义**：向量的缩放

#### 点积（内积）

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = \mathbf{u}^T \mathbf{v}$$

**性质**：
- 交换律：$\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$
- 分配律：$\mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w}$

**几何意义**：$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$

#### 叉积（外积，仅3D）

$$\mathbf{u} \times \mathbf{v} = \begin{bmatrix} u_2v_3 - u_3v_2 \\ u_3v_1 - u_1v_3 \\ u_1v_2 - u_2v_1 \end{bmatrix}$$

**几何意义**：垂直于两个向量的平面，大小等于平行四边形面积

### 1.3 向量的范数

**L2范数（欧几里得范数）**：

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{\mathbf{v}^T \mathbf{v}}$$

**L1范数**：

$$\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|$$

**L∞范数**：

$$\|\mathbf{v}\|_\infty = \max_i |v_i|$$

### 1.4 单位向量与方向

**单位向量**：

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

**方向余弦**：向量与坐标轴的夹角余弦

---

## 2. 矩阵基础

### 2.1 矩阵的定义

**矩阵**是由数字排列成的矩形阵列。一个 $m \times n$ 矩阵 $\mathbf{A}$：

$$\mathbf{A} = \begin{bmatrix} 
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

### 2.2 矩阵的基本运算

#### 矩阵加法

$$\mathbf{A} + \mathbf{B} = [a_{ij} + b_{ij}]$$

**条件**：同型矩阵（相同维度）

#### 标量乘法

$$c\mathbf{A} = [ca_{ij}]$$

#### 矩阵乘法

$$\mathbf{C} = \mathbf{A}\mathbf{B}$$

其中 $c_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$

**条件**：$\mathbf{A}$ 的列数 = $\mathbf{B}$ 的行数

**重要性质**：
- 不满足交换律：$\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$（一般情况）
- 满足结合律：$(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$
- 满足分配律：$\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{A}\mathbf{B} + \mathbf{A}\mathbf{C}$

### 2.3 矩阵的转置

$$\mathbf{A}^T = [a_{ji}]$$

**性质**：
- $(\mathbf{A}^T)^T = \mathbf{A}$
- $(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$
- $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T$

### 2.4 矩阵的迹

对于方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$：

$$\text{tr}(\mathbf{A}) = \sum_{i=1}^{n} a_{ii}$$

**性质**：
- $\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})$
- $\text{tr}(\mathbf{A}\mathbf{B}) = \text{tr}(\mathbf{B}\mathbf{A})$
- $\text{tr}(\mathbf{A}^T) = \text{tr}(\mathbf{A})$

---

## 3. 矩阵运算

### 3.1 矩阵的逆

对于方阵 $\mathbf{A}$，如果存在 $\mathbf{A}^{-1}$ 使得：

$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$

则称 $\mathbf{A}$ 可逆（非奇异）。

**求逆方法**：
- 高斯消元法
- 伴随矩阵法：$\mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})} \text{adj}(\mathbf{A})$
- LU分解

**性质**：
- $(\mathbf{A}^{-1})^{-1} = \mathbf{A}$
- $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$
- $(\mathbf{A}^T)^{-1} = (\mathbf{A}^{-1})^T$

### 3.2 行列式

对于 $2 \times 2$ 矩阵：

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

对于 $3 \times 3$ 矩阵（萨吕法则）：

$$\det(\mathbf{A}) = a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})$$

**性质**：
- $\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A})\det(\mathbf{B})$
- $\det(\mathbf{A}^T) = \det(\mathbf{A})$
- $\det(\mathbf{A}^{-1}) = \frac{1}{\det(\mathbf{A})}$

### 3.3 矩阵的秩

**秩**是矩阵中线性无关的行（或列）的最大数量。

**性质**：
- $\text{rank}(\mathbf{A}) = \text{rank}(\mathbf{A}^T)$
- $\text{rank}(\mathbf{A}\mathbf{B}) \leq \min(\text{rank}(\mathbf{A}), \text{rank}(\mathbf{B}))$
- 对于方阵，$\mathbf{A}$ 可逆当且仅当 $\text{rank}(\mathbf{A}) = n$

### 3.4 伪逆（Moore-Penrose逆）

对于非方阵或奇异矩阵，定义伪逆 $\mathbf{A}^+$：

$$\mathbf{A}^+ = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T \quad \text{(当 } \mathbf{A}^T\mathbf{A} \text{ 可逆时)}$$

**应用**：最小二乘问题

---

## 4. 特殊矩阵

### 4.1 单位矩阵

$$\mathbf{I} = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

**性质**：$\mathbf{A}\mathbf{I} = \mathbf{I}\mathbf{A} = \mathbf{A}$

### 4.2 对角矩阵

$$\mathbf{D} = \text{diag}(d_1, d_2, \ldots, d_n) = \begin{bmatrix} d_1 & 0 & \cdots & 0 \\ 0 & d_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & d_n \end{bmatrix}$$

**性质**：
- 乘法简单：$\mathbf{D}_1\mathbf{D}_2 = \text{diag}(d_{1i}d_{2i})$
- 求逆简单：$\mathbf{D}^{-1} = \text{diag}(1/d_i)$

### 4.3 对称矩阵

$$\mathbf{A} = \mathbf{A}^T$$

**性质**：
- 特征值为实数
- 特征向量正交

### 4.4 正交矩阵

$$\mathbf{Q}^T\mathbf{Q} = \mathbf{Q}\mathbf{Q}^T = \mathbf{I}$$

即 $\mathbf{Q}^{-1} = \mathbf{Q}^T$

**性质**：
- 保持向量长度：$\|\mathbf{Q}\mathbf{v}\| = \|\mathbf{v}\|$
- 保持角度

**应用**：旋转矩阵

### 4.5 正定矩阵

对于对称矩阵 $\mathbf{A}$，如果对任意非零向量 $\mathbf{x}$：

$$\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$$

则称 $\mathbf{A}$ 正定。

**性质**：
- 所有特征值为正
- 行列式为正
- 可逆

**应用**：协方差矩阵、Hessian矩阵

---

## 5. 在具身智能中的应用

### 5.1 坐标变换

**旋转矩阵**：正交矩阵，用于坐标系旋转

$$\mathbf{R} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**齐次变换矩阵**：

$$\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

### 5.2 状态估计

**卡尔曼滤波**：使用矩阵运算进行状态估计

$$\mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k + \mathbf{B}\mathbf{u}_k$$

### 5.3 神经网络

**前向传播**：矩阵乘法

$$\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}$$

**反向传播**：矩阵求导

### 5.4 最小二乘问题

求解 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 的最小二乘解：

$$\mathbf{x} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{b} = \mathbf{A}^+\mathbf{b}$$

---

## 6. 相关论文

### 基础理论

1. **"Matrix Analysis"** - Horn & Johnson (经典教材)
   - 矩阵理论的权威参考书

2. **"The Matrix Cookbook"** - Petersen & Pedersen
   - 矩阵运算公式速查手册

3. **"A Survey of Matrix Theory and Matrix Inequalities"** - Marcus & Minc
   - 矩阵理论综述

### 数值计算

4. **"Numerical Linear Algebra"** - Trefethen & Bau
   - 数值线性代数经典教材

5. **"Matrix Computations"** - Golub & Van Loan
   - 矩阵计算算法权威参考书

### 应用相关

6. **"The Singular Value Decomposition: Anatomy of Optimizing an Algorithm for Extreme Scale"** - Dongarra et al.
   - SVD算法优化

7. **"Randomized Algorithms for Matrices and Data"** - Mahoney
   - 大规模矩阵随机算法

---

## 7. 实践练习

### 练习1：基础运算

使用Python/NumPy实现：

```python
import numpy as np

# 1. 向量运算
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# 计算点积、叉积、范数

# 2. 矩阵运算
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 计算矩阵乘法、转置、逆、行列式
```

### 练习2：坐标变换

实现2D旋转矩阵：

```python
def rotation_matrix_2d(theta):
    """
    创建2D旋转矩阵
    theta: 旋转角度（弧度）
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

# 验证正交性
# 验证旋转后向量长度不变
```

### 练习3：最小二乘

实现线性回归：

```python
def linear_regression_normal_equation(X, y):
    """
    使用正规方程求解线性回归
    X: 特征矩阵 (n_samples, n_features)
    y: 目标向量 (n_samples,)
    """
    # 添加偏置项
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # 正规方程: theta = (X^T X)^(-1) X^T y
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta
```

### 练习4：矩阵分解验证

```python
# 验证 LU 分解
from scipy.linalg import lu

A = np.array([[4, 3], [6, 3]])
P, L, U = lu(A)

# 验证 P @ A = L @ U
```

---

**下一步**：[矩阵分解](02-matrix-decomposition.md)
