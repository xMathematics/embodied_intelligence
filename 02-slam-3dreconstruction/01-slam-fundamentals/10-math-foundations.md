# 1.10 数学基础补充

## 1. 概述

SLAM和三维重建涉及多个数学分支。本章总结性的介绍SLAM所需的核心数学工具，为后续章节提供参考。

## 2. 线性代数

### 2.1 矩阵分解

**奇异值分解（SVD）**：

$$ \mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T $$

其中 $\mathbf{U}^T \mathbf{U} = \mathbf{I}$，$\mathbf{V}^T \mathbf{V} = \mathbf{I}$，$\mathbf{\Sigma} = \text{diag}(\sigma_1, \ldots, \sigma_r)$。

SVD在SLAM中的应用：
- 本质矩阵/基础矩阵的求解
- 最小二乘问题的求解
- 点云配准（ICP中SVD求解）
- 矩阵伪逆计算

**特征值分解**：

$$ \mathbf{A} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^{-1} $$

用于：主成分分析（PCA）、协方差分析、不确定性椭圆。

**Cholesky分解**（对称正定矩阵）：

$$ \mathbf{A} = \mathbf{L} \mathbf{L}^T $$

用于：稀疏线性系统的快速求解。

**QR分解**：

$$ \mathbf{A} = \mathbf{Q} \mathbf{R} $$

用于：最小二乘的数值稳定求解，iSAM中的增量更新。

### 2.2 舒尔补（Schur Complement）

对于分块矩阵 $\mathbf{M} = \begin{bmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{bmatrix}$：

舒尔补为 $\mathbf{S} = \mathbf{D} - \mathbf{C} \mathbf{A}^{-1} \mathbf{B}$

舒尔补在BA中的作用：
- 将大系统分解为相机位姿和地图点两个子系统
- 通过边缘化地图点得到缩小的相机系统

## 3. 稀疏线性代数

### 3.1 稀疏矩阵

SLAM中的信息矩阵/海森矩阵是高度稀疏的，因为：
- 每个观测只关联一个位姿和一个地图点
- 位姿之间只通过运动约束相邻连接

### 3.2 稀疏求解器

常用稀疏线性系统求解器：

| 求解器 | 特点 | 适用范围 |
|--------|------|----------|
| CHOLMOD | Cholesky分解，多线程 | 大规模稀疏正定系统 |
| SuiteSparse | 多种分解方法 | 通用稀疏问题 |
| Eigen Sparse | 模板化，易用 | 中小规模 |
| CSparse | 轻量级 | 教学和原型 |

### 3.3 变量消元与重排序

变量消元顺序对求解效率影响巨大：
- COLAMD（Column Approximate Minimum Degree）
- METIS（基于图划分）

## 4. 概率与统计

### 4.1 重要概率分布

**高斯分布**：SLAM中最基本的分布

$$ P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) $$

**卡方分布**：用于一致性检验和匹配验证

$$ \chi^2_k = \sum_{i=1}^k Z_i^2, \quad Z_i \sim \mathcal{N}(0,1) $$

**Wishart分布**：协方差矩阵的先验分布

### 4.2 假设检验

**卡方检验**：用于判断观测是否为内点

$$ T = \mathbf{e}^T \mathbf{\Sigma}^{-1} \mathbf{e} \sim \chi^2_d $$

如果 $T > \chi^2_{d, \alpha}$，则拒绝假设（视为外点）。

## 5. 优化理论

### 5.1 最小二乘问题

**线性最小二乘**：

$$ \mathbf{x}^* = \arg\min_{\mathbf{x}} \|\mathbf{A} \mathbf{x} - \mathbf{b}\|^2 $$

正规方程解：$\mathbf{x}^* = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{b}$

**非线性最小二乘**：

$$ \mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_i \|\mathbf{e}_i(\mathbf{x})\|^2 $$

### 5.2 信赖域方法

在信赖域半径内最小化模型函数：

$$ \min_{\Delta \mathbf{x}} m(\Delta \mathbf{x}) = \frac{1}{2} \|\mathbf{e} + \mathbf{J} \Delta \mathbf{x}\|^2, \quad \text{s.t. } \|\Delta \mathbf{x}\| \leq \Delta $$

### 5.3 线搜索方法

沿搜索方向逐步减小步长：

$$ \mathbf{x}^{t+1} = \mathbf{x}^t + \alpha \mathbf{d}^t $$

## 6. 数值计算注意事项

### 6.1 数值稳定性

- 避免矩阵求逆（使用分解代替）
- 使用归一化提高数值稳定性
- 注意浮点数精度限制

### 6.2 常用数值技巧

- **阻尼因子**（Levenberg-Marquardt）：在正规方程对角线加阻尼
- **自适应步长**：Armijo条件、Wolfe条件
- **有限差分**：当解析雅可比不可用时

## 7. 参考文献

1. Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
2. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
3. Nocedal, J., & Wright, S. (2006). *Numerical Optimization* (2nd ed.). Springer.
4. Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM.
