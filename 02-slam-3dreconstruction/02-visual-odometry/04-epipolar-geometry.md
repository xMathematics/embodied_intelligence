# 2.4 对极几何

## 1. 概述

对极几何（Epipolar Geometry）描述了同一场景在不同视角下拍摄的两幅图像之间的几何关系。它是视觉SLAM中运动估计的数学基础。

## 2. 对极几何基础

### 2.1 基本概念

**对极点（Epipole）**：相机光心在另一图像平面上的投影点。

**对极线（Epipolar Line）**：包含对极点的直线，空间点在另一图像上的投影必位于对应的对极线上。

**对极平面（Epipolar Plane）**：空间点和两个相机光心确定的平面。

### 2.2 对极约束

对于两幅图像中的对应点 $\mathbf{x}_1$ 和 $\mathbf{x}_2$：

$$ \mathbf{x}_2^T \mathbf{E} \mathbf{x}_1 = 0 \quad \text{（已标定相机）} $$
$$ \mathbf{x}_2^T \mathbf{F} \mathbf{x}_1 = 0 \quad \text{（未标定相机）} $$

## 3. 本质矩阵（Essential Matrix）

### 3.1 定义

本质矩阵 $\mathbf{E} \in \mathbb{R}^{3 \times 3}$ 描述了已标定相机之间的对极几何关系：

$$ \mathbf{E} = [\mathbf{t}]_\times \mathbf{R} $$

其中 $\mathbf{R}$ 是旋转矩阵，$\mathbf{t}$ 是平移向量。

### 3.2 性质

- **秩**：$\text{rank}(\mathbf{E}) = 2$
- **自由度**：5（旋转3 + 平移2，尺度不可确定）
- **奇异值**：$\mathbf{E} = \mathbf{U} \text{diag}(\sigma, \sigma, 0) \mathbf{V}^T$

### 3.3 从本质矩阵恢复位姿

$$\mathbf{E} = \mathbf{U} \text{diag}(\sigma, \sigma, 0) \mathbf{V}^T$$

四种可能的 $\mathbf{R}, \mathbf{t}$ 组合：

$$ \mathbf{t}_1 = \mathbf{U}_{:,3}, \quad \mathbf{t}_2 = -\mathbf{U}_{:,3} $$
$$ \mathbf{R}_1 = \mathbf{U} \mathbf{W} \mathbf{V}^T, \quad \mathbf{R}_2 = \mathbf{U} \mathbf{W}^T \mathbf{V}^T $$

其中 $\mathbf{W} = \begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}$。

通过三角化验证选择正确的解（两个相机前方有最多3D点的那组解）。

## 4. 基础矩阵（Fundamental Matrix）

### 4.1 定义

基础矩阵 $\mathbf{F} \in \mathbb{R}^{3 \times 3}$ 描述了未标定相机之间的对极几何关系：

$$ \mathbf{F} = \mathbf{K}_2^{-T} \mathbf{E} \mathbf{K}_1^{-1} $$

### 4.2 性质

- **秩**：$\text{rank}(\mathbf{F}) = 2$
- **自由度**：7（3×3矩阵减去尺度一致性再减去秩2约束）
- **对极线**：$\mathbf{l}_2 = \mathbf{F} \mathbf{x}_1$，$\mathbf{l}_1 = \mathbf{F}^T \mathbf{x}_2$

## 5. 单应矩阵（Homography Matrix）

### 5.1 定义

当场景是平面或相机纯旋转时，使用单应矩阵 $\mathbf{H} \in \mathbb{R}^{3 \times 3}$：

$$ \mathbf{x}_2 = \mathbf{H} \mathbf{x}_1 $$

### 5.2 平面诱导的单应

对于平面 $\mathbf{n}^T \mathbf{X} = d$：

$$ \mathbf{H} = \mathbf{K}_2 \left( \mathbf{R} - \frac{\mathbf{t} \mathbf{n}^T}{d} \right) \mathbf{K}_1^{-1} $$

### 5.3 单应矩阵分解

从 $\mathbf{H}$ 可以恢复4组可能的 $\mathbf{R}, \mathbf{t}, \mathbf{n}, d$。

## 6. 三焦张量（Trifocal Tensor）

三焦张量 $\mathcal{T}$ 描述了三视图之间的几何关系，是三视图的对极几何约束：

$$ \sum_{i=1}^{3} \mathbf{x}_1^i (\mathbf{x}_2 \times \sum_{k=1}^{3} \mathcal{T}_i^{jk} \mathbf{x}_3^k) = 0 $$

三焦张量在点-线-点对应中提供了比两视图更强的约束。

## 7. 退化情况分析

| 情况 | 可用模型 | 问题 |
|------|----------|------|
| 纯旋转 | $\mathbf{H}$ | $\mathbf{E}$ 无解 |
| 平面场景 | $\mathbf{H}$ | $\mathbf{E}$ 退化 |
| 极短基线 | $\mathbf{H}$ | $\mathbf{E}$ 不稳定 |
| 一般运动 | $\mathbf{E}$ 或 $\mathbf{F}$ | 无 |

## 8. 参考文献

1. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
2. Longuet-Higgins, H. C. (1981). A computer algorithm for reconstructing a scene from two projections. *Nature*, 293, 133-135.
3. Nistér, D. (2004). An efficient solution to the five-point relative pose problem. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 26(6), 756-770.
4. Ma, Y., Soatto, S., Košecká, J., & Sastry, S. S. (2004). *An Invitation to 3-D Vision*. Springer.
