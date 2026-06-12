# 3.4 光束法平差（Bundle Adjustment）

## 1. 概述

光束法平差（Bundle Adjustment, BA）是SLAM和三维重建中最核心的优化技术。它同时优化所有相机位姿和3D点坐标，最小化观测值与预测值之间的重投影误差。

## 2. BA问题形式化

### 2.1 重投影误差

$$ \mathbf{e}_{ij} = \mathbf{z}_{ij} - \pi(\mathbf{x}_i, \mathbf{y}_j) $$

其中 $\mathbf{z}_{ij}$ 是第 $i$ 个相机观测到的第 $j$ 个特征点的像素坐标，$\pi$ 是投影函数。

### 2.2 BA目标函数

$$ \min_{\{\mathbf{x}_i\}, \{\mathbf{y}_j\}} \sum_{i=1}^{m} \sum_{j \in \mathcal{V}(i)} \|\mathbf{z}_{ij} - \pi(\mathbf{x}_i, \mathbf{y}_j)\|^2_{\mathbf{\Sigma}_{ij}} $$

其中 $\mathcal{V}(i)$ 是第 $i$ 个相机可见的特征点集合。

## 3. BA的稀疏结构

### 3.1 海森矩阵的稀疏性

BA的海森矩阵具有箭形稀疏结构：

$$ \mathbf{H} = \begin{bmatrix} \mathbf{H}_{cc} & \mathbf{H}_{cp} \\ \mathbf{H}_{pc} & \mathbf{H}_{pp} \end{bmatrix} $$

其中 $\mathbf{H}_{cc}$（相机-相机）是对角块矩阵，$\mathbf{H}_{pp}$（点-点）是对角矩阵，$\mathbf{H}_{cp}$（相机-点）是稀疏块矩阵。

### 3.2 舒尔补

利用舒尔补将BA分解为两个子系统：

**第一步**：求解缩小的相机系统

$$ [\mathbf{H}_{cc} - \mathbf{H}_{cp} \mathbf{H}_{pp}^{-1} \mathbf{H}_{pc}] \Delta \mathbf{x}_c = \mathbf{b}_c - \mathbf{H}_{cp} \mathbf{H}_{pp}^{-1} \mathbf{b}_p $$

**第二步**：反求点的增量

$$ \Delta \mathbf{x}_p = \mathbf{H}_{pp}^{-1} (\mathbf{b}_p - \mathbf{H}_{pc} \Delta \mathbf{x}_c) $$

## 4. BA的变体

### 4.1 按优化范围分类

| 类型 | 优化范围 | 计算量 | 精度 | 应用 |
|------|----------|--------|------|------|
| 全局BA | 所有相机和点 | 最高 | 最高 | 离线优化 |
| 局部BA | 局部窗口内 | 中 | 高 | 实时SLAM |
| 滑动窗口BA | 固定大小窗口 | 低 | 中 | 实时VIO |
| 增量BA | 增量更新 | 低 | 高 | 大规模SLAM |

### 4.2 运动BA（Motion-Only BA）

只优化相机位姿，固定3D点位置：

$$ \min_{\{\mathbf{x}_i\}} \sum_{i,j} \|\mathbf{z}_{ij} - \pi(\mathbf{x}_i, \mathbf{y}_j)\|^2 $$

### 4.3 结构BA（Structure-Only BA）

只优化3D点位置，固定相机位姿：

$$ \min_{\{\mathbf{y}_j\}} \sum_{i,j} \|\mathbf{z}_{ij} - \pi(\mathbf{x}_i, \mathbf{y}_j)\|^2 $$

## 5. BA的鲁棒性

### 5.1 鲁棒核函数

在BA中使用鲁棒核函数降低误匹配的影响：

$$ \min \sum_{i,j} \rho(\|\mathbf{z}_{ij} - \pi(\mathbf{x}_i, \mathbf{y}_j)\|) $$

### 5.2 外点剔除

在BA之前/期间剔除高重投影误差的观测：
- **χ²检验**：$\mathbf{e}^T \mathbf{\Sigma}^{-1} \mathbf{e} < \chi^2_{d, 0.95}$
- **Covariance加权**：降低不可靠观测的权重

## 6. 大规模BA

| 方法 | 核心技术 | 时间复杂度 | 适用规模 |
|------|---------|-----------|---------|
| SBA | 稀疏BA | O(N³) | 中等 |
| SSBA | 超稀疏BA | O(N²) | 大 |
| PCG | 预条件共轭梯度 | 近似线性 | 超大 |
| 分块BA | 分布式计算 | 线性 | 超大 |

## 7. BA优化库对比

| 库 | 语言 | 自动微分 | 稀疏求解 | 特点 |
|---|------|---------|---------|------|
| Ceres Solver | C++ | 是 | CHOLMOD | Google出品，功能丰富 |
| g2o | C++ | 否 | CSparse/CHOLMOD | SLAM专用，接口友好 |
| GTSAM | C++ | 是 | Eigen/CHOLMOD | 因子图，增量优化 |
| PyTorch BA | Python | 是 | 可微BA | 深度学习集成 |

## 8. 参考文献

1. Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). Bundle adjustment—a modern synthesis. *Vision Algorithms: Theory and Practice*.
2. Lourakis, M. I., & Argyros, A. A. (2009). SBA: A software package for generic sparse bundle adjustment. *ACM Transactions on Mathematical Software*, 36(1), 1-30.
3. Agarwal, S., et al. (2010). Building Rome in a day. *Communications of the ACM*, 54(10), 105-112.
4. Wu, C., Agarwal, S., Curless, B., & Seitz, S. M. (2011). Multicore bundle adjustment. *CVPR*.
