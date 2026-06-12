# 7.1 增量式SfM

## 1. 概述

结构从运动（Structure from Motion, SfM）是从无序图像集合中恢复相机位姿和场景3D结构的技术。增量式SfM是最经典的SfM方法，通过逐步添加图像来建图和定位。

## 2. SfM基本原理

### 2.1 问题定义

给定 $N$ 幅图像 $I_1, \ldots, I_N$，估计：
- 每幅图像的相机位姿 $\mathbf{P}_i = \mathbf{K}_i [\mathbf{R}_i \mid \mathbf{t}_i]$
- 场景的3D结构 $\mathbf{X}_j \in \mathbb{R}^3$

### 2.2 核心步骤

1. **特征提取与匹配**：在所有图像对之间匹配特征点
2. **几何验证**：验证匹配对的几何一致性（使用基础矩阵或单应矩阵）
3. **场景初始化**：选择一个好的初始图像对
4. **增量注册**：逐步添加新图像
5. **三角化**：为新图像中的匹配点计算3D坐标
6. **BA优化**：定期执行光束法平差

## 3. 增量式SfM流程

### 3.1 初始化图像对选择

选择标准：
- 足够多的匹配点（>100）
- 充足的视差（基础矩阵的RANSAC内点比单应矩阵更多）
- 大基线（提高三角化精度）

### 3.2 图像注册

使用PnP算法将新图像注册到现有模型中：

$$ \mathbf{P}_i = \text{PnP}(\mathbf{X}_j, \mathbf{x}_{ij}) $$

其中 $\mathbf{X}_j$ 是已三角化的3D点，$\mathbf{x}_{ij}$ 是这些点在图像 $i$ 中的2D投影。

### 3.3 三角化

为新匹配的2D对应点计算3D位置：

$$ \mathbf{X}_j = \text{triangulate}(\mathbf{x}_{ij}, \mathbf{x}_{i'j}, \mathbf{P}_i, \mathbf{P}_{i'}) $$

### 3.4 过滤策略

- **高重投影误差**点被剔除：$\|\mathbf{x}_{ij} - \pi(\mathbf{P}_i, \mathbf{X}_j)\| > \text{threshold}$
- **小三角化角度**：< 3° 的点被剔除
- **大重投影误差**的匹配被剔除

## 4. 增量式SfM的挑战

- **漂移累积**：前期的误差会传播到后续
- **计算效率**：每次添加图像后BA，$O(N^3)$
- **初始化依赖**：初始化图像对的选择很重要
- **局部最优**：可能陷入局部最优

## 5. COLMAP的实现

COLMAP（Schönberger & Frahm, 2016）是目前最流行的SfM工具：

**关键改进**：
- 场景不重叠的图像对分组（Scene Graph）
- 基于投票的匹配点筛选
- 按图像重建难度排序的注册策略
- 高效的过滤策略
- 分层BA减少计算量

## 6. 参考文献

1. Snavely, N., Seitz, S. M., & Szeliski, R. (2006). Photo tourism: Exploring photo collections in 3D. *ACM SIGGRAPH*.
2. Schonberger, J. L., & Frahm, J.-M. (2016). Structure-from-motion revisited. *CVPR*.
3. Wu, C. (2013). Towards linear-time incremental structure from motion. *3DV*.
4. Agarwal, S., et al. (2011). Building Rome in a day. *Communications of the ACM*, 54(10), 105-112.
