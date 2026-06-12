# 8.3 3D高斯泼溅

## 1. 概述

3D高斯泼溅（3D Gaussian Splatting, 3DGS）是Kerbl等人（SIGGRAPH 2023）提出的新范式，使用3D高斯椭球体显式表示场景。它在保持高质量的同时实现了实时渲染，迅速成为NeRF的重要替代方案。

## 2. 核心原理

### 2.1 高斯椭球体表示

每个高斯由以下参数定义：

- **位置** $\boldsymbol{\mu} \in \mathbb{R}^3$：高斯中心
- **协方差** $\mathbf{\Sigma} \in \mathbb{R}^{3 \times 3}$：椭球形状和大小
- **颜色** $\mathbf{c} \in \mathbb{R}^3$：RGB颜色（或球谐系数）
- **不透明度** $\alpha \in [0, 1]$

$$ G(\mathbf{x}) = \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right) $$

### 2.2 协方差分解

为了保证 $\mathbf{\Sigma}$ 的正定性，将其分解为旋转矩阵 $\mathbf{R}$ 和缩放矩阵 $\mathbf{S}$：

$$ \mathbf{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^T \mathbf{R}^T $$

- 旋转：使用四元数 $\mathbf{q} \in \mathbb{R}^4$ 表示
- 缩放：使用向量 $\mathbf{s} \in \mathbb{R}^3$ 表示

### 2.3 可微光栅化

使用**基于平铺的光栅化器**（Tile-based Rasterizer）：

1. 将屏幕划分为16×16的平铺
2. 将高斯投影到图像平面
3. 每个平铺维护可见高斯列表
4. 按深度排序并混合

**混合公式**：

$$ C = \sum_{i=1}^{N} \alpha_i' \mathbf{c}_i \prod_{j=1}^{i-1} (1 - \alpha_j') $$

其中 $\alpha_i' = \alpha_i G_i^{\text{proj}}(\mathbf{x})$ 是投影后的透明度。

## 3. 自适应密度控制

### 3.1 初始化

从SfM点云初始化高斯：每个SfM点创建一个高斯，协方差初始化为小值。

### 3.2 分裂与克隆

**梯度触发的控制策略**：

- **克隆**：梯度大的高斯 → 在该位置创建新高斯
- **分裂**：过大高斯 → 分裂为两个小高斯
- **剔除**：不透明度太低的高斯 → 移除

### 3.3 透明度重置

定期将高斯的透明度重置，防止陷入局部最优。

## 4. 训练

### 4.1 损失函数

$$ \mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}} $$

其中 $\mathcal{L}_1$ 是L1损失，$\mathcal{L}_{\text{D-SSIM}}$ 是结构相似性损失。

### 4.2 优化

- Adam优化器
- 学习率指数衰减
- 逐属性学习率调度

## 5. 3DGS vs NeRF

| 特性 | 3D高斯泼溅 | NeRF |
|------|-----------|------|
| 表示方式 | 显式（3D高斯） | 隐式（MLP） |
| 渲染速度 | 实时（>100fps） | 慢（>1秒/帧） |
| 训练时间 | 30分钟-1小时 | 数小时 |
| 内存占用 | 高（数千万高斯） | 低（MLP权重） |
| 几何提取 | 需要处理 | 隐式 |
| 编辑能力 | 好（显式操作） | 差 |
| 渲染质量 | 高 | 高 |

## 6. 3DGS的扩展

### 6.1 4D高斯泼溅

将3DGS扩展到动态场景，添加时间维度。

### 6.2 Mip-Splatting

处理多尺度问题，引入3D平滑滤波和2D Mip滤波。

### 6.3 压缩方法

- **LightGaussian**：剪枝 + 量化
- **EfficientGS**：压缩到1/10大小

### 6.4 在SLAM中的应用

- **SplaTAM**：3DGS作为SLAM地图表示
- **GS-SLAM**：实时SLAM系统
- **SIGMA**：语义SLAM + 3DGS

## 7. 参考文献

1. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). 3D Gaussian splatting for real-time radiance field rendering. *ACM Transactions on Graphics*, 42(4).
2. Kopanas, G., et al. (2021). Point-based neural rendering with per-view optimization. *Computer Graphics Forum*.
3. Keetha, N., et al. (2024). SplaTAM: Splat, track & map 3D Gaussians for dense RGB-D SLAM. *CVPR*.
4. Yan, C., et al. (2024). GS-SLAM: Dense visual SLAM with 3D Gaussian splatting. *ICRA*.
