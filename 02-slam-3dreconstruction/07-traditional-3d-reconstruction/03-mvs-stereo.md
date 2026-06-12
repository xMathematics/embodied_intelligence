# 7.3 多视图立体视觉（MVS）

## 1. 概述

多视图立体视觉（Multi-View Stereo, MVS）利用多幅已知位姿的图像重建场景的稠密3D结构。MVS是SfM的后续步骤，将稀疏点云升级为稠密模型。

## 2. MVS问题定义

给定 $N$ 幅已知位姿的图像 $\{(I_i, \mathbf{P}_i)\}_{i=1}^N$，重建场景的稠密3D几何。

## 3. MVS方法分类

### 3.1 基于体素的方法

将空间离散为体素网格，为每个体素分配占据/空闲标签。

**代表方法**：
- **Voxel Coloring**：检查体素的色彩一致性
- **Space Carving**：逐步剔除不一致的体素

**优缺点**：
- 适合拓扑复杂的场景
- 内存与场景体积成正比

### 3.2 基于面片的方法

**PMVS（Patch-based MVS）**：

使用面片（Patch）作为基本重建单元：

1. **初始特征匹配**：生成稀疏面片
2. **面片膨胀**：从种子点向周边扩张
3. **面片过滤**：剔除不一致的面片

**面片优化**：最小化光度一致性误差

$$ \mathbf{p}^* = \arg\min_{\mathbf{p}} \sum_{i \in V(\mathbf{p})} \|I_i(\mathbf{x}_i) - \bar{I}(\mathbf{p})\|^2 $$

### 3.3 基于深度图的方法

为每幅图像估计深度图，然后融合：

1. 为每幅图像生成深度图
2. 深度融合
3. 表面重建

**代表方法**：
- **COLMAP MVS**：像素级视图选择 + 深度估计 + 深度融合
- **PatchMatch MVS**：随机搜索 + 传播的深度估计
- **Gipuma**：GPU加速的MVS

## 4. PatchMatch MVS

### 4.1 基本思想

PatchMatch MVS利用随机搜索和传播进行高效深度估计：

1. **随机初始化**：为每个像素赋予随机深度和法线
2. **传播**：在相邻像素间传播好的深度值
3. **随机搜索**：在当前最佳值附近随机采样
4. **视图选择**：选择最佳参考视图

### 4.2 匹配代价

$$ C(p, d, \mathbf{n}) = 1 - \text{NCC}(p, d, \mathbf{n}) $$

使用归一化互相关（NCC）作为光度一致性度量。

## 5. 深度融合

### 5.1 深度图一致性检查

验证不同视角深度图的一致性：

$$ \|\mathbf{x}_i - \pi(\mathbf{P}_i, \mathbf{X}_j)\| < \text{threshold} $$

### 5.2 深度图融合方法

- **基于可见性**：选择可见性好的深度
- **基于置信度**：根据匹配质量加权
- **基于几何一致性**：检查跨视角几何一致性

## 6. MVS方法对比

| 方法 | 类型 | 精度 | 完整性 | 计算量 |
|------|------|------|--------|--------|
| PMVS | 面片 | 高 | 中 | 高 |
| COLMAP MVS | 深度图 | 高 | 高 | 中 |
| PatchMatch MVS | 深度图 | 中 | 高 | 快 |
| Gipuma | 深度图 | 高 | 高 | GPU快 |
| Voxel Hunter | 体素 | 中 | 中 | 中 |

## 7. 参考文献

1. Furukawa, Y., & Ponce, J. (2010). Accurate, dense, and robust multiview stereopsis. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 32(8), 1362-1376.
2. Schonberger, J. L., Zheng, E., Frahm, J.-M., & Pollefeys, M. (2016). Pixelwise view selection for unstructured multi-view stereo. *ECCV*.
3. Barnes, C., Shechtman, E., Finkelstein, A., & Goldman, D. B. (2009). PatchMatch: A randomized correspondence algorithm for structural image editing. *ACM Transactions on Graphics*, 28(3).
4. Seitz, S. M., Curless, B., Diebel, J., Scharstein, D., & Szeliski, R. (2006). A comparison and evaluation of multi-view stereo reconstruction algorithms. *CVPR*.
