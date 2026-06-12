# 8.2 NeRF变体与改进

## 1. 概述

NeRF激发了大量后续研究工作，从速度、质量、泛化、动态场景等多个方向进行了改进。本章详细介绍重要的NeRF变体。

## 2. 速度优化

### 2.1 Instant-NGP

**Müller et al., SIGGRAPH 2022** - 极速NeRF

**多分辨率哈希编码**：
- 多级哈希表存储特征向量
- 每级独立分辨率，从粗到细
- 可微的查找和插值

**优势**：
- 训练时间从数小时缩短到数秒
- 1080p渲染可达30fps
- 小幅降低质量

### 2.2 Plenoxels

**Fridovich-Keil et al., CVPR 2022** - 不需要神经网络的辐射场

**稀疏体素网格**：
- 每个体素存储球谐系数
- 使用TV正则化
- 完全可微优化

**优势**：
- 不需要神经网络，完全体素表示
- 训练仅需几分钟
- 适合快速重建

### 2.3 TensoRF

**Chen et al., CVPR 2022** - 张量分解

**VM分解**：
$$ \mathcal{A} = \sum_{r=1}^{R} \mathbf{v}_r^1 \circ \mathbf{v}_r^2 \circ \mathbf{v}_r^3 \quad \text{(CP分解)} $$
$$ \mathcal{A} = \sum_{r=1}^{R} \mathbf{v}_r^1 \circ \mathbf{M}_r^{2,3} \quad \text{(VM分解)} $$

将4D张量分解为向量和矩阵的乘积，大幅降低存储。

## 3. 质量改进

### 3.1 Mip-NeRF

**Barron et al., ICCV 2021** - 多尺度抗锯齿NeRF

**集成位置编码（IPE）**：
- 用高斯锥体代替光线
- 沿锥体计算预期特征
- 自然的多尺度分辨率

**优势**：
- 消除锯齿伪影
- 支持连续缩放级别
- 提高远距离细节

### 3.2 Mip-NeRF 360

**Barron et al., CVPR 2022** - 无边界场景

- 非线性参数化（场景收缩）
- 基于提议的采样（proposal sampling）
- 无边界场景的统一处理

## 4. 复杂场景

### 4.1 NeRF-W（NeRF in the Wild）

**Martin-Brualla et al., CVPR 2021** - 互联网照片集

**关键问题**：
- 光照变化
-  transient（瞬时）物体（行人、车辆）
- 不同的相机和处理管线

**解决方案**：
- 每图像外观嵌入（Appearance Embedding）
- Transient网络分支处理瞬时物体
- 不确定性建模

### 4.2 NeRF++

**Zhang et al., 2020** - 处理无边界场景

**双球体参数化**：
- 内部：标准NeRF
- 外部：反向球体参数化

### 4.3 Ref-NeRF

**Verbin et al., SIGGRAPH 2022** - 反射和镜面效果

- 将外观分解为漫反射和镜面反射
- 使用法向量集成提高表面质量
- 更好的反射重建

## 5. 动态场景

### 5.1 D-NeRF

**Pumarola et al., CVPR 2021** - 动态场景NeRF

**变形场**：
- 使用额外的MLP预测变形
- 将动态点映射到规范空间
- 在规范空间中渲染

$$ \mathbf{x}' = \mathbf{x} + \Delta \mathbf{x}(t) $$

### 5.2 Nerfies

**Park et al., CVPR 2021** - 自拍视频重建

- 弹性正则化
- 背景-前景分解
- 适用于自拍视频

## 6. NeRF变体对比

| 方法 | 速度 | 质量 | 动态 | 无边界 | 泛化 |
|------|------|------|------|--------|------|
| NeRF原版 | 慢 | 高 | 否 | 否 | 否 |
| Instant-NGP | 极快 | 高 | 否 | 否 | 否 |
| Mip-NeRF | 中 | 极高 | 否 | 否 | 否 |
| NeRF-W | 慢 | 中 | 部分 | 否 | 否 |
| D-NeRF | 慢 | 中 | 是 | 否 | 否 |
| Plenoxels | 快 | 高 | 否 | 否 | 否 |
| TensoRF | 快 | 高 | 否 | 否 | 否 |

## 7. 参考文献

1. Müller, T., Evans, A., Schied, C., & Keller, A. (2022). Instant neural graphics primitives with a multiresolution hash encoding. *SIGGRAPH*.
2. Barron, J. T., Mildenhall, B., Tancik, M., Hedman, P., Martin-Brualla, R., & Srinivasan, P. P. (2021). Mip-NeRF: A multiscale representation for anti-aliasing neural radiance fields. *ICCV*.
3. Fridovich-Keil, S., et al. (2022). Plenoxels: Radiance fields without neural networks. *CVPR*.
4. Chen, A., Xu, Z., Geiger, A., Yu, J., & Su, H. (2022). TensoRF: Tensorial radiance fields. *CVPR*.
5. Martin-Brualla, R., et al. (2021). NeRF in the Wild: Neural radiance fields for unconstrained photo collections. *CVPR*.
