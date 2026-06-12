# 5.3 深度学习方法与NetVLAD

## 1. 概述

深度学习方法在回环检测中的应用极大地提升了在复杂环境下的性能。NetVLAD是其中最经典的工作，将传统VLAD聚合用可微的神经网络层实现。

## 2. NetVLAD

### 2.1 核心思想

NetVLAD（Arandjelovic et al., CVPR 2016）将CNN特征和VLAD聚合结合为一个端到端可训练的模型。

**VLAD回顾**：将局部描述子聚合为全局描述子

$$ V(j, k) = \sum_{i=1}^{N} a_k(\mathbf{x}_i)(\mathbf{x}_i(j) - \mathbf{c}_k(j)) $$

**NetVLAD改进**：用可微的soft分配替代硬分配

$$ \tilde{a}_k(\mathbf{x}_i) = \frac{e^{\mathbf{w}_k^T \mathbf{x}_i + b_k}}{\sum_{k'} e^{\mathbf{w}_{k'}^T \mathbf{x}_i + b_{k'}}} $$

### 2.2 网络架构

```
输入图像 → CNN → 卷积特征图 → NetVLAD层 → 全连接 → 全局描述子
```

### 2.3 训练

- **损失函数**：三元组排名损失
- **数据**：Google Street View Time Machine
- **采样**：正样本（相同地点不同时间）+ 负样本（不同地点）

## 3. 其他深度学习方法

### 3.1 CALC（2018）

使用自编码器学习紧凑的图像描述子，适合大规模回环检测。

### 3.2 DenseVLAD（2018）

使用稠密SIFT和VLAD，在视角变化大的场景表现优秀。

### 3.3 SuperPoint+SuperGlue（2020）

使用学习型特征点和匹配器进行回环检测：
- SuperPoint检测特征点
- SuperGlue进行跨图像匹配
- 匹配数量作为回环得分

### 3.4 Patch-NetVLAD（2021）

在NetVLAD基础上引入局部块信息，融合全局和局部特征。

### 3.5 SFV（Scene Feature Vector, 2023）

基于Transformer的全局场景描述子，利用自注意力机制捕获长距离依赖关系。

## 4. 几何验证

深度学习回环检测通常需要结合几何验证提高精度：

**几何验证步骤**：
1. 匹配特征点
2. 使用RANSAC估计基础矩阵/单应矩阵
3. 检查内点比例
4. 确认回环

## 5. 词袋 vs 深度方法对比

| 特性 | 词袋(DBoW) | NetVLAD | SuperPoint/SuperGlue |
|------|-----------|---------|---------------------|
| 特征 | 手工ORB | CNN学习 | CNN学习 |
| 描述子 | 二进制 | 浮点 | 浮点 |
| 速度 | 极快 | GPU快 | GPU中 |
| 视角鲁棒 | 中 | 高 | 高 |
| 光照鲁棒 | 中 | 高 | 中 |
| 精度(复杂场景) | 中 | 高 | 高 |

## 6. 参考文献

1. Arandjelovic, R., Gronat, P., Torii, A., Pajdla, T., & Sivic, J. (2016). NetVLAD: CNN architecture for weakly supervised place recognition. *CVPR*.
2. DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). SuperPoint: Self-supervised interest point detection and description. *CVPR Workshop*.
3. Sarlin, P. E., et al. (2020). SuperGlue: Learning feature matching with graph neural networks. *CVPR*.
4. Hausler, S., et al. (2021). Patch-NetVLAD: Multi-scale fusion of locally-global descriptors for place recognition. *CVPR*.
