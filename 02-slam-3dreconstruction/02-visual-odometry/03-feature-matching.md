# 2.3 特征匹配

## 1. 概述

特征匹配是视觉SLAM中建立帧间数据关联的关键步骤。高质量的匹配是实现准确位姿估计的前提。本章全面介绍从传统匹配方法到现代学习型匹配方法的技术体系。

## 2. 基本匹配策略

### 2.1 暴力匹配（Brute-Force Matching）

对特征集 $\mathcal{F}_1$ 中的每个描述子 $\mathbf{d}_1$，与 $\mathcal{F}_2$ 中的所有描述子计算距离，选择最近邻：

$$ \text{NN}(\mathbf{d}_1) = \arg\min_{\mathbf{d}_2 \in \mathcal{F}_2} \text{dist}(\mathbf{d}_1, \mathbf{d}_2) $$

**复杂度**：$O(N_1 N_2 D)$，其中 $D$ 是描述子维度。

### 2.2 快速近似最近邻（FLANN）

FLANN使用KD-tree或层次聚类树加速匹配：

- **KD-tree**：适用于低维数据（$D \leq 10$）
- **分层k-means树**：适用于高维数据
- **自动选择**：自动选择最优算法和参数

### 2.3 Lowe's Ratio Test

Lowe提出的比率测试用于筛选匹配点：

$$ \text{ratio} = \frac{\text{dist}(\mathbf{d}_1, \text{NN}_1)}{\text{dist}(\mathbf{d}_1, \text{NN}_2)} < \text{threshold} $$

典型阈值：0.6-0.8。阈值越小，匹配越严格，匹配数量越少。

### 2.4 交叉验证匹配

双向匹配一致性检查：从 $\mathcal{F}_1$ 到 $\mathcal{F}_2$ 的最近邻和从 $\mathcal{F}_2$ 到 $\mathcal{F}_1$ 的最近邻互为对方。

## 3. 几何验证

### 3.1 RANSAC（Random Sample Consensus）

RANSAC是最常用的鲁棒估计方法，用于去除误匹配。

**算法流程**：
1. 随机选择最小样本集（例如4对点估计单应矩阵）
2. 使用样本集估计模型参数
3. 计算所有匹配点到模型的内点距离
4. 统计内点数量
5. 重复步骤1-4，选择内点最多的模型
6. 使用所有内点重新估计模型

**RANSAC变体**：
| 算法 | 特点 | 效率 |
|------|------|------|
| RANSAC | 标准方法 | 中 |
| MSAC | 使用M-estimator损失 | 中 |
| PROSAC | 从高置信度样本开始采样 | 高 |
| MLESAC | 最大化似然估计 | 低 |
| MAGSAC | 自适应阈值 | 高 |

### 3.2 PROSAC（Progressive Sample Consensus）

PROSAC利用匹配质量的先验信息指导采样过程，优先采样高质量匹配，加速收敛。

### 3.3 MAGSAC（Marginalizing Sample Consensus）

MAGSAC通过边缘化内点阈值实现无需人工设定阈值的RANSAC。

## 4. 学习型匹配方法

### 4.1 SuperGlue

SuperGlue（CVPR 2020）使用图神经网络和注意力机制进行特征匹配。

**架构**：
1. **特征编码**：将特征点位置和描述子编码为节点特征
2. **图神经网络**：使用注意力聚合消息传递
3. **最优传输**：使用Sinkhorn算法求解分配问题

**注意力机制**：
- **自注意力**（Self-Attention）：同一图像内部的特征交互
- **交叉注意力**（Cross-Attention）：跨图像的特征交互

### 4.2 LoFTR

LoFTR（CVPR 2021）是无检测器的匹配方法，直接在密集特征图上进行匹配。

**关键创新**：
- 使用Transformer处理密集特征
- 线性注意力机制降低计算复杂度
- 粗到细的匹配策略

### 4.3 LightGlue

LightGlue（2023）是SuperGlue的加速版本，根据问题难度自适应计算深度。

**自适应策略**：
- 早期退出：简单图像对在浅层网络即可完成匹配
- 置信度预测：预测是否需要继续深层计算

### 4.4 DKM（Dense Kernelized Feature Matching）

DKM通过密集核化特征匹配，在像素级别进行匹配，达到非常高的精度。

## 5. 极线引导匹配

### 5.1 极线约束

利用极线几何缩小搜索范围：

$$ \mathbf{x}_2^T \mathbf{F} \mathbf{x}_1 = 0 $$

匹配只需在对应极线上搜索，将2D搜索降为1D搜索。

### 5.2 极线搜索

在估计得到基础矩阵后，沿极线搜索匹配点：
- 结合归一化互相关（NCC）
- 半全局匹配（SGM）

## 6. 匹配质量评估

### 6.1 内点率

内点数与总匹配数的比值：

$$ \text{inlier\_ratio} = \frac{N_{\text{inlier}}}{N_{\text{total}}} $$

### 6.2 重投影误差

$$ \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \|\mathbf{x}_i - \pi(\mathbf{X}_i)\|^2} $$

## 7. 参考文献

1. Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. *Communications of the ACM*, 24(6), 381-395.
2. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *IJCV*, 60(2), 91-110.
3. Sarlin, P. E., et al. (2020). SuperGlue: Learning feature matching with graph neural networks. *CVPR*.
4. Sun, J., et al. (2021). LoFTR: Detector-free local feature matching with transformers. *CVPR*.
5. Lindenberger, P., et al. (2023). LightGlue: Local feature matching at light speed. *ICCV*.
