# 2.7 ICP与点云配准

## 1. 概述

ICP（Iterative Closest Point）是点云配准中最经典的算法，用于将两个点云对齐到同一个坐标系。在RGB-D SLAM、LiDAR SLAM和三维重建中广泛应用。

## 2. 经典ICP

### 2.1 问题定义

给定源点云 $\mathcal{P} = \{\mathbf{p}_1, \ldots, \mathbf{p}_N\}$ 和目标点云 $\mathcal{Q} = \{\mathbf{q}_1, \ldots, \mathbf{q}_M\}$，寻找最优刚体变换 $\mathbf{T} = (\mathbf{R}, \mathbf{t})$：

$$ \mathbf{R}^*, \mathbf{t}^* = \arg\min_{\mathbf{R}, \mathbf{t}} \sum_{i=1}^{N} \|\mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i\|^2 $$

### 2.2 ICP算法步骤

1. **匹配**：对每个 $\mathbf{p}_i$，找到 $\mathcal{Q}$ 中最近的 $\mathbf{q}_j$
2. **求解**：计算最优变换 $(\mathbf{R}, \mathbf{t})$
3. **应用**：用求解的变换更新点云
4. **迭代**：重复直到收敛

**收敛判断**：均方误差变化小于阈值，或达到最大迭代次数。

### 2.3 SVD求解ICP

1. 计算质心：$\bar{\mathbf{p}} = \frac{1}{N}\sum \mathbf{p}_i$，$\bar{\mathbf{q}} = \frac{1}{N}\sum \mathbf{q}_i$ 
2. 中心化点云：$\mathbf{p}_i' = \mathbf{p}_i - \bar{\mathbf{p}}$，$\mathbf{q}_i' = \mathbf{q}_i - \bar{\mathbf{q}}$
3. 计算协方差矩阵：$\mathbf{H} = \sum \mathbf{p}_i' \mathbf{q}_i'^T$
4. SVD分解：$\mathbf{H} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$
5. 旋转：$\mathbf{R} = \mathbf{V} \mathbf{U}^T$
6. 平移：$\mathbf{t} = \bar{\mathbf{q}} - \mathbf{R} \bar{\mathbf{p}}$

**行列式检查**：确保 $\det(\mathbf{R}) = +1$，否则需要反射修正。

## 3. ICP变体

### 3.1 Point-to-Plane ICP

$$ \mathbf{R}^*, \mathbf{t}^* = \arg\min \sum_{i} ((\mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i)^T \mathbf{n}_i)^2 $$

其中 $\mathbf{n}_i$ 是目标点 $\mathbf{q}_i$ 的法向量。精度更高，但需要法向量信息。

### 3.2 Plane-to-Plane ICP

$$ \mathbf{R}^*, \mathbf{t}^* = \arg\min \sum_{i} \|(\mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i)^T \mathbf{n}_i\|^2 + \|\mathbf{R} \mathbf{n}_i^p - \mathbf{n}_i^q\|^2 $$

同时最小化点和法向量的误差。

### 3.3 Generalized ICP（GICP）

GICP将ICP统一为概率框架，Point-to-Point和Point-to-Plane均是其特例。

**GICP模型**：假设每个点都有协方差 $\mathbf{C}_i^P$ 和 $\mathbf{C}_i^Q$：

$$ \mathbf{T}^* = \arg\min_{\mathbf{T}} \sum_i d_i^T (\mathbf{C}_i^Q + \mathbf{R} \mathbf{C}_i^P \mathbf{R}^T)^{-1} d_i $$

其中 $d_i = \mathbf{q}_i - \mathbf{R} \mathbf{p}_i - \mathbf{t}$。

### 3.4 Colored ICP

利用颜色信息增强配准：

$$ \mathbf{T}^* = \arg\min \sum_i (\|\mathbf{e}_g\|^2 + w_c \|\mathbf{e}_c\|^2) $$

其中 $\mathbf{e}_g$ 是几何误差，$\mathbf{e}_c$ 是颜色误差。

## 4. NDT（Normal Distributions Transform）

NDT将点云概率建模为正态分布混合：

**步骤**：
1. 将空间划分为网格（voxel）
2. 每个网格计算正态分布 $(\boldsymbol{\mu}_i, \mathbf{\Sigma}_i)$
3. 优化扫描点在目标分布下的概率

$$ \mathbf{T}^* = \arg\min_{\mathbf{T}} \sum_i -\log\left(\sum_j w_j \mathcal{N}(\mathbf{T} \mathbf{p}_i; \boldsymbol{\mu}_j, \mathbf{\Sigma}_j)\right) $$

NDT不需要显式的匹配步骤，计算稳定，收敛域宽。

## 5. 鲁棒ICP

### 5.1 稀疏ICP

使用L1范数替代L2范数，对异常值更鲁棒：

$$ \mathbf{R}^*, \mathbf{t}^* = \arg\min \sum_i \|\mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i\|_1 $$

### 5.2 Robust ICP with M-Estimator

使用M-estimator降低外点权重：

$$ \mathbf{R}^*, \mathbf{t}^* = \arg\min \sum_i \rho(\|\mathbf{R} \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i\|) $$

### 5.3 剔除异常匹配

最大距离阈值、边界一致性、法向量一致性等策略。

## 6. ICP算法对比

| 算法 | 精度 | 收敛域 | 鲁棒性 | 计算量 | 适用场景 |
|------|------|--------|--------|--------|----------|
| Point-to-Point | 中 | 小 | 低 | 低 | 粗糙配准 |
| Point-to-Plane | 高 | 中 | 中 | 中 | 室内场景 |
| GICP | 高 | 中 | 高 | 中 | 一般场景 |
| NDT | 中 | 大 | 高 | 中 | 室外场景 |
| Colored ICP | 高 | 大 | 高 | 高 | RGB-D配准 |
| Sparse ICP | 中 | 小 | 最高 | 高 | 有外点场景 |

## 7. 参考文献

1. Besl, P. J., & McKay, N. D. (1992). A method for registration of 3-D shapes. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 14(2), 239-256.
2. Chen, Y., & Medioni, G. (1992). Object modelling by registration of multiple range images. *Image and Vision Computing*, 10(3), 145-155.
3. Segal, A., Haehnel, D., & Thrun, S. (2009). Generalized-ICP. *Robotics: Science and Systems*.
4. Biber, P., & Strasser, W. (2003). The normal distributions transform: A new approach to laser scan matching. *IROS*.
5. Park, J., Zhou, Q.-Y., & Koltun, V. (2017). Colored point cloud registration revisited. *ICCV*.
