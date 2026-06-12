# 3.2 粒子滤波SLAM

## 1. 概述

粒子滤波SLAM使用Rao-Blackwellized粒子滤波（RBPF）将SLAM问题分解为位姿估计和地图构建两部分。这种方法可以处理非高斯、非线性的SLAM问题。

## 2. Rao-Blackwellized粒子滤波

### 2.1 分解

RBPF-SLAM利用条件独立性将SLAM后验分解：

$$ P(\mathbf{x}_{1:k}, \mathbf{m} \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k}) = P(\mathbf{m} \mid \mathbf{x}_{1:k}, \mathbf{z}_{1:k}) \cdot P(\mathbf{x}_{1:k} \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k}) $$

其中：
- $P(\mathbf{x}_{1:k} \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k})$：位姿部分，用粒子滤波估计
- $P(\mathbf{m} \mid \mathbf{x}_{1:k}, \mathbf{z}_{1:k})$：地图部分，在位姿已知条件下解析计算

### 2.2 算法流程

1. 从提议分布采样新粒子位姿
2. 根据观测更新粒子权重
3. 根据粒子的位姿轨迹更新地图（每个粒子维护独立地图）
4. 重采样

## 3. FastSLAM

### 3.1 FastSLAM 1.0

FastSLAM 1.0（Montemerlo et al., 2002）是第一个实用的RBPF-SLAM系统。

**提议分布**：使用里程计模型：

$$ \mathbf{x}_k^{[i]} \sim P(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{[i]}, \mathbf{u}_k) $$

**权重更新**：

$$ w_k^{[i]} = w_{k-1}^{[i]} \cdot \int P(\mathbf{z}_k \mid \mathbf{x}_k^{[i]}, \mathbf{m}_{k-1}^{[i]}) P(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{[i]}, \mathbf{u}_k) d\mathbf{x}_k $$

**地图维护**：每个粒子使用EKF维护特征地图。

### 3.2 FastSLAM 2.0

FastSLAM 2.0（Montemerlo et al., 2003）使用改进的提议分布，考虑当前观测：

$$ \mathbf{x}_k^{[i]} \sim P(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{[i]}, \mathbf{u}_k, \mathbf{z}_k) $$

**优势**：
- 需要更少的粒子
- 在高精度传感器下表现更好
- 权重分布更集中，减少粒子退化

## 4. 提议分布设计

提议分布的选择对粒子滤波性能至关重要：

| 提议分布 | 优势 | 劣势 | 适用场景 |
|----------|------|------|----------|
| 里程计模型 | 简单 | 需要大量粒子 | 低精度传感器 |
| 观测模型最优 | 粒子少 | 复杂解析 | 高精度传感器 |
| 自适应混合 | 平衡 | 实现复杂 | 通用 |

## 5. 重采样策略

### 5.1 重采样方法

| 方法 | 原理 | 方差 | 复杂度 |
|------|------|------|--------|
| 多项式重采样 | 多项式分布采样 | 高 | O(N) |
| 残差重采样 | 先取整数部分再随机 | 中 | O(N) |
| 分层重采样 | 系统划分区间 | 低 | O(N) |
| 系统重采样 | 等间距采样 | 最低 | O(N) |

### 5.2 自适应重采样

根据有效粒子数决定是否重采样：

$$ N_{\text{eff}} = \frac{1}{\sum_i (w^{[i]})^2} $$

当 $N_{\text{eff}} < N_{\text{th}}$ 时执行重采样。

## 6. 粒子退化与贫化

**退化（Degeneracy）**：几乎所有粒子权重接近0，只有少数粒子有显著权重。

**贫化（Impoverishment）**：重采样后粒子多样性丧失，大量粒子完全相同。

## 7. FastSLAM系统对比

| 特性 | FastSLAM 1.0 | FastSLAM 2.0 |
|------|-------------|-------------|
| 提议分布 | 里程计模型 | 考虑观测的最优分布 |
| 所需粒子数 | 多(1000+) | 少(100+) |
| 传感器噪声 | 适合低精度 | 适合高精度 |
| 地图精度 | 中 | 高 |
| 计算复杂度 | 高 | 中 |

## 8. 参考文献

1. Montemerlo, M., Thrun, S., Koller, D., & Wegbreit, B. (2002). FastSLAM: A factored solution to the simultaneous localization and mapping problem. *AAAI*.
2. Montemerlo, M., Thrun, S., Koller, D., & Wegbreit, B. (2003). FastSLAM 2.0: An improved particle filtering algorithm for simultaneous localization and mapping that provably converges. *IJCAI*.
3. Doucet, A., de Freitas, N., Murphy, K., & Russell, S. (2000). Rao-Blackwellised particle filtering for dynamic Bayesian networks. *UAI*.
4. Grisetti, G., Stachniss, C., & Burgard, W. (2007). Improved techniques for grid mapping with Rao-Blackwellized particle filters. *IEEE Transactions on Robotics*, 23(1), 34-46.
