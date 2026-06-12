# 3.1 EKF-SLAM滤波方法

## 1. 概述

扩展卡尔曼滤波（Extended Kalman Filter, EKF）是SLAM领域最早和最经典的方法之一。EKF-SLAM将机器人位姿和所有地图特征点联合建模为高斯分布，通过预测和更新步骤递归估计状态。

## 2. EKF-SLAM推导

### 2.1 状态向量

$$ \mathbf{x} = [\mathbf{x}_v^T, \mathbf{y}_1^T, \mathbf{y}_2^T, \ldots, \mathbf{y}_n^T]^T $$

其中 $\mathbf{x}_v$ 是机器人位姿，$\mathbf{y}_i$ 是第 $i$ 个特征。

### 2.2 状态协方差

$$ \mathbf{P} = \begin{bmatrix} \mathbf{P}_{vv} & \mathbf{P}_{vm} \\ \mathbf{P}_{mv} & \mathbf{P}_{mm} \end{bmatrix} $$

- $\mathbf{P}_{vv}$：机器人位姿的不确定性
- $\mathbf{P}_{mm}$：地图的不确定性
- $\mathbf{P}_{vm}, \mathbf{P}_{mv}$：机器人和地图的互协方差

### 2.3 预测步骤

$$ \hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1|k-1}, \mathbf{u}_k) $$
$$ \mathbf{P}_{k|k-1} = \mathbf{F}_k \mathbf{P}_{k-1|k-1} \mathbf{F}_k^T + \mathbf{Q}_k $$

其中 $\mathbf{F}_k = \frac{\partial f}{\partial \mathbf{x}}\big|_{\hat{\mathbf{x}}_{k-1|k-1}}$。

### 2.4 观测更新

$$ \mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k)^{-1} $$
$$ \hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1})) $$
$$ \mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} $$

### 2.5 地图扩展

当观测到新特征时，需要将新特征添加到状态向量中：

$$ \mathbf{x}_{\text{new}} = g(\mathbf{x}_v, \mathbf{z}_{\text{new}}) $$
$$ \mathbf{P}_{\text{new}} = \begin{bmatrix} \mathbf{P} & \mathbf{P} \mathbf{G}_x^T \\ \mathbf{G}_x \mathbf{P} & \mathbf{G}_x \mathbf{P} \mathbf{G}_x^T + \mathbf{G}_z \mathbf{R} \mathbf{G}_z^T \end{bmatrix} $$

## 3. EKF-SLAM的计算复杂度

EKF-SLAM的主要计算瓶颈在于协方差矩阵的维护：

- 协方差矩阵维度：$(N_v + 3N_m) \times (N_v + 3N_m)$
- 更新复杂度：$O(N_m^2)$
- 大规模场景下不可行

## 4. EKF-SLAM的一致性分析

### 4.1 不一致性问题

EKF-SLAM已知存在一致性问题：估计的协方差总是低估真实不确定性，导致滤波结果过度自信。

### 4.2 原因分析

1. **线性化误差**：在估计值处线性化，但真实值偏离估计值
2. **可观测性失真**：线性化后系统不可观测子空间的维度与真实系统不同
3. **虚假信息**：EKF更新隐含地引入了不存在的约束

### 4.3 FEJ（First Estimate Jacobian）

FEJ方法在**首次估计值**处计算雅可比，而不是在更新后的估计值处：

$$ \mathbf{H}_k = \frac{\partial h}{\partial \mathbf{x}}\bigg|_{\bar{\mathbf{x}}_0} $$

FEJ恢复系统的真实可观测性属性，显著改善一致性。

## 5. 其他滤波方法

### 5.1 UKF-SLAM

无迹卡尔曼滤波（UKF）使用Sigma点传播，无需线性化：

$$ \boldsymbol{\mathcal{X}} = [\hat{\mathbf{x}}, \hat{\mathbf{x}} \pm \sqrt{(n+\lambda)\mathbf{P}}] $$

精度高于EKF，但计算量与EKF相同。

### 5.2 SEIF（Sparse Extended Information Filter）

SEIF利用信息矩阵的近似稀疏性，将计算复杂度从 $O(N^2)$ 降低到 $O(N)$。

### 5.3 D&C SLAM（Divide and Conquer）

将地图分块处理，各块独立维护协方差，定期合并。

## 6. EKF-SLAM的历史地位

EKF-SLAM虽然在现代已被图优化方法取代，但：
- 确立了SLAM问题的概率框架
- 证明了SLAM问题的收敛性
- 提供了实时的SLAM解决方案
- 深刻影响了后续所有SLAM研究

## 7. 参考文献

1. Smith, R., Self, M., & Cheeseman, P. (1986). Estimating uncertain spatial relationships in robotics. *IEEE Conference on Robotics and Automation*.
2. Dissanayake, G., Newman, P., Clark, S., Durrant-Whyte, H., & Csorba, M. (2001). A solution to the simultaneous localization and map building (SLAM) problem. *IEEE Transactions on Robotics and Automation*, 17(3), 229-241.
3. Huang, G. P., Mourikis, A. I., & Roumeliotis, S. I. (2008). Analysis and improvement of the consistency of extended Kalman filter based SLAM. *ICRA*.
4. Thrun, S., et al. (2004). The sparse extended information filter. *Robotics: Science and Systems*.
