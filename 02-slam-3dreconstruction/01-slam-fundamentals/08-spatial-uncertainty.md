# 1.8 空间不确定性

## 1. 概述

空间不确定性是SLAM和机器人感知中的核心概念。由于传感器噪声、运动误差和计算近似，机器人的位姿估计和地图构建总是伴随着不确定性。正确建模和管理不确定性对于SLAM系统的鲁棒性和一致性至关重要。

## 2. 不确定性来源

### 2.1 主要来源

SLAM中的不确定性来自多个方面：

1. **传感器噪声**：所有传感器都存在测量噪声
2. **模型误差**：简化的数学模型无法完全描述真实物理过程
3. **线性化误差**：非线性系统的线性化近似引入误差
4. **数据关联错误**：错误的特征匹配导致估计偏差
5. **数值误差**：浮点数计算精度限制

### 2.2 不确定性的传播

不确定性的传播遵循误差传播定律。设 $\mathbf{y} = f(\mathbf{x})$，则：

$$ \mathbf{\Sigma}_y \approx \mathbf{J}_f \mathbf{\Sigma}_x \mathbf{J}_f^T $$

其中 $\mathbf{J}_f$ 是 $f$ 的雅可比矩阵。

## 3. 协方差表示

### 3.1 位姿协方差

对于二维位姿 $\mathbf{x} = (x, y, \theta)^T$：

$$ \mathbf{\Sigma}_{\text{pose}} = \begin{bmatrix} \sigma_x^2 & \sigma_{xy} & \sigma_{x\theta} \\ \sigma_{xy} & \sigma_y^2 & \sigma_{y\theta} \\ \sigma_{x\theta} & \sigma_{y\theta} & \sigma_\theta^2 \end{bmatrix} $$

### 3.2 三维位姿不确定性

对于三维位姿 $\mathbf{T} \in \text{SE}(3)$，协方差定义在李代数空间 $\mathbb{R}^6$ 上：

$$ \boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma}_{\text{pose3}}) $$

其中 $\mathbf{\Sigma}_{\text{pose3}} \in \mathbb{R}^{6 \times 6}$ 是正定对称矩阵。

### 3.3 空间点的不确定性

环境特征点 $\mathbf{p}$ 的不确定性：

$$ \mathbf{p} \sim \mathcal{N}(\hat{\mathbf{p}}, \mathbf{\Sigma}_p) $$

其中 $\mathbf{\Sigma}_p \in \mathbb{R}^{3 \times 3}$。

## 4. 流形上的不确定性

### 4.1 旋转的不确定性

旋转 $\mathbf{R} \in \text{SO}(3)$ 的不确定性不能直接在旋转矩阵上建模，而是在李代数上建模：

$$ \mathbf{R} = \bar{\mathbf{R}} \exp(\boldsymbol{\phi}^\wedge), \quad \boldsymbol{\phi} \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma}_\phi) $$

其中 $\boldsymbol{\phi} \in \mathbb{R}^3$ 是旋转扰动。

### 4.2 各向同性 vs 各向异性

- **各向同性**（Isotopic）：在所有方向上不确定性相同（球形高斯分布）
- **各向异性**（Anisotropic）：在不同方向上不确定性不同（椭球形高斯分布）

## 5. 一阶误差传播

### 5.1 变换的误差传播

对于点 $\mathbf{p}$ 经过变换 $\mathbf{T}$：

$$ \mathbf{p}' = \mathbf{T} \mathbf{p} $$

考虑 $\mathbf{T}$ 和 $\mathbf{p}$ 都有不确定性：

$$ \mathbf{\Sigma}_{p'} \approx \mathbf{R} \mathbf{\Sigma}_p \mathbf{R}^T + [\mathbf{p}]_\times^\wedge \mathbf{\Sigma}_\phi [\mathbf{p}]_\times^{\wedge T} $$

### 5.2 位姿链的误差累积

对于一系列位姿变换 $\mathbf{T}_{0n} = \mathbf{T}_{01} \mathbf{T}_{12} \cdots \mathbf{T}_{n-1,n}$，累积协方差为：

$$ \mathbf{\Sigma}_{0n} = \sum_{i=1}^{n} \text{Ad}(\mathbf{T}_{i,n}) \mathbf{\Sigma}_{i-1,i} \text{Ad}(\mathbf{T}_{i,n})^T $$

这解释了为什么SLAM中位姿误差随着运动距离增加而累积（漂移）。

## 6. 不确定性椭圆/椭球

### 6.1 置信椭圆（2D）

对于 $n\sigma$ 置信椭圆：

$$ (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = n^2 $$

椭圆半轴长度为 $\lambda_i n$，方向为对应的特征向量 $\mathbf{v}_i$，其中 $\lambda_i$ 是 $\mathbf{\Sigma}$ 的特征值。

### 6.2 置信椭球（3D）

类似地，3D置信椭球的半轴为 $\sqrt{\lambda_i} n$。

## 7. 信息矩阵

### 7.1 定义

信息矩阵（Information Matrix / Precision Matrix）是协方差矩阵的逆：

$$ \mathbf{\Omega} = \mathbf{\Sigma}^{-1} $$

### 7.2 信息形式的优势

1. **稀疏性**：条件独立变量对应的信息矩阵元素为0
2. **可加性**：独立观测的信息可以相加
3. **高效更新**：增量更新只需修改局部信息矩阵

### 7.3 图SLAM中的信息矩阵

在图SLAM中，全局信息矩阵具有高度的稀疏性，这是因为每个观测只关联少量的变量：

$$ \mathbf{\Omega} = \sum_{i} \mathbf{J}_i^T \mathbf{\Omega}_i \mathbf{J}_i $$

## 8. 一致性分析

### 8.1 估计一致性

估计 $\hat{\mathbf{x}}$ 被认为是一致的（consistent），如果：

$$ \mathbb{E}[(\mathbf{x} - \hat{\mathbf{x}})(\mathbf{x} - \hat{\mathbf{x}})^T] \leq \mathbf{\Sigma}_{\hat{x}} $$

即估计的协方差不会低估真实不确定性。

### 8.2 EKF-SLAM的不一致性

EKF-SLAM存在已知的不一致性问题，主要源于：
1. **线性化误差**：线性化点偏离真实值
2. **虚假信息**：线性化导致的信息错误累积
3. **可观性失真**：线性化改变了系统的可观性结构

### 8.3 FEJ（First-Estimate Jacobian）

为解决EKF-SLAM的不一致性，Huang等人提出FEJ方法——在首次估计值处计算雅可比矩阵，而不是在当前的更新值处计算。

## 9. 参考文献

1. Huang, G. P., Mourikis, A. I., & Roumeliotis, S. I. (2008). Analysis and improvement of the consistency of extended Kalman filter based SLAM. *IEEE International Conference on Robotics and Automation*.
2. Censi, A. (2007). On achievable accuracy for pose estimation. *IEEE International Conference on Robotics and Automation*.
3. Barfoot, T. D. (2017). *State Estimation for Robotics*. Cambridge University Press.
4. Smith, R., Self, M., & Cheeseman, P. (1986). Estimating uncertain spatial relationships in robotics. *IEEE Conference on Robotics and Automation*.
