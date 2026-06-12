# 3.10 边缘化与舒尔补

## 1. 概述

边缘化（Marginalization）是SLAM系统中的关键技术，用于在固定计算预算下移除旧状态同时保留其信息。舒尔补（Schur Complement）是实现高效边缘化的数学工具。

## 2. 舒尔补

### 2.1 定义

对于分块矩阵 $\mathbf{M} = \begin{bmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{bmatrix}$：

舒尔补为：$\mathbf{S} = \mathbf{D} - \mathbf{C} \mathbf{A}^{-1} \mathbf{B}$

### 2.2 矩阵求逆引理

$$ \begin{bmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{bmatrix}^{-1} = \begin{bmatrix} \mathbf{A}^{-1} + \mathbf{A}^{-1} \mathbf{B} \mathbf{S}^{-1} \mathbf{C} \mathbf{A}^{-1} & -\mathbf{A}^{-1} \mathbf{B} \mathbf{S}^{-1} \\ -\mathbf{S}^{-1} \mathbf{C} \mathbf{A}^{-1} & \mathbf{S}^{-1} \end{bmatrix} $$

## 3. SLAM中的边缘化

### 3.1 为什么需要边缘化

在滑动窗口SLAM（如VINS-Mono, DSO）中：
- 计算资源有限
- 需要移除旧状态
- 同时保留旧状态的信息（约束）

### 3.2 边缘化过程

假设我们有以下最小二乘问题：

$$ \min_{\mathbf{x}_m, \mathbf{x}_r} \|\mathbf{e}(\mathbf{x}_m, \mathbf{x}_r)\|^2 $$

其中 $\mathbf{x}_m$ 是要边缘化的变量，$\mathbf{x}_r$ 是保留变量。

线性化后的正规方程：

$$ \begin{bmatrix} \mathbf{H}_{mm} & \mathbf{H}_{mr} \\ \mathbf{H}_{rm} & \mathbf{H}_{rr} \end{bmatrix} \begin{bmatrix} \Delta \mathbf{x}_m \\ \Delta \mathbf{x}_r \end{bmatrix} = \begin{bmatrix} \mathbf{b}_m \\ \mathbf{b}_r \end{bmatrix} $$

通过舒尔补边缘化 $\Delta \mathbf{x}_m$：

$$ (\mathbf{H}_{rr} - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{H}_{mr}) \Delta \mathbf{x}_r = \mathbf{b}_r - \mathbf{H}_{rm} \mathbf{H}_{mm}^{-1} \mathbf{b}_m $$

即：

$$ \tilde{\mathbf{H}}_{rr} \Delta \mathbf{x}_r = \tilde{\mathbf{b}}_r $$

其中 $\tilde{\mathbf{H}}_{rr}$ 和 $\tilde{\mathbf{b}}_r$ 包含了被边缘化变量的信息，作为保留变量的先验约束。

## 4. 滑动窗口滤波

### 4.1 基本原理

滑动窗口滤波（Sliding Window Filter）是VIO中的标准方法：

1. 维护一个固定大小的关键帧窗口
2. 当新关键帧加入时，最旧的关键帧被边缘化
3. 边缘化将旧帧的信息转换为先验约束

### 4.2 关键帧选择与边缘化策略

常见策略：
- **Marge最旧帧**：保留短时约束（适合低速运动）
- **Marge共同可见最少的帧**：保留信息丰富的帧
- **Marge视差最小的帧**：避免退化

## 5. Fill-in问题

### 5.1 填充现象

边缘化引入的填充（Fill-in）：

边缘化前（稀疏）：$\mathbf{H}$ 具有箭形稀疏结构
边缘化后（变密）：先验信息矩阵 $\tilde{\mathbf{H}}_{rr}$ 变得稠密

### 5.2 缓解策略

- **近似边缘化**：忽略部分弱填充项
- **稀疏化**：对信息矩阵进行稀疏近似
- **自适应边缘化**：选择最小化填充的边缘化顺序

## 6. First Estimate Jacobian (FEJ)

### 6.1 问题

边缘化时雅可比矩阵的计算点选择会影响系统的一致性和可观性。

### 6.2 FEJ原则

FEJ要求在**首次估计值**处计算雅可比矩阵：

$$ \mathbf{J} = \frac{\partial \mathbf{e}}{\partial \mathbf{x}} \bigg|_{\mathbf{x}_0} $$

而非在当前的迭代估计值处计算。

### 6.3 为什么FEJ重要

- 维持系统的真实可观性子空间
- 防止虚假信息注入
- 提高估计一致性

## 7. 参考文献

1. Sibley, G., Matthies, L., & Sukhatme, G. (2010). Sliding window filter with application to planetary landing. *Journal of Field Robotics*, 27(5), 587-608.
2. Huang, G. P., Mourikis, A. I., & Roumeliotis, S. I. (2008). Analysis and improvement of the consistency of extended Kalman filter based SLAM. *ICRA*.
3. Leutenegger, S., Lynen, S., Bosse, M., Siegwart, R., & Furgale, P. (2015). Keyframe-based visual-inertial odometry using nonlinear optimization. *The International Journal of Robotics Research*, 34(3), 314-334.
4. Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A robust and versatile monocular visual-inertial state estimator. *IEEE Transactions on Robotics*, 34(4), 1004-1020.
