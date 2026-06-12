# 4.3 IMU预积分

## 1. 概述

IMU预积分（IMU Preintegration）是视觉-惯性SLAM中的关键技术。它将高频IMU测量值在两个关键帧之间预积分为相对运动约束，避免了在优化过程中重复积分IMU测量。

## 2. IMU运动学

### 2.1 连续时间模型

$$ \begin{aligned} \dot{\mathbf{R}} &= \mathbf{R} [\boldsymbol{\omega}_m - \mathbf{b}_g - \mathbf{n}_g]_\times \\ \dot{\mathbf{v}} &= \mathbf{R} (\mathbf{a}_m - \mathbf{b}_a - \mathbf{n}_a) + \mathbf{g} \\ \dot{\mathbf{p}} &= \mathbf{v} \\ \dot{\mathbf{b}}_g &= \mathbf{n}_{bg} \\ \dot{\mathbf{b}}_a &= \mathbf{n}_{ba} \end{aligned} $$

### 2.2 离散时间传播

在两个关键帧 $i$ 和 $j$ 之间：

$$ \begin{aligned} \mathbf{R}_j &= \mathbf{R}_i \prod_{k=i}^{j-1} \exp((\boldsymbol{\omega}_k - \mathbf{b}_{g,k}) \Delta t) \\ \mathbf{v}_j &= \mathbf{v}_i + \mathbf{g} \Delta t_{ij} + \sum_{k=i}^{j-1} \mathbf{R}_k (\mathbf{a}_k - \mathbf{b}_{a,k}) \Delta t \\ \mathbf{p}_j &= \mathbf{p}_i + \sum_{k=i}^{j-1} \left[ \mathbf{v}_k \Delta t + \frac{1}{2} \mathbf{g} \Delta t^2 + \frac{1}{2} \mathbf{R}_k (\mathbf{a}_k - \mathbf{b}_{a,k}) \Delta t^2 \right] \end{aligned} $$

## 3. 预积分公式

### 3.1 预积分测量值

将参考帧从世界坐标系改为关键帧 $i$：

$$ \begin{aligned} \Delta \mathbf{R}_{ij} &= \mathbf{R}_i^T \mathbf{R}_j = \prod_{k=i}^{j-1} \exp((\boldsymbol{\omega}_k - \mathbf{b}_{g,k}) \Delta t) \\ \Delta \mathbf{v}_{ij} &= \mathbf{R}_i^T (\mathbf{v}_j - \mathbf{v}_i - \mathbf{g} \Delta t_{ij}) = \sum_{k=i}^{j-1} \Delta \mathbf{R}_{ik} (\mathbf{a}_k - \mathbf{b}_{a,k}) \Delta t \\ \Delta \mathbf{p}_{ij} &= \mathbf{R}_i^T (\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i \Delta t_{ij} - \frac{1}{2} \mathbf{g} \Delta t_{ij}^2) \\ &= \sum_{k=i}^{j-1} \left[ \Delta \mathbf{v}_{ik} \Delta t + \frac{1}{2} \Delta \mathbf{R}_{ik} (\mathbf{a}_k - \mathbf{b}_{a,k}) \Delta t^2 \right] \end{aligned} $$

### 3.2 噪声传播

预积分测量值受IMU噪声影响，定义噪声传播的协方差矩阵 $\mathbf{\Sigma}_{ij}$，用于后续优化的信息矩阵。

## 4. 零偏更新校正

当IMU零偏估计在优化过程中更新时，无需重新预积分，使用一阶近似校正：

$$ \begin{aligned} \Delta \tilde{\mathbf{R}}_{ij}(\mathbf{b}_g) &\approx \Delta \tilde{\mathbf{R}}_{ij}(\bar{\mathbf{b}}_g) \exp\left(\frac{\partial \Delta \mathbf{R}_{ij}}{\partial \mathbf{b}_g} \delta \mathbf{b}_g\right) \\ \Delta \tilde{\mathbf{v}}_{ij}(\mathbf{b}_g, \mathbf{b}_a) &\approx \Delta \tilde{\mathbf{v}}_{ij}(\bar{\mathbf{b}}_g, \bar{\mathbf{b}}_a) + \frac{\partial \Delta \mathbf{v}_{ij}}{\partial \mathbf{b}_g} \delta \mathbf{b}_g + \frac{\partial \Delta \mathbf{v}_{ij}}{\partial \mathbf{b}_a} \delta \mathbf{b}_a \\ \Delta \tilde{\mathbf{p}}_{ij}(\mathbf{b}_g, \mathbf{b}_a) &\approx \Delta \tilde{\mathbf{p}}_{ij}(\bar{\mathbf{b}}_g, \bar{\mathbf{b}}_a) + \frac{\partial \Delta \mathbf{p}_{ij}}{\partial \mathbf{b}_g} \delta \mathbf{b}_g + \frac{\partial \Delta \mathbf{p}_{ij}}{\partial \mathbf{b}_a} \delta \mathbf{b}_a \end{aligned} $$

## 5. 预积分残差

IMU预积分因子在因子图中的残差：

$$ \mathbf{r}_{ij} = \begin{bmatrix} \ln(\Delta \tilde{\mathbf{R}}_{ij}^T \mathbf{R}_i^T \mathbf{R}_j)^\vee \\ \mathbf{R}_i^T (\mathbf{v}_j - \mathbf{v}_i - \mathbf{g} \Delta t_{ij}) - \Delta \tilde{\mathbf{v}}_{ij} \\ \mathbf{R}_i^T (\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i \Delta t_{ij} - \frac{1}{2} \mathbf{g} \Delta t_{ij}^2) - \Delta \tilde{\mathbf{p}}_{ij} \\ \mathbf{b}_{g,j} - \mathbf{b}_{g,i} \\ \mathbf{b}_{a,j} - \mathbf{b}_{a,i} \end{bmatrix} $$

## 6. 参考文献

1. Lupton, T., & Sukkarieh, S. (2012). Visual-inertial-aided navigation for high-dynamic motion in built environments without initial conditions. *IEEE Transactions on Robotics*, 28(1), 61-76.
2. Forster, C., Carlone, L., Dellaert, F., & Scaramuzza, D. (2017). On-manifold preintegration for real-time visual-inertial odometry. *IEEE Transactions on Robotics*, 33(1), 1-21.
3. Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A robust and versatile monocular visual-inertial state estimator. *IEEE Transactions on Robotics*, 34(4), 1004-1020.
