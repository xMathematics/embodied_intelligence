# 1.5 运动模型

## 1. 概述

运动模型描述了系统状态随时间演变的规律。在SLAM中，运动模型用于预测机器人（或相机）在连续时刻之间的位姿变化。准确的运动模型对于SLAM系统的预测步骤、数据关联和状态估计至关重要。

## 2. 机器人运动模型

### 2.1 速度运动模型（Velocity Motion Model）

假设控制输入为线速度 $v$ 和角速度 $\omega$：

$$ \mathbf{u}_k = \begin{bmatrix} v_k \\ \omega_k \end{bmatrix} $$

**状态更新**：

$$ \begin{aligned} x_{k} &= x_{k-1} - \frac{v_k}{\omega_k} \sin\theta_{k-1} + \frac{v_k}{\omega_k} \sin(\theta_{k-1} + \omega_k \Delta t) \\ y_{k} &= y_{k-1} + \frac{v_k}{\omega_k} \cos\theta_{k-1} - \frac{v_k}{\omega_k} \cos(\theta_{k-1} + \omega_k \Delta t) \\ \theta_{k} &= \theta_{k-1} + \omega_k \Delta t \end{aligned} $$

当 $\omega_k \approx 0$（直线运动）时：

$$ \begin{aligned} x_{k} &= x_{k-1} + v_k \Delta t \cos\theta_{k-1} \\ y_{k} &= y_{k-1} + v_k \Delta t \sin\theta_{k-1} \\ \theta_{k} &= \theta_{k-1} \end{aligned} $$

**噪声模型**：对控制输入加入噪声

$$ \begin{aligned} \hat{v}_k &= v_k + \varepsilon_{\alpha_1 v^2 + \alpha_2 \omega^2} \\ \hat{\omega}_k &= \omega_k + \varepsilon_{\alpha_3 v^2 + \alpha_4 \omega^2} \\ \hat{\gamma}_k &= \varepsilon_{\alpha_5 v^2 + \alpha_6 \omega^2} \end{aligned} $$

其中 $\varepsilon_{\sigma^2} \sim \mathcal{N}(0, \sigma^2)$，$\hat{\gamma}_k$ 是最终旋转的额外噪声。

### 2.2 里程计运动模型（Odometry Motion Model）

假设控制输入为里程计测量值 $\mathbf{u}_k = (\delta_{\text{rot1}}, \delta_{\text{trans}}, \delta_{\text{rot2}})$：

$$ \begin{aligned} \delta_{\text{rot1}} &= \text{atan2}(\bar{y} - y, \bar{x} - x) - \theta \\ \delta_{\text{trans}} &= \sqrt{(\bar{x} - x)^2 + (\bar{y} - y)^2} \\ \delta_{\text{rot2}} &= \bar{\theta} - \theta - \delta_{\text{rot1}} \end{aligned} $$

**状态更新**：

$$ \begin{aligned} x' &= x + \hat{\delta}_{\text{trans}} \cos(\theta + \hat{\delta}_{\text{rot1}}) \\ y' &= y + \hat{\delta}_{\text{trans}} \sin(\theta + \hat{\delta}_{\text{rot1}}) \\ \theta' &= \theta + \hat{\delta}_{\text{rot1}} + \hat{\delta}_{\text{rot2}} \end{aligned} $$

## 3. 相机运动模型

### 3.1 恒速模型（Constant Velocity Model）

假设相机角速度和线速度恒定：

$$ \mathbf{v}_k = \mathbf{v}_{k-1} + \mathbf{w}_v $$
$$ \mathbf{T}_k = \mathbf{T}_{k-1} \circ \exp(\mathbf{v}_k \Delta t) $$

这是视觉SLAM中最常用的运动模型，用于提供初始位姿估计。

### 3.2 恒加速度模型

假设相机加速度恒定（角加速度和线加速度）：

$$ \begin{aligned} \mathbf{a}_k &= \mathbf{a}_{k-1} + \mathbf{w}_a \\ \mathbf{v}_k &= \mathbf{v}_{k-1} + \mathbf{a}_k \Delta t \\ \mathbf{T}_k &= \mathbf{T}_{k-1} \circ \exp(\mathbf{v}_k \Delta t) \end{aligned} $$

### 3.3 恒转向率模型（Constant Turn Rate Model）

假设角速度 $\omega$ 和线速度 $v$ 恒定：

$$ \begin{aligned} \mathbf{T}_{k} &= \mathbf{T}_{k-1} \cdot \begin{bmatrix} \mathbf{R}(\omega \Delta t) & \mathbf{t}(v, \omega, \Delta t) \\ \mathbf{0}^T & 1 \end{bmatrix} \end{aligned} $$

## 4. IMU运动模型

### 4.1 IMU运动学方程

IMU的运动学方程描述了位姿、速度和IMU测量值之间的关系：

$$ \begin{aligned} \dot{\mathbf{R}} &= \mathbf{R} [\boldsymbol{\omega}_m - \mathbf{b}_g - \mathbf{n}_g]_\times \\ \dot{\mathbf{v}} &= \mathbf{R} (\mathbf{a}_m - \mathbf{b}_a - \mathbf{n}_a) + \mathbf{g} \\ \dot{\mathbf{p}} &= \mathbf{v} \\ \dot{\mathbf{b}}_g &= \mathbf{n}_{bg} \\ \dot{\mathbf{b}}_a &= \mathbf{n}_{ba} \end{aligned} $$

### 4.2 IMU预积分

IMU预积分（Preintegration）是视觉-惯性SLAM中的关键技术，它将高频IMU测量值预积分为两个关键帧之间的相对运动约束：

$$ \begin{aligned} \Delta \mathbf{R}_{ij} &= \mathbf{R}_i^T \mathbf{R}_j \\ \Delta \mathbf{v}_{ij} &= \mathbf{R}_i^T (\mathbf{v}_j - \mathbf{v}_i - \mathbf{g} \Delta t_{ij}) \\ \Delta \mathbf{p}_{ij} &= \mathbf{R}_i^T (\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i \Delta t_{ij} - \frac{1}{2} \mathbf{g} \Delta t_{ij}^2) \end{aligned} $$

**预积分测量值的计算**：

$$ \begin{aligned} \Delta \tilde{\mathbf{R}}_{ij} &= \prod_{k=i}^{j-1} \exp\left((\boldsymbol{\omega}_k - \mathbf{b}_g) \Delta t\right) \\ \Delta \tilde{\mathbf{v}}_{ij} &= \sum_{k=i}^{j-1} \Delta \tilde{\mathbf{R}}_{ik} (\mathbf{a}_k - \mathbf{b}_a) \Delta t \\ \Delta \tilde{\mathbf{p}}_{ij} &= \sum_{k=i}^{j-1} \left[ \Delta \tilde{\mathbf{v}}_{ik} \Delta t + \frac{1}{2} \Delta \tilde{\mathbf{R}}_{ik} (\mathbf{a}_k - \mathbf{b}_a) \Delta t^2 \right] \end{aligned} $$

### 4.3 零偏更新时的预积分校正

当IMU零偏估计发生变化时，无需重新计算预积分，使用一阶线性校正：

$$ \begin{aligned} \Delta \mathbf{R}_{ij}(\mathbf{b}_g) &\approx \Delta \tilde{\mathbf{R}}_{ij} \exp\left(\frac{\partial \Delta \mathbf{R}_{ij}}{\partial \mathbf{b}_g} \delta \mathbf{b}_g\right) \\ \Delta \mathbf{v}_{ij}(\mathbf{b}_g, \mathbf{b}_a) &\approx \Delta \tilde{\mathbf{v}}_{ij} + \frac{\partial \Delta \mathbf{v}_{ij}}{\partial \mathbf{b}_g} \delta \mathbf{b}_g + \frac{\partial \Delta \mathbf{v}_{ij}}{\partial \mathbf{b}_a} \delta \mathbf{b}_a \\ \Delta \mathbf{p}_{ij}(\mathbf{b}_g, \mathbf{b}_a) &\approx \Delta \tilde{\mathbf{p}}_{ij} + \frac{\partial \Delta \mathbf{p}_{ij}}{\partial \mathbf{b}_g} \delta \mathbf{b}_g + \frac{\partial \Delta \mathbf{p}_{ij}}{\partial \mathbf{b}_a} \delta \mathbf{b}_a \end{aligned} $$

## 5. 运动模型的对比与选择

| 模型 | 适用传感器 | 精度 | 计算量 | 适用场景 |
|------|-----------|------|--------|----------|
| 速度运动模型 | 轮式机器人 | 中 | 低 | 地面机器人 |
| 里程计运动模型 | 轮式编码器 | 高 | 低 | 有编码器的机器人 |
| 恒速模型 | 相机 | 低 | 极低 | 视觉SLAM初始估计 |
| IMU预积分 | IMU | 高 | 高 | VIO/视觉-惯性SLAM |
| 恒转向率 | 车辆 | 中 | 低 | 自动驾驶 |

## 6. 参考文献

1. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
2. Forster, C., Carlone, L., Dellaert, F., & Scaramuzza, D. (2017). On-manifold preintegration for real-time visual-inertial odometry. *IEEE Transactions on Robotics*, 33(1), 1-21.
3. Lupton, T., & Sukkarieh, S. (2012). Visual-inertial-aided navigation for high-dynamic motion in built environments without initial conditions. *IEEE Transactions on Robotics*, 28(1), 61-76.
4. Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A robust and versatile monocular visual-inertial state estimator. *IEEE Transactions on Robotics*, 34(4), 1004-1020.
