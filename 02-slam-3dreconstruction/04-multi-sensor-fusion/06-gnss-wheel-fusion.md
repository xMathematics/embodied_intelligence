# 4.6 GPS/轮式里程计融合

## 1. 概述

除了视觉和IMU外，GPS和轮式里程计也是SLAM系统中常用的辅助传感器。GPS提供全局位置约束消除累积漂移，轮式里程计提供低成本的运动估计。

## 2. GPS-视觉融合

### 2.1 GPS定位原理

GPS通过测量卫星信号传输时间计算接收器位置：

$$ \rho_i = \sqrt{(x_i - x)^2 + (y_i - y)^2 + (z_i - z)^2} + c \cdot \delta t $$

需要至少4颗卫星求解3D位置和时间偏移。

### 2.2 GPS误差源

- **电离层延迟**：单频GPS最大误差5-10m
- **多径效应**：城市峡谷中可达数十米
- **卫星几何**：DOP值影响精度
- **信号遮挡**：室内/隧道不可用

### 2.3 RTK GPS

实时动态差分GPS（RTK）通过基站差分校正，可达厘米级精度：

$$ \mathbf{p}_{\text{RTK}} = \mathbf{p}_{\text{base}} + \Delta \mathbf{p} $$

### 2.4 GPS-视觉融合策略

- **松耦合**：GPS作为全局位置因子加入图优化
- **紧耦合**：使用伪距/载波相位原始观测融合

## 3. 轮式里程计融合

### 3.1 差速轮模型

两轮差速机器人的里程计：

$$ \begin{aligned} \omega &= \frac{v_r - v_l}{b} \\ v &= \frac{v_r + v_l}{2} \end{aligned} $$

### 3.2 轮式里程计标定

通过Umbilical标定法：

$$ \begin{bmatrix} \delta S_r \\ \delta S_l \\ b \end{bmatrix} = \arg\min \sum_i \|\mathbf{p}_i - \mathbf{p}_i^{\text{gt}}\|^2 $$

### 3.3 轮式-视觉融合

- 轮式里程计提供短时可靠的位移约束
- 视觉提供长时稳定的方向约束
- 在视觉失效时（如纯旋转）轮式里程计提供支持

## 4. 多传感器融合框架

**VINS-Fusion**：支持GPS、轮式里程计的可扩展VIO框架
**LIO-SAM**：支持LiDAR、IMU、GPS的因子图融合
**MaRS**：多传感器融合框架

## 5. 参考文献

1. Parkison, S., et al. (1996). *Global Positioning System: Theory and Applications*. AIAA.
2. Qin, T., Cao, S., Pan, J., & Shen, S. (2019). A general optimization-based framework for local odometry estimation with multiple sensors. *arXiv preprint arXiv:1901.03638*.
3. Soloviev, A. (2008). Tight coupling of GPS and INS for urban navigation. *IEEE Transactions on Aerospace and Electronic Systems*, 44(3).
