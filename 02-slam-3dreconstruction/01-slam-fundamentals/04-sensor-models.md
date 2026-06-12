# 1.4 传感器模型

## 1. 概述

传感器模型是SLAM系统的基石，它描述了传感器如何感知环境以及如何从传感器数据中提取信息。准确理解传感器模型对于设计鲁棒的SLAM系统至关重要。

## 2. 相机模型

### 2.1 针孔相机模型

针孔模型是最常用的相机模型，描述了三维空间点到二维图像平面的投影过程。

**投影方程**：

$$ \lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} $$

其中相机内参矩阵 $\mathbf{K}$ 为：

$$ \mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} $$

$f_x, f_y$ 是焦距（像素单位），$(c_x, c_y)$ 是主点坐标。

**归一化坐标**：

$$ \begin{bmatrix} x_n \\ y_n \end{bmatrix} = \begin{bmatrix} X/Z \\ Y/Z \end{bmatrix} $$

**像素坐标**：

$$ \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x x_n + c_x \\ f_y y_n + c_y \end{bmatrix} $$

### 2.2 畸变模型

实际相机存在镜头畸变，主要包括径向畸变和切向畸变。

**径向畸变**（由镜头形状引起）：

$$ \begin{aligned} x_{\text{dist}} &= x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \\ y_{\text{dist}} &= y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) \end{aligned} $$

其中 $r^2 = x_n^2 + y_n^2$，$k_1, k_2, k_3$ 是径向畸变系数。

**切向畸变**（由镜头与成像平面不平行引起）：

$$ \begin{aligned} x_{\text{dist}} &= x_n + [2p_1 x_n y_n + p_2 (r^2 + 2x_n^2)] \\ y_{\text{dist}} &= y_n + [p_1 (r^2 + 2y_n^2) + 2p_2 x_n y_n] \end{aligned} $$

其中 $p_1, p_2$ 是切向畸变系数。

### 2.3 双目相机模型

双目相机通过视差计算深度：

$$ Z = \frac{f \cdot b}{d} $$

其中 $b$ 是基线距离，$d = u_L - u_R$ 是视差。

### 2.4 RGB-D相机模型

RGB-D相机直接提供深度信息，主要有两种技术：

**结构光法**（如Kinect v1）：
- 投影红外散斑图案
- 通过图案变形计算深度
- 精度较高，但受环境光影响大

**飞行时间法（ToF）**（如Kinect v2/Azure Kinect）：
- 发射调制红外光
- 测量光飞行时间
- 受环境光影响较小
- 存在多径干扰问题

### 2.5 全景相机模型

全景相机（如鱼眼、Omnidirectional）提供宽视场角：

**统一球面模型**（Unified Sphere Model）：
将点投影到单位球面，再投影到图像平面：

$$ \begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} f_x \frac{x}{\xi \sqrt{x^2+y^2+z^2}+z} + c_x \\ f_y \frac{y}{\xi \sqrt{x^2+y^2+z^2}+z} + c_y \end{bmatrix} $$

其中 $\xi$ 是镜面参数。

## 3. IMU模型

### 3.1 测量模型

**加速度计**：

$$ \mathbf{a}_m = \mathbf{a} + \mathbf{b}_a + \mathbf{n}_a + \mathbf{R}^T \mathbf{g} $$

其中 $\mathbf{a}_m$ 是测量值，$\mathbf{a}$ 是真实加速度，$\mathbf{b}_a$ 是零偏，$\mathbf{n}_a$ 是噪声，$\mathbf{g}$ 是重力。

**陀螺仪**：

$$ \boldsymbol{\omega}_m = \boldsymbol{\omega} + \mathbf{b}_g + \mathbf{n}_g $$

其中 $\boldsymbol{\omega}_m$ 是测量值，$\boldsymbol{\omega}$ 是真实角速度，$\mathbf{b}_g$ 是零偏，$\mathbf{n}_g$ 是噪声。

### 3.2 噪声特性

IMU噪声主要分为两部分：

**白噪声**（White Noise / AWGN）：
$$ \mathbb{E}[\mathbf{n}(t)] = 0, \quad \mathbb{E}[\mathbf{n}(t_1)\mathbf{n}(t_2)^T] = \sigma^2 \delta(t_1-t_2) $$

**零偏随机游走**（Bias Random Walk）：
$$ \dot{\mathbf{b}} = \boldsymbol{\eta}_b, \quad \boldsymbol{\eta}_b \sim \mathcal{N}(0, \sigma_b^2) $$

### 3.3 Allan方差分析

Allan方差是分析IMU噪声特性的标准工具，可以识别出：

| 噪声类型 | Allan方差斜率 | 对应参数 |
|----------|---------------|----------|
| 量化噪声 | $-1$ | $\sigma_Q$ |
| 白噪声 | $-1/2$ | $\sigma_{AW}$ |
| 零偏不稳定 | $0$ | $\sigma_{BI}$ |
| 随机游走 | $+1/2$ | $\sigma_{RW}$ |
| 速率斜坡 | $+1$ | $\sigma_{RR}$ |

## 4. LiDAR模型

### 4.1 光束模型（Beam Model）

$$ P(z_t \mid x_t, m) = \begin{cases} w_{\text{hit}} P_{\text{hit}} \\ w_{\text{short}} P_{\text{short}} \\ w_{\text{max}} P_{\text{max}} \\ w_{\text{rand}} P_{\text{rand}} \end{cases} $$

### 4.2 似然场模型（Likelihood Field Model）

$$ P(z_t \mid x_t, m) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{d^2}{2\sigma^2}\right) $$

其中 $d$ 是测量点到最近障碍物的距离。

### 4.3 LiDAR传感器的类型

| 类型 | 工作原理 | 典型产品 | 特点 |
|------|----------|----------|------|
| 机械旋转 | 多线激光器旋转 | Velodyne HDL-64E | 360°FOV, 机械磨损 |
| 固态（Flash） | 面阵探测器 | Ouster OS系列 | 无运动部件 |
| MEMS振镜 | 微振镜扫描 | Livox Horizon | 视场角较小 |
| 光学相控阵（OPA） | 光学天线阵列 | 尚未量产 | 纯固态, 潜力大 |

## 5. 其它传感器模型

### 5.1 GPS模型

$$ \mathbf{z}_{\text{GPS}} = \mathbf{p} + \mathbf{n}, \quad \mathbf{n} \sim \mathcal{N}(0, \mathbf{\Sigma}_{\text{GPS}}) $$

GPS精度受多种因素影响：
- 多径效应（城市峡谷）
- 大气延迟
- 卫星几何分布（DOP值）
- 信号遮挡（室内不可用）

### 5.2 轮式里程计模型

$$ \begin{bmatrix} \Delta x \\ \Delta y \\ \Delta \theta \end{bmatrix} = \begin{bmatrix} \frac{\Delta s_r + \Delta s_l}{2} \cos(\theta + \frac{\Delta s_r - \Delta s_l}{2b}) \\ \frac{\Delta s_r + \Delta s_l}{2} \sin(\theta + \frac{\Delta s_r - \Delta s_l}{2b}) \\ \frac{\Delta s_r - \Delta s_l}{b} \end{bmatrix} $$

其中 $\Delta s_r, \Delta s_l$ 是左右轮的移动距离，$b$ 是轮距。

### 5.3 声纳/超声波模型

声纳传感器通过飞行时间测量距离，具有锥形波束特性：

$$ z = \frac{ct}{2} $$

其中 $c$ 是声速，$t$ 是飞行时间。

## 6. 多传感器标定

多传感器标定是传感器融合的前提：

| 标定类型 | 方法 | 精度 |
|----------|------|------|
| 相机内参标定 | 棋盘格法（Zhang's method） | 高 |
| 双目立体标定 | 双目标定 | 高 |
| 相机-IMU外参标定 | Kalibr, IMU-Camera calibration | 中 |
| 相机-LiDAR标定 | 标定板法, 互信息法 | 中 |
| LiDAR-IMU标定 | 运动约束法 | 中 |

## 7. 参考文献

1. Zhang, Z. (2000). A flexible new technique for camera calibration. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), 1330-1334.
2. Scaramuzza, D., & Fraundorfer, F. (2011). Visual odometry: Part I: The first 30 years and fundamentals. *IEEE Robotics & Automation Magazine*, 18(4), 80-92.
3. Forster, C., Carlone, L., Dellaert, F., & Scaramuzza, D. (2017). On-manifold preintegration for real-time visual-inertial odometry. *IEEE Transactions on Robotics*, 33(1), 1-21.
4. El-Sheimy, N., Hou, H., & Niu, X. (2008). Analysis and modeling of inertial sensors using Allan variance. *IEEE Transactions on Instrumentation and Measurement*, 57(1), 140-149.
