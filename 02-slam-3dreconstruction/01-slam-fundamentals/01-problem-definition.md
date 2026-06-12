# 1.1 SLAM问题定义

## 1. 概述

同时定位与地图构建（Simultaneous Localization and Mapping，SLAM）是机器人学和计算机视觉领域的核心问题之一。SLAM问题描述的是：**一个机器人在未知环境中从未知初始位置出发，在运动过程中根据自身携带的传感器观测信息，同时估计自身的位姿（定位）和构建环境的地图（建图）**。

这一问题的本质在于：定位需要地图（已知的地标来估计位姿），而建图又需要精确的位姿（已知的位姿来放置地图特征）。这种"鸡和蛋"的循环依赖关系使得SLAM成为极具挑战性的问题。

### 1.1 SLAM的历史发展

| 时期 | 里程碑 | 贡献 |
|------|--------|------|
| 1986-1990 | 概念形成期 | Smith, Cheeseman和Durrant-Whyte将统计估计理论引入机器人定位和建图问题 |
| 1991-1995 | 理论奠基期 | Smith, Self和Cheeseman提出"Stochastic Map"，建立EKF-SLAM框架 |
| 1996-2000 | 收敛性分析 | Dissanayake证明EKF-SLAM的收敛性，奠定理论基础 |
| 2001-2005 | 高效方法 | Montemerlo提出FastSLAM（Rao-Blackwellized粒子滤波），Thrun提出GraphSLAM |
| 2006-2010 | 开源时代 | Klein提出PTAM（Parallel Tracking and Mapping），开创并行化SLAM |
| 2011-2015 | 成熟应用期 | ORB-SLAM发布，LSD-SLAM开创直接法，大规模SLAM系统涌现 |
| 2016-2020 | 多传感器融合 | ORB-SLAM2/3, VINS-Mono, DSO, 深度学习辅助SLAM |
| 2021-至今 | 神经SLAM时代 | NeRF-SLAM, 3D Gaussian Splatting SLAM, 基础模型辅助SLAM |

### 1.2 SLAM问题的数学建模

#### 1.2.1 基本变量定义

在典型的SLAM问题中，我们定义以下变量：

$$\begin{aligned}
\mathbf{x}_k &: \text{机器人（相机）在时刻} k \text{的位姿}, k = 1, \ldots, K \\
\mathbf{u}_k &: \text{从时刻} k-1 \text{到时刻} k \text{的控制输入} \\
\mathbf{m}_j &: \text{第} j \text{个环境特征的坐标}, j = 1, \ldots, N \\
\mathbf{z}_{kj} &: \text{在时刻} k \text{对特征} j \text{的观测}
\end{aligned}$$

#### 1.2.2 运动模型

运动模型描述机器人状态如何随时间演变：

$$\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k) + \mathbf{w}_k, \quad \mathbf{w}_k \sim \mathcal{N}(0, \mathbf{Q}_k)$$

其中 $\mathbf{w}_k$ 是过程噪声，$\mathbf{Q}_k$ 是噪声协方差矩阵。

#### 1.2.3 观测模型

观测模型描述机器人如何感知环境：

$$\mathbf{z}_{kj} = h(\mathbf{x}_k, \mathbf{m}_j) + \mathbf{v}_{kj}, \quad \mathbf{v}_{kj} \sim \mathcal{N}(0, \mathbf{R}_{kj})$$

其中 $\mathbf{v}_{kj}$ 是观测噪声，$\mathbf{R}_{kj}$ 是噪声协方差矩阵。

#### 1.2.4 SLAM问题的概率公式

SLAM问题的目标是估计所有变量在给定所有观测和控制输入下的后验概率：

$$P(\mathbf{x}_{1:K}, \mathbf{m} \mid \mathbf{z}_{1:K}, \mathbf{u}_{1:K})$$

这一概率分布可以通过贝叶斯滤波框架递归求解（见下一章）。

### 1.3 完整SLAM与在线SLAM

根据估计目标的不同，SLAM问题可以分为两种形式：

**完整SLAM（Full SLAM）**：
估计所有时刻的机器人位姿和地图，即求解 $P(\mathbf{x}_{1:K}, \mathbf{m} \mid \mathbf{z}_{1:K}, \mathbf{u}_{1:K})$

**在线SLAM（Online SLAM）**：
只估计当前时刻的机器人位姿和地图，即求解 $P(\mathbf{x}_k, \mathbf{m} \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k})$

### 1.4 SLAM问题的可观测性

SLAM问题的可观测性分析对理解其难度至关重要：

- **单目相机**：尺度不可观测（无法获得绝对尺度信息）
- **双目相机**：尺度可观测（基线提供尺度参考）
- **RGB-D相机**：尺度可观测（直接深度测量）
- **单目+IMU**：尺度弱可观测（需要激励足够的运动）
- **立体+IMU**：完全可观测

### 1.5 SLAM系统的通用框架

一个完整的SLAM系统通常包含以下组件：

```
传感器数据 → [前端(前端)] → [后端(后端)] → [地图]
                  ↓                          ↑
             [回环检测] → 优化信息 → [全局优化]
```

1. **前端（Visual Odometry / Front-end）**：
   - 传感器数据预处理
   - 特征提取与匹配
   - 帧间运动估计
   - 关键帧选择

2. **后端（Back-end / Optimization）**：
   - 接收前端估计的位姿和观测约束
   - 执行滤波或优化
   - 维护全局一致性

3. **回环检测（Loop Closure）**：
   - 识别机器人回到之前访问过的位置
   - 提供全局约束以消除累积漂移

4. **建图（Mapping）**：
   - 根据优化的位姿构建地图
   - 地图可以是稀疏特征点、稠密点云、网格、语义地图等形式

### 1.6 SLAM的分类

#### 按传感器类型

| 类型 | 传感器 | 代表系统 | 特点 |
|------|--------|----------|------|
| 视觉SLAM | 单/双/RGB-D相机 | ORB-SLAM, DSO | 成本低，信息丰富 |
| 激光SLAM | 2D/3D LiDAR | Cartographer, LOAM | 精度高，受光照影响小 |
| 视觉-惯性SLAM | 相机+IMU | VINS-Mono, OKVIS | 鲁棒性好 |
| 多传感器融合SLAM | 多种传感器 | LIO-SAM, R3LIVE | 最鲁棒 |

#### 按方法范式

| 范式 | 核心思想 | 代表系统 |
|------|----------|----------|
| 滤波SLAM | 递归贝叶斯估计 | EKF-SLAM, FastSLAM |
| 图优化SLAM | 非线性最小二乘 | ORB-SLAM, g2o |
| 直接法SLAM | 光度误差最小化 | LSD-SLAM, DSO |
| 学习法SLAM | 深度神经网络 | D2SLAM, DP-SLAM |

### 1.7 SLAM的核心挑战

1. **累积漂移**：位姿估计误差随时间累积
2. **数据关联**：正确建立观测与地图特征之间的对应关系
3. **计算效率**：实时处理大量传感器数据
4. **动态环境**：处理移动物体
5. **光照变化**：适应不同光照条件
6. **低纹理场景**：在缺乏纹理的环境中进行定位
7. **大规模环境**：处理大规模长期运行的SLAM
8. **感知混叠**：不同位置看起来相似导致的误匹配

### 1.8 参考文献

1. Smith, R., Self, M., & Cheeseman, P. (1986). Estimating uncertain spatial relationships in robotics. *IEEE Conference on Robotics and Automation*.
2. Durrant-Whyte, H., & Bailey, T. (2006). Simultaneous localization and mapping: part I. *IEEE Robotics & Automation Magazine*, 13(2), 99-110.
3. Bailey, T., & Durrant-Whyte, H. (2006). Simultaneous localization and mapping (SLAM): part II. *IEEE Robotics & Automation Magazine*, 13(3), 108-117.
4. Dissanayake, G., Newman, P., Clark, S., Durrant-Whyte, H., & Csorba, M. (2001). A solution to the simultaneous localization and map building (SLAM) problem. *IEEE Transactions on Robotics and Automation*, 17(3), 229-241.
5. Cadena, C., Carlone, L., Carrillo, H., et al. (2016). Past, present, and future of simultaneous localization and mapping: Toward the robust-perception age. *IEEE Transactions on Robotics*, 32(6), 1309-1332.
