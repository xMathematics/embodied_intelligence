# 4.4 视觉-惯性里程计（VIO）

## 1. 概述

视觉-惯性里程计（Visual-Inertial Odometry, VIO）融合相机和IMU，利用两种传感器的互补特性实现鲁棒的状态估计。相机提供丰富的视觉信息但受光照和运动影响，IMU提供高频运动信息但存在漂移。

## 2. VIO系统分类

### 2.1 按融合框架

| 类型 | 方法 | 代表系统 |
|------|------|----------|
| 滤波 | 扩展卡尔曼滤波 | MSCKF, ROVIO |
| 优化 | 滑动窗口非线性优化 | VINS-Mono, OKVIS |
| 混合 | 滤波+优化 | HybVIO |

### 2.2 按相机类型

- **单目VIO**：尺度弱可观，需IMU激励
- **双目VIO**：尺度可观，精度更高
- **RGB-D VIO**：直接深度测量

## 3. MSCKF

**Multi-State Constraint Kalman Filter**（Mourikis & Roumeliotis, 2007）

**核心思想**：
- EKF框架，状态包含滑动窗口中的多个相机位姿
- 特征点不加入状态，而是用于构建多状态约束
- 计算复杂度与特征点数量无关

**状态向量**：
$$ \mathbf{x} = [\mathbf{x}_{IMU}^T, \mathbf{x}_{C1}^T, \mathbf{x}_{C2}^T, \ldots, \mathbf{x}_{CN}^T]^T $$

**观测更新**：对新特征点的多帧观测联合更新，避免了特征点入状态。

## 4. OKVIS

**Open Keyframe-based Visual-Inertial SLAM**（Leutenegger et al., 2015）

**核心思想**：
- 基于关键帧的滑动窗口优化
- 视觉重投影误差 + IMU预积分误差
- 边缘化移除旧帧

## 5. VINS-Mono

**VINS-Mono**（Qin, Li, Shen, 2018）

**完整系统**：
- IMU预积分
- 视觉特征跟踪
- 滑动窗口优化
- 重定位检测
- 全局图优化

**初始化**：
1. 纯视觉SfM估计初始结构
2. 视觉-惯性对齐恢复尺度、重力方向、速度
3. 在线标定IMU零偏

## 6. VIO初始化

### 6.1 为什么需要初始化

单目VIO的初始化需要解决：
- 尺度模糊性
- 重力方向估计
- IMU零偏估计
- 速度初始化

### 6.2 初始化流程

1. 视觉SfM：纯视觉运动恢复结构
2. 陀螺仪零偏估计：利用旋转约束
3. 速度、重力、尺度估计：求解线性系统
4. 加速度零偏估计：进一步精化

## 7. VIO退化分析

| 情况 | 影响 | 对策 |
|------|------|------|
| 静止 | 位置不可观 | 零速检测 |
| 匀速 | 尺度不可观 | 充分激励 |
| 纯旋转 | 平移不可观 | 避免纯旋转 |
| 加速不足 | 重力和尺度耦合 | 等待充分加速 |

## 8. 参考文献

1. Mourikis, A. I., & Roumeliotis, S. I. (2007). A multi-state constraint Kalman filter for vision-aided inertial navigation. *ICRA*.
2. Leutenegger, S., Lynen, S., Bosse, M., Siegwart, R., & Furgale, P. (2015). Keyframe-based visual-inertial odometry using nonlinear optimization. *The International Journal of Robotics Research*, 34(3), 314-334.
3. Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A robust and versatile monocular visual-inertial state estimator. *IEEE Transactions on Robotics*, 34(4), 1004-1020.
4. Bloesch, M., Omari, S., Hutter, M., & Siegwart, R. (2015). Robust visual inertial odometry using a direct EKF-based approach. *IROS*.
