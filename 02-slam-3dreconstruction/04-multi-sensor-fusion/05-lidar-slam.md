# 4.5 LiDAR SLAM

## 1. 概述

LiDAR SLAM使用激光雷达作为主要传感器进行定位和建图。LiDAR提供精确的距离测量，不受光照影响，在自动驾驶和机器人领域广泛应用。

## 2. LiDAR SLAM基础

### 2.1 LiDAR数据类型

- **2D LiDAR**：单线扫描，用于室内导航
- **3D LiDAR**：多线扫描（16/32/64/128线），用于自动驾驶和室外场景
- **固态LiDAR**：非旋转式，视场角有限

### 2.2 点云预处理

- **降采样**：体素网格滤波
- **地面去除**：RANSAC平面拟合
- **点云分割**：去除动态物体

## 3. LOAM系列

### 3.1 LOAM（Lidar Odometry and Mapping）

**作者**：Zhang & Singh, RSS 2014

**核心架构**：
- **里程计**（高频，10Hz）：帧间配准
- **建图**（低频，1Hz）：扫描到地图配准

**特征提取**：
根据局部平滑度 $c$ 分类：

$$ c = \frac{1}{|\mathcal{S}| \|\mathbf{X}_i\|} \sum_{j \in \mathcal{S}, j \neq i} \|\mathbf{X}_j - \mathbf{X}_i\| $$

- **边缘点**：$c$ 大（尖锐特征）
- **平面点**：$c$ 小（平坦区域）

**扫描匹配**：
- 边缘点到边缘线的距离
- 平面点到平面片的距离

### 3.2 LeGO-LOAM

**改进**（2018）：
- 地面分割（基于深度图像的快速分割）
- 两步LM优化（先优化地面，后优化非地面）
- 基于ICP的回环检测
- 轻量化设计

### 3.3 LIO-SAM

**LiDAR-IMU融合**（2020）：
- 因子图框架
- LiDAR里程计因子
- IMU预积分因子
- GPS因子（可选）
- 回环因子

## 4. Cartographer

Google开发的2D/3D LiDAR SLAM系统（2016）：

**核心方法**：
- **局部建图**：使用scan-to-submap匹配（基于Ceres优化）
- **全局优化**：基于SPA（Sparse Pose Adjustment）
- **回环检测**：基于分支定界的快速相关扫描匹配

## 5. LiDAR-视觉融合

### 5.1 深度融合方法

| 系统 | 方法 | 特点 |
|------|------|------|
| V-LOAM | LOAM + VO | 视觉辅助LiDAR |
| DEMO | 深度增强 | LiDAR深度辅助VO |
| LVI-SAM | 因子图 | LiDAR-Visual-IMU |
| R3LIVE | 紧耦合 | 实时RGB-D LiDAR |
| FAST-LIO | 迭代卡尔曼 | 高效LiDAR-IMU |

### 5.2 数据关联

- **几何关联**：LiDAR点投影到图像
- **语义关联**：共享语义标签
- **深度关联**：LiDAR为视觉提供深度先验

## 6. LiDAR SLAM对比

| 系统 | 传感器 | 方法 | 回环 | 实时性 |
|------|--------|------|------|--------|
| LOAM | LiDAR | 特征法 | 无 | 实时 |
| LeGO-LOAM | LiDAR | 特征法 | ICP | 实时 |
| LIO-SAM | LiDAR+IMU | 因子图 | ICP | 实时 |
| Cartographer | LiDAR | CSM | 分支定界 | 实时 |
| HDL-Graph-SLAM | LiDAR | 位姿图 | 特征 | 近实时 |

## 7. 参考文献

1. Zhang, J., & Singh, S. (2014). LOAM: Lidar odometry and mapping in real-time. *RSS*.
2. Shan, T., & Englot, B. (2018). LeGO-LOAM: Lightweight and ground-optimized lidar odometry and mapping on variable terrain. *IROS*.
3. Shan, T., Englot, B., Meyers, D., Wang, W., Ratti, C., & Rus, D. (2020). LIO-SAM: Tightly coupled lidar inertial odometry via smoothing and mapping. *IROS*.
4. Hess, W., Kohler, D., Rapp, H., & Andor, D. (2016). Real-time loop closure in 2D LIDAR SLAM. *ICRA*.
5. Xu, W., et al. (2021). R3LIVE: A robust, real-time, RGB-colored, LiDAR-Inertial-Visual tightly-coupled state estimation and mapping package. *ICRA*.
