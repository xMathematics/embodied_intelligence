# 3.9 主流SLAM系统详解

## 1. 概述

本章详细介绍最具影响力的主流SLAM系统，包括其架构、算法特点和应用场景。

## 2. ORB-SLAM系列

### 2.1 ORB-SLAM（2015）

**作者**：Mur-Artal, Montiel, Tardós
**会议**：IEEE Transactions on Robotics

**核心贡献**：
- 统一的单目SLAM框架
- 三线程并行架构（跟踪、局部建图、回环检测）
- 基于ORB特征的完整SLAM系统

**系统架构**：
```
输入帧 → 跟踪线程 → [位姿估计 → 关键帧决策]
                  → 局部建图 → [局部BA → 剔除 → 新点三角化]
                  → 回环检测 → [DBoW搜索 → 闭环融合 → 全局BA]
```

### 2.2 ORB-SLAM2（2017）

**扩展**：
- 支持单目、双目和RGB-D相机
- 地图保存和加载
- 改进的局部BA

### 2.3 ORB-SLAM3（2020）

**创新**：
- 视觉-惯性紧耦合（IMU预积分）
- 多地图系统（Atlas）
- 改进的回环检测
- 完整的视觉-惯性BA

**多地图系统**：
```
Atlas
├── 活跃地图 (Active Map)
└── 非活跃地图 (Non-active Maps)
    ├── Map 1
    ├── Map 2
    └── ...
```

## 3. VINS-Mono

**作者**：Qin, Li, Shen
**会议**：IEEE Transactions on Robotics (2018)

**核心创新**：
- 单目视觉-惯性紧耦合
- 滑动窗口非线性优化
- IMU预积分因子
- 在线标定

**系统流程**：
```
IMU预积分 → 视觉跟踪 → 滑动窗口优化 → 回环检测 → 全局优化
```

## 4. DSO

**作者**：Engel, Koltun, Cremers
**会议**：TPAMI (2017)

**核心创新**：
- 稀疏直接法
- 光度标定
- 滑动窗口优化
- 无回环检测（纯里程计）

## 5. LSD-SLAM

**作者**：Engel, Schöps, Cremers
**会议**：ECCV (2014)

**核心创新**：
- 大规模直接法单目SLAM
- 半稠密深度地图
- 基于Sim(3)的尺度感知优化

## 6. Cartographer

**作者**：Google
**发布**：2016, 开源

**核心创新**：
- 基于激光雷达的SLAM
- 局部 submaps 和全局优化
- 支持2D和3D
- 回环检测使用扫描匹配

## 7. RTAB-Map

**作者**：Labbé, Michaud
**发布**：2013

**核心创新**：
- 基于外观的实时回环检测
- 增量式内存管理
- 多传感器融合
- 长期运行SLAM

## 8. LOAM系列

### 8.1 LOAM（2014）

**作者**：Zhang, Singh
**会议**：RSS

**核心创新**：
- 基于LiDAR的实时里程计和建图
- 高频低精度里程计 + 低频高精度建图
- 特征提取（边缘点、平面点）

### 8.2 LeGO-LOAM（2018）

- 地面分割
- 轻量化特征提取
- 更高效的优化

### 8.3 LIO-SAM（2020）

- LiDAR-IMU紧耦合
- 因子图优化框架
- 回环检测

## 9. 系统对比

| 系统 | 传感器 | 方法 | 回环 | 地图 | 年份 |
|------|--------|------|------|------|------|
| ORB-SLAM3 | Mono/Stereo/RGBD+IMU | 特征法 | DBoW2 | 稀疏 | 2020 |
| VINS-Mono | Mono+IMU | 特征法 | DBoW2 | 稀疏 | 2018 |
| DSO | Mono | 直接法 | 无 | 稀疏 | 2017 |
| LSD-SLAM | Mono | 直接法 | 直接 | 半稠密 | 2014 |
| Cartographer | LiDAR | 扫描匹配 | CSM | 网格 | 2016 |
| LOAM | LiDAR | 特征法 | 无 | 点云 | 2014 |
| LIO-SAM | LiDAR+IMU | 因子图 | ICP | 点云 | 2020 |

## 10. 参考文献

1. Mur-Artal, R., & Tardós, J. D. (2017). ORB-SLAM2: An open-source SLAM system for monocular, stereo, and RGB-D cameras. *IEEE Transactions on Robotics*, 33(5), 1255-1262.
2. Campos, C., Elvira, R., et al. (2021). ORB-SLAM3: An accurate open-source library for visual, visual-inertial, and multimap SLAM. *IEEE Transactions on Robotics*, 37(6), 1874-1890.
3. Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A robust and versatile monocular visual-inertial state estimator. *IEEE Transactions on Robotics*, 34(4), 1004-1020.
4. Zhang, J., & Singh, S. (2014). LOAM: Lidar odometry and mapping in real-time. *RSS*.
