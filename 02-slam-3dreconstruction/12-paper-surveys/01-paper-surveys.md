# 12. 论文详解

## 1. 概述

本章汇总SLAM和三维重建领域的经典和前沿论文的详细分析。

## 2. SLAM经典论文

### 2.1 ORB-SLAM系列

**ORB-SLAM (2015)**：
- Title: ORB-SLAM: A Versatile and Accurate Monocular SLAM System
- Authors: Raul Mur-Artal, J. M. M. Montiel, Juan D. Tardós
- Venue: IEEE Transactions on Robotics

**ORB-SLAM2 (2017)**：
- Title: ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras
- Authors: Raul Mur-Artal, Juan D. Tardós
- Venue: IEEE Transactions on Robotics

**ORB-SLAM3 (2020)**：
- Title: ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM
- Authors: Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, et al.
- Venue: IEEE Transactions on Robotics

### 2.2 DSO (2016)

**Title**: Direct Sparse Odometry
**Authors**: Jakob Engel, Vladlen Koltun, Daniel Cremers
**Venue**: IEEE Transactions on Pattern Analysis and Machine Intelligence

**关键创新**：
- 稀疏直接法视觉里程计
- 光度标定
- 滑动窗口优化
- 高效关键帧管理

### 2.3 LSD-SLAM (2014)

**Title**: LSD-SLAM: Large-Scale Direct Monocular SLAM
**Authors**: Jakob Engel, Thomas Schöps, Daniel Cremers
**Venue**: ECCV

### 2.4 PTAM (2007)

**Title**: Parallel Tracking and Mapping for Small AR Workspaces
**Authors**: Georg Klein, David Murray
**Venue**: ISMAR

### 2.5 SVO (2014)

**Title**: SVO: Fast Semi-Direct Monocular Visual Odometry
**Authors**: Christian Forster, Matia Pizzoli, Davide Scaramuzza
**Venue**: ICRA

## 3. VIO论文

### 3.1 MSCKF (2007)

**A Multi-State Constraint Kalman Filter for Vision-Aided Inertial Navigation**
- Authors: Anastasios I. Mourikis, Stergios I. Roumeliotis
- Venue: ICRA

### 3.2 OKVIS (2015)

**Keyframe-Based Visual-Inertial Odometry Using Nonlinear Optimization**
- Authors: Stefan Leutenegger, Simon Lynen, Michael Bosse, et al.
- Venue: IJRR

### 3.3 VINS-Mono (2018)

**VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator**
- Authors: Tong Qin, Peiliang Li, Shaojie Shen
- Venue: IEEE Transactions on Robotics

## 4. LiDAR SLAM论文

### 4.1 LOAM (2014)

**LOAM: Lidar Odometry and Mapping in Real-time**
- Authors: Ji Zhang, Sanjiv Singh
- Venue: RSS

### 4.2 Cartographer (2016)

**Real-Time Loop Closure in 2D LIDAR SLAM**
- Authors: Wolfgang Hess, Damon Kohler, Holger Rapp, Daniel Andor
- Venue: ICRA

### 4.3 LIO-SAM (2020)

**LIO-SAM: Tightly Coupled Lidar Inertial Odometry via Smoothing and Mapping**
- Authors: Tixiao Shan, Brendan Englot, Drew Meyers, et al.
- Venue: IROS

## 5. 三维重建论文

### 5.1 SfM (2016)

**Structure-from-Motion Revisited**
- Authors: Johannes L. Schönberger, Jan-Michael Frahm
- Venue: CVPR

### 5.2 COLMAP (2016)

**Pixelwise View Selection for Unstructured Multi-View Stereo**
- Authors: Johannes L. Schönberger, Enliang Zheng, Jan-Michael Frahm, Marc Pollefeys
- Venue: ECCV

### 5.3 KinectFusion (2011)

**KinectFusion: Real-Time Dense Surface Mapping and Tracking**
- Authors: Richard A. Newcombe, et al.
- Venue: ISMAR

### 5.4 BundleFusion (2017)

**BundleFusion: Real-Time Globally Consistent 3D Reconstruction Using On-the-Fly Surface Reintegration**
- Authors: Angela Dai, Matthias Nießner, Michael Zollhöfer, et al.
- Venue: ACM Transactions on Graphics

## 6. 神经重建论文

### 6.1 NeRF (2020)

**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**
- Authors: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, et al.
- Venue: ECCV

### 6.2 Instant-NGP (2022)

**Instant Neural Graphics Primitives with a Multiresolution Hash Encoding**
- Authors: Thomas Müller, Alex Evans, Christoph Schied, Alexander Keller
- Venue: SIGGRAPH

### 6.3 3D Gaussian Splatting (2023)

**3D Gaussian Splatting for Real-Time Radiance Field Rendering**
- Authors: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
- Venue: ACM Transactions on Graphics

### 6.4 NeuS (2021)

**NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-View Reconstruction**
- Authors: Peng Wang, Lingjie Liu, Yuan Liu, et al.
- Venue: NeurIPS

## 7. 神经SLAM论文

### 7.1 iMAP (2021)

**iMAP: Implicit Mapping and Positioning in Real-Time**
- Authors: Edgar Sucar, Shikun Liu, Joseph Ortiz, Andrew J. Davison
- Venue: ICCV

### 7.2 NICE-SLAM (2022)

**NICE-SLAM: Neural Implicit Scalable Encoding for SLAM**
- Authors: Zihan Zhu, Songyou Peng, Viktor Larsson, et al.
- Venue: CVPR

### 7.3 DROID-SLAM (2021)

**DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras**
- Authors: Zachary Teed, Jia Deng
- Venue: NeurIPS

### 7.4 SplaTAM (2024)

**SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM**
- Authors: Nikhil Keetha, Jay Karhade, Krishna Murthy Jatavallabhula, et al.
- Venue: CVPR

## 8. 回环检测论文

### 8.1 DBoW2 (2012)

**Bags of Binary Words for Fast Place Recognition in Image Sequences**
- Authors: Dorian Gálvez-López, Juan D. Tardós
- Venue: IEEE Transactions on Robotics

### 8.2 NetVLAD (2016)

**NetVLAD: CNN Architecture for Weakly Supervised Place Recognition**
- Authors: Relja Arandjelovic, Petr Gronat, Akihiko Torii, et al.
- Venue: CVPR

## 9. 综述论文

1. Cadena, C., et al. (2016). Past, present, and future of simultaneous localization and mapping: Toward the robust-perception age. *IEEE Transactions on Robotics*, 32(6), 1309-1332.
2. Durrant-Whyte, H., & Bailey, T. (2006). Simultaneous localization and mapping: Part I. *IEEE Robotics & Automation Magazine*, 13(2), 99-110.
3. Bailey, T., & Durrant-Whyte, H. (2006). Simultaneous localization and mapping (SLAM): Part II. *IEEE Robotics & Automation Magazine*, 13(3), 108-117.

**作者**: Mildenhall et al.

**发表会议**: ECCV

**核心贡献**:
- 神经辐射场
- 体素渲染
- 位置编码
- 视图合成

**架构**:
- MLP网络
- 分层采样
- 体积渲染
- 位置编码

**解决的问题**:
- 高质量视图合成
- 隐式场景表示

**缺陷**:
- 训练慢
- 推理慢

---

### 9. Instant-NGP (2022)

**作者**: Müller et al.

**发表会议**: SIGGRAPH

**核心贡献**:
- 实时神经渲染
- 多分辨率哈希编码
- 快速训练
- 高效推理

**架构**:
- 哈希编码
- 小型MLP
- 光线步进
- 多分辨率

**解决的问题**:
- NeRF速度慢
- 实时交互

**缺陷**:
- 内存占用

---

### 10. COLMAP (2016)

**作者**: Schönberger et al.

**发表期刊**: ACM TOG

**核心贡献**:
- 结构从运动系统
- 多视图立体视觉
- 开源工具
- 高精度重建

**架构**:
- 特征提取与匹配
- 增量重建
- 稠密重建
- 表面重建

**解决的问题**:
- 大规模场景重建
- 易用性

**缺陷**:
- 计算时间长