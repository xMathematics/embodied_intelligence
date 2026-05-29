# SLAM与三维重建模块

## 模块概述

同时定位与地图构建（SLAM）是具身智能的核心技术之一，本模块涵盖视觉里程计、SLAM算法、三维重建等内容，并详细讲解了大量相关论文。

## 学习目标

完成本模块学习后，您将能够：
- 理解视觉里程计原理
- 掌握主流SLAM算法
- 了解三维重建技术
- 理解NeRF等神经渲染方法

## 学习路径

### 第一部分：视觉里程计 ✅（1周）

| 章节 | 内容 | 难度 | 链接 |
|------|------|------|------|
| 1.1 | 特征提取与匹配 | ⭐⭐⭐ | [查看](visual-odometry/01-feature-extraction.md) |
| 1.2 | 对极几何与本质矩阵 | ⭐⭐⭐⭐ | [查看](visual-odometry/02-epipolar-geometry.md) |
| 1.3 | 直接法视觉里程计 | ⭐⭐⭐⭐ | [查看](visual-odometry/03-direct-method-vo.md) |

**核心内容**：
- SIFT、SURF、ORB特征提取算法
- 特征匹配与RANSAC
- 对极几何、本质矩阵、基础矩阵
- 直接法vs稀疏法
- 光流估计（Lucas-Kanade、Horn-Schunck）

**重要论文**：
1. "Distinctive Image Features from Scale-Invariant Keypoints" (SIFT) - Lowe, 2004
2. "ORB: An efficient alternative to SIFT or SURF" - Rublee et al., 2011
3. "An efficient solution to the five-point relative pose problem" - Nister, 2003
4. "LSD-SLAM: Large-Scale Direct Monocular SLAM" - Engel et al., 2014
5. "DSO: Direct Sparse Odometry" - Engel et al., 2016

---

### 第二部分：SLAM算法 ✅（2周）

| 章节 | 内容 | 难度 | 链接 |
|------|------|------|------|
| 2.1 | SLAM算法基础 | ⭐⭐⭐⭐ | [查看](slam-algorithms/01-slam-basics.md) |
| 2.2 | 主流SLAM系统详解 | ⭐⭐⭐⭐⭐ | [查看](slam-algorithms/02-mainstream-slam.md) |

**核心内容**：
- EKF-SLAM、粒子滤波SLAM
- 光束法平差（Bundle Adjustment）
- 因子图优化、位姿图优化
- 回环检测与闭环融合
- ORB-SLAM系列、LSD-SLAM、DSO、Cartographer

**重要论文**：
1. "A Solution to the Simultaneous Localization and Map Building (SLAM) Problem" - Leonard & Durrant-Whyte, 1995
2. "Bundle Adjustment - A Modern Synthesis" - Triggs et al., 2000
3. "iSAM: Incremental Smoothing and Mapping" - Kaess et al., 2008
4. "ORB-SLAM: A Versatile and Accurate Monocular SLAM System" - Mur-Artal et al., 2015
5. "ORB-SLAM3: An Accurate Open-Source Library for Visual-Inertial SLAM" - Campos et al., 2020
6. "Cartographer: Real-Time Concurrent Localization and Mapping in 2D and 3D" - Google, 2016

---

### 第三部分：三维重建 ✅（1周）

| 章节 | 内容 | 难度 | 链接 |
|------|------|------|------|
| 3.1 | 结构从运动与MVS | ⭐⭐⭐⭐ | [查看](3d-reconstruction/01-sfm-mvs.md) |
| 3.2 | 神经辐射场（NeRF） | ⭐⭐⭐⭐⭐ | [查看](3d-reconstruction/02-nerf.md) |

**核心内容**：
- 结构从运动（SfM）
- 多视图立体视觉（MVS）
- 深度估计与表面重建
- NeRF原理与体素渲染
- NeRF变体（Instant-NGP、NeRF-W、Mip-NeRF）

**重要论文**：
1. "Structure from Motion Revisited" - Schönberger & Frahm, 2016
2. "Pixelwise View Selection for Unstructured Multi-View Stereo" - Schönberger et al., 2017
3. "Poisson Surface Reconstruction" - Kazhdan et al., 2006
4. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" - Mildenhall et al., 2020
5. "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" - Müller et al., 2022
6. "NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections" - Martin-Brualla et al., 2021

---

### 论文详解（单独文件）

| 论文 | 作者 | 发表年份 | 链接 |
|------|------|---------|------|
| SIFT | David G. Lowe | 2004 | [查看](papers/sift.md) |
| ORB | Rublee et al. | 2011 | [查看](papers/orb.md) |
| DSO | Engel et al. | 2016 | [查看](papers/dso.md) |
| Bundle Adjustment | Triggs et al. | 2000 | [查看](papers/bundle-adjustment.md) |
| ORB-SLAM3 | Campos et al. | 2020 | [查看](papers/orb-slam3.md) |
| NeRF | Mildenhall et al. | 2020 | [查看](papers/nerf.md) |
| Instant-NGP | Müller et al. | 2022 | [查看](papers/instant-ngp.md) |

---

## 推荐学习资源

### 书籍
1. 《SLAM十四讲》- 高翔
2. 《Multiple View Geometry in Computer Vision》- Hartley & Zisserman
3. 《Probabilistic Robotics》- Thrun et al.
4. 《Visual SLAM: From Theory to Practice》- 沈邵劼

### 开源项目
1. **ORB-SLAM3** - https://github.com/UZ-SLAMLab/ORB_SLAM3
2. **COLMAP** - https://colmap.github.io/
3. **g2o** - https://github.com/RainerKuemmerle/g2o
4. **NeRF** - https://github.com/bmild/nerf
5. **Instant-NGP** - https://github.com/NVlabs/instant-ngp

### 数据集
1. **KITTI Odometry** - http://www.cvlibs.net/datasets/kitti/eval_odometry.php
2. **TUM RGB-D** - https://vision.in.tum.de/data/datasets/rgbd-dataset
3. **EuRoC MAV** - https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
4. **DTU MVS** - https://roboimagedata.compute.dtu.dk/?page_id=36
5. **NeRF Synthetic** - https://www.matthewtancik.com/nerf

### 论文综述
1. "Visual SLAM: A Survey from Deep Learning Perspective" - Liu et al., 2021
2. "Neural Radiance Fields: A Survey" - Liu et al., 2022

---

## 学习评估

### 自测题
1. 解释EKF-SLAM与优化-based SLAM的区别
2. 什么是光束法平差？它在SLAM中的作用是什么？
3. 比较SIFT、SURF、ORB三种特征提取算法的优缺点
4. 什么是对极约束？如何利用它估计相机位姿？
5. 解释NeRF的工作原理，包括体素渲染和位置编码

### 实践项目
1. 使用ORB-SLAM3处理KITTI数据集
2. 使用COLMAP重建三维场景
3. 运行NeRF进行场景渲染
4. 实现简单的特征匹配和对极几何估计
5. 使用Open3D进行点云处理和表面重建

---

**下一个模块**：[机器学习基础](../03-machine-learning/README.md)