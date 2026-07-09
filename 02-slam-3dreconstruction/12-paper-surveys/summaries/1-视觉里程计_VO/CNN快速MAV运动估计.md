# CNN快速MAV运动估计

## 基本信息
- **英文标题**: CNN-based Ego-Motion Estimation for Fast MAV Maneuvers
- **作者**: Yingfu Xu, Guido C. H. E. de Croon
- **发表会议/期刊**: IEEE RA-L 2021
- **关键词**: MAV,快速机动,运动模糊,单应性,自监督

## 研究背景（前提）
MAV快速机动中，大视差和运动模糊使传统特征点法VO鲁棒性严重下降。快速飞行时特征点移出视野，帧间跟踪困难。

## 问题提出（由什么问题引出）
CNN在标准VO数据集表现优秀，但尚无系统评估在MAV快速机动中的性能。PRGFlow只研究低速飞行(~0.5m/s)。

## 要解决的问题
如何在显著运动模糊和大视差的MAV快速机动中，用CNN从单目相机(面向平面)和IMU辅助估计3D平移？

---

## 采用的方法
基于ICSTN，采用级联网络块逐步精化位姿。金字塔特征图扩大感受野。IMU提供姿态辅助去旋转。支持有监督(Charbonnier损失)和自监督(光度误差)训练。

## 理论依据
单应性变换H=R+tn^T/d联系平面场景两帧间像素坐标。IMU提供姿态后仅需估计平移/距离比。

## 核心公式推导
- **单应性变换**: $x_2 = Hx_1, H=R+\frac{tn^T}{d}$
  - 平面场景的两视图关系

- **级联精化**: $\hat{t} = \sum \Delta t_k$
  - K个级联块逐步精化

- **Charbonnier损失**: $\rho(x)=\sqrt{x^2+\epsilon^2}$
  - 鲁棒损失函数



---

## 实验结果
在快速机动中显著优于传统特征点法和现有CNN方法。自监督学习优于有监督。

## 尚未解决的问题（后续方向）
假设平面场景限制应用；需IMU辅助；主要估计平移。

---

## 原始摘要
Micro Air Vehicles (MAVs), fast maneuvers stay challenging
mainly because of the big visual disparity and motion blur.
In the pursuit of higher robustness, we study convolutional
neural networks (CNNs) that predict the relative pose between
subsequent images from a fast-moving monocular camera
facing a planar scene. Aided by the Inertial Measurement Unit
(IMU), we mainly focus on translational motion. The networks
we study have similar small model sizes (around 1.35MB) and
high inference speeds (around 10 milliseconds on a mobile
GPU). Images for training and testing have realistic motion blur.
Departing from a network framework that iteratively warps
the ﬁrst image to match the second with cascaded network
blocks, we study different network architectures and training
strategies. Simulated datasets and a self-collected MAV ﬂight
dataset are used for evaluation. The proposed setup shows
better accuracy over existing networks and traditional feature-
point-based methods during fast maneuvers. Moreover, self-
supervised learning outperforms supervised learning. Videos
and open-sourced code are available at https://github.
com/tudelft/PoseNet_Planar
I. INTRODUCTION
Indoor ﬂight of Micro Air Vehicles (MAVs) is an attractive
but challenging task. Towards the goal of autonomy, robust
state estimation is one of the most essential modules of the
MAV’s ﬂight control system. A camera captures rich infor-
mation in a big ﬁeld of view. Since being small and power-
efﬁcient, it is an ideal
