# ViPE-视频位姿引擎

## 基本信息
- **英文标题**: ViPE: Video Pose Engine for 3D Geometric Perception
- **作者**: Jiahui Huang, Qunjie Zhou, Hesam Rabeti, Aleksandr Korovko, Huan Ling, Xuanchi Ren, Tianchang Shen, Jun Gao, Dmitry Slepichev, Chen-Hsuan Lin, Jiawei Ren, Kevin Xie, Joydeep Biswas, Laura Leal-Taixe, Sanja Fidler
- **发表会议/期刊**: NVIDIA Technical Report 2025
- **关键词**: 视频位姿,3D感知,SLAM,深度估计

## 研究背景（前提）
精确的3D几何感知是空间AI的重要前提。从野外视频获取一致且精确的3D标注仍然是关键挑战。

## 问题提出（由什么问题引出）
现有方法依赖大规模训练数据，但获取in-the-wild视频的一致精确3D标注困难。

## 要解决的问题
如何构建一个高效、鲁棒的视频处理引擎，从无约束原始视频估计相机内参、相机运动和密集近度量深度图？

---

## 采用的方法
ViPE集成经典SLAM(效率、可扩展性)和学习模型(鲁棒性)的优势。支持多种相机模型(针孔、广角、全景)，处理动态自拍、电影镜头、行车记录仪等多样场景。

## 理论依据
结合经典SLAM的效率和可扩展性与学习模型的鲁棒性。通过连续帧特征跟踪和全局优化实现鲁棒位姿和深度估计。

## 核心公式推导
- **全局优化**: $\min_{T,D} \sum ||\text{reproject}(T_i, D_i)||^2 + L_{smooth}$
  - 联合优化位姿和深度



---

## 实验结果
在TUM/KITTI上超越现有未标定位姿估计基线18%/50%。在单GPU上以3-5FPS运行。标注了约96M帧的数据集。

## 尚未解决的问题（后续方向）
在快速运动和大旋转场景中仍有挑战；深度为近度量级(非精确度量)。

---

## 原始摘要
Accurate 3D geometric perception is an important prerequisite for a wide range of spatial AI systems.
While state-of-the-art methods depend on large-scale training data, acquiring consistent and precise 3D
annotations from in-the-wild videos remains a key challenge. In this work, we introduce ViPE, a handy
and versatile video processing engine designed to bridge this gap. ViPE efficiently estimates camera
intrinsics, camera motion, and dense, near-metric depth maps from unconstrained raw videos. It is
robust to diverse scenarios, including dynamic selfie videos, cinematic shots, or dashcams, and supports
various camera models such as pinhole, wide-angle, and 360° panoramas. We have benchmarked ViPE
on multiple benchmarks. Notably, it outperforms existing uncalibrated pose estimation baselines by
18%/50% on TUM/KITTI sequences, and runs at 3-5FPS on a single GPU for standard input resolutions.
We use ViPE to annotate a large-scale collection of videos. This collection includes around 100K real-world
internet videos, 1M high-quality AI-generated videos, and 2K panoramic videos, totaling approximately
96M frames – all annotated with accurate camera poses and dense depth maps. We open-source ViPE
and the annotated dataset with the hope of accelerating the development of spatial AI systems.
