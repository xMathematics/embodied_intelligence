# PVO-全景视觉里程计

## 基本信息
- **英文标题**: PVO: Panoptic Visual Odometry
- **作者**: Weicai Ye, Xinyue Lan, Shuo Chen, Ming Ouyang, Xinlong Wang, Xiaolong Li, Yan Lu, Hao Zhao
- **发表会议/期刊**: CVPR 2023
- **关键词**: 全景分割,多任务学习,语义VO

## 研究背景（前提）
传统VO仅输出相机轨迹，缺乏对场景的语义理解。语义VO输出语义但缺乏实例级理解。

## 问题提出（由什么问题引出）
能否同时实现VO(位姿估计)、语义分割和实例分割的联合学习，输出完整的全景理解？

## 要解决的问题
如何将VO与全景分割(语义+实例分割)统一到一个框架中？

---

## 采用的方法
提出PVO：统一框架同时估计相机位姿、深度图、语义标签和实例掩码。设计多任务网络结构实现几何-语义联合学习。

## 理论依据
运动估计和全景分割可共享特征表示。静态场景结构辅助分割，语义信息辅助运动估计。

## 核心公式推导
- **全景VO损失**: $L = L_{pose} + L_{depth} + L_{sem} + L_{inst}$
  - 位姿、深度、语义、实例的联合损失



---

## 实验结果
在多个基准上同时实现了VO、语义分割和实例分割的SOTA性能。

## 尚未解决的问题（后续方向）
多任务训练需仔细平衡权重；全景分割依赖大规模标注数据。

---

## 原始摘要
We present PVO, a novel panoptic visual odometry frame-
work to achieve more comprehensive modeling of the scene
motion, geometry, and panoptic segmentation information.
Our PVO models visual odometry (VO) and video panop-
tic segmentation (VPS) in a unified view, which makes the
two tasks mutually beneficial. Specifically, we introduce
a panoptic update module into the VO Module with the
guidance of image panoptic segmentation. This Panoptic-
Enhanced VO Module can alleviate the impact of dynamic
objects in the camera pose estimation with a panoptic-aware
dynamic mask. On the other hand, the VO-Enhanced VPS
Module also improves the segmentation accuracy by fusing
the panoptic segmentation result of the current frame on the
fly to the adjacent frames, using geometric information such
as camera pose, depth, and optical flow obtained from the
VO Module. These two modules contribute to each other
through recurrent iterative optimization. Extensive exper-
iments demonstrate that PVO outperforms state-of-the-art
methods in both visual odometry and video panoptic segmen-
tation tasks.
∗indicates equal contribution. † indicates the corresponding author.
