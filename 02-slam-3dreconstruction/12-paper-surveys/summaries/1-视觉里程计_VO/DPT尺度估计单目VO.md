# DPT尺度估计单目VO

## 基本信息
- **英文标题**: Dense Prediction Transformer for Scale Estimation in Monocular Visual Odometry
- **作者**: André O. Françani, Marcos R. O. A. Maximo
- **发表会议/期刊**: LARS/SBR/WRE 2022
- **关键词**: 尺度估计,Vision Transformer,深度估计

## 研究背景（前提）
单目VO因深度信息缺失导致尺度模糊和尺度漂移问题，严重影响位姿估计精度。

## 问题提出（由什么问题引出）
Dense Prediction Transformer(DPT)在单目深度估计中表现出色。能否将其应用于单目VO的尺度估计？

## 要解决的问题
如何利用DPT模型估计的深度图来校正单目VO中的尺度漂移？

---

## 采用的方法
将DPT深度估计模型集成到单目VO流水线中，用DPT估计的深度图恢复绝对尺度，减少尺度漂移。

## 理论依据
DPT(Vision Transformer based)可从单张图像估计出具有尺度信息的深度图。将此深度信息作为VO的尺度参考。

## 核心公式推导
- **尺度校正**: $s_t = \text{scale}(D_{DPT}, D_{VO})$
  - 用DPT深度校正VO尺度



---

## 实验结果
通过DPT精确的深度估计，有效减少了单目VO的尺度漂移，达到与SOTA竞争的性能。

## 尚未解决的问题（后续方向）
依赖DPT的深度估计质量；DPT推理速度慢，影响实时性。

---

## 原始摘要
tion of the position of an agent through images of a single camera,
and it is applied in autonomous vehicles, medical robots, and
augmented reality. However, monocular systems suffer from the
scale ambiguity problem due to the lack of depth information in
2D frames. This paper contributes by showing an application of
the dense prediction transformer model for scale estimation in
monocular visual odometry systems. Experimental results show
that the scale drift problem of monocular systems can be reduced
through the accurate estimation of the depth map by this model,
achieving competitive state-of-the-art performance on a visual
odometry benchmark.
Index Terms—monocular visual odometry, scale estimation,
deep learning, monocular depth estimation, vision transformer
I. INTRODUCTION
Visual odometry (VO) is an accurate and classical process
of estimating the camera pose and motion from a sequence
of images. It is a tracking problem widely applied in mobile
robots and autonomous vehicles [1].
Monocular visual odometry (MVO) systems use a single
camera to capture the images, while the stereo VO algorithms
utilize a stereo camera pair which allows the computation of
feature depth between image frames. In this work, we deal
with MVO mainly due to its simplicity, i.e. monocular systems
have simpler hardware than stereo and are more accessible
in society, especially via mobile devices. Nevertheless, MVO
systems lack the depth information and scale of the objects
since the three-dimensio
