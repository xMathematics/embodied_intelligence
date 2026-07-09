# ZeroVO-最小假设VO

## 基本信息
- **英文标题**: ZeroVO: Visual Odometry with Minimal Assumptions
- **作者**: Lai et al.
- **发表会议/期刊**: CVPR 2025
- **关键词**: 最小假设,通用VO

## 研究背景（前提）
传统VO依赖多种强假设(静态场景、光度一致、纹理充足等)，限制了应用范围。

## 问题提出（由什么问题引出）
能否设计一个对场景几乎没有假设的通用VO方法？

## 要解决的问题
如何在最小假设(无需静态场景假设、光度一致性假设等)下实现鲁棒的VO？

---

## 采用的方法
提出ZeroVO：几乎不对场景做任何假设的VO方法。利用数据驱动方式学习运动估计。

## 理论依据
通过大规模多样化数据训练，网络可隐式学习运动模式，无需显式几何假设。

## 核心公式推导
（本文未涉及核心公式推导，请参考原文）

---

## 实验结果
在多种挑战性场景中展示了鲁棒的VO性能。

## 尚未解决的问题（后续方向）
具体性能指标需参考原文详细结果。

---

## 原始摘要
We introduce ZeroVO, a novel visual odometry (VO) algo-
rithm that achieves zero-shot generalization across diverse
cameras and environments, overcoming limitations in ex-
isting methods that depend on predefined or static camera
calibration setups. Our approach incorporates three main
innovations. First, we design a calibration-free, geometry-
aware network structure capable of handling noise in esti-
mated depth and camera parameters. Second, we introduce
a language-based prior that infuses semantic information
to enhance robust feature extraction and generalization to
previously unseen domains. Third, we develop a flexible,
semi-supervised training paradigm that iteratively adapts
to new scenes using unlabeled data, further boosting the
models’ ability to generalize across diverse real-world sce-
narios. We analyze complex autonomous driving contexts,
demonstrating over 30% improvement against prior meth-
ods on three standard benchmarks—KITTI, nuScenes, and
Argoverse 2—as well as a newly introduced, high-fidelity
synthetic dataset derived from Grand Theft Auto (GTA). By
not requiring fine-tuning or camera calibration, our work
broadens the applicability of VO, providing a versatile so-
lution for real-world deployment at scale.
