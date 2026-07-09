# DINO-VO-视觉基础模型VO

## 基本信息
- **英文标题**: DINO-VO: A Feature-based Visual Odometry Leveraging a Visual Foundation Model
- **作者**: Maulana Bisyir Azhari, David Hyunchul Shim
- **发表会议/期刊**: IEEE RA-L 2025
- **关键词**: 视觉基础模型,DINOv2,特征匹配,可微分PnP

## 研究背景（前提）
学习型单目VO面临鲁棒性、泛化性和效率挑战。DINOv2等视觉基础模型改善了鲁棒性和泛化性。

## 问题提出（由什么问题引出）
DINOv2的特征粒度过粗，直接用于VO的稀疏特征匹配效果有限。如何将视觉基础模型适配到VO中？

## 要解决的问题
如何利用DINOv2视觉基础模型的鲁棒语义特征，结合细粒度几何特征，构建高性能特征VO系统？

---

## 采用的方法
提出DINO-VO：1)针对DINOv2粗特征设计显著关键点检测器；2)将DINOv2鲁棒语义特征与细粒度几何特征互补；3)基于Transformer的匹配器和可微分位姿估计层。

## 理论依据
视觉基础模型(DINOv2)提供鲁棒的语义特征表示。结合传统细粒度几何特征可弥补粗特征的空间分辨率不足。Transformer匹配器可学习高质量匹配。

## 核心公式推导
- **特征融合**: $F = f_{DINO}(I) \oplus f_{geo}(I)$
  - DINOv2语义特征与几何特征融合

- **可微分位姿**: $\hat{T} = \arg\min \sum ||\text{proj}(\hat{T}, X) - x||^2$
  - 可微分PnP层



---

## 实验结果
在TartanAir和KITTI上超越现有帧到帧VO方法。72FPS速度，<1GB显存使用。

## 尚未解决的问题（后续方向）
DINOv2推理有一定计算量；语义特征在极弱纹理区域仍有挑战。

---

## 原始摘要
poses robustness, generalization, and efficiency challenges in
robotics. Recent advances in visual foundation models, such
as DINOv2, have improved robustness and generalization in
various vision tasks, yet their integration in VO remains limited
due to coarse feature granularity. In this paper, we present
DINO-VO, a feature-based VO system leveraging DINOv2 visual
foundation model for its sparse feature matching. To address the
integration challenge, we propose a salient keypoints detector tai-
lored to DINOv2’s coarse features. Furthermore, we complement
DINOv2’s robust-semantic features with fine-grained geometric
features, resulting in more localizable representations. Finally,
a transformer-based matcher and differentiable pose estimation
layer enable precise camera motion estimation by learning good
matches. Against prior detector-descriptor networks like Super-
Point, DINO-VO demonstrates greater robustness in challenging
environments. Furthermore, we show superior accuracy and gen-
eralization of the proposed feature descriptors against standalone
DINOv2 coarse features. DINO-VO outperforms prior frame-to-
frame VO methods on the TartanAir and KITTI datasets and
is competitive on EuRoC dataset, while running efficiently at
72 FPS with less than 1GB of memory usage on a single GPU.
Moreover, it performs competitively against Visual SLAM sys-
tems on outdoor driving scenarios, showcasing its generalization
capabilities.
Index Terms—Deep Learning Methods; Localization; V
