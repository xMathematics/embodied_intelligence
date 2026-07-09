# CoProU-VO-投影不确定性VO

## 基本信息
- **英文标题**: CoProU-VO: Combining Projected Uncertainty for End-to-End Unsupervised Monocular Visual Odometry
- **作者**: Jingchao Xie, Oussema Dhaouadi, Weirong Chen, Johannes Meier, Jacques Kaiser, Daniel Cremers
- **发表会议/期刊**: arXiv 2025
- **关键词**: 不确定性,无监督VO,概率建模

## 研究背景（前提）
无监督单目VO利用光度一致性作为监督信号，但在低纹理、动态区域和光照变化中光度损失不可靠。

## 问题提出（由什么问题引出）
光度损失在低纹理/动态/光照变化区域不可靠。能否建模和传播不确定性来缓解这些问题？

## 要解决的问题
如何将投影不确定性(不确定性的传播)结合到端到端无监督单目VO框架中？

---

## 采用的方法
提出不确定性感知的无监督单目VO框架。建模深度和位姿的不确定性，并将其通过投影过程传播到图像空间，在损失函数中加权处理不可靠区域。

## 理论依据
不确定性可估计深度和位姿预测的置信度。通过不确定性的投影传播，可在图像空间识别不可靠区域并在损失中降低其权重。

## 核心公式推导
- **不确定性传播**: $\Sigma_{proj} = J \Sigma_{depth} J^T + \Sigma_{pose}$
  - 从深度和位姿不确定性传播到投影不确定性

- **不确定性加权损失**: $L = \sum w(\Sigma_{proj}) \cdot \rho(I_t, \hat{I}_t)$
  - 用不确定性权重调制光度损失



---

## 实验结果
在多个基准测试中优于现有无监督单目VO方法。

## 尚未解决的问题（后续方向）
不确定性建模增加了计算复杂度；不确定性估计的准确性依赖模型设计。

---

## 原始摘要
igation, robotics, and augmented reality, with unsupervised approaches
eliminating the need for expensive ground-truth labels. However, these
methods struggle when dynamic objects violate the static scene assump-
tion, leading to erroneous pose estimations. We tackle this problem by
uncertainty modeling, which is a commonly used technique that creates
robust masks to filter out dynamic objects and occlusions without re-
quiring explicit motion segmentation. Traditional uncertainty modeling
considers only single-frame information, overlooking the uncertainties
across consecutive frames. Our key insight is that uncertainty must be
propagated and combined across temporal frames to effectively identify
unreliable regions, particularly in dynamic scenes. To address this chal-
lenge, we introduce Combined Projected Uncertainty VO (CoProU-VO),
a novel end-to-end approach that combines target frame uncertainty with
projected reference frame uncertainty using a principled probabilistic
formulation. Built upon vision transformer backbones, our model si-
multaneously learns depth, uncertainty estimation, and camera poses.
Consequently, experiments on the KITTI and nuScenes datasets demon-
strate significant improvements over previous unsupervised monocular
end-to-end two-frame-based methods and exhibit strong performance
in challenging highway scenes where other approaches often fail. Ad-
ditionally, comprehensive ablation studies validate the effectiveness of
cross-frame uncertainty pr
