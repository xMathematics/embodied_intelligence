# UnDeepVO-无监督单目VO

## 基本信息
- **英文标题**: UnDeepVO: Monocular Visual Odometry through Unsupervised Deep Learning
- **作者**: Ruihao Li, Sen Wang, Zhiqiang Long, Dongbing Gu
- **发表会议/期刊**: ICRA 2018
- **关键词**: 无监督学习,单目VO,绝对尺度恢复,立体视觉

## 研究背景（前提）
传统几何VO对相机参数敏感，在弱纹理、运动模糊等场景中脆弱。有监督深度学习方法需要大量位姿/深度标注数据，获取成本高昂。

## 问题提出（由什么问题引出）
现有无监督VO主要聚焦深度估计，无法恢复绝对尺度。仅使用单目序列训练无法解决VO的尺度模糊问题。

## 要解决的问题
如何在不使用标注数据的情况下，实现单目VO的6-DoF位姿估计和深度估计，并恢复绝对尺度？

---

## 采用的方法
提出UnDeepVO，由VGG-based位姿估计网络和编码器-解码器深度估计网络组成。训练阶段使用立体图像对，利用空间(左右目)和时间(连续帧)几何一致性构造无监督损失；测试仅用单目图像。位姿网络将平移和旋转分离为两个全连接层。

## 理论依据
基于立体视觉几何约束：空间一致性利用双目基线的投影几何；时间一致性利用连续帧的投影几何。通过已知双目基线恢复绝对尺度。

## 核心公式推导
- **视差-深度关系**: $D_p = Bf / D_{dep}$
  - B为基线，f为焦距，D_{dep}为深度

- **光度损失(SSIM+L1)**: $L_{pho} = \lambda_s L_{SSIM} + (1-\lambda_s)L_{l1}$
  - 合成图像与原始图像的联合光度损失

- **视差一致性**: $D_{dis} = D_p \times I_W$
  - I_W为图像宽度

- **总损失**: $L_{total} = L_{spatial} + L_{temporal}$
  - 空间损失(立体对)+时间损失(连续帧)



---

## 实验结果
在KITTI数据集上，UnDeepVO在无监督方法中取得有竞争力位姿精度，成功恢复绝对尺度。

## 尚未解决的问题（后续方向）
训练需立体图像对；深度质量有限；剧烈光照和快速运动时性能下降。

---

## 原始摘要
(VO) system called UnDeepVO in this paper. UnDeepVO is
able to estimate the 6-DoF pose of a monocular camera and
the depth of its view by using deep neural networks. There
are two salient features of the proposed UnDeepVO: one is
the unsupervised deep learning scheme, and the other is the
absolute scale recovery. Speciﬁcally, we train UnDeepVO by
using stereo image pairs to recover the scale but test it by
using consecutive monocular images. Thus, UnDeepVO is a
monocular system. The loss function deﬁned for training the
networks is based on spatial and temporal dense information. A
system overview is shown in Fig. 1. The experiments on KITTI
dataset show our UnDeepVO achieves good performance in
terms of pose accuracy.
I. INTRODUCTION
Visual odometry (VO) enables a robot to localize itself in
various environments by only using low-cost cameras. In the
past few decades, model-based VO or geometric VO has been
widely studied and its two paradigms, feature-based method
[1]–[3] and direct method [4]–[6], have both achieved great
success. However, model-based methods tend to be sensitive
to camera parameters and fragile in challenging settings, e.g.,
featureless places, motion blurs and lighting changes.
In recent years, data-driven VO or deep learning based
VO has drawn signiﬁcant attention due to its potentials in
learning capability and the robustness to camera parameters
and challenging environments. Starting from the relocaliza-
tion problem with the use of supervised learnin
