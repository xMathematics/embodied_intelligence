# 显著稀疏位姿监督VO

## 基本信息
- **英文标题**: Salient Sparse Visual Odometry with Pose-only Supervision
- **作者**: Siyu Chen, Kangcheng Liu, Chen Wang, Shenghai Yuan, Jianfei Yang, Lihua Xie
- **发表会议/期刊**: IEEE RA-L 2024
- **关键词**: 位姿监督,单应预训练,显著点,混合方法

## 研究背景（前提）
传统VO在光照变化和运动模糊中困难。深度学习VO更鲁棒但泛化性有问题。

## 问题提出（由什么问题引出）
现有方法需密集光流标签。能否仅用位姿标签(易获得)训练，无需密集光流标注？如何提高泛化性？

## 要解决的问题
如何仅使用位姿监督(无需密集光流标签)训练鲁棒且泛化的VO系统？

---

## 采用的方法
提出混合VO框架：1)自监督单应预训练增强光流学习；2)随机块显著点检测策略提取光流块。仅需位姿标签训练。

## 理论依据
单应性预训练可学习鲁棒的光流表示。显著点检测在稀疏区域提取高质量跟踪点。

## 核心公式推导
- **单应预训练**: $L_{homo} = ||I_1 - \text{warp}(I_2, H)||$
  - 自监督单应性学习

- **仅位姿监督**: $L_{pose} = ||\hat{T} - T_{gt}||$
  - 仅使用位姿真值监督



---

## 实验结果
在标准数据集上达到有竞争力性能，在极端和未见场景中具有更强鲁棒性和泛化性。

## 尚未解决的问题（后续方向）
位姿标签在某些场景仍需外部传感器；极端运动下性能下降。

---

## 原始摘要
autonomous systems, providing accurate position and orientation
estimates at reasonable costs. While traditional VO methods excel
in some conditions, they struggle with challenges like variable
lighting and motion blur. Deep learning-based VO, though more
adaptable, can face generalization problems in new environments.
Addressing these drawbacks, this paper presents a novel hybrid
visual odometry (VO) framework that leverages pose-only super-
vision, offering a balanced solution between robustness and the
need for extensive labeling. We propose two cost-effective and
innovative designs: a self-supervised homographic pre-training
for enhancing optical flow learning from pose-only labels and
a random patch-based salient point detection strategy for more
accurate optical flow patch extraction. These designs eliminate
the need for dense optical flow labels for training and significantly
improve the generalization capability of the system in diverse
and challenging environments. Our pose-only supervised method
achieves competitive performance on standard datasets and
greater robustness and generalization ability in extreme and
unseen scenarios, even compared to dense optical flow-supervised
state-of-the-art methods.
I. INTRODUCTION
Visual odometry (VO) is a fundamental technique in en-
abling autonomous systems to estimate position and orien-
tation with respect to their surrounding environment. It has
wide applications in autonomous driving, AR/VR, and mobile
robotics due to its 
