# DytanVO-动态环境VO

## 基本信息
- **英文标题**: DytanVO: Joint Refinement of Visual Odometry and Motion Segmentation in Dynamic Environments
- **作者**: Shihao Shen, Yilin Cai, Wenshan Wang, Sebastian Scherer
- **发表会议/期刊**: CoRL 2022
- **关键词**: 动态环境,运动分割,联合优化

## 研究背景（前提）
学习型VO算法在静态场景表现优异，但在动态、拥挤的环境中容易失败。

## 问题提出（由什么问题引出）
语义分割丢弃动态关联但也丢弃了静态特征，且难以扩展到未见过的类别。能否利用自运动与运动分割的相互依赖关系？

## 要解决的问题
如何在动态环境中联合优化相机自运动估计和运动分割？

---

## 采用的方法
第一个基于监督学习的动态环境VO方法。利用自运动与运动分割的相互依赖关系，设计联合优化框架，同时估计相机位姿和运动物体分割。

## 理论依据
相机自运动(ego-motion)和物体运动(motion segmentation)之间存在相互约束关系。正确的自运动估计需要排除动态点，正确的运动分割需要准确的自运动。

## 核心公式推导
- **联合优化**: $\min_{T, M} L_{pose}(T) + L_{seg}(M) + L_{joint}(T,M)$
  - 位姿、分割和联合损失的优化



---

## 实验结果
在动态场景数据集上显著优于现有的VO方法。有效处理了动态环境中的位姿估计。

## 尚未解决的问题（后续方向）
需要运动分割标注数据训练；极端密集动态场景仍有挑战。

---

## 原始摘要
achieve remarkable performance on common static scenes,
beneﬁting from high-capacity models and massive annotated
data, but tend to fail in dynamic, populated environments.
Semantic segmentation is largely used to discard dynamic
associations before estimating camera motions but at the cost
of discarding static features and is hard to scale up to unseen
categories. In this paper, we leverage the mutual dependence
between camera ego-motion and motion segmentation and
show that both can be jointly reﬁned in a single learning-
based framework. In particular, we present DytanVO, the
ﬁrst supervised learning-based VO method that deals with
dynamic environments. It takes two consecutive monocular
frames in real-time and predicts camera ego-motion in an
iterative fashion. Our method achieves an average improvement
of 27.7% in ATE over state-of-the-art VO solutions in real-world
dynamic environments, and even performs competitively among
dynamic visual SLAM systems which optimize the trajectory
on the backend. Experiments on plentiful unseen environments
also demonstrate our method’s generalizability.
I. INTRODUCTION
Visual odometry (VO), one of the most essential com-
ponents for pose estimation in the visual Simultaneous
Localization and Mapping (SLAM) system, has attracted
signiﬁcant interest in robotic applications over past few
years [1]. A lot of research works have been conducted
to develop an accurate and robust monocular VO system
using both geometry-based methods [2], [3]. 
