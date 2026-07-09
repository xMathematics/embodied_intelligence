# DeepAVO-特征蒸馏深度VO

## 基本信息
- **英文标题**: DeepAVO: Efficient Pose Refining with Feature Distilling for Deep Visual Odometry
- **作者**: Ran Zhu, Mingkun Yang, Wang Liu, Rujun Song, Bo Yan, Zhuoling Xiao
- **发表会议/期刊**: IEEE RA-L 2021
- **关键词**: 特征蒸馏,注意力机制,分支网络

## 研究背景（前提）
学习型VO方法中，特征对运动模式的贡献不同但被等同对待。不同运动模式(旋转/平移)需要不同特征。

## 问题提出（由什么问题引出）
特征对不同运动模式有区分性贡献。能否设计注意力机制让网络自动关注与特定运动相关的特征？

## 要解决的问题
如何通过特征蒸馏和注意力机制，让网络学习针对不同运动模式(旋转/平移)的特征选择？

---

## 采用的方法
提出四分支网络，每分支关注光流输入的不同象限，分别学习旋转和平移。引入通道-空间注意力机制强制每分支显式蒸馏与特定运动相关的特征。

## 理论依据
光流场不同区域对不同运动模式有差异化响应。通过注意力机制和分支结构实现特征的差异化学习。

## 核心公式推导
- **四分支结构**: $R = \sum f_R(\text{quad}_i), t = \sum f_t(\text{quad}_i)$
  - 各分支分别贡献旋转和平移

- **通道注意力**: $M_c = \sigma(\text{MLP}(\text{GAP}(F)))$
  -  squeeze-and-excitation通道注意力



---

## 实验结果
在KITTI数据集上验证了特征蒸馏的有效性，位姿估计精度优于基线方法。

## 尚未解决的问题（后续方向）
四分支结构增加计算量；光流作为输入限制了端到端特性。

---

## 原始摘要
The technology for Visual Odometry (VO) that estimates the position and orientation of the moving object
through analyzing the image sequences captured by on-board cameras, has been well investigated with the
rising interest in autonomous driving.
This paper studies monocular VO from the perspective of Deep
Learning (DL). Unlike most current learning-based methods, our approach, called DeepAVO, is established
on the intuition that features contribute discriminately to diﬀerent motion patterns. Speciﬁcally, we present a
novel four-branch network to learn the rotation and translation by leveraging Convolutional Neural Networks
(CNNs) to focus on diﬀerent quadrants of optical ﬂow input. To enhance the ability of feature selection, we
further introduce an eﬀective channel-spatial attention mechanism to force each branch to explicitly distill
related information for speciﬁc Frame to Frame (F2F) motion estimation. Experiments on various datasets
involving outdoor driving and indoor walking scenarios show that the proposed DeepAVO outperforms the
state-of-the-art monocular methods by a large margin, demonstrating competitive performance to the stereo
VO algorithm and verifying promising potential for generalization.
Keywords:
Visual odometry, neural network, attention mechanism, monocular camera
