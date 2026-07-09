# DF-VO-重访视觉里程计

## 基本信息
- **英文标题**: Visual Odometry Revisited: What Should Be Learnt?
- **作者**: Huangying Zhan, Chamara Saroj Weerasekera, Jia-Wang Bian, Ian Reid
- **发表会议/期刊**: ICRA 2020
- **关键词**: 混合方法,对极几何,PnP,尺度恢复,光流

## 研究背景（前提）
端到端深度学习VO性能仍不及几何方法。纯几何VO在有利条件下精确但受尺度漂移困扰。DL可直接预测真实尺度。

## 问题提出（由什么问题引出）
如何正确地将深度学习融入几何VO框架？哪些模块应该学习？应该学什么？如何取长补短？

## 要解决的问题
如何结合深度学习(单视图深度CNN+双视图光流CNN)与经典多视图几何(对极几何、PnP)，构建简单鲁棒的帧到帧VO？

---

## 采用的方法
提出DF-VO：训练CNN用于单视图深度和双视图光流。用前向-后向光流一致性筛选高质量2D匹配。大光流时用对极几何+三角化+深度缩放恢复尺度；小光流时用PnP。

## 理论依据
对极几何通过本质矩阵E恢复相对位姿。PnP通过3D-2D对应求解位姿。用CNN深度作参考恢复绝对尺度。

## 核心公式推导
- **对极约束**: $p_2^T K^{-T} E K^{-1} p_1 = 0, E=[t]_\times R$
  - 2D-2D匹配的几何约束

- **PnP重投影误差**: $e = \sum ||K(RX_i+t) - p_i||^2$
  - 最小化3D到2D的投影误差

- **尺度恢复**: $s = \text{median}(D_{cnn} / D_{tri})$
  - CNN深度与三角化深度之比的中值

- **光流一致性**: $\text{err} = |-F_{fw} - F_{bw}|$
  - 前向与后向光流差异



---

## 实验结果
在KITTI上超越纯深度学习和纯几何方法。深度CNN辅助解决尺度漂移。

## 尚未解决的问题（后续方向）
纯旋转时对极几何退化；单视图深度估计精度仍有限。

---

## 原始摘要
odometry (VO) algorithm which leverages geometry-based
methods and deep learning. Most existing VO/SLAM systems
with superior performance are based on geometry and have
to be carefully designed for different application scenarios.
Moreover, most monocular systems suffer from scale-drift issue.
Some recent deep learning works learn VO in an end-to-end
manner but the performance of these deep systems is still not
comparable to geometry-based methods. In this work, we revisit
the basics of VO and explore the right way for integrating
deep learning with epipolar geometry and Perspective-n-Point
(PnP) method. Speciﬁcally, we train two convolutional neural
networks (CNNs) for estimating single-view depths and two-
view optical ﬂows as intermediate outputs. With the deep
predictions, we design a simple but robust frame-to-frame VO
algorithm (DF-VO) which outperforms pure deep learning-
based and geometry-based methods. More importantly, our
system does not suffer from the scale-drift issue being aided by a
scale consistent single-view depth CNN. Extensive experiments
on KITTI dataset shows the robustness of our system and a
detailed ablation study shows the effect of different factors in
our system. Code is available at here: DF-VO.
I. INTRODUCTION
The ability for an autonomous robot to know its where-
abouts and its surroundings is of utmost importance for tasks
such as object manipulation and navigation. Vision-based
localisation and mapping is often the preferred choice due
to fa
