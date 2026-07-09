# RNN无监督视频VO

## 基本信息
- **英文标题**: Recurrent Neural Network for (Un-)supervised Learning of Monocular Video Visual Odometry
- **作者**: Sen Wang, Ronald Clark, Hongkai Wen, Niki Trigoni
- **发表会议/期刊**: CVPR 2019
- **关键词**: RNN,无监督学习,光度损失,序列建模

## 研究背景（前提）
DeepVO[ICRA 2017]提出有监督端到端VO，但需大量位姿标注。无监督VO通过光度损失训练但缺乏时序建模。

## 问题提出（由什么问题引出）
能否结合无监督学习和RNN时序建模优势，在无标注数据上学习序列运动模式？

## 要解决的问题
如何在无监督(无需位姿真值)框架下，用RNN建模视频序列的时序运动信息进行VO？

---

## 采用的方法
提出无监督+有监督联合训练框架。无监督部分使用光度一致性损失，RNN建模帧间时序关系。可仅用无监督模式训练。

## 理论依据
光度一致性损失提供自监督信号。RNN隐式学习序列的运动动力学。

## 核心公式推导
- **无监督光度损失**: $L_{unsup} = \sum ||I_t - \hat{I}_t||$
  - 视图重建的光度一致性损失

- **联合损失**: $L = L_{unsup} + \lambda L_{sup}$
  - 无监督与有监督损失可组合



---

## 实验结果
在KITTI上验证了结合RNN和无监督学习的有效性。无监督模式取得了与有监督模式可比的性能。

## 尚未解决的问题（后续方向）
无监督模式的精度仍低于有监督模式；光度损失在挑战场景中不可靠。

---

## 原始摘要
Deep learning-based,
single-view depth estimation
methods have recently shown highly promising results.
However, such methods ignore one of the most important
features for determining depth in the human vision sys-
tem, which is motion. We propose a learning-based, multi-
view dense depth map and odometry estimation method that
uses Recurrent Neural Networks (RNN) and trains utilizing
multi-view image reprojection and forward-backward ﬂow-
consistency losses. Our model can be trained in a super-
vised or even unsupervised mode. It is designed for depth
and visual odometry estimation from video where the in-
put frames are temporally correlated. However, it also gen-
eralizes to single-view depth estimation. Our method pro-
duces superior results to the state-of-the-art approaches for
single-view and multi-view learning-based depth estimation
on the KITTI driving dataset.
