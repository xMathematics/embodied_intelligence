# 开放世界在线自适应VO

## 基本信息
- **英文标题**: Generalizing to the Open World: Deep Visual Odometry with Online Adaptation
- **作者**: Shunkai Li, Xin Wang, Yingdian Cao, Fei Xue, Zike Yan, Hongbin Zha
- **发表会议/期刊**: CVPR 2021
- **关键词**: 在线自适应,泛化性,测试时训练

## 研究背景（前提）
学习型VO在训练环境外泛化能力有限，难以适应开放世界场景。

## 问题提出（由什么问题引出）
如何在测试时通过在线自适应让VO快速适应新环境？

## 要解决的问题
如何设计具备在线自适应能力的深度VO，使其能泛化到未见过的开放世界场景？

---

## 采用的方法
提出测试时在线自适应VO框架。设计自监督自适应机制，在推理阶段通过自监督信号(光度一致性)快速微调网络参数适应新环境。

## 理论依据
在测试时通过自监督损失(如光度一致性)对网络进行快速微调，可使模型适应目标场景的数据分布。无需目标场景的真值标签。

## 核心公式推导
- **自适应损失**: $L_{adapt} = L_{photo}(\hat{I}_t, I_t) + \lambda L_{smooth}(D)$
  - 推理时的自监督自适应损失

- **在线更新**: $\theta_{t+1} = \theta_t - \eta \nabla L_{adapt}(\theta_t)$
  - 每个测试场景进行少量梯度更新



---

## 实验结果
在跨数据集泛化实验中，在线自适应方法显著改善了VO在未见场景中的性能。

## 尚未解决的问题（后续方向）
在线自适应增加了推理时间；自适应可能在某些极端场景失败。

---

## 原始摘要
Despite learning-based visual odometry (VO) has shown
impressive results in recent years, the pretrained networks
may easily collapse in unseen environments.
The large
domain gap between training and testing data makes them
difﬁcult to generalize to new scenes.
In this paper, we
propose an online adaptation framework for deep VO with
the assistance of scene-agnostic geometric computations
and Bayesian inference. In contrast to learning-based pose
estimation, our method solves pose from optical ﬂow and
depth while the single-view depth estimation is continuously
improved with new observations by online learned uncer-
tainties. Meanwhile, an online learned photometric uncer-
tainty is used for further depth and pose optimization by
a differentiable Gauss-Newton layer. Our method enables
fast adaptation of deep VO networks to unseen environ-
ments in a self-supervised manner. Extensive experiments
including Cityscapes to KITTI and outdoor KITTI to indoor
TUM demonstrate that our method achieves state-of-the-art
generalization ability among self-supervised VO methods.
