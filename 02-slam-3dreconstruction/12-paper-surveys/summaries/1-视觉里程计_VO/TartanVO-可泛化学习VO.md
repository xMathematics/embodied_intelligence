# TartanVO-可泛化学习VO

## 基本信息
- **英文标题**: TartanVO: A Generalizable Learning-based VO
- **作者**: Wenshan Wang, Yaoyu Hu, Sebastian Scherer
- **发表会议/期刊**: CoRL 2021
- **关键词**: 泛化性,合成数据,上尺度损失,相机内参

## 研究背景（前提）
现有学习型VO只能工作在训练数据集的场景中，难以泛化到新数据集和真实世界场景。

## 问题提出（由什么问题引出）
能否利用大规模的多样化合成数据(TartanAir)训练出可跨数据集泛化的VO模型？

## 要解决的问题
如何训练一个泛化到多种数据集和真实世界场景的学习型VO模型？

---

## 采用的方法
利用TartanAir大规模多样化合成数据(含挑战环境)训练。提出up-to-scale损失函数，将相机内参作为网络输入以实现跨数据集泛化。

## 理论依据
大规模多样化合成数据可覆盖广泛的场景和运动模式。上尺度不变的损失函数使网络不依赖绝对尺度。内参作为输入使网络适应不同相机。

## 核心公式推导
- **Up-to-scale损失**: $L = \min_s ||\hat{T} - s \cdot T_{gt}||$
  - 消除尺度模糊的损失函数

- **内参条件化**: $\hat{T} = f(I_1, I_2, K)$
  - 将相机内参K作为网络输入



---

## 实验结果
仅用合成数据训练，无需微调，即可在多个真实数据集(KITTI/EuRoC等)上泛化，在挑战场景中超越几何方法。

## 尚未解决的问题（后续方向）
合成数据与真实数据的域差距仍然存在；极端真实场景中可能退化。

---

## 原始摘要
which generalizes to multiple datasets and real-world scenarios, and outperforms
geometry-based methods in challenging scenes. We achieve this by leveraging
the SLAM dataset TartanAir, which provides a large amount of diverse synthetic
data in challenging environments. Furthermore, to make our VO model generalize
across datasets, we propose an up-to-scale loss function and incorporate the cam-
era intrinsic parameters into the model. Experiments show that a single model,
TartanVO, trained only on synthetic data, without any ﬁnetuning, can be general-
ized to real-world datasets such as KITTI and EuRoC, demonstrating signiﬁcant
advantages over the geometry-based methods on challenging trajectories. Our
code is available at https://github.com/castacks/tartanvo.
Keywords: Visual Odometry, Generalization, Deep Learning, Optical Flow
1
