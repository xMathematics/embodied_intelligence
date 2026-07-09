# Transformer引导几何模型VO

## 基本信息
- **英文标题**: Transformer Guided Geometry Model for Flow-Based Unsupervised Visual Odometry
- **作者**: Xiangyu Li, Yonghong Hou, Pichao Wang, Zhimin Gao, Mingliang Xu, Wanqing Li
- **发表会议/期刊**: AAAI 2021
- **关键词**: Transformer,无监督VO,光流,注意力机制

## 研究背景（前提）
现有无监督VO方法或仅用成对图像匹配，或用RNN整合长序列信息。前者不精确，后者训练耗时且误差累积。

## 问题提出（由什么问题引出）
RNN中单一隐状态存储的历史信息导致严重误差累积。能否用Transformer在局部时间窗口内建立几何模型？

## 要解决的问题
如何利用Transformer的注意机制在短时窗口内建立深度-光流-位姿的几何关联，减少误差累积？

---

## 采用的方法
提出双估计器：TAPE(Transformer辅助位姿估计器)处理局部时间窗口序列，F2FPE(光流到位姿估计器)处理成对图像。两者通过一致性损失约束，F2FPE为推理主估计器。

## 理论依据
Transformer的全注意力机制可在局部时间窗口内建模深度、光流和位姿的几何相关性和时间依赖性。将VO视为序列到序列的翻译问题。

## 核心公式推导
- **DF-Group**: $G_t = [D_t, F_{t-1\to t}, F_{t\to t+1}]$
  - 深度和光流的组合输入

- **一致性损失**: $L_{cons} = ||T_{F2FPE} - T_{TAPE}||$
  - 两个估计器位姿一致性约束

- **Transformer注意力**: $\text{Attn}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d})V$
  - 缩放点积注意力



---

## 实验结果
在KITTI和Malaga数据集上大幅超越无监督方法，与有监督和传统方法可比。显著减少误差累积。

## 尚未解决的问题（后续方向）
Transformer推理速度较慢；需要预计算深度和光流作为输入。

---

## 原始摘要
Existing unsupervised visual odometry (VO) methods ei-
ther match pairwise images or integrate the temporal in-
formation using recurrent neural networks over a long se-
quence of images.
They are either not accurate, time-
consuming in training or error accumulative. In this paper,
we propose a method consisting of two camera pose esti-
mators that deal with the information from pairwise images
and a short sequence of images respectively. For image se-
quences, a Transformer-like structure is adopted to build
a geometry model over a local temporal window, referred
to as Transformer-based Auxiliary Pose Estimator (TAPE).
Meanwhile, a Flow-to-Flow Pose Estimator (F2FPE) is
proposed to exploit the relationship between pairwise im-
ages. The two estimators are constrained through a simple
yet effective consistency loss in training. Empirical evalu-
ation has shown that the proposed method outperforms the
state-of-the-art unsupervised learning-based methods by a
large margin and performs comparably to supervised and
traditional ones on the KITTI and Malaga dataset.
