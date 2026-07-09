# DPVO-深度块VO

## 基本信息
- **英文标题**: Deep Patch Visual Odometry
- **作者**: Zachary Teed, Lahav Lipson, Jia Deng
- **发表会议/期刊**: NeurIPS 2023
- **关键词**: 图像块,可微分BA,联合优化

## 研究背景（前提）
现有VO方法或依赖稀疏特征点或预测密集光流，各有优缺点。

## 问题提出（由什么问题引出）
能否结合稀疏特征(高效)和密集匹配(鲁棒)的优点，在图像块(patch)级别进行VO？

## 要解决的问题
如何设计基于深度图像块的VO方法，利用可微分优化实现高精度帧间跟踪？

---

## 采用的方法
提出DPVO：将图像划分为重叠块(patch)，为每个块建立深度特征表示。通过可微分优化层联合优化所有块的位姿和深度。将VO表述为块级别优化问题。

## 理论依据
图像块保留了局部结构信息。同时优化多块的位姿约束可提供更鲁棒的运动估计。可微分优化将传统BA嵌入网络。

## 核心公式推导
- **块级特征**: $F_p = f_{CNN}(I, p)$
  - 提取每个图像块的特征表示

- **可微分BA**: $\min_{T,D} \sum_p ||\pi(T, D_p) - f_p||^2_{\Sigma}$
  - 块级光束法平差



---

## 实验结果
在多个VO基准上达到SOTA精度，特别是在长期漂移方面显著优于现有方法。

## 尚未解决的问题（后续方向）
块的数量影响计算效率；大运动下块间匹配困难。

---

## 原始摘要
We propose Deep Patch Visual Odometry (DPVO), a new deep learning system
for monocular Visual Odometry (VO). DPVO uses a novel recurrent network
architecture designed for tracking image patches across time. Recent approaches
to VO have significantly improved the state-of-the-art accuracy by using deep
networks to predict dense flow between video frames. However, using dense flow
incurs a large computational cost, making these previous methods impractical for
many use cases. Despite this, it has been assumed that dense flow is important
as it provides additional redundancy against incorrect matches. DPVO disproves
this assumption, showing that it is possible to get the best accuracy and efficiency
by exploiting the advantages of sparse patch-based matching over dense flow.
DPVO introduces a novel recurrent update operator for patch based correspondence
coupled with differentiable bundle adjustment. On Standard benchmarks, DPVO
outperforms all prior work, including the learning-based state-of-the-art VO-system
(DROID) using a third of the memory while running 3x faster on average. Code is
available at https://github.com/princeton-vl/DPVO
1
