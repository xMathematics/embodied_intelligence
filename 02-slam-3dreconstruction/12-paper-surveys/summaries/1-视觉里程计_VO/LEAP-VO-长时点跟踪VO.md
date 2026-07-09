# LEAP-VO-长时点跟踪VO

## 基本信息
- **英文标题**: LEAP-VO: Long-term Effective Any Point Tracking for Visual Odometry
- **作者**: Weirong Chen, Le Chen, Rui Wang, Marc Pollefeys
- **发表会议/期刊**: CVPR 2024
- **关键词**: 长时点跟踪,时序概率,遮挡处理

## 研究背景（前提）
现有VO主要聚焦双视点跟踪，忽略了图像序列中丰富的时序上下文和全局运动模式。

## 问题提出（由什么问题引出）
双视点方法无法处理遮挡，不提供轨迹可靠性评估。长时点跟踪可利用时序上下文应对遮挡。

## 要解决的问题
如何利用长时点跟踪解决VO中的遮挡、动态物体和低纹理场景？

---

## 采用的方法
提出LEAP模块：结合视觉、跟踪间和时序线索，基于锚点的动态跟踪估计。时序概率公式将分布更新集成到可学习迭代精化模块中。作为VO前端使用。

## 理论依据
长时点跟踪跨多帧跟踪查询点，即使发生遮挡也能可靠跟踪。时序概率建模可推理逐点不确定性。

## 核心公式推导
- **长时跟踪**: $\text{traj}(p) = f_{LEAP}(I_1,...,I_T, p)$
  - 跨多帧的像素轨迹估计

- **概率精化**: $p(\text{pos}_t | I_{1:t}) \propto p(I_t | \text{pos}_t) p(\text{pos}_t | \text{pos}_{t-1})$
  - 时序概率跟踪



---

## 实验结果
在多个VO基准测试中显著优于现有基线方法。有效处理遮挡和动态场景。

## 尚未解决的问题（后续方向）
长时跟踪计算量较大；需大量训练数据。

---

## 原始摘要
Visual odometry estimates the motion of a moving cam-
era based on visual input. Existing methods, mostly focus-
ing on two-view point tracking, often ignore the rich tempo-
ral context in the image sequence, thereby overlooking the
global motion patterns and providing no assessment of the
full trajectory reliability. These shortcomings hinder per-
formance in scenarios with occlusion, dynamic objects, and
low-texture areas. To address these challenges, we present
the Long-term Effective Any Point Tracking (LEAP) mod-
ule. LEAP innovatively combines visual, inter-track, and
temporal cues with mindfully selected anchors for dynamic
track estimation. Moreover, LEAP’s temporal probabilistic
formulation integrates distribution updates into a learnable
iterative refinement module to reason about point-wise un-
certainty. Based on these traits, we develop LEAP-VO, a
robust visual odometry system adept at handling occlusions
and dynamic scenes. Our mindful integration showcases a
novel practice by employing long-term point tracking as the
front-end. Extensive experiments demonstrate that the pro-
posed pipeline significantly outperforms existing baselines
across various visual odometry benchmarks.
