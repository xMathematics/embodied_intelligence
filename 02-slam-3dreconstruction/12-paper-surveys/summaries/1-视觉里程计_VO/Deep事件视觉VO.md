# Deep事件视觉VO

## 基本信息
- **英文标题**: Deep Event Visual Odometry
- **作者**: Simon Klenk, Marvin Motzet, Lukas Koestler, Daniel Cremers
- **发表会议/期刊**: 3DV 2024
- **关键词**: 事件相机,深度VO,端到端学习

## 研究背景（前提）
事件相机能在高速运动和弱光条件下跟踪位姿，但现有基于事件的单目VO在基准测试中性能有限。

## 问题提出（由什么问题引出）
现有事件VO方法在基准测试中表现有限，部分方法甚至诉诸多传感器(IMU/深度)辅助。能否设计纯事件相机的深度VO？

## 要解决的问题
如何设计基于端到端深度学习的纯事件相机单目VO？

---

## 采用的方法
提出纯事件相机的深度VO框架，利用事件数据的时间丰富性进行端到端位姿估计。

## 理论依据
事件流的高时间分辨率(μs级)提供了丰富的运动信息，可通过神经网络直接回归位姿。

## 核心公式推导
（本文未涉及核心公式推导，请参考原文）

---

## 实验结果
在事件VO基准测试中达到纯事件方法的SOTA性能。

## 尚未解决的问题（后续方向）
事件相机分辨率低；弱纹理场景仍有挑战；需专门的硬件。

---

## 原始摘要
Event cameras offer the exciting possibility of tracking
the camera’s pose during high-speed motion and in adverse
lighting conditions. Despite this promise, existing event-
based monocular visual odometry (VO) approaches demon-
strate limited performance on recent benchmarks. To ad-
dress this limitation, some methods resort to additional sen-
sors such as IMUs, stereo event cameras, or frame-based
cameras. Nonetheless, these additional sensors limit the ap-
plication of event cameras in real-world devices since they
increase cost and complicate system requirements. More-
over, relying on a frame-based camera makes the system
susceptible to motion blur and HDR. To remove the depen-
dency on additional sensors and to push the limits of us-
ing only a single event camera, we present Deep Event VO
(DEVO), the first monocular event-only system with strong
performance on a large number of real-world benchmarks.
DEVO sparsely tracks selected event patches over time. A
key component of DEVO is a novel deep patch selection
mechanism tailored to event data. We significantly decrease
the pose tracking error on seven real-world benchmarks by
up to 97% compared to event-only methods and often sur-
pass or are close to stereo or inertial methods.
