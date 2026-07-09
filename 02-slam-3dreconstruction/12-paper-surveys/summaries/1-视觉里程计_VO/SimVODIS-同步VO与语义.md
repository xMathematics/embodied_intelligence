# SimVODIS-同步VO与语义

## 基本信息
- **英文标题**: SimVODIS: Simultaneous Visual Odometry, Object Detection, and Instance Segmentation
- **作者**: Ue-Hwan Kim, Se-Ho Kim, Jong-Hwan Kim
- **发表会议/期刊**: IEEE TPAMI
- **关键词**: 多任务学习,语义VO,实例分割,无监督学习

## 研究背景（前提）
智能体需同时理解几何和语义信息。传统方法分别运行VO/SLAM和检测/分割模块，计算量大、架构复杂。

## 问题提出（由什么问题引出）
现有数据驱动VO无法提供语义信息。语义Mapping/SLAM需额外运行识别线程，增加系统复杂度。

## 要解决的问题
能否设计单线程神经网络同时实现VO(位姿+深度)和目标检测+实例分割？

---

## 采用的方法
基于Mask-RCNN架构扩展多任务分支：共享特征图上添加位姿估计和深度预测分支。无监督学习，光度一致性(视图合成)作为监督信号。

## 理论依据
Mask-RCNN共享特征图同时支持几何和语义任务。视图重建损失：利用估计位姿和深度从相邻帧合成当前帧，与原始帧比较。

## 核心公式推导
- **运动向量**: $u = [t^T, r^T]^T$
  - t为平移，r为欧拉角旋转

- **多任务损失**: $L = L_{pose} + L_{depth} + L_{det} + L_{seg}$
  - 所有任务的损失加权和



---

## 实验结果
在位姿估计、深度预测、目标检测、实例分割方面均达SOTA，单线程完成所有任务。

## 尚未解决的问题（后续方向）
多任务间可能竞争；模型规模大；语义任务依赖预训练Mask-RCNN。

---

## 原始摘要
with humans. The agents should perceive geometric features as well as semantic entities inherent in the environment. Contemporary
methods in general provide one type of information regarding the environment at a time, making it difﬁcult to conduct high-level tasks.
Moreover, running two types of methods and associating two resultant information requires a lot of computation and complicates the
software architecture. To overcome these limitations, we propose a neural architecture that simultaneously performs both geometric
and semantic tasks in a single thread: simultaneous visual odometry, object detection, and instance segmentation (SimVODIS).
Training SimVODIS requires unlabeled video sequences and the photometric consistency between input image frames generates
self-supervision signals. The performance of SimVODIS outperforms or matches the state-of-the-art performance in pose estimation,
depth map prediction, object detection, and instance segmentation tasks while completing all the tasks in a single thread. We expect
SimVODIS would enhance the autonomy of intelligent agents and let the agents provide effective services to humans.
Index Terms—Visual odometry (VO), data-driven VO, visual SLAM, semantic VO, semantic SLAM, semantic mapping, monocular
video, depth map prediction, depth estimation, ego-motion estimation, unsupervised learning, deep convolutional neural network
(CNN).
!
1
