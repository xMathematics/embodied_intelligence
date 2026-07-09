# RAUM-VO-旋转校正无监督VO

## 基本信息
- **英文标题**: RAUM-VO: Rotational Adjusted Unsupervised Monocular Visual Odometry
- **作者**: Claudio Cimarelli, Hriday Bavle, Jose Luis Sanchez-Lopez, Holger Voos
- **发表会议/期刊**: Sensors (MDPI) 2022
- **关键词**: 旋转校正,无监督VO,光度损失

## 研究背景（前提）
无监督单目VO在快速旋转时性能显著下降，因为光度损失假设对大旋转不鲁棒。

## 问题提出（由什么问题引出）
无监督VO的弱点是旋转估计精度不足。能否通过显式的旋转校正机制改善？

## 要解决的问题
如何在校正旋转效应的基础上进行无监督单目VO训练？

---

## 采用的方法
提出旋转校正的无监督VO框架。在光度损失计算前，先通过旋转估计和校正减少帧间大旋转的影响。

## 理论依据
先估计并补偿帧间旋转，剩余平移运动可通过光度一致性更鲁棒地估计。

## 核心公式推导
- **旋转校正**: $I'_t = \text{warp}(I_t, R_{t\to t+1})$
  - 用估计旋转校正图像

- **校正后光度损失**: $L = ||I_{t+1} - \text{warp}(I'_t, t_{t\to t+1})||$
  - 在校正图像上计算平移的光度损失



---

## 实验结果
在快速旋转场景中显著改善了无监督VO的性能。

## 尚未解决的问题（后续方向）
旋转估计本身仍有误差；极端旋转中校正可能不准确。

---

## 原始摘要
gained popularity over traditional methods, which rely on epipolar geometry or non-linear opti-
mization. Notably, deep learning can overcome many issues of monocular vision, such as perceptual
aliasing, low-textured areas, scale drift, and degenerate motions. In addition, concerning supervised
learning, we can fully leverage video stream data without the need for depth or motion labels. How-
ever, in this work, we note that rotational motion can limit the accuracy of the unsupervised pose
networks more than the translational component. Therefore, we present RAUM-VO, an approach
based on a model-free epipolar constraint for frame-to-frame motion estimation (F2F) to adjust the ro-
tation during training and online inference. To this end, we match 2D keypoints between consecutive
frames using pre-trained deep networks, Superpoint and Superglue, while training a network for
depth and pose estimation using an unsupervised training protocol. Then, we adjust the predicted
rotation with the motion estimated by F2F using the 2D matches and initializing the solver with
the pose network prediction. Ultimately, RAUM-VO shows a considerable accuracy improvement
compared to other unsupervised pose networks on the KITTI dataset, while reducing the complexity
of other hybrid or traditional approaches and achieving comparable state-of-the-art results.
Keywords: visual odometry; depth estimation; unsupervised learning; deep learning
