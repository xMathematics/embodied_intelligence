# MAC-VO-度量感知立体VO

## 基本信息
- **英文标题**: MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual Odometry
- **作者**: Yuheng Qiu, Yutian Chen, Zihao Zhang, Wenshan Wang, Sebastian Scherer
- **发表会议/期刊**: CoRL 2024
- **关键词**: 不确定性估计,关键点选择,立体VO,协方差

## 研究背景（前提）
传统几何方法偏好纹理丰富的边缘特征；学习型方法使用尺度无关的权重矩阵。

## 问题提出（由什么问题引出）
能否训练度量感知的不确定性模型，同时实现关键点选择和位姿图优化中的残差加权？

## 要解决的问题
如何利用学习的不确定性模型实现1)关键点选择(剔除低质量特征)和2)位姿图优化中的协方差加权？

---

## 采用的方法
训练度量感知不确定性模型预测关键点的空间误差。该不确定性同时用于关键点选择和SLAM后端优化中的协方差矩阵。

## 理论依据
学习的不确定性可反映关键点的空间定位误差。低不确定性=高质量关键点。不确定性还可作为优化中的信息矩阵(协方差逆)。

## 核心公式推导
- **不确定性预测**: $\Sigma = f_\theta(\text{keypoint})$
  - 预测关键点的空间协方差

- **协方差加权优化**: $\min \sum e_i^T \Sigma_i^{-1} e_i$
  - 用不确定性加权残差



---

## 实验结果
在立体VO基准测试中优于传统几何方法和现有学习方法。

## 尚未解决的问题（后续方向）
不确定性预测的质量依赖训练数据分布；需要立体相机。

---

## 原始摘要
stereo visual odometry (VO) framework that trains a metrics-
aware uncertainty model to serve two critical functions: select-
ing keypoints and weighting residuals in pose graph optimiza-
tion. Unlike traditional geometric methods that favor texture-
rich features like edges, our keypoint selector leverages this
learned uncertainty model to eliminate low-quality features
based on global inconsistency. In contrast to learning-based
approaches that rely on scale-agnostic weight matrices for
covariance, our metrics-aware covariance model—derived from
the learned uncertainty—captures spatial errors in keypoint
registration and inter-axis correlations. By embedding this co-
variance model into pose graph optimization, MAC-VO achieves
superior robustness and accuracy in pose estimation, excelling
in challenging environments with varying illumination, feature
density, and motion patterns. Evaluations on public benchmark
datasets demonstrate that MAC-VO surpasses existing VO
algorithms and even some SLAM systems in difficult scenarios.
Additionally, the uncertainty map offers valuable insights for
decision-making.
Index Terms— SLAM, Learning VO, Covariance Estimation
I. INTRODUCTION
V
ISUAL Odometry (VO) predicts the relative cam-
era pose from image sequences and often serves as
the front-end of Simultaneous Localization and Mapping
(SLAM) systems. Over the past few decades, both geometric
and learning-based methods have been developed with sig-
nificant advances in generalizability
