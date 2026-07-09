# D3VO-深度位姿不确定性

## 基本信息
- **英文标题**: D3VO: Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry
- **作者**: Nan Yang, Lukas von Stumberg, Rui Wang, Daniel Cremers
- **发表会议/期刊**: CVPR 2020
- **关键词**: 深度估计,位姿估计,不确定性,直接法VO

## 研究背景（前提）
直接法VO依赖光度一致性假设，在光照变化、曝光变化等场景中易失败。

## 问题提出（由什么问题引出）
能否用三个深度网络分别预测深度、相对位姿变换和不确定性，并集成到传统直接法VO中提升鲁棒性？

## 要解决的问题
如何将深度学习的深度估计、位姿估计和不确定性预测与直接法VO(如DSO)融合？

---

## 采用的方法
提出D3VO：三个网络分别预测单目深度(自监督)、帧间相对位姿和光度不确定性。预测的深度作为直接法VO的虚拟立体约束，不确定性作为光度残差的权重。

## 理论依据
自监督深度网络提供度量深度作为虚拟立体约束。不确定性网络预测光度误差的方差，作为优化中的自适应权重。

## 核心公式推导
- **深度网络**: $D_t = f_{depth}(I_t)$
  - 自监督单目深度估计

- **位姿网络**: $T_{t\to s} = f_{pose}(I_t, I_s)$
  - 帧间相对位姿估计

- **不确定性**: $\Sigma_t = f_{unc}(I_t)$
  - 预测光度不确定性的方差

- **加权光度损失**: $L = \sum \frac{||I_t - \hat{I}_t||^2}{\Sigma_t} + \log \Sigma_t$
  - 不确定性加权的光度损失



---

## 实验结果
在多个VO基准上超越DSO等传统直接法。不确定性加权显著提升了光度跟踪的鲁棒性。

## 尚未解决的问题（后续方向）
三个网络增加了系统复杂性；实时性受网络推理速度限制。

---

## 原始摘要
We propose D3VO as a novel framework for monocu-
lar visual odometry that exploits deep networks on three
levels – deep depth, pose and uncertainty estimation. We
ﬁrst propose a novel self-supervised monocular depth es-
timation network trained on stereo videos without any ex-
ternal supervision. In particular, it aligns the training im-
age pairs into similar lighting condition with predictive
brightness transformation parameters. Besides, we model
the photometric uncertainties of pixels on the input images,
which improves the depth estimation accuracy and provides
a learned weighting function for the photometric residu-
als in direct (feature-less) visual odometry. Evaluation re-
sults show that the proposed network outperforms state-of-
the-art self-supervised depth estimation networks. D3VO
tightly incorporates the predicted depth, pose and uncer-
tainty into a direct visual odometry method to boost both
the front-end tracking as well as the back-end non-linear
optimization. We evaluate D3VO in terms of monocular vi-
sual odometry on both the KITTI odometry benchmark and
the EuRoC MAV dataset. The results show that D3VO out-
performs state-of-the-art traditional monocular VO meth-
ods by a large margin.
It also achieves comparable re-
sults to state-of-the-art stereo/LiDAR odometry on KITTI
and to the state-of-the-art visual-inertial odometry on Eu-
RoC MAV, while using only a single camera.
