# AFT-VO-异步融合Transformer

## 基本信息
- **英文标题**: AFT-VO: Asynchronous Fusion Transformers for Multi-View Visual Odometry Estimation
- **作者**: Nimet Kaygusuz, Oscar Mendez, Richard Bowden
- **发表会议/期刊**: IEEE RA-L 2023
- **关键词**: 传感器融合,异步,Transformer,多视角

## 研究背景（前提）
传感器融合方法(如卡尔曼滤波)假设传感器同步，但低成本硬件中不同步问题普遍存在。

## 问题提出（由什么问题引出）
现有深度融合方法通常假设传感器同步，这在低成本硬件中不切实际。如何融合异步多视角相机？

## 要解决的问题
如何用Transformer融合异步多视角相机的预测，处理时间差异实现鲁棒的VO估计？

---

## 采用的方法
提出基于Transformer的异步融合架构，将多视角相机的时间戳差异编码为位置信息输入Transformer，融合各相机独立VO预测。

## 理论依据
Transformer的自注意力机制能够处理非对齐的序列数据。时间戳差异可作为位置编码，让网络学习跨相机的时间关系。

## 核心公式推导
- **异步位置编码**: $PE(\Delta t) = f(\Delta t)$
  - 将时间差异编码为Transformer位置编码

- **跨模态注意力**: $\text{CrossAttn}(Q_i, K_j, V_j)$
  - 不同相机特征之间的交叉注意力



---

## 实验结果
在异步多视角VO数据集上优于现有融合方法。

## 尚未解决的问题（后续方向）
需要精确的时间戳信息；计算复杂度随相机数量增加。

---

## 原始摘要
sensor fusion techniques, such as the Kalman Filter, to han-
dle individual sensor failures. More recently, deep learning-
based fusion approaches have been proposed, increasing the
performance and requiring less model-speciﬁc implementations.
However, current deep fusion approaches often assume that
sensors are synchronised, which is not always practical, es-
pecially for low-cost hardware. To address this limitation, in
this work, we propose AFT-VO, a novel transformer-based
sensor fusion architecture to estimate VO from multiple sensors.
Our framework combines predictions from asynchronous multi-
view cameras and accounts for the time discrepancies of
measurements coming from different sources.
Our approach ﬁrst employs a Mixture Density Network
(MDN) to estimate the probability distributions of the 6-DoF
poses for every camera in the system. Then a novel transformer-
based fusion module, AFT-VO, is introduced, which combines
these asynchronous pose estimations, along with their conﬁ-
dences. More speciﬁcally, we introduce Discretiser and Source
Encoding techniques which enable the fusion of multi-source
asynchronous signals.
We evaluate our approach on the popular nuScenes and
KITTI datasets. Our experiments demonstrate that multi-view
fusion for VO estimation provides robust and accurate trajec-
tories, outperforming the state of the art in both challenging
weather and lighting conditions.
I. INTRODUCTION
Visual Odometry (VO) can be described as the process
of estimating
