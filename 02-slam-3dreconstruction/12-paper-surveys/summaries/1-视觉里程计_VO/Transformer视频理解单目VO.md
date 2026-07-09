# Transformer视频理解单目VO

## 基本信息
- **英文标题**: Transformer-Based Model for Monocular Visual Odometry: A Video Understanding Approach
- **作者**: André O. Françani, Marcos R. O. A. Maximo
- **发表会议/期刊**: IEEE Access 2025
- **关键词**: Transformer,视频理解,单目VO

## 研究背景（前提）
单目VO传统上依赖几何方法，需针对特定场景大量工程调参。

## 问题提出（由什么问题引出）
视频理解领域的Transformer模型可处理序列信息。能否将其应用于单目VO的视频序列建模？

## 要解决的问题
如何用Transformer模型从视频理解的角度解决单目VO问题？

---

## 采用的方法
将VO视为视频理解任务，使用Transformer架构处理连续图像序列，捕捉帧间时空关系。

## 理论依据
Transformer的自注意力机制可建模长距离帧间依赖关系，从视频序列中学习运动模式。

## 核心公式推导
- **帧间注意力**: $\text{Attn}(Q_t, K_{t-k:t+k}, V_{t-k:t+k})$
  - 当前帧与周围帧的注意力计算



---

## 实验结果
在VO基准上取得了有竞争力的性能，展示了Transformer解决VO的潜力。

## 尚未解决的问题（后续方向）
Transformer推理速度较慢；需要大量训练数据。

---

## 原始摘要
robots and autonomous vehicles. This problem is called monocular visual odometry and often relies on
geometric approaches that require considerable engineering effort for a specific scenario. Deep learning
methods have been shown to be generalizable after proper training and with a large amount of available
data. Transformer-based architectures have dominated the state-of-the-art in natural language processing and
computer vision tasks, such as image and video understanding. In this work, we deal with the monocular
visual odometry as a video understanding task to estimate the 6 degrees of freedom of a camera’s pose.
We contribute by presenting the TSformer-VO model based on spatio-temporal self-attention mechanisms
to extract features from clips and estimate the motions in an end-to-end manner. Our approach achieved
competitive state-of-the-art performance compared with geometry-based and deep learning-based methods
on the KITTI visual odometry dataset, outperforming the DeepVO implementation highly accepted in the
visual odometry community. The code is publicly available at https://github.com/aofrancani/TSformer-VO.
INDEX TERMS Deep learning, monocular visual odometry, transformer, video understanding.
I. INTRODUCTION
Determining the location of a robot in an environment is
a classical task for mobile robots and autonomous vehicles
applications [1]. Visual odometry (VO) consists of estimating
the camera’s pose and motion given a sequence of frames, i.e.
using visual sensors.
