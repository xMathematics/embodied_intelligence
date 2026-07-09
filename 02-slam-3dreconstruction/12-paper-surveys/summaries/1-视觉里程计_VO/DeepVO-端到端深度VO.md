# DeepVO-端到端深度VO

## 基本信息
- **英文标题**: DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks
- **作者**: Sen Wang, Ronald Clark, Hongkai Wen, Niki Trigoni
- **发表会议/期刊**: ICRA 2017
- **关键词**: 端到端学习,循环卷积网络,序列建模

## 研究背景（前提）
传统VO需要精心设计流水线(特征提取、匹配、运动估计、优化等)，各模块需仔细调参，且对场景变化敏感。

## 问题提出（由什么问题引出）
现有DL方法主要面向识别/分类，学习外观特征而非几何特征。VO本质上需要几何推理和时间序列建模，纯CNN无法满足。尚无端到端的深度学习VO方法。

## 要解决的问题
如何设计端到端深度学习框架，直接从原始RGB图像序列中估计6-DoF位姿，无需任何传统VO流水线模块和相机标定？

---

## 采用的方法
提出基于深度RCNN的端到端VO。CNN部分(类FlowNet)提取几何特征，RNN部分(两层LSTM)建模序列运动动力学。输入连续帧堆叠，输出6-DoF位姿。

## 理论依据
CNN自动学习与运动估计相关的几何特征；RNN隐式建模图像序列的时间依赖性和运动动力学。

## 核心公式推导
- **序列建模**: $p_t = f_{RCNN}(I_t, I_{t-1}, ..., I_{t-n})$
  - 从n+1帧图像估计t时刻位姿

- **LSTM输入门**: $i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)$
  - 控制当前输入流入

- **LSTM遗忘门**: $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$
  - 控制历史状态保留

- **LSTM输出门**: $o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$
  - 控制状态输出

- **细胞状态更新**: $c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + U_c h_{t-1} + b_c)$
  - 



---

## 实验结果
在KITTI VO数据集上与经典方法(VISO2,ORB-SLAM)竞争力相当。首次证明端到端DL可作为传统VO补充。

## 尚未解决的问题（后续方向）
泛化到新环境仍有挑战；精度低于最先进几何方法；缺乏明确尺度恢复机制。

---

## 原始摘要
(VO) problem. Most of existing VO algorithms are developed
under a standard pipeline including feature extraction, feature
matching, motion estimation, local optimisation, etc. Although
some of them have demonstrated superior performance, they
usually need to be carefully designed and speciﬁcally ﬁne-tuned
to work well in different environments. Some prior knowledge
is also required to recover an absolute scale for monocular
VO. This paper presents a novel end-to-end framework for
monocular VO by using deep Recurrent Convolutional Neural
Networks (RCNNs) 1. Since it is trained and deployed in an
end-to-end manner, it infers poses directly from a sequence
of raw RGB images (videos) without adopting any module in
the conventional VO pipeline. Based on the RCNNs, it not
only automatically learns effective feature representation for
the VO problem through Convolutional Neural Networks, but
also implicitly models sequential dynamics and relations using
deep Recurrent Neural Networks. Extensive experiments on the
KITTI VO dataset show competitive performance to state-of-
the-art methods, verifying that the end-to-end Deep Learning
technique can be a viable complement to the traditional VO
systems.
I. INTRODUCTION
Visual odometry (VO), as one of the most essential
techniques for pose estimation and robot localisation, has
attracted signiﬁcant interest in both the computer vision and
robotics communities over the past few decades [1]. It has
been widely applied to various robots as a
