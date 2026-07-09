# ViT-VO-混合架构

## 基本信息
- **英文标题**: ViT VO: A Visual Odometry Technique Using CNN-Transformer Hybrid Architecture
- **作者**: Jayaraj P. B., Ebin J., Karthik R., Pournami P. N.
- **发表会议/期刊**: I3CS 2023
- **关键词**: CNN,Transformer,混合架构

## 研究背景（前提）
CNN在特征提取方面有效但感受野有限，Transformer有全局感受野但计算量大。

## 问题提出（由什么问题引出）
能否结合CNN的局部特征提取和Transformer的全局上下文建模优势进行VO？

## 要解决的问题
如何设计CNN-Transformer混合架构同时利用局部和全局视觉信息进行VO？

---

## 采用的方法
提出CNN-Transformer混合VO架构。CNN模块提取局部特征，Transformer模块建模全局空间-时间关系。

## 理论依据
CNN高效提取局部视觉特征，Transformer的自注意力机制建模长距离依赖和全局上下文。

## 核心公式推导
（本文未涉及核心公式推导，请参考原文）

---

## 实验结果
在标准VO基准上验证了混合架构的有效性。

## 尚未解决的问题（后续方向）
Transformer部分增加计算开销；模型复杂度较高。

---

## 原始摘要
autonomous agents (e.g., vehicle, robot etc.). It allows them to be able to track 
their paths and properly detect and avoid obstacles. Visual Odometry (VO) is 
one of the techniques used for agent localization. VO involves estimating the 
motion of an agent using the images taken by cameras attached to it. Conven- 
tional VO algorithms require specific workarounds for challenges posed by the 
working environment and the captured sensor data. On the other hand, Deep 
Learning approaches have shown tremendous eĨĨŝciency and accuracy in tasks 
that require high degree of adaptability and scalability. In this work, a novel 
deep learning model is proposed to perform VO tasks for space robotic applica- 
tions. The model consists of an optical flow estimation module which abstracts 
away scene-specific details from the input video sequence and produces an in- 
termediate representation. The CNN module which follows next learn relative 
poses from the optical flow estimates. The final module is a state-of-the-art 
Vision Transformer, which learn absolute pose from the relative pose learnt by 
the CNN module. The model is trained on the KITTI dataset and has obtained 
a promising accuracy of approximately 2%. It has outperformed the baseline 
model, MagicVO, in a few sequences in the dataset. 
Keywords: Visual Odometry, Deep Learning, Optical Flow, Convolutional 
Neural Networks, Generative Adversarial Networks, Sequence-based Models
