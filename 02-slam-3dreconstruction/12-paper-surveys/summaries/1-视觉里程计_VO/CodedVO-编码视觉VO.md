# CodedVO-编码视觉VO

## 基本信息
- **英文标题**: CodedVO: Coded Visual Odometry
- **作者**: Sachin Shah, Naitri Rajyaguru, Chahat Deep Singh, Christopher Metzler, Yiannis Aloimonos
- **发表会议/期刊**: IEEE RA-L 2024
- **关键词**: 编码光学,尺度恢复,计算成像

## 研究背景（前提）
单目VO面临尺度模糊问题——无法从单张2D图像确定物体的绝对尺度。

## 问题提出（由什么问题引出）
传统方法通过多视图几何或额外传感器解决尺度模糊。能否通过定制光学元件将度量深度信息物理编码到图像中？

## 要解决的问题
如何利用编码光学(定制掩码/衍射元件)将深度信息编码到2D图像中，实现单目VO的尺度恢复？

---

## 采用的方法
在相机镜头前放置编码掩码(衍射光学元件)，将深度信息物理编码到捕获的PSF(点扩散函数)中。VO流水线同时处理编码图像并解码深度信息。

## 理论依据
编码光学元件使PSF随深度变化，从而将深度信息编码到图像中。通过解码PSF可恢复每像素的度量深度，解决尺度模糊。

## 核心公式推导
- **编码成像**: $I_{coded} = I_{sharp} * PSF(z) + n$
  - 深度z调制点扩散函数

- **深度解码**: $\hat{z} = f_{dec}(I_{coded})$
  - 从编码图像解码深度



---

## 实验结果
实现了单目VO的SOTA性能，成功克服尺度模糊问题。

## 尚未解决的问题（后续方向）
需要定制光学硬件；编码光学降低图像质量；PSF对场景依赖。

---

## 原始摘要
robots
often
rely
on
monocular
cameras for odometry estimation and navigation. However, the
scale ambiguity problem presents a critical barrier to effective
monocular visual odometry. In this paper, we present CodedVO,
a novel monocular visual odometry method that overcomes
the scale ambiguity problem by employing custom optics to
physically encode metric depth information into imagery. By
incorporating this information into our odometry pipeline,
we achieve state-of-the-art performance in monocular visual
odometry with a known scale. We evaluate our method in
diverse indoor environments and demonstrate its robustness and
adaptability. We achieve a 0.08m average trajectory error in
odometry evaluation on the ICL-NUIM indoor odometry dataset.
I. INTRODUCTION
Over 3.8 billion years of genetic evolution, nature has
witnessed remarkable transformations, with a significant focus
on the development of visual systems. This journey has taken
us from the earliest photoreceptors to the intricate and diverse
array of eye structures observed in species like frogs [1] and
cuttlefish [2]. This evolutionary trajectory has been purposive
and parsimonious [3], predominantly influenced by sensory
behaviors
and
environmental
interactions.
Evolutionary
processes have resulted in these animals and their sensory
organs becoming more and more specialized. By contrast,
today in the field of robotics a general-purpose philosophy
to sensor design is the norm. For instance, a similar set of
cameras is 
