# 模态不变具身VO

## 基本信息
- **英文标题**: Modality-invariant Visual Odometry for Embodied Vision
- **作者**: Marius Memmel, Roman Bachmann, Amir Zamir
- **发表会议/期刊**: CVPR 2023
- **关键词**: 模态不变,跨模态,域对抗,具身视觉

## 研究背景（前提）
具身视觉系统中，不同平台(机器人、无人机等)的传感器配置不同，导致VO模型在不同平台间难以迁移。

## 问题提出（由什么问题引出）
能否学习一个模态不变的VO表示，使模型在不同传感器配置(不同相机、分辨率等)间零样本迁移？

## 要解决的问题
如何学习对传感器模态变化(相机参数、分辨率等)不变的VO特征表示？

---

## 采用的方法
提出跨模态VO框架，通过跨模态对比学习和域对抗训练，学习对传感器模态变化不变的VO特征。

## 理论依据
跨模态对比学习将不同模态但相同位姿的图像映射到相似的潜空间表示。域对抗训练强制特征分布对齐。

## 核心公式推导
- **跨模态对比**: $L = -\log \frac{e^{\text{sim}(f_A, f_B)/\tau}}{\sum e^{\text{sim}(f_A, f_k)/\tau}}$
  - 拉近同一位姿的不同模态特征

- **域对抗**: $\min_f \max_d L_{domain}(d(f(I)))$
  - 对抗训练使特征分布跨域对齐



---

## 实验结果
在不同传感器配置间展示了零样本迁移能力，有效实现了模态不变的VO。

## 尚未解决的问题（后续方向）
极端不同的模态间对齐仍有挑战；对比学习需大量负样本。

---

## 原始摘要
Effectively localizing an agent in a realistic, noisy setting
is crucial for many embodied vision tasks. Visual Odome-
try (VO) is a practical substitute for unreliable GPS and
compass sensors, especially in indoor environments. While
SLAM-based methods show a solid performance without
large data requirements, they are less flexible and robust
w.r.t. to noise and changes in the sensor suite compared
to learning-based approaches.
Recent deep VO models,
however, limit themselves to a fixed set of input modalities,
e.g., RGB and depth, while training on millions of sam-
ples. When sensors fail, sensor suites change, or modali-
ties are intentionally looped out due to available resources,
e.g., power consumption, the models fail catastrophically.
Furthermore, training these models from scratch is even
more expensive without simulator access or suitable exist-
ing models that can be fine-tuned. While such scenarios
get mostly ignored in simulation, they commonly hinder a
model’s reusability in real-world applications. We propose
a Transformer-based modality-invariant VO approach that
can deal with diverse or changing sensor suites of naviga-
tion agents. Our model outperforms previous methods while
training on only a fraction of the data. We hope this method
opens the door to a broader range of real-world applica-
tions that can benefit from flexible and learned VO models.
