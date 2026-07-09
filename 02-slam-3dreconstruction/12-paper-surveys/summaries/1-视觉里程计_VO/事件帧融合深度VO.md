# 事件帧融合深度VO

## 基本信息
- **英文标题**: Deep Visual Odometry with Events and Frames
- **作者**: Roberto Pellerito, Marco Cannici, Daniel Gehrig, Joris Belhadj, Olivier Dubois-Matra, Massimo Casasco, Davide Scaramuzza
- **发表会议/期刊**: IROS 2024
- **关键词**: 事件相机,多模态融合,深度VO

## 研究背景（前提）
事件相机在低光、高速运动中表现出色，提供异步事件流。标准相机提供密集易跟踪特征。

## 问题提出（由什么问题引出）
现有事件+帧的VO仍主要依赖模型方法，尚未充分利用端到端学习架构的优势。如何无缝融合异步事件和同步帧？

## 要解决的问题
如何将事件相机和标准相机的互补优势结合到端到端深度VO框架中？

---

## 采用的方法
提出深度VO框架同时处理事件流和标准图像帧。设计双模态特征提取和融合网络，处理事件(异步)和帧(同步)的不同时间特性。

## 理论依据
事件相机输出异步事件流，标准相机输出同步帧。通过时序特征提取和对齐实现双模态融合。

## 核心公式推导
- **双模态融合**: $F_{vo} = f(E_{events}, I_{frames})$
  - 融合事件和帧的特征进行VO



---

## 实验结果
在行星地形模拟数据集和标准基准上，融合方法优于单一模态方法。

## 尚未解决的问题（后续方向）
需要事件相机硬件；实时处理事件流计算量大。

---

## 原始摘要
robotic navigation, especially in GPS-denied environments like
planetary terrains. To improve robustness, recent model-based
VO systems have begun combining standard and event-based
cameras. While event cameras excel in low-light and high-speed
motion, standard cameras provide dense and easier-to-track
features. However, the field of image- and event-based VO
still predominantly relies on model-based methods and is yet
to fully integrate recent image-only advancements leveraging
end-to-end learning-based architectures. Seamlessly integrating
the two modalities remains challenging due to their different
nature, one asynchronous, the other not, limiting the potential
for a more effective image- and event-based VO. We introduce
RAMP-VO, the first end-to-end learned image- and event-
based VO system. It leverages novel Recurrent, Asynchronous,
and Massively Parallel (RAMP) encoders capable of fusing
asynchronous events with image data, providing 8× faster
inference and 33% more accurate predictions than existing
solutions. Despite being trained only in simulation, RAMP-VO
outperforms previous methods on the newly introduced Apollo
and Malapert datasets, and on existing benchmarks, where it
improves image- and event-based methods by 58.8% and 30.6%,
paving the way for robust and asynchronous VO in space.
Multimedial Material: For code and datasets visit https:
//github.com/uzh-rpg/rampvo.
I. INTRODUCTION
Visual Odometry (VO) is vital for robotic platforms
but often fails in challe
