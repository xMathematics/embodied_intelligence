# 8.1 深度学习目标检测

## 1. 为什么需要目标检测

**问题**：图像分类只回答"有什么"，目标检测需要回答"有什么、在哪里"。

**核心**：在图像中定位物体并分类，是机器人感知的基础。

## 2. 两阶段检测器

**R-CNN家族演进**：R-CNN(2014) → Fast R-CNN(2015) → Faster R-CNN(2015) → Mask R-CNN(2017)

**为什么分两阶段**：第一阶段生成候选区域（RPN），第二阶段精细分类和回归。

**Faster R-CNN架构**：
1. 骨干网络（ResNet）提取特征
2. RPN生成候选区域
3. RoI Align提取区域特征
4. 分类和回归头

## 3. 单阶段检测器

**YOLO系列**：从YOLOv1(2015)到YOLOv10(2024)，逐步改进。

**YOLO核心思想**：将检测视为回归问题，一次前向直接输出边界框。

**RetinaNet**：Focal Loss解决正负样本不均衡。

## 4. DETR——端到端检测器

**论文**：Carion et al., 2020 — ECCV

**为什么重要**：去除手工设计组件（锚框、NMS），用Transformer+集合预测。

**Deformable DETR**：可变形注意力加速收敛。

## 5. 在具身智能中的应用

- **抓取物体检测**：检测可抓取的物体
- **机器人导航**：检测门、走廊、障碍物
- **人机交互**：检测人手和物体
- **三维感知**：2D检测映射到3D空间

## 6. 参考文献

1. Ren, S., et al. (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. *NeurIPS*.
2. Redmon, J., et al. (2016). You only look once: Unified, real-time object detection. *CVPR*.
3. Carion, N., et al. (2020). End-to-end object detection with transformers. *ECCV*.
