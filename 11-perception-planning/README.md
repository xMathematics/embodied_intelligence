
# 感知与规划控制模块

## 模块概述

感知与规划控制是具身智能的核心闭环，本模块涵盖感知算法、路径规划、任务规划等内容。

## 学习目标

完成本模块学习后，您将能够：
- 理解感知算法原理
- 掌握路径规划方法
- 了解任务规划技术

## 学习路径

### 第一部分：感知（2周）

#### 1.1 目标检测

**核心概念：**
- 两阶段检测器（Faster R-CNN）
- 一阶段检测器（YOLO、SSD）
- Transformer检测器（DETR）

**推荐论文：**
- "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
- "You Only Look Once: Unified, Real-Time Object Detection" (YOLO)
- "End-to-End Object Detection with Transformers" (DETR)

**实践练习：**
- 使用YOLO进行目标检测
- 理解不同检测器的优缺点

#### 1.2 语义分割

**核心概念：**
- FCN、U-Net
- DeepLab系列
- 全景分割

**推荐论文：**
- "Fully Convolutional Networks for Semantic Segmentation" (FCN)
- "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs"

**实践练习：**
- 使用U-Net进行图像分割
- 理解分割原理

#### 1.3 深度估计

**核心概念：**
- 单目深度估计
- 双目立体视觉
- 深度补全

**推荐论文：**
- "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
- "Monodepth2: Digging Into Self-Supervised Monocular Depth Estimation"

**实践练习：**
- 使用Monodepth2进行深度估计

---

### 第二部分：路径规划（2周）

#### 2.1 全局路径规划

**核心概念：**
- 栅格地图
- A*算法
- 快速搜索随机树（RRT）

**推荐论文：**
- "A Formal Basis for the Heuristic Determination of Minimum Cost Paths" (A*)
- "Rapidly-exploring Random Trees: A New Tool for Path Planning" (RRT)

**实践练习：**
- 实现A*算法
- 实现RRT算法

#### 2.2 局部路径规划

**核心概念：**
- DWA（动态窗口法）
- TEb（时序弹性带）
- 人工势场法

**推荐论文：**
- "The Dynamic Window Approach to Collision Avoidance"

**实践练习：**
- 理解DWA原理

---

### 第三部分：任务规划（2周）

#### 3.1 符号规划

**核心概念：**
- PDDL（规划域定义语言）
- STRIPS规划
- HTN（分层任务网络）

**推荐资源：**
- 《Automated Planning: Theory & Practice》- Ghallab et al.

**实践练习：**
- 理解PDDL语法
- 设计简单的规划问题

#### 3.2 学习型规划

**核心概念：**
- 基于RL的规划
- 模仿学习
- 大模型规划

**推荐论文：**
- "Learning to Act: Deep Learning for Planning"
- "Planning with Large Language Models"

**实践练习：**
- 研究LLM规划方法

---

## 推荐学习资源

### 书籍
1. 《Planning Algorithms》- LaValle
2. 《Automated Planning: Theory & Practice》- Ghallab et al.
3. 《Computer Vision: Algorithms and Applications》- Szeliski

### 在线课程
1. Coursera - AI for Robotics
2. edX - Robotics Planning and Control

### 工具
- OpenCV - 计算机视觉
- MoveIt! - 运动规划
- ROS Navigation - 导航栈

## 学习评估

### 自测题
1. 比较YOLO与Faster R-CNN的优缺点
2. 解释A*算法的启发式函数
3. 说明任务规划与路径规划的区别

### 实践项目
1. 使用YOLO进行实时目标检测
2. 实现A*路径规划
3. 设计简单的PDDL规划问题

---

**下一个模块：** [具身系统综合](../12-embodied-systems/README.md)
