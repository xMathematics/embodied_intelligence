# 计算机视觉模块

## 概述

本模块介绍计算机视觉的核心技术，包括图像分类、目标检测、语义分割、实例分割和关键点检测。

## 章节结构

### 第一部分：基础概念

1. **[图像分类](01-image-classification.md)**
   - 图像分类概述
   - 传统方法（SIFT、HOG）
   - 深度学习方法（CNN）
   - 数据增强技术
   - 实践练习

2. **[目标检测](02-object-detection.md)**
   - 目标检测概述
   - 两阶段检测器（Faster R-CNN）
   - 一阶段检测器（YOLO、SSD）
   - Transformer检测器（DETR）
   - 实践练习

3. **[语义分割](03-semantic-segmentation.md)**
   - 语义分割概述
   - FCN、U-Net架构
   - DeepLab系列
   - Transformer-based方法
   - 实践练习

4. **[实例分割](04-instance-segmentation.md)**
   - 实例分割概述
   - Mask R-CNN
   - YOLACT、SOLO
   - 实践练习

5. **[关键点检测](05-keypoint-detection.md)**
   - 关键点检测概述
   - HRNet、Simple Baseline
   - CenterNet
   - 实践练习

## 学习目标

完成本模块学习后，你将能够：

- 理解计算机视觉的核心任务和评价指标
- 实现经典的图像分类模型
- 掌握目标检测的主流方法
- 实现语义分割和实例分割模型
- 完成关键点检测任务

## 前置知识

- Python编程基础
- PyTorch框架基础
- 卷积神经网络原理

## 实践项目

每个章节包含3个实践练习，帮助巩固理论知识：

1. 基础模型实现
2. 损失函数实现
3. 评价指标计算

---

**下一模块**：[感知-规划集成](../02-perception-planning-integration)