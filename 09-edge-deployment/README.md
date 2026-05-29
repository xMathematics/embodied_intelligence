
# 边缘部署与CUDA模块

## 模块概述

边缘部署是将AI模型部署到嵌入式设备的关键技术，本模块涵盖CUDA编程、并行计算、模型优化等内容。

## 学习目标

完成本模块学习后，您将能够：
- 理解GPU并行计算原理
- 掌握CUDA编程基础
- 能够优化和部署深度学习模型

## 学习路径

### 第一部分：并行计算基础（1周）

#### 1.1 并行计算概念

**核心概念：**
- SIMD、MIMD架构
- GPU vs CPU架构
- 并行算法设计

**推荐资源：**
- 《Programming Massively Parallel Processors》- Kirk & Hwu
- CUDA官方文档

**实践练习：**
- 理解并行计算原理

#### 1.2 CUDA架构

**核心概念：**
- CUDA核心
- 线程层次结构
- 内存层次

**实践练习：**
- 理解CUDA编程模型

---

### 第二部分：CUDA编程（2周）

#### 2.1 CUDA基础

**核心概念：**
- kernel函数
- 线程与块
- 内存访问模式

**推荐资源：**
- 《CUDA Programming Guide》
- 《Hands-On GPU Programming with Python and CUDA》

**实践练习：**
- 编写简单的CUDA程序
- 实现向量加法

#### 2.2 高级CUDA技巧

**核心概念：**
- 共享内存优化
- 合并内存访问
- warp优化

**实践练习：**
- 优化矩阵乘法
- 使用共享内存

---

### 第三部分：模型优化（1周）

#### 3.1 模型压缩

**核心概念：**
- 量化（Quantization）
- 剪枝（Pruning）
- 知识蒸馏

**推荐论文：**
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
- "Learning Efficient Convolutional Networks through Network Slimming"

**实践练习：**
- 使用PyTorch进行模型量化

#### 3.2 推理框架

**核心概念：**
- TensorRT
- ONNX Runtime
- OpenVINO

**实践练习：**
- 使用TensorRT优化模型
- 部署模型到边缘设备

---

## 推荐学习资源

### 书籍
1. 《Programming Massively Parallel Processors》- Kirk & Hwu
2. 《CUDA Programming: A Developer's Guide to Parallel Computing with GPUs》
3. 《Hands-On GPU Programming with Python and CUDA》

### 在线课程
1. NVIDIA CUDA Training
2. Udacity - Intro to Parallel Programming

### 工具
- CUDA Toolkit
- TensorRT
- ONNX Runtime

## 学习评估

### 自测题
1. 解释CUDA线程层次结构
2. 比较量化与剪枝的区别
3. 说明TensorRT的作用

### 实践项目
1. 使用CUDA实现矩阵乘法
2. 使用TensorRT优化深度学习模型
3. 将模型部署到边缘设备

---

**下一个模块：** [大模型与世界模型](../10-large-models/README.md)
