# 论文与综述

## 概述

本模块介绍边缘部署与高性能计算领域的重要论文和综述。

## 内容列表

### 1. 计算机体系结构论文
- CPU架构
- GPU架构
- 存储层次

### 2. 并行计算论文
- CUDA编程
- 并行算法
- 分布式计算

### 3. 模型优化论文
- 模型压缩
- 量化技术
- 知识蒸馏

### 4. 推理框架论文
- TensorRT
- TVM
- ONNX

### 5. 边缘部署论文
- 边缘计算
- 模型部署
- 性能优化

## 重要论文详解

### 1. Computer Architecture: A Quantitative Approach (2017)

**作者**: Hennessy & Patterson

**发表期刊**: Morgan Kaufmann

**核心贡献**:
- 系统介绍计算机体系结构
- 量化分析方法
- 现代CPU/GPU架构

**架构**:
- 指令级并行
- 存储层次设计
- 多核架构

**解决的问题**:
- 性能提升方法
- 能效优化

---

### 2. Programming Massively Parallel Processors (2016)

**作者**: Kirk & Hwu

**发表期刊**: Morgan Kaufmann

**核心贡献**:
- CUDA编程指南
- 并行编程方法
- GPU优化技巧

**架构**:
- 线程层次结构
- 内存优化
- 性能分析

**解决的问题**:
- GPU编程入门
- 并行算法设计

---

### 3. TVM: An Automated End-to-End Optimizing Compiler (2018)

**作者**: Chen et al.

**发表会议**: OSDI

**核心贡献**:
- 深度学习编译器
- 自动优化
- 跨平台部署

**架构**:
- 张量表达式语言
- 自动调优
- 异构后端

**解决的问题**:
- 模型部署复杂性
- 跨平台优化

---

### 4. Quantization and Training of Neural Networks (2017)

**作者**: Jacob et al.

**发表会议**: CVPR

**核心贡献**:
- 量化训练方法
- INT8推理
- 精度保持

**架构**:
- 量化感知训练
- 权重量化
- 激活量化

**解决的问题**:
- 模型压缩
- 边缘部署

---

### 5. Efficient Inference on Edge Devices (2020)

**作者**: Wang et al.

**发表期刊**: IEEE Transactions on Mobile Computing

**核心贡献**:
- 边缘推理优化
- 资源感知调度
- 动态适配

**架构**:
- 轻量级模型设计
- 自适应推理
- 能耗优化

**解决的问题**:
- 边缘设备限制
- 实时推理