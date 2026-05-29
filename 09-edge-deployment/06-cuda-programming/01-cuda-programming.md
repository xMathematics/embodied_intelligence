# CUDA编程

## 概述

本模块介绍CUDA编程模型和优化技术。

## 内容列表

### 6.1 CUDA编程模型
- CUDA执行模型
- 线程层次（Grid/Block/Thread）
- Kernel函数
- 设备内存管理

### 6.2 Kernel编程
- 线程索引计算
- 数据并行处理
- 线程束（Warp）
- 分支发散

### 6.3 内存层次优化
- 全局内存
- 共享内存
- 常量内存
- 纹理内存

### 6.4 Warp优化
- Warp shuffle
- Warp同步
- 内存合并
- Bank冲突避免

### 6.5 高级CUDA技术
- CUDA Streams
- 异步操作
- 多GPU编程
- CUDA Graph