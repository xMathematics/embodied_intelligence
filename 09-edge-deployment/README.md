# 边缘部署与高性能计算模块

## 模块概述

本模块系统性地覆盖从计算机底层硬件架构、并行计算原理、GPU编程到深度学习模型优化、推理加速以及边缘设备部署的完整知识链。**核心目标**是让读者深入理解如何将具身智能机器人中的深度学习模型（感知、控制、规划等）高效部署到资源受限的边缘计算平台上，实现实时、低功耗、高可靠的机器人智能系统。

### 为什么需要边缘部署？

| 挑战 | 说明 | 具身智能中的具体场景 |
|------|------|---------------------|
| **实时性要求** | 云端推理延迟不可控（>100ms） | 机器人避障需<10ms响应 |
| **带宽限制** | 高清视频/点云数据量巨大 | RealSense相机每秒产生~400MB数据 |
| **隐私安全** | 数据不宜上传云端 | 家庭服务机器人的视觉数据 |
| **离线运行** | 网络不可靠或无网络 | 野外勘探、深海/太空机器人 |
| **功耗约束** | 电池供电，算力有限 | 无人机续航仅30分钟 |

---

## 模块结构（重构版）

```
09-edge-deployment/
├── 01-computer-architecture/      # 计算机体系结构基础
├── 02-cpu-architecture/           # CPU处理器架构与指令优化
├── 03-gpu-architecture/           # GPU架构与异构计算
├── 04-parallel-computing/         # 并行计算原理与模型
├── 05-high-performance/           # 高性能计算与云端基础设施
├── 06-cuda-programming/           # CUDA与GPU编程实战
├── 07-model-optimization/         # 模型压缩、量化与轻量化
├── 08-inference-frameworks/       # 推理引擎与编译优化
├── 09-edge-deployment/            # 边缘部署与具身智能系统
└── 10-paper-surveys/              # 前沿论文与综述精读
```

---

## 学习路径与具身智能映射

### 第一阶段：硬件基础（第1-3章）

```
计算机体系结构 ──→ CPU架构 ──→ GPU架构
     │                 │            │
     ▼                 ▼            ▼
  冯·诺依曼瓶颈     流水线冲突    SIMT执行模型
  内存层次优化       SIMD向量化     Tensor Core
  NUMA/NUMA感知     分支预测       GPU内存层次
```

**具身智能映射**：
- 理解**内存层次** → 优化机器人感知管线中的数据搬运
- 掌握**SIMD/向量化** → 加速机器人运动学正逆解计算
- 认识**GPU架构** → 为视觉大模型推理选择最优硬件

### 第二阶段：并行计算（第4-6章）

```
并行计算原理 ──→ 高性能计算 ──→ CUDA编程实战
     │                 │            │
     ▼                 ▼            ▼
  Amdahl定律         MPI通信       Kernel优化
  数据并行策略       GPU集群       共享内存tile
  任务调度           Slurm调度      Warp优化
```

**具身智能映射**：
- **并行归约** → 加速点云配准（ICP算法）
- **CUDA Stream** → 多传感器流水线并行处理
- **MPI/分布式** → 机器人集群协同计算

### 第三阶段：模型优化（第7-8章）

```
模型压缩优化 ──→ 推理框架与编译器
     │                 │
     ▼                 ▼
  量化(INT8/FP16)    TensorRT
  剪枝(结构化)       ONNX Runtime
  知识蒸馏           TVM / OpenVINO
  NAS轻量化搜索      MNN/TNN移动端
```

**具身智能映射**：
- **INT8量化** → 将YOLO检测模型从30ms降至5ms
- **TensorRT** → 优化BEV感知模型在Jetson上的推理
- **知识蒸馏** → 将VLA大模型蒸馏为轻量级策略网络

### 第四阶段：系统部署（第9-10章）

```
边缘部署系统 ──→ 前沿论文精读
     │                 │
     ▼                 ▼
  Jetson/FPGA/NPU    量化算法论文
  实时推理管线       模型编译器论文
  功耗优化           边缘SLAM论文
  多模态融合部署     具身VLA部署论文
```

---

## 各章节知识点速览

| 章节 | 核心知识点 | 在具身智能中的作用 | 解决的关键问题 |
|------|-----------|-------------------|---------------|
| 01 | 冯·诺依曼架构、存储层次、总线、多核架构 | 理解机器人计算平台的硬件瓶颈 | 计算与访存带宽不匹配 |
| 02 | 流水线、超标量、SIMD、分支预测 | 优化底层控制循环和感知前处理 | 控制频率受限、数据搬运慢 |
| 03 | SM架构、Tensor Core、显存层次、移动GPU | GPU加速视觉/点云推理 | 大规模并行计算需求 |
| 04 | Amdahl、并行模式、同步原语 | 设计多传感器融合并行管线 | 串行瓶颈导致延迟累积 |
| 05 | MPI/OpenMP、GPU集群、容器化 | 分布式训练、仿真集群 | 单机算力不足以训练大模型 |
| 06 | CUDA Kernel、共享内存、Stream | 自定义算子加速机器人感知 | 标准库算子性能不足 |
| 07 | 量化、剪枝、蒸馏、NAS | 模型轻量化适配边缘设备 | 边缘设备显存/算力受限 |
| 08 | TensorRT、ONNX、TVM、OpenVINO | 跨平台模型部署与加速 | 模型-硬件适配效率低 |
| 09 | Jetson、FPGA、实时管线、功耗优化 | 完整机器人系统部署 | 系统集成与工程落地 |
| 10 | 各方向经典论文精读 | 把握前沿技术动态 | 理论与实践的桥梁 |

---

## 推荐学习资源

### 经典书籍
1. **Computer Architecture: A Quantitative Approach** (Hennessy & Patterson, 6th Ed.)
2. **Programming Massively Parallel Processors** (Kirk & Hwu, 4th Ed.)
3. **CUDA C++ Programming Guide** - NVIDIA Official
4. **Deep Learning with PyTorch: Model Optimization and Deployment**
5. **Efficient Processing of Deep Neural Networks** (Vivienne Sze et al.)

### 在线课程
1. **MIT 6.004 - Computation Structures**
2. **UIUC ECE 408 - GPU Programming**
3. **NVIDIA DLI - CUDA/TensorRT/Jetson 认证课程**
4. **Stanford CS231n - Efficient AI Inference**
5. **TinyML and Efficient Deep Learning** (Harvard)

### 工具与框架
| 工具 | 用途 | 在具身智能中的典型应用 |
|------|------|----------------------|
| CUDA Toolkit | GPU通用计算 | 加速点云处理、图像预处理 |
| TensorRT | 推理优化 | Jetson上部署YOLO/BEV感知 |
| ONNX Runtime | 跨平台推理 | 模型在不同机器人平台间迁移 |
| TVM | 深度学习编译器 | 自动调优适配新硬件 |
| NVIDIA Jetson | 边缘AI平台 | 机器人主控/协处理器 |
| Intel OpenVINO | Intel硬件优化 | 在Intel NUC上部署感知模型 |
| TensorFlow Lite | 移动端推理 | 在树莓派上运行轻量模型 |
| ROS2 + micro-ROS | 机器人中间件 | 边缘节点与主控通信 |

---

## 模块关系与前置知识

```
具身智能知识体系
├── 04-deep-learning           ◄── 前置：深度学习基础（CNN/Transformer/VLM）
├── 05-reinforcement-learning  ◄── 前置：强化学习（策略网络训练）
├── 06-robotics                ◄── 前置：机器人学基础（运动学/动力学）
├── 08-robot-hardware          ◄── 相关：传感器/执行器硬件
├── 09-edge-deployment         ◄── 当前模块
├── 10-large-models            ◄── 后续：大模型基础架构
├── 11-perception-planning     ◄── 后续：感知与规划算法
└── 12-embodied-systems        ◄── 综合：完整具身智能系统
```

**建议学习顺序**：
1. 先完成 **04-deep-learning** 深度学习基础
2. 再完成 **06-robotics** 机器人学基础
3. 学习本模块 **09-edge-deployment**
4. 最后进入 **12-embodied-systems** 综合实践

---

## 学习目标与评估

### 学习目标
- ✅ 理解计算机底层硬件结构对AI推理性能的影响
- ✅ 掌握CPU/GPU架构差异及其在机器人中的应用场景
- ✅ 能够设计和实现并行算法加速机器人感知管线
- ✅ 掌握模型量化、剪枝、蒸馏等轻量化技术
- ✅ 熟练使用TensorRT/ONNX/TVM进行推理加速
- ✅ 能够在Jetson等边缘设备上完整部署机器人AI模型
- ✅ 了解前沿的边缘部署与具身智能论文

### 自测题
1. 冯·诺依曼瓶颈是什么？在机器人实时控制中如何缓解？
2. GPU的SIMT执行模型与CPU的SIMD有何本质区别？
3. 在Jetson Orin上部署一个7B参数的大模型需要哪些优化步骤？
4. INT8量化对感知模型精度有多大影响？如何补偿精度损失？
5. 对比TensorRT的算子融合与TVM的图优化，各有什么优劣？
6. 设计一个多传感器（RGB+LiDAR+IMU）的实时推理管线，如何利用CUDA Stream并行？

### 实践项目
| 项目 | 难度 | 描述 |
|------|------|------|
| CUDA矩阵乘法优化 | ⭐⭐⭐ | 使用共享内存和tile优化 |
| YOLO TensorRT部署 | ⭐⭐⭐⭐ | 在Jetson上部署并调优 |
| 模型量化对比实验 | ⭐⭐⭐ | 比较INT8/FP16/BF16效果 |
| VLA模型蒸馏 | ⭐⭐⭐⭐⭐ | 将VLA蒸馏为轻量策略 |
| 多传感器融合管线 | ⭐⭐⭐⭐ | 使用CUDA Stream并行处理 |

---

> **上一个模块**：[机器人硬件](../08-robot-hardware/README.md)
> **下一个模块**：[大模型与世界模型](../10-large-models/README.md)