# 边缘部署与高性能计算模块

## 模块概述

本模块涵盖计算机底层硬件结构、并行计算原理、CPU/GPU架构以及边缘设备上的AI模型部署。从计算机体系结构基础开始，深入讲解并行编程、GPU计算、模型优化和边缘部署技术。

## 学习目标

完成本模块学习后，您将能够：
- 理解计算机底层硬件结构和CPU架构
- 掌握GPU架构和CUDA编程
- 理解并行计算原理和算法设计
- 掌握深度学习模型优化技术
- 能够部署模型到边缘设备
- 进行性能分析和调优

---

## 模块结构

```
09-edge-deployment/
├── 01-computer-architecture/   # 计算机体系结构
├── 02-cpu-architecture/        # CPU架构
├── 03-gpu-architecture/        # GPU架构
├── 04-parallel-computing/      # 并行计算
├── 05-high-performance/        # 高性能计算
├── 06-cuda-programming/        # CUDA编程
├── 07-model-optimization/      # 模型优化
├── 08-inference-frameworks/    # 推理框架
├── 09-edge-deployment/         # 边缘部署
└── 10-paper-surveys/         # 论文与综述
```

---

## 学习路径

### 第一部分：计算机体系结构（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 1.1 | 冯·诺依曼架构 | ⭐⭐⭐ |
| 1.2 | 指令集架构 | ⭐⭐⭐⭐ |
| 1.3 | 存储层次 | ⭐⭐⭐⭐ |
| 1.4 | 总线与接口 | ⭐⭐⭐ |
| 1.5 | 多核与众核 | ⭐⭐⭐⭐ |

**核心内容**：
- 冯·诺依曼瓶颈
- CISC vs RISC
- 缓存层次（L1/L2/L3）
- NUMA架构
- 内存一致性模型

---

### 第二部分：CPU架构（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 2.1 | CPU微架构 | ⭐⭐⭐⭐ |
| 2.2 | 指令级并行 | ⭐⭐⭐⭐⭐ |
| 2.3 | 超标量执行 | ⭐⭐⭐⭐⭐ |
| 2.4 | 分支预测 | ⭐⭐⭐⭐ |
| 2.5 | SIMD指令 | ⭐⭐⭐⭐ |

**核心内容**：
- 流水线设计
- 乱序执行
- 推测执行
- AVX/SSE指令集
- 多线程与超线程

---

### 第三部分：GPU架构（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 3.1 | GPU vs CPU | ⭐⭐⭐ |
| 3.2 | NVIDIA GPU架构 | ⭐⭐⭐⭐ |
| 3.3 | AMD GPU架构 | ⭐⭐⭐⭐ |
| 3.4 | 移动GPU | ⭐⭐⭐ |
| 3.5 | 专用AI芯片 | ⭐⭐⭐⭐ |

**核心内容**：
- 流式多处理器（SM）
- CUDA核心与Tensor核心
- ROCm架构
- Mobile GPU（Adreno、Mali）
- TPU/NPU/ASIC

---

### 第四部分：并行计算（2周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 4.1 | 并行计算基础 | ⭐⭐⭐ |
| 4.2 | 并行编程模型 | ⭐⭐⭐⭐ |
| 4.3 | 并行算法设计 | ⭐⭐⭐⭐ |
| 4.4 | 同步与通信 | ⭐⭐⭐⭐ |
| 4.5 | 并行模式 | ⭐⭐⭐⭐ |

**核心内容**：
- SIMD、MIMD、SPMD
- OpenMP、MPI
- 数据并行与任务并行
- 锁与同步原语
- MapReduce、数据流

---

### 第五部分：高性能计算（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 5.1 | HPC集群 | ⭐⭐⭐⭐ |
| 5.2 | 分布式计算 | ⭐⭐⭐⭐ |
| 5.3 | 云计算平台 | ⭐⭐⭐ |
| 5.4 | 超级计算机 | ⭐⭐⭐⭐ |
| 5.5 | 性能度量 | ⭐⭐⭐⭐ |

**核心内容**：
- 集群架构
- InfiniBand网络
- AWS/GCP/Azure GPU实例
- TOP500榜单
- FLOPS、能效比

---

### 第六部分：CUDA编程（2周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 6.1 | CUDA编程模型 | ⭐⭐⭐⭐ |
| 6.2 | Kernel编程 | ⭐⭐⭐⭐ |
| 6.3 | 内存层次优化 | ⭐⭐⭐⭐⭐ |
| 6.4 | Warp优化 | ⭐⭐⭐⭐⭐ |
| 6.5 | 高级CUDA技术 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 线程层次（Grid/Block/Thread）
- 共享内存、常量内存
- 合并内存访问
- Warp shuffle
- CUDA Streams、Async

---

### 第七部分：模型优化（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 7.1 | 模型压缩 | ⭐⭐⭐⭐ |
| 7.2 | 量化技术 | ⭐⭐⭐⭐ |
| 7.3 | 剪枝与稀疏化 | ⭐⭐⭐⭐ |
| 7.4 | 知识蒸馏 | ⭐⭐⭐⭐ |
| 7.5 | 神经架构搜索 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 权重共享、低秩分解
- INT8/FP16量化
- 结构化/非结构化剪枝
- Teacher-Student训练
- NAS搜索空间

---

### 第八部分：推理框架（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 8.1 | TensorRT | ⭐⭐⭐⭐ |
| 8.2 | ONNX Runtime | ⭐⭐⭐⭐ |
| 8.3 | OpenVINO | ⭐⭐⭐⭐ |
| 8.4 | TVM | ⭐⭐⭐⭐⭐ |
| 8.5 | MNN/TNN | ⭐⭐⭐⭐ |

**核心内容**：
- 图优化、算子融合
- 量化感知训练
- 异构推理
- AutoTVM优化
- 移动端框架

---

### 第九部分：边缘部署（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 9.1 | 边缘设备 | ⭐⭐⭐ |
| 9.2 | 部署流程 | ⭐⭐⭐⭐ |
| 9.3 | 性能优化 | ⭐⭐⭐⭐ |
| 9.4 | 低功耗设计 | ⭐⭐⭐⭐ |
| 9.5 | 实际案例 | ⭐⭐⭐⭐ |

**核心内容**：
- Jetson、RK3588、NPU
- 模型转换与优化
- 内存优化、延迟优化
- 电源管理
- 机器人/自动驾驶部署

---

## 推荐学习资源

### 书籍
1. 《Computer Architecture: A Quantitative Approach》- Hennessy & Patterson
2. 《Programming Massively Parallel Processors》- Kirk & Hwu
3. 《CUDA Programming Guide》- NVIDIA
4. 《High Performance Computing》- Gropp et al.
5. 《Deep Learning Optimization for Edge Deployment》- Wang et al.

### 在线课程
1. **MIT 6.004 - Computation Structures**
2. **UC Berkeley CS267 - Parallel Computing**
3. **NVIDIA CUDA Training**
4. **HPC University Courses**
5. **Edge AI Certification**

### 工具与框架
1. **CUDA Toolkit** - GPU编程
2. **TensorRT** - 推理优化
3. **ONNX Runtime** - 跨平台推理
4. **TVM** - 深度学习编译器
5. **NVIDIA Jetson** - 边缘平台

---

## 学习评估

### 自测题
1. 解释冯·诺依曼架构的瓶颈
2. CPU和GPU的架构差异是什么？
3. CUDA线程层次结构是怎样的？
4. 常用的模型压缩技术有哪些？
5. TensorRT如何优化深度学习模型？

### 实践项目
1. 使用CUDA实现矩阵乘法
2. 使用OpenMP实现并行排序
3. 使用TensorRT优化并部署模型
4. 在Jetson上部署YOLO模型
5. 对比不同量化方法的效果

---

## 模块关系

```
具身智能
├── 04-deep-learning (深度学习) ← 前置模块
├── 08-robot-hardware (机器人硬件) ← 相关模块
├── 09-edge-deployment (边缘部署) ← 当前模块
└── 12-embodied-systems (具身系统) ← 综合应用
```

**学习顺序建议**：
1. 先学习深度学习（04-deep-learning）
2. 学习机器人硬件（08-robot-hardware）
3. 学习本模块（09-edge-deployment）
4. 最后学习具身系统（12-embodied-systems）

---

**上一个模块**：[机器人硬件](../08-robot-hardware/README.md)
**下一个模块**：[大模型与世界模型](../10-large-models/README.md)