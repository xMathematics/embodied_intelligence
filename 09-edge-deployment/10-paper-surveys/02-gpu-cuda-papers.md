# 10.2 GPU架构与CUDA优化

## 1. "NVIDIA Tesla: A Unified Graphics and Computing Architecture"

| 项目 | 内容 |
|------|------|
| 作者 | Lindholm et al. |
| 发表 | IEEE Micro 2008 |
| 核心 | 统一着色器架构, SIMT执行模型 |
| 机器人价值 | 理解GPU如何加速机器人算法 |

## 2. "Dissecting the Ampere GPU Architecture"

| 项目 | 内容 |
|------|------|
| 发表 | Hot Chips 2021 |
| 核心 | Tensor Core Gen3, MIG, NVSwitch |
| 机器人价值 | Jetson Orin的GPU基础 |

## 3. "Tile-Level Parallelism for DL Inferencing on GPUs"

| 项目 | 内容 |
|------|------|
| 发表 | HPCA 2021 |
| 核心 | Tile级并行策略, 共享内存缓存 |
| 机器人价值 | Jetson上优化卷积推理 |

## 4. "CUDA Warp Optimization for Real-Time Sensor Processing"

| 项目 | 内容 |
|------|------|
| 发表 | IEEE Access 2022 |
| 核心 | Warp Shuffle, 合并访问, 分支避免 |
| 机器人价值 | 加速LiDAR点云预处理 |
