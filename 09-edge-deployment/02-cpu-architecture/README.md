# 第2章：CPU处理器架构与指令优化

## 章节概览

本章深入讲解CPU微架构的核心技术，并探讨如何将这些知识应用于具身智能机器人的性能优化。共分6个文件：

| 文件 | 内容 | 机器人中的核心价值 |
|------|------|------------------|
| [01-pipeline-microarchitecture.md](01-pipeline-microarchitecture.md) | CPU流水线与微架构 | 理解控制循环的指令执行过程 |
| [02-superscalar-out-of-order.md](02-superscalar-out-of-order.md) | 超标量与乱序执行 | 利用ILP加速机器人多任务 |
| [03-branch-prediction.md](03-branch-prediction.md) | 分支预测 | 优化行为树和碰撞检测 |
| [04-simd-vectorization.md](04-simd-vectorization.md) | SIMD向量化 | 4-16×加速感知预处理 |
| [05-cache-optimization.md](05-cache-optimization.md) | 缓存优化 | 解决数据搬运瓶颈 |
| [06-robotics-optimization.md](06-robotics-optimization.md) | 机器人中的CPU综合优化 | 完整案例实践 |
