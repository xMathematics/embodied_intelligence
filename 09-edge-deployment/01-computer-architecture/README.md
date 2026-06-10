# 第1章：计算机体系结构基础

## 章节概览

本章是边缘部署模块的基石，系统性地介绍计算机体系结构的核心概念，并深入探讨每个知识点在**具身智能机器人**中的实际应用。本章共分为6个文件：

| 文件 | 内容 | 在具身智能中的核心价值 |
|------|------|----------------------|
| [01-von-neumann-architecture.md](01-von-neumann-architecture.md) | 冯·诺依曼架构与瓶颈 | 理解机器人实时控制的延迟根源 |
| [02-instruction-set-architecture.md](02-instruction-set-architecture.md) | 指令集架构 (ISA) | ARM vs x86选型决策 |
| [03-memory-hierarchy.md](03-memory-hierarchy.md) | 存储层次结构 | 解决机器人感知的数据搬运瓶颈 |
| [04-bus-interface-technology.md](04-bus-interface-technology.md) | 总线与接口技术 | 多传感器数据采集带宽规划 |
| [05-multi-core-numa.md](05-multi-core-numa.md) | 多核架构与缓存一致性 | 机器人多任务并行执行 |
| [06-embodied-applications.md](06-embodied-applications.md) | 具身智能综合应用 | 完整系统设计案例 |

## 为什么机器人学家需要学计算机体系结构？

具身智能机器人的三大核心挑战都直接与计算机体系结构相关：

1. **实时性** (Real-time)：控制环路需在1-10kHz频率下运行，任何延迟抖动都可能导致机器人失稳
2. **数据吞吐** (Data Throughput)：多传感器（RGB-D+LiDAR+IMU）每秒产生GB级数据
3. **功耗约束** (Power Constraint)：电池供电限制了可用算力

→ **不理解体系结构，就无法优化机器人系统的端到端性能。**

## 学习路径

```
冯·诺依曼架构 ──→ ISA理解 ──→ 存储层次优化 ──→ 总线规划 ──→ 多核调度
      │              │             │              │             │
      ▼              ▼             ▼              ▼             ▼
  控制延迟        ARM选型       数据搬运       传感器带宽     任务隔离
  的根源         决策           优化            规划          策略
```

## 前置知识
- 基本的数字电路知识（与或非门、触发器）
- 基本的编程概念（变量、函数、内存）
