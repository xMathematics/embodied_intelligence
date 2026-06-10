# 4.2 并行编程模型

## 1. 模型分类

| 模型 | 说明 | 代表 | 适合机器人算法 |
|------|------|------|--------------|
| **SIMD** | 单指令多数据 | SSE/AVX/NEON | 图像像素处理 |
| **SIMT** | SIMD的GPU变体 | CUDA | 神经网络推理 |
| **SPMD** | 单程序多数据 | MPI/CUDA | 点云并行处理 |
| **MPI** | 消息传递 | MPI库 | 多机器人协同 |
| **OpenMP** | 共享内存多线程 | #pragma omp | CPU多核加速 |
| **任务并行** | 不同任务并行 | TBB/Pthread | 多传感器 |

## 2. 在机器人中的选择

```
你的并行粒度是什么？
├── 数据级并行 (相同操作, 大量数据)
│   ├── CPU上 → SIMD向量化 (循环)
│   ├── GPU上 → CUDA Kernel
│   └── 多核CPU → OpenMP
│
├── 任务级并行 (不同操作, 流水线)
│   └── → Pthread / TBB / ROS2多节点
│
└── 分布式并行 (多机器/多进程)
    └── → MPI / gRPC
```

## 3. OpenMP示例 (CPU多核加速点云)

```cpp
#include <omp.h>

void process_pointcloud(Point* cloud, int n) {
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < n; i++) {
        // 每个点独立处理
        cloud[i].x = transform(cloud[i]);
    }
}
```

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Parallel Computing Models Survey" | ACM 1998 | 并行模型综述 |
| "OpenMP for Embedded Robotics" | ICRA 2021 | OpenMP在机器人中 |
