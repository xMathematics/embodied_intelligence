# 5.2 MPI与OpenMP编程

## 1. MPI (消息传递接口)

### 基本概念
- **进程级并行**：每个进程独立内存空间
- **消息传递**：通过 Send/Recv 通信
- **适合**：分布式系统、多机器人

### 机器人协同MPI示例

```cpp
#include <mpi.h>

// 多机器人协同建图
void distributed_slam() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // 机器人ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // 机器人总数
    
    // 每个机器人建局部地图
    LocalMap local = build_local_map(rank);
    
    // 同步所有局部地图
    int map_size = local.data_size();
    std::vector<float> all_maps(map_size * size);
    MPI_Allgather(local.data(), map_size, MPI_FLOAT,
                  all_maps.data(), map_size, MPI_FLOAT,
                  MPI_COMM_WORLD);
    
    // 融合为全局地图
    GlobalMap global = fuse(all_maps, size);
}
```

## 2. OpenMP (共享内存)

### 在机器人中的应用

```cpp
// 并行处理多传感器数据
#pragma omp parallel sections
{
    #pragma omp section
    { process_rgb_image(rgb); }    // 线程1
    
    #pragma omp section
    { process_depth(depth); }      // 线程2
    
    #pragma omp section
    { process_lidar(lidar); }      // 线程3
}
```

## 3. 混合编程模式

```
MPI + OpenMP + CUDA:
├── MPI: 多节点/多机器人通信 (粗粒度)
├── OpenMP: 单节点CPU多核 (中等粒度)
└── CUDA: GPU加速 (细粒度)
```

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "MPI Standard" | MPI Forum 2021 | MPI规范 |
| "OpenMP for Robotics" | ICRA 2022 | OpenMP机器人应用 |
| "Hybrid Programming for Distributed Robots" | IEEE RAM 2022 | 混合编程方法 |
