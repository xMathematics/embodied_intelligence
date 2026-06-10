# 6.1 CUDA编程模型与Kernel

## 1. 线程层次

```
Grid (1D/2D/3D)
└── Block (1D/2D/3D)  ← 运行在同一个SM上, 可通过共享内存通信
    └── Thread (最小编程单元)
        └── Warp (32线程) ← 硬件执行单元
```

## 2. Kernel函数

```cpp
// 向量加法Kernel
__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    // 线程索引计算
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // 每个线程处理一个元素
    }
}

// 启动配置
int N = 1 << 20;  // 1M元素
const int THREADS = 256;
const int BLOCKS = (N + THREADS - 1) / THREADS;  // 4096 blocks

vec_add<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, N);
```

## 3. 在机器人中的应用

| 操作 | 线程映射 | Kernel逻辑 | 加速比 |
|------|---------|-----------|-------|
| 图像归一化 | 1线程/像素 | 逐像素归一化 | 100× |
| 点云变换 | 1线程/点 | 矩阵×向量 | 200× |
| 体素滤波 | 1线程/体素 | 均值计算 | 50× |

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "CUDA Programming Guide" | NVIDIA 2023 | 官方指南 |
| "Programming Massively Parallel Processors" | Kirk & Hwu 2022 | CUDA教材 |
