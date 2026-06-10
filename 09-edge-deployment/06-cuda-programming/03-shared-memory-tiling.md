# 6.3 共享内存与Tiling

## 1. Tiling矩阵乘法

```cpp
#define TILE 16

__global__ void matmul_tiled(float* A, float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    
    for (int t = 0; t < N / TILE; t++) {
        // 协作加载tile到共享内存
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();
        
        // 在共享内存中计算
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}
```

## 2. 在机器人中的应用

| 应用 | 共享内存用途 | 加速比 |
|------|------------|-------|
| 图像滤波(3×3卷积) | 存储输入tile | 10× |
| 点云ICP | 局部点云块 | 5× |
| 体素哈希 | 体素表缓存 | 8× |

## 3. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Shared Memory Optimization for ICP" | ICRA 2022 | GPU加速ICP |
| "Tile-Level Parallelism for DL Inference" | HPCA 2021 | Tile级并行 |
