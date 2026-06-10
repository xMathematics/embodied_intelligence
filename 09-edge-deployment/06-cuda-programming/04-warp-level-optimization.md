# 6.4 Warp级优化

## 1. 分支发散

```cpp
// ❌ Warp内分支发散
if (threadIdx.x < 16) {
    // 前半Warp执行, 后半Warp等待 (浪费50%算力)
} else {
    // 后半Warp执行, 前半Warp等待
}

// ✅ 以Block粒度分支
if (blockIdx.x % 2 == 0) {
    // 整个Block的所有Warp走同一路径
}
```

## 2. 合并内存访问

```cpp
// ❌ 非合并 (stride)
for (int i = 0; i < 32; i++)
    val = data[threadIdx.x + i * blockDim.x];  // 32次事务

// ✅ 合并 (连续)
val = data[threadIdx.x];  // 1次宽事务
```

## 3. Warp Shuffle

```cpp
// Warp内归约 (无需共享内存)
__global__ void warp_reduce(float* data) {
    int tid = threadIdx.x;
    float val = data[tid];
    
    // Warp Shuffle归约
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    
    if (tid == 0) data[blockIdx.x] = val;
}
```

## 4. 在机器人中的应用

- **点云下采样**: Warp shuffle计算体素质心
- **LayerNorm**: Warp级归约加速归一化
- **Softmax**: Warp级归约+广播

## 5. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Warp-Level Parallelism for Robotics" | IROS 2021 | Warp级优化 |
| "CUDA Warp Optimization for Sensor Processing" | IEEE Access 2022 | 传感器处理 |
