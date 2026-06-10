# 3.3 GPU内存层次

## 1. GPU内存层次总览

```
GPU内存层次 (Ampere Architecture)
                   
                 寄存器 (Register)
                 ~64K × 32-bit / SM
                 ~1 cycle 延迟
                        │
              ┌─────────┴─────────┐
              │  L1 / Shared Memory │
              │  128-192KB / SM    │
              │  ~30 cycle 延迟    │
              └─────────┬─────────┘
                        │
              ┌─────────┴─────────┐
              │    L2 Cache        │
              │    6-40 MB         │
              │    ~200 cycle      │
              └─────────┬─────────┘
                        │
              ┌─────────┴─────────┐
              │  Global Memory     │
              │  (GDDR6/HBM)      │
              │  8-80 GB          │
              │  ~400-800 cycle   │
              └───────────────────┘
```

### 详细参数对比

| 内存类型 | 位置 | 典型大小 | 延迟 | 带宽 | 作用域 | 缓存 |
|---------|------|---------|------|------|-------|------|
| **寄存器** | 每个SM片上 | 64K×32-bit | ~1 cycle | ~100 TB/s | 单线程 | — |
| **共享内存** | 每个SM片上 | 48-164 KB | ~30 cycle | ~10-20 TB/s | 同Block | 手动管理 |
| **L1 Cache** | 每个SM片上 | 同共享内存 | ~30 cycle | — | 同Block | 自动 |
| **L2 Cache** | 芯片上 | 6-40 MB | ~200 cycle | ~3-5 TB/s | 所有SM | 自动 |
| **全局内存** | 板载GDDR | 8-80 GB | ~400-800 cycle | ~1-2 TB/s | 所有线程 | L1+L2 |
| **常量内存** | 芯片上 | 64 KB | ~200 cycle | 广播 | 所有线程 | 专用 |
| **纹理内存** | 板载+缓存 | 同全局 | ~200-400 | 空间缓存 | 所有线程 | 纹理Cache |

---

## 2. 共享内存详解

### 2.1 作用

共享内存是GPU上**最重要的性能优化资源** — 比全局内存快10-20倍。

### 2.2 Bank Conflict

```
共享内存分为32个Bank (每个Bank 32-bit宽):

Bank:  0   1   2   3  ...  31
       │   │   │   │       │
线程0: [0]                   → 访问Bank 0 ✓
线程1:     [1]               → 访问Bank 1 ✓
线程2:     [1]               → 访问Bank 1 ❌ (Conflict! 串行化)
线程3:         [3]           → 访问Bank 3 ✓

规则:
· 不同线程访问不同Bank → 并行 (32路)
· 同线程访问同Bank不同地址 → 串行 (N路冲突)
· 所有线程访问同Bank同地址 → 广播 (1路)
```

### 2.3 避免Bank Conflict

```cpp
// ❌ 有Bank Conflict
__shared__ float data[32][32];
// 按列访问: data[threadIdx.x][threadIdx.y]
// thread 0: 访问 data[0][0] (Bank 0)
// thread 1: 访问 data[1][0] (Bank 0) ← Conflict!

// ✅ 使用Padding避免
__shared__ float data[32][33];  // +1 padding
// data[threadIdx.x][threadIdx.y]
// 现在每个元素在不同Bank
```

---

## 3. 合并内存访问 (Coalesced Access)

GPU访问全局内存时，**相邻线程访问相邻地址**效率最高。

```cpp
// ❌ 非合并访问 (stride)
int idx = threadIdx.x;
float val = data[idx * 32];  // thread 0: offset 0
                              // thread 1: offset 32
                              // 需要32次独立事务!

// ✅ 合并访问 (连续)
float val = data[idx];  // thread 0: offset 0
                         // thread 1: offset 1
                         // 合并为1次宽事务!
```

### 在机器人中的影响

```cpp
// 点云处理中的合并访问
// ❌ AoS模式 (非合并)
struct Point { float x, y, z; };
__global__ void scale_points_bad(Point* pts, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pts[i].x *= s;  // 读取32字节, 只用4字节
        pts[i].y *= s;  // 浪费带宽!
        pts[i].z *= s;
    }
}

// ✅ SoA模式 (合并)
__global__ void scale_points_good(float* x, float* y, float* z, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= s;  // 连续读取, 完全合并
    }
}
// 带宽利用率: 25% (AoS) → 100% (SoA)
```

---

## 4. 内存优化策略在机器人中的应用

| 机器人算法 | 关键优化 | 加速比 |
|-----------|---------|-------|
| 图像预处理 | 全局内存合并访问 | 5-10× |
| 点云配准 (ICP) | 共享内存存储最近点 | 3-5× |
| 体素滤波 | 共享内存减少全局访问 | 5-8× |
| YOLO检测 | Tensor Core混合精度 | 3-4× |

---

## 5. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Memory Access Optimization for GPU Robotics" | ICRA 2021 | GPU机器人内存优化 |
| "Understanding GPU Memory Hierarchy" | ISPASS 2022 | GPU内存层次详解 |
| "Efficient Data Layout for NN on GPUs" | PACT 2020 | 神经网络数据布局 |
