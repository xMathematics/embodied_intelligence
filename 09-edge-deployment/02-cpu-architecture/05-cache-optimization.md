# 2.5 缓存优化策略

## 1. 问题陈述

CPU的缓存是最重要的性能优化资源。大多数机器人算法是**内存受限**而非计算受限。

### 典型场景分析

```
处理一帧 1920×1080 RGB 图像 (约6MB):
├── L1 Cache (64KB):   ❌ 只能容纳 ~1% 的图像
├── L2 Cache (512KB):  ❌ ~8%
├── L3 Cache (8MB):    ✅ 可容纳完整帧
└── 内存 (16GB):       ✅

结论: 逐像素操作时, 每次访问从L3加载 (15ns)
      vs 如果数据在L1 (1ns) → 15倍差距
```

---

## 2. 数据布局优化

### 2.1 SoA vs AoS (最关键)

```cpp
// ❌ AoS (Array of Structures) — Cache不友好
struct Point {
    float x, y, z;  // 12 bytes
    float nx, ny, nz; // 法向量
    float intensity;   // 强度
    uint8_t rgb[3];   // 颜色
}; // 共 31 bytes
Point cloud[N];  // N个点连续存储

// 当只需要处理x,y,z时:
float sum_x = 0;
for (int i = 0; i < N; i++)
    sum_x += cloud[i].x;  // 读取31字节, 只用4字节
    // Cache利用率: 4/31 = 13% ❌

// ✅ SoA (Structure of Arrays) — Cache友好
struct PointCloud {
    float x[N], y[N], z[N];       // 位置 (连续)
    float nx[N], ny[N], nz[N];    // 法向量
    float intensity[N];            // 强度
    uint8_t r[N], g[N], b[N];     // 颜色
};

// 处理x坐标:
for (int i = 0; i < N; i++)
    sum_x += cloud.x[i];  // 连续读取4字节, Cache利用率100% ✅
```

**在机器人中的实际影响**：

| 操作 | AoS Cache利用率 | SoA Cache利用率 | 性能差距 |
|------|----------------|----------------|---------|
| 点云坐标变换 | 25% | 100% | 2-4× |
| 点云法向量计算 | 20% | 100% | 3-5× |
| 图像逐像素 | 100% (天然SoA) | — | 基准 |

### 2.2 结构体对齐

```cpp
// ❌ 未对齐 — 跨Cache Line
struct alignas(64) RobotJoint {
    float position;    // 4 bytes (offset 0)
    float velocity;    // 4 bytes (offset 4) 
    float torque;      // 4 bytes (offset 8)
    // 仅12字节, 但浪费52字节padding到64
};

// ✅ 紧凑对齐 — 一Cache Line放多个
struct JointCompact {
    float pos, vel, torque;  // 12 bytes
};
// 64/12 ≈ 5个joint/Cache Line

// ✅ 显式对齐到Cache Line
struct alignas(64) JointCacheLine {
    float data[16];  // 刚好64字节 = 1 Cache Line
};
```

### 2.3 False Sharing (伪共享) 避免

```cpp
// ❌ 伪共享 — 多核频繁更新同一Cache Line的不同部分
struct SharedCounters {
    int counter_a;  // 核心0更新
    int counter_b;  // 核心1更新
    // 两个int在同一个Cache Line!
};

// ✅ 填充到不同Cache Line
struct alignas(64) CounterCore0 { int val; char pad[60]; };
struct alignas(64) CounterCore1 { int val; char pad[60]; };
// 每个计数器独立Cache Line
```

---

## 3. 循环优化

### 3.1 循环分块 (Loop Tiling)

```cpp
// ❌ 大矩阵乘法 — 整块处理, 频繁Cache Miss
void matmul_bad(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
    // B按列访问! Cache Miss率极高
}

// ✅ 分块处理 — 子块在Cache中完成
#define BLOCK 64  // 适配L1 Cache
void matmul_tiled(float* A, float* B, float* C, int N) {
    for (int ii = 0; ii < N; ii += BLOCK)
        for (int jj = 0; jj < N; jj += BLOCK)
            for (int kk = 0; kk < N; kk += BLOCK)
                for (int i = ii; i < ii+BLOCK; i++)
                    for (int j = jj; j < jj+BLOCK; j++) {
                        float sum = 0;
                        for (int k = kk; k < kk+BLOCK; k++)
                            sum += A[i*N+k] * B[k*N+j];
                        C[i*N+j] += sum;
                    }
    // A和B的子块都留在Cache中
}
```

### 3.2 循环合并

```cpp
// ❌ 两个独立循环 — 重复遍历数据
for (int i = 0; i < N; i++) cloud[i].x *= scale;  // 1次遍历
for (int i = 0; i < N; i++) cloud[i].y *= scale;  // 2次遍历
for (int i = 0; i < N; i++) cloud[i].z *= scale;  // 3次遍历

// ✅ 合并 — 1次遍历
for (int i = 0; i < N; i++) {
    cloud[i].x *= scale;  // 一次Cache Line加载
    cloud[i].y *= scale;  // 用了三次
    cloud[i].z *= scale;  // 充分利用Cache Line
}
```

---

## 4. 预取 (Prefetching)

```cpp
// 软件预取 — 提前加载数据到Cache
void process_frames_gpu_prefetch(uint8_t* frames, int n_frames) {
    for (int i = 0; i < n_frames; i++) {
        // 预取第i+2帧
        __builtin_prefetch(&frames[(i+2) * FRAME_SIZE], 0, 3);
        // 0=读, 3=高局部性
        
        process(frames + i * FRAME_SIZE);
        // 处理时, 下一帧已在Cache中
    }
}
```

---

## 5. 在机器人中的综合应用

| 场景 | 优化策略 | 预期提升 |
|------|---------|---------|
| 点云处理 | SoA数据布局 | 2-4× |
| 图像卷积 | 循环分块 + 缓存行复用 | 2-3× |
| 状态估计 (EKF) | 矩阵分块 | 2-5× |
| 碰撞检测 | AABB树缓存优化 | 2-3× |
| 多传感器数据 | SoA + 合并循环 | 2-4× |

---

## 6. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Cache-Oblivious Algorithms" | Frigo et al., FOCS 1999 | Cache无关算法 |
| "What Every Programmer Should Know About Memory" | Drepper, 2007 | 经典指南 |
| "Optimizing Memory Access for Robotic Perception" | IROS 2021 | 机器人感知内存优化 |
