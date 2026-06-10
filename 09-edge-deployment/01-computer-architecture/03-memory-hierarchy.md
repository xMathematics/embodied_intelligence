# 1.3 存储层次结构

## 1. 基本概念

存储层次结构利用**局部性原理**来弥合CPU和内存之间的速度差距。

### 局部性原理

| 类型 | 定义 | 在机器人中的体现 |
|------|------|----------------|
| **时间局部性** | 最近访问的数据很可能被再次访问 | 控制循环中的PD参数反复读取 |
| **空间局部性** | 访问某个地址后，邻近地址很可能被访问 | 图像像素的连续读取 |
| **顺序局部性** | 指令通常是顺序执行的 | 程序代码的顺序执行 |

### 存储层次金字塔

```
                  寄存器的  ~0.3ns, ~1KB
                 L1 Cache  ~1ns,   ~64KB
               L2 Cache    ~4ns,   ~512KB
             L3 Cache      ~15ns,  ~8MB
          主存 (DRAM)      ~80ns,  ~16GB
         SSD (NVMe)        ~100μs, ~512GB
       HDD                 ~10ms,  ~2TB
```

### 各层次详细参数

| 层级 | 典型大小 | 延迟 | 带宽 | 实现技术 | 每比特成本 |
|------|---------|------|------|---------|-----------|
| 寄存器 | 512B-1KB | ~0.3ns | — | Flip-flop | $$$$$ |
| L1 Cache (指令) | 32-64KB | ~0.5-1ns | ~2TB/s | SRAM | $$$ |
| L1 Cache (数据) | 32-64KB | ~0.5-1ns | ~2TB/s | SRAM | $$$ |
| L2 Cache | 256KB-1MB | ~3-5ns | ~1TB/s | SRAM | $$$ |
| L3 Cache | 2-32MB | ~10-20ns | ~500GB/s | SRAM | $$ |
| 主存 (DDR5) | 8-64GB | ~50-100ns | ~100GB/s | DRAM | $ |
| NVMe SSD | 256GB-2TB | ~100μs | ~5GB/s | NAND Flash | ¢ |
| HDD | 1-20TB | ~10ms | ~200MB/s | Magnetic | ¢ |

---

## 2. Cache的工作原理

### 2.1 Cache映射方式

| 映射方式 | 说明 | 硬件复杂度 | 冲突率 |
|---------|------|-----------|-------|
| **直接映射** | 每个内存地址映射到唯一Cache行 | 低 | 高 |
| **全相连** | 任何地址可映射到任何Cache行 | 高 | 低 |
| **组相连** | 折中方案，N路组相连 | 中 | 中 |

### 2.2 Cache行 (Cache Line)

Cache行是Cache与内存之间的数据交换单位，通常为 **64字节**（16个float）。

```c
// 一个Cache Line可以容纳:
// 16个 float   (4×16=64)
// 8个  double   (8×8=64)
// 64个 uint8_t  (1×64=64)
```

### 2.3 写策略

| 策略 | 说明 | 性能 | 一致性 |
|------|------|------|-------|
| **Write-through** | 同时写Cache和内存 | 慢 | 强 |
| **Write-back** | 先写Cache，替换时写回内存 | 快 | 需维护 |
| **Write-allocate** | 写miss时加载到Cache | 配合Write-back | — |
| **No-write-allocate** | 写miss时直接写内存 | 配合Write-through | — |

---

## 3. Cache Miss的三种类型

| Miss类型 | 定义 | 在机器人中的表现 |
|---------|------|----------------|
| **Compulsory Miss** (冷启动) | 第一次访问某个数据 | 程序刚启动时的首次数据访问 |
| **Capacity Miss** | Cache容量不足 | 处理大图像时，图像数据超Cache容量 |
| **Conflict Miss** | 多个地址映射到同一Cache行 | 两个大数组的起始地址相距恰好是Cache大小的整数倍 |

### 机器人的Cache Miss分析

```
案例：YOLO推理中的Cache行为

输入层 (640×640×3 = 1.2MB)
├── L1 Cache (64KB):  ❌ 远超容量, 反复Capacity Miss
├── L2 Cache (512KB): ❌ 仍超容量
└── L3 Cache (8MB):   ✅ 可容纳, 但延迟高

第一层卷积权重 (3×3×3×16 = 432字节)
├── L1 Cache: ✅ 完全容纳
└── Cache Miss率: 低

全连接层权重 (1024×1024×4 = 4MB)
├── L3 Cache (8MB): ✅ 可容纳
└── 但若另一个模型也在用L3, 则可能Conflict Miss
```

---

## 4. 在机器人中的Cache优化策略

### 4.1 循环分块 (Loop Tiling / Cache Blocking)

```c
// ❌ 差: 大矩阵导致大量Cache Miss
for (int i = 0; i < N; i++)
    for (int j = 0; j < M; j++)
        sum += A[i][j] * B[j][i];

// ✅ 好: 分块使子矩阵在Cache中完成
#define BLOCK 64  // 适配L1 Cache大小
for (int ii = 0; ii < N; ii += BLOCK)
    for (int jj = 0; jj < M; jj += BLOCK)
        for (int i = ii; i < ii+BLOCK; i++)
            for (int j = jj; j < jj+BLOCK; j++)
                sum += A[i][j] * B[j][i];
                // 整个BLOCK × BLOCK子矩阵在Cache中
```

### 4.2 数据对齐

```c
// 对齐到Cache Line边界 (64字节)
struct alignas(64) RobotState {
    float position[3];    // 12字节
    float velocity[3];    // 12字节
    float orientation[4]; // 16字节
    // ... 填充到64字节
};
// 好处: 每个RobotState刚好占一个Cache Line
// 多核访问不同RobotState不会相互干扰 (避免False Sharing)
```

### 4.3 数据布局优化

```c
// ❌ AoS (Array of Structures) - Cache不友好
struct Point { float x, y, z; };
Point points[N];  // x1 y1 z1 x2 y2 z2 ...
// 处理x坐标时, 只用了1/3的Cache Line

// ✅ SoA (Structure of Arrays) - Cache友好
struct Points {
    float x[N];  // x1 x2 x3 ... 连续存放
    float y[N];  // y1 y2 y3 ... 连续存放
    float z[N];  // z1 z2 z3 ... 连续存放
};
// 处理x坐标时, 整个Cache Line都是有用的x数据
```

### 4.4 软件预取 (Software Prefetch)

```cpp
// 在机器人控制循环中预取下一帧数据
void control_loop() {
    while (running) {
        // 预取下一帧传感器数据到Cache
        __builtin_prefetch(&next_frame, 0, 3);  // 0=读取, 3=高局部性
        process(current_frame);
        current_frame = next_frame;
        next_frame = sensor_read();  // 后台DMA传输
    }
}
```

---

## 5. TLB (Translation Lookaside Buffer)

### 5.1 概念

TLB是虚拟地址到物理地址转换的**快速缓存**。

```
虚拟地址 → [TLB查找] → 命中 → 物理地址 → 访问内存
              ↓ miss
            [页表遍历] (慢100×)
              ↓
            物理地址 → 访问内存
```

### 5.2 在机器人中的影响

```c
// 大内存分配导致TLB压力
float* big_array = new float[100 * 1024 * 1024];  // 400MB
// 若页大小为4KB, 需要 400MB/4KB = 100,000个页表项
// TLB通常只有 64-1024个条目 → 频繁TLB Miss

// 缓解: 使用大页 (Huge Pages)
// Linux透明大页或显式 2MB/1GB 页
// 400MB/2MB = 200个页表项 → 可放入TLB
```

---

## 6. 相关论文

| 论文 | 发表 | 核心贡献 |
|------|------|---------|
| "The Memory Wall" | Wulf & McKee, ACM SIGARCH 1995 | 首次提出"存储墙"概念 |
| "What Every Programmer Should Know About Memory" | Drepper, 2007 | 经典内存优化实践指南 |
| "Cache-Oblivious Algorithms" | Frigo et al., FOCS 1999 | Cache无关算法设计理论 |
| "A Case for ML-Aware Memory Systems" | ISCA 2021 | 针对ML负载的存储系统设计 |
| "Optimizing Memory Access for Robotic Perception" | IROS 2021 | 机器人感知中的内存优化 |

---

## 7. 本章小结

### 核心要点
1. **存储墙是机器人性能的首要瓶颈** — 大多数机器人算法是内存受限而非计算受限
2. **Cache Miss是性能杀手** — 一次L3 Cache Miss = 200+周期浪费
3. **优化数据布局比优化计算更重要** — SoA > AoS, 分块 > 整块
4. **TLB Miss在大型数据处理中不可忽视** — 考虑使用Huge Pages

### 自测题
1. 解释三种Cache Miss类型。在图像处理中各有什么表现？
2. 为什么SoA数据布局比AoS更Cache友好？
3. 分块算法的Block大小应该如何选择？
4. 在Jetson上处理1080p视频流时，如何估计Cache Miss率？
