# 1.5 多核架构与缓存一致性

## 1. 基本概念

### 1.1 为什么需要多核？

```
单核性能增长放缓:
┌────┬────┬────┬────┬────┬────┐
│P4  │Core2│i7  │i7-8│i9-12│     │
│3GHz│3GHz │3.4 │4.2 │5.0  │频率 │
│1核 │2核  │4核 │6核 │16核 │核心数│
└────┴────┴────┴────┴────┴────┘
2000  2006  2010  2017  2022  年份

结论：提升核心数成为提升性能的主要手段
```

### 1.2 机器人中的多核需求

具身智能机器人需要同时运行多个任务：

| 任务 | 核心需求 | 典型占用 |
|------|---------|---------|
| ROS2通信 | 低延迟消息传递 | 1个核心 |
| 视觉感知 | 高计算量 | 2-4个核心 |
| SLAM | 中计算量 | 1-2个核心 |
| 控制循环 | 实时保证 | 1个核心 (独占) |
| 人机交互 | 可变负载 | 1个核心 |
| **总计** | | **6-10个核心** |

---

## 2. SMP vs NUMA

### 2.1 SMP (对称多处理)

```
           ┌─────────────────────┐
           │    系统互联总线       │
           └──┬──┬──┬──┬──┬──┬──┘
              │  │  │  │  │  │
             C0 C1 C2 C3 C4 C5
              │  │  │  │  │  │
           ┌──┴──┴──┴──┴──┴──┴──┐
           │    共享内存 (UMA)     │
           │    统一延迟访问       │
           └─────────────────────┘
```

**特点**：
- 所有核心访问内存的延迟相同
- 编程简单
- 核心数增加时，总线竞争加剧
- 通常用于 ≤8核系统

### 2.2 NUMA (非统一内存访问)

```
┌───────────┐    ┌───────────┐
│  Node 0   │    │  Node 1   │
│ C0 C1 C2 C3│    │ C4 C5 C6 C7│
│    │      │    │    │      │
│ ┌──┴──┐  │    │ ┌──┴──┐  │
│ │MEM0 │  │    │ │MEM1 │  │
│ └─────┘  │    │ └─────┘  │
└────┬─────┘    └────┬─────┘
     │               │
     └───互联总线────┘
```

**特点**：
- 核心访问本地内存延迟低，访问远端内存延迟高
- 可扩展性强 (可达数百核心)
- 编程需要考虑数据放置
- 常用于 ≥8核系统

**NUMA延迟**：
```
Node0访问MEM0: ~100ns (本地)
Node0访问MEM1: ~180ns (远端, 通过互联)
Node1访问MEM0: ~180ns (远端)
Node1访问MEM1: ~100ns (本地)
```

### 2.3 在Jetson上的NUMA拓扑

```bash
# Jetson Orin AGX (12核) 查看NUMA拓扑
$ numactl --hardware
available: 2 nodes (0-1)
node 0 cpus: 0-5    # 6个Cortex-A78AE (性能核)
node 0 size: 8192 MB
node 1 cpus: 6-11   # 6个Cortex-A78AE (性能核)
node 1 size: 8192 MB
node distances:
  node   0   1
    0   10  20    # Node0访问Node1内存延迟×2
    1   20  10    # 需要NUMA感知编程
```

---

## 3. big.LITTLE / 大小核架构

### 3.1 概念

```
Jetson Orin的12核CPU:
┌─────────────────────────────────────┐
│           ARM DynamIQ               │
│  ┌──────────────────────────────┐   │
│  │    Cortex-A78AE 集群         │   │
│  │  C0  C1  C2  ...  C11       │   │
│  │  └─── 12个高性能核 ───┘      │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │    共享L3 Cache (4MB)        │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

**大小核架构的核心理念**：
- 大核 (Performance): 高频率，高性能
- 小核 (Efficiency): 低频率，低功耗
- 由操作系统调度器动态分配任务

### 3.2 在机器人中的应用策略

```c
// Linux核心隔离：将控制任务绑定到专用核心
// 方法1: 通过内核启动参数隔离核心
// 在/boot/extlinux/extlinux.conf中添加:
// isolcpus=0-1 nohz_full=0-1 rcu_nocbs=0-1

// 方法2: 运行时绑定
#include <sched.h>
#include <pthread.h>

void bind_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

// 控制循环绑定到Core 0
void control_thread() {
    bind_to_core(0);  // 独占Core 0
    while (running) {
        // 控制循环不会被其他任务干扰
        read_sensors();
        compute_control();
        actuate_motors();
    }
}

// 感知任务绑定到其他核心
void perception_thread() {
    bind_to_core(2);  // Core 2-5处理感知
    while (running) {
        process_image();
        run_detection();
    }
}
```

---

## 4. 缓存一致性协议

### 4.1 问题：多核数据不一致

```
核心0                    核心1
L1 Cache: x=10          L1 Cache: x=10
  │                        │
  ├── x = x + 5 ──────────┤─── (未通知核心0)
  │   x=15                 │   x被修改?
  ▼                        ▼
内存: x=10 (过时了!)
```

### 4.2 MESI协议

MESI是经典的缓存一致性协议：

| 状态 | 含义 | 该行在Cache中 | 内存中 | 其他Cache中 |
|------|------|-------------|-------|-----------|
| **M** (Modified) | 已修改 | ✅ 最新 | ❌ 过时 | ❌ 不存在 |
| **E** (Exclusive) | 独占 | ✅ 最新 | ✅ 最新 | ❌ 不存在 |
| **S** (Shared) | 共享 | ✅ 可能最新 | ✅ 最新 | ✅ 可能存在 |
| **I** (Invalid) | 无效 | ❌ 不可用 | — | — |

**工作原理**：
```
核心0读取x:   Cache Miss → 从内存加载 → E状态
核心1读取x:   监听核心0的Cache → 两者变为S状态
核心0写x:     发送Invalidate → 核心1的Cache变为I状态 → 自己变为M状态
核心1读x:     Cache Miss → 从核心0的Cache获取最新值 → 两者变S状态
```

### 4.3 伪共享 (False Sharing)

**经典的多核性能陷阱**：

```c
// ❌ 伪共享示例
struct Data {
    int counter0;  // 核心0使用
    // ... 56字节填充 ...
    int counter1;  // 核心1使用
    // counter0和counter1在同一个Cache Line (64字节)中
};

Data data;
// 核心0频繁写data.counter0
// 核心1频繁写data.counter1
// 尽管不冲突, 但因共享Cache Line导致持续的Invalidate通信

// ✅ 解决方法: 对齐到不同Cache Line
struct alignas(64) DataCore0 { int counter; char padding[60]; };
struct alignas(64) DataCore1 { int counter; char padding[60]; };
DataCore0 data0;  // Cache Line 0-63
DataCore1 data1;  // Cache Line 64-127
// 两个计数器在不同Cache Line, 互不干扰
```

---

## 5. 内存一致性模型

| 模型 | 说明 | 编程难度 | 性能 |
|------|------|---------|------|
| **顺序一致性** (SC) | 所有操作按程序顺序执行 | 低 | 低 |
| **TSO** (x86) | 写后读可重排 | 中 | 中 |
| **弱一致性** (ARM) | 所有操作可重排 | 高 | 高 |
| **释放一致性** | acquire/release语义 | 中 | 高 |

### ARM vs x86 内存模型

```c
// x86 (TSO): 写操作不会重排到读之后
// 但写-写可重排
core0: store(x, 1); load(y);
core1: store(y, 1); load(x);
// x86上至少有一方读到1

// ARM (Weak): 所有操作都可能重排
// 需要使用内存屏障:
core0: store(x, 1); __sync_synchronize(); load(y);
core1: store(y, 1); __sync_synchronize(); load(x);
```

---

## 6. 在具身智能中的应用

### 6.1 核心分配策略

```
Jetson Orin 12核分配 (推荐):
┌──────────┬─────────┬──────────────────────┐
│ 核心编号  │ 职责     │ 说明                  │
├──────────┼─────────┼──────────────────────┤
│ Core 0   │ 控制循环 │ 独占, 隔离, 最高优先级 │
│ Core 1   │ ROS2通信 │ ROS2 Spin + 消息路由   │
│ Core 2-5 │ 感知处理  │ 图像预处理, 特征提取   │
│ Core 6-7 │ SLAM     │ 前端+后端优化          │
│ Core 8-9 │ 导航规划  │ 路径规划 + 避障       │
│ Core 10  │ 人机交互  │ 语音/UI               │
│ Core 11  │ 系统服务  │ 日志,监控,OTA         │
└──────────┴─────────┴──────────────────────┘
```

### 6.2 实时性保障

```bash
# Linux实时配置 (PREEMPT_RT)
# 1. 核心隔离
isolcpus=0 nohz_full=0 rcu_nocbs=0

# 2. 设置实时调度
chrt -rr 99 ./control_loop  # 实时循环优先级99

# 3. 内存锁 (防止页面换出)
mlockall(MCL_CURRENT | MCL_FUTURE);
```

---

## 7. 相关论文

| 论文 | 发表 | 核心内容 |
|------|------|---------|
| "NUMA-Aware Memory Management for Real-Time Systems" | RTSS 2019 | NUMA感知的实时内存管理 |
| "Cache Coherence Protocols: A Survey" | ACM CS 2020 | 缓存一致性协议综述 |
| "A Study of big.LITTLE Architecture for Robotics" | ICRA 2022 | 大小核架构在机器人中的应用 |
| "Real-Time Scheduling on Heterogeneous Multi-Core" | ECRTS 2021 | 异构多核实时调度 |

---

## 8. 本章小结

### 核心要点
1. **NUMA感知编程**在多核机器人平台上至关重要
2. **核心隔离**是保证控制循环实时性的关键手段
3. **伪共享 (False Sharing)** 是多核编程最常见的性能陷阱
4. **ARM弱一致性模型**需要谨慎使用内存屏障
5. **大小核架构**为机器人提供了性能-功耗的弹性平衡

### 自测题
1. SMP和NUMA的区别是什么？在Jetson Orin上属于哪种？
2. 什么是伪共享？如何在机器人编程中避免？
3. 为什么实时控制需要使用核心隔离？
4. ARM和x86的内存模型有何不同？对编程有什么影响？
