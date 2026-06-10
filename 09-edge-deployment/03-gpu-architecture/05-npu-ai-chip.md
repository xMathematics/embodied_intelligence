# 3.5 专用AI芯片 (NPU/TPU/ASIC)

## 1. 为什么需要专用AI芯片

| 对比 | GPU | NPU/TPU | 优势 |
|------|-----|---------|------|
| **功耗** | 15-450W | 0.5-35W | NPU 10-100×更好 |
| **利用率** | 30-60% (AI推理) | 80-95% | NPU利用率更高 |
| **成本** | 高 | 低 (集成SoC) | NPU成本优势 |
| **尺寸** | 大 (独立芯片) | 小 (SoC集成) | NPU节省空间 |

**在机器人中的应用**：对于功耗受限（无人机）或成本敏感（消费级机器人）的场景，NPU是优于GPU的选择。

---

## 2. 主要NPU产品对比

| 芯片 | 厂商 | 架构 | 算力 | 功耗 | TOPS/W | 适用机器人 |
|------|------|------|------|------|--------|-----------|
| **TPU v4** | Google | MXU脉动阵列 | 275 TFLOPS | ~200W | 1.4 | 云端训练 |
| **Ascend 310** | 华为 | DaVinci Core | 16 TOPS | 8W | 2.0 | 华为机器人 |
| **RK3588 NPU** | Rockchip | 三核NPU | 6 TOPS | ~1W | 6.0 | 国产SBC |
| **地平线J6** | 地平线 | BPU (纳什) | 128 TOPS | 35W | 3.7 | 自动驾驶 |
| **Hailo-8** | Hailo | 数据流架构 | 26 TOPS | 2.5W | 10.4 | 工业检测 |
| **Intel Movidius 2** | Intel | SHAVE向量 | 1 TOPS | 1-2W | 0.5-1 | 无人机 |
| **Kneron KL720** | Kneron | NPU+CNN | 1.5 TOPS | ~0.5W | 3.0 | 超低功耗 |
| **NVIDIA DLA** | NVIDIA | 专用加速器 | 10-50 TOPS | ~1-3W | ~10 | Jetson内建 |

---

## 3. 主要NPU架构解析

### 3.1 Google TPU

```
TPU v4核心: MXU (Matrix Unit) = 脉动阵列
┌────────────────────────────────────┐
│            128×128 MAC阵列           │
│  ┌──┬──┬──┬──┐  ┌──┬──┬──┬──┐    │
│  │M │M │M │M │  │M │M │M │M │    │
│  ├──┤  │  │  │  ├──┤  │  │  │    │
│  │M │M │M │M │  │M │M │M │M │    │
│  ├──┤  │  │  │  ├──┤  │  │  │    │
│  │M │M │M │M │→ │M │M │M │M │    │
│  ├──┤  │  │  │  ├──┤  │  │  │    │
│  │M │M │M │M │  │M │M │M │M │    │
│  └──┴──┴──┴──┘  └──┴──┴──┴──┘    │
│  ← 数据流入    结果流出一 →       │
└────────────────────────────────────┘
```

### 3.2 华为DaVinci

```
DaVinci Core (Ascend 310):
┌─────────────────────────┐
│   3D Cube (矩阵计算)     │
│   16×16×16 MAC阵列      │
├─────────────────────────┤
│   Vector Unit           │
│   128-bit SIMD          │
├─────────────────────────┤
│   Scalar Unit           │
│   通用处理器             │
├─────────────────────────┤
│   缓存: L0A + L0B + L1 │
│   共享内存: 32KB        │
└─────────────────────────┘
```

### 3.3 Hailo-8 (数据流架构)

Hailo-8采用**静态数据流**架构，编译时将神经网络映射为硬件流水线：
- 无动态调度开销
- 每层计算单元同时工作
- 极低延迟 (一批次感知延迟<1ms)

---

## 4. 在机器人中的选型指南

### 选型决策

```
你的机器人需要AI推理吗?
├── 是, 且算力需求 > 50 TOPS
│   └── → GPU (Jetson Orin)
│
├── 是, 5-50 TOPS, 功耗<15W
│   └── → NPU (Hailo-8 / 地平线J6)
│       └── 配合ARM CPU使用
│
├── 是, <5 TOPS, 功耗<3W
│   └── → 集成NPU (RK3588 / Qualcomm)
│       └── 单芯片方案
│
└── 仅需简单分类, <1 TOPS
    └── → MCU级NPU (Kneron)
```

### 典型配置

| 机器人类型 | AI加速方案 | 理由 |
|-----------|-----------|------|
| 无人机 | Qualcomm RB5 | 低功耗+视觉ISP集成 |
| 轮式服务 | RK3588 NPU | 成本低, 6TOPS够用 |
| 工业检测 | Jetson Orin + Hailo-8 | GPU+NPU互补 |
| 人形机器人 | Jetson Orin AGX | 需要大模型推理 |

---

## 5. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "TPU v4: An Optically Reconfigurable Supercomputer" | ISCA 2021 | TPU v4架构 |
| "DaVinci: A Scalable Architecture for NN" | Huawei 2020 | DaVinci架构 |
| "Hailo-8: Dataflow Architecture for Edge AI" | Hailo 2021 | Hailo-8白皮书 |
| "NPU Survey for Edge AI" | ACM CS 2022 | NPU综述 |
