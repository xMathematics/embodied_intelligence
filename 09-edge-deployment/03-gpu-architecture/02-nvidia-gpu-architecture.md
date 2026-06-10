# 3.2 NVIDIA GPU架构详解

## 1. GPU代际演进

| 代际 | 微架构 | SM配置 | Tensor Core | 代表产品 | 机器人平台 |
|------|-------|-------|------------|---------|-----------|
| Maxwell (2014) | GMxxx | 128 FP32/SM | ❌ | GTX 980 | Jetson Nano |
| Pascal (2016) | GP10x | 128 FP32/SM | ❌ | GTX 1080 | Jetson TX2 |
| Volta (2017) | GV100 | 64 FP32+64 INT32 | 第1代 | V100, Titan V | Jetson Xavier |
| Turing (2018) | TU10x | 64 FP32+64 INT32 | 第2代 | RTX 20, T4 | — |
| Ampere (2020) | GA10x | 64 FP32+64 INT32 | 第3代 | RTX 30, A100 | **Jetson Orin** |
| Hopper (2022) | GH100 | 128 FP32+64 INT32 | 第4代 | H100 | — |
| Ada (2022) | AD10x | 128 FP32+64 INT32 | 第4代 | RTX 4090 | 工作站 |
| Blackwell (2024) | GB200 | 增强 | 第5代 | B200 | — |

---

## 2. SM内部结构 (以Ampere为例)

```
SM (Ampere架构)
│
├── 4个处理块 (Processing Block)
│   │
│   ├── 16 FP32 CUDA Cores
│   ├── 16 INT32 CUDA Cores  
│   ├── 1× 第3代 Tensor Core
│   ├── Warp调度器 + 寄存器 (最多64K 32-bit)
│   └── 共享内存 / L1 Cache (最大192KB)
│
├── 总CUDA Core: 64 FP32 + 64 INT32 = 128/SM
├── 总Tensor Core: 4/SM
└── 总寄存器: 64K 32-bit/SM
```

### Warp执行模型

```
GPU执行的基本单位: Warp = 32线程

┌─────────────────────────────────────────────┐
│ Warp调度器 → 每时钟周期选择一个Warp执行      │
│                                             │
│ Warp 0: ──┬──┬──┬──┬── ... ──┬── (32线程)  │
│ Warp 1: ──┬──┬──┬──┬── ... ──┬──           │
│ ...                                        │
│ Warp 31: ──┬──┬──┬──┬── ... ──┬──          │
│                                             │
│ 总Warp/SM: 32 (Ampere)                      │
│ 并发Warp上限: 16 (受寄存器/共享内存限制)     │
└─────────────────────────────────────────────┘
```

---

## 3. Tensor Core深度解析

### 3.1 原理

Tensor Core执行 **D = A × B + C**，一次完成4×4矩阵乘法+加法：

```
Tensor Core一次操作:
┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐
│ A00..A03 │   │ B00..B03 │   │ C00..C03 │   │ D00..D03 │
│ A10..A13 │ × │ B10..B13 │ + │ C10..C13 │ = │ D10..D13 │
│ A20..A23 │   │ B20..B23 │   │ C20..C23 │   │ D20..D23 │
│ A30..A33 │   │ B30..B33 │   │ C30..C33 │   │ D30..D33 │
└──────┘     └──────┘     └──────┘     └──────┘
```

### 3.2 Tensor Core代际对比

| 代际 | 架构 | 支持精度 | 峰值TFLOPS (A100/H100) |
|------|------|---------|----------------------|
| 第1代 | Volta | FP16 | 125 FP16 |
| 第2代 | Turing | FP16 | — |
| 第3代 | Ampere | FP16, BF16, INT8, INT4 | 312 FP16 / 624 INT8 |
| 第4代 | Hopper | FP16, BF16, FP8, INT8 | 990 FP16 / 1979 FP8 |
| 第5代 | Blackwell | FP16, FP8, FP4, INT8 | 更高 |

### 3.3 在机器人感知中的加速

| 模型 | FP32 (无Tensor Core) | FP16 (Tensor Core) | 加速比 |
|------|--------------------|-------------------|-------|
| ResNet-50 | 6ms | 1.5ms (Jetson Orin) | 4× |
| YOLOv8m | 30ms | 8ms | 3.75× |
| ViT-B | 50ms | 15ms | 3.3× |

---

## 4. Jetson GPU规格

| 平台 | GPU架构 | CUDA Core | Tensor Core | FP16 TFLOPS | INT8 TOPS |
|------|--------|----------|------------|------------|----------|
| Jetson Nano | Maxwell 128-core | 128 | — | 0.25 | — |
| Jetson TX2 | Pascal 256-core | 256 | — | 0.75 | — |
| Xavier NX | Volta 384-core | 384 | 48 (Gen2) | 7.5 | 15 |
| Orin NX 16GB | Ampere 1024-core | 1024 | 32 (Gen3) | 35 | 70 |
| Orin AGX 64GB | Ampere 2048-core | 2048 | 64 (Gen3) | 138 | 275 |

---

## 5. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "NVIDIA Tesla Architecture" | IEEE Micro 2008 | 统一GPU架构 |
| "Tensor Cores for AI" | NVIDIA 2020 | Tensor Core技术白皮书 |
| "Jetson Orin Platform" | NVIDIA 2022 | Jetson Orin架构 |
| "Dissecting Ampere GPU" | Hot Chips 2021 | Ampere架构详解 |
