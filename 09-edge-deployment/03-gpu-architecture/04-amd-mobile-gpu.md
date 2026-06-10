# 3.4 AMD与移动GPU

## 1. AMD GPU架构

### 1.1 RDNA (游戏) vs CDNA (计算)

| 特性 | RDNA 3 | CDNA 3 |
|------|--------|--------|
| 目标 | 图形/游戏 | HPC/AI计算 |
| 核心单元 | WGP (双CU) | Matrix Core |
| 计算单元(CU) | 支持SIMD | 支持Matrix |
| 矩阵加速 | INT8/INT4 | FP16/BF16/INT8 |
| 软件栈 | ROCm | ROCm |
| 代表产品 | RX 7900 XTX | MI300X |
| AI算力 | 61 TFLOPS (FP16) | 130 TFLOPS (FP16) |

### 1.2 ROCm平台

ROCm (Radeon Open Compute) 是AMD的GPU计算平台。

```
ROCm软件栈:
┌─────────────────────────┐
│ TensorFlow / PyTorch    │
├─────────────────────────┤
│    MIOpen (深度学习库)   │
│   rocBLAS (BLAS库)      │
│   rocFFT (FFT库)        │
├─────────────────────────┤
│    HIP (异构编程接口)    │
│   (CUDA兼容层)          │
├─────────────────────────┤
│    ROCk Kernel驱动       │
└─────────────────────────┘
```

### 1.3 HIP: CUDA到AMD的桥梁

```cpp
// CUDA代码
__global__ void vec_add(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}

// HIP代码 (几乎一样)
__global__ void vec_add(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}

// HIP有工具 hipify 可自动将CUDA代码转为HIP
```

---

## 2. 移动GPU

### 2.1 Qualcomm Adreno

| 代际 | 架构 | 特性 | 机器人平台 |
|------|------|------|-----------|
| Adreno 6xx | 统一着色器 | OpenCL/Vulkan | RB5 |
| Adreno 7xx | 改进着色器 | INT8加速 | RB6 |
| Adreno 8xx | 最新 | AI引擎 | 未来平台 |

**RB5/RB6机器人平台**：
- 集成Adreno GPU + Hexagon DSP (AI加速)
- 专为无人机和移动机器人设计
- 功耗5-15W, 可运行轻量CNN模型

### 2.2 ARM Mali

| 代际 | 架构 | 特性 | 机器人平台 |
|------|------|------|-----------|
| Mali-G52 | Bifrost | 2-4核 | RK3588等 |
| Mali-G610 | Valhall | 可变核数 | RK3588 |
| Mali-G720 | 第5代 | 延迟着色 | 下一代 |

**在RK3588中的应用**：
- 4-16个着色器核心
- 支持OpenGL ES 3.2, Vulkan 1.1
- 可运行轻量级推理 (GPU Compute)
- 配合内置NPU使用

### 2.3 Apple GPU

| 芯片 | GPU核心 | 统一内存 | AI算力 |
|------|--------|---------|-------|
| M1 | 8核 | 16GB | 11 TFLOPS |
| M2 Max | 38核 | 96GB | 13.6 TFLOPS |
| M3 Ultra | 80核 | 192GB | ~32 TFLOPS |

**在机器人开发中的应用**：
- 机器人仿真和算法开发
- 通过CoreML部署模型
- 统一内存架构省去CPU-GPU数据传输

---

## 3. 非NVIDIA平台的机器人适配

| 平台 | GPU | 推理框架 | 适用范围 |
|------|-----|---------|---------|
| AMD工作站 | RX/MI系列 | ONNX Runtime + ROCm | 训练/仿真 |
| Qualcomm RB5 | Adreno + DSP | QNN/SNPE | 无人机 |
| RK3588 | Mali | RKNN Toolkit | 国产机器人 |
| Apple Silicon | Apple GPU | CoreML | 开发/仿真 |
| 树莓派 | VideoCore | NCNN/TFLite | 教育/轻量 |

---

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "AMD CDNA Architecture for HPC" | Hot Chips 2022 | CDNA架构 |
| "ROCm Platform Overview" | AMD 2020 | ROCm平台 |
| "Mobile GPU Architectures Survey" | IEEE Access 2021 | 移动GPU综述 |
| "Qualcomm RB5 Robotics Platform" | Qualcomm 2021 | RB5平台 |
