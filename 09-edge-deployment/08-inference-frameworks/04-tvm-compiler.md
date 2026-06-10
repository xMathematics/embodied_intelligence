# 8.4 TVM编译器

## 1. 架构

```
深度学习框架 (PyTorch/TF/ONNX)
         │
         ▼
    TVM Relay IR
         │
    ┌────┴────┐
    │ AutoTVM  │ ← 自动算子搜索
    │ AutoScheduler │
    └────┬────┘
         │
    ┌────┴────┐
    │ 代码生成  │ → CUDA, OpenCL, Metal, ...
    └─────────┘
```

## 2. TVM vs TensorRT

| 对比 | TensorRT | TVM |
|------|---------|-----|
| 适用硬件 | NVIDIA GPU+DLA | 几乎所有硬件 |
| 优化方式 | 手工调优Kernel库 | 自动搜索+代码生成 |
| 灵活性 | NVIDIA生态 | 可扩展任意后端 |
| 性能(NVIDIA) | ✅ 极致 | 良好 |
| 性能(非NVIDIA) | ❌ 不支持 | ✅ 好 |
| 部署复杂度 | 简单 | 较复杂 |

## 3. 在机器人中的应用

- **非NVIDIA硬件**: AMD GPU/FPGA/NPU
- **自定义加速器**: 专用芯片代码生成
- **异形算子**: 自定义注意力机制

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "TVM: End-to-End Optimizing Compiler" | Chen et al., OSDI 2018 | TVM原始论文 |
| "AutoScheduler" | Zheng et al., OSDI 2020 | 自动调度 |
| "Ansor" | Zheng et al., OSDI 2020 | 自动调度器 |
