# 8.1 推理优化技术总览

## 1. 核心优化技术

| 技术 | 原理 | 加速效果 | 说明 |
|------|------|---------|------|
| **图优化** | 简化计算图, 消除无用操作 | 1-2× | 常数折叠, 死代码消除 |
| **算子融合** | 合并相邻算子(Conv+BN+ReLU) | 2-3× | 减少Kernel launch |
| **量化** | FP32→INT8 | 3-4× | Tensor Core加速 |
| **内存优化** | 内存池, in-place操作 | 1-2× | 减少分配拷贝 |
| **自动调优** | 搜索最优kernel配置 | 1-3× | 适配具体硬件 |
| **动态形状** | 优化动态输入 | 灵活 | 适配变分辨率 |

## 2. 算子融合示例

```
融合前: Conv → BN → ReLU → Pool
        4次Kernel launch + 4次内存读写

融合后: Conv+BN+ReLU → Pool
        2次Kernel launch + 2次内存读写
       (BN吸收进Conv权重 → 1次Kernel)
```

## 3. 不同框架性能对比 (Jetson Orin)

| 框架 | YOLOv8m | ResNet-50 | ViT-B |
|------|---------|-----------|-------|
| PyTorch (Eager) | 30ms | 15ms | 50ms |
| TorchScript | 25ms | 12ms | 42ms |
| ONNX Runtime | 18ms | 8ms | 30ms |
| TensorRT | **5ms** | **3ms** | **10ms** |

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Efficient Processing of DNN" | Sze et al., IEEE 2017 | DNN高效处理综述 |
| "Survey of DL Optimization Frameworks" | IEEE Access 2022 | 框架综述 |
