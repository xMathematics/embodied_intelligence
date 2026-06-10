# 8.7 推理框架选型指南

## 1. 选型决策树

```
你的目标硬件是什么？
├── NVIDIA GPU (Jetson/RTX)
│   └── → TensorRT (首选)
│       └── → ONNX Runtime + TensorRT EP
│
├── Intel CPU/iGPU (NUC)
│   └── → OpenVINO (首选)
│       └── → ONNX Runtime + OpenVINO EP
│
├── ARM CPU (树莓派/RK3588)
│   └── → NCNN/MNN (ARM优化)
│       └── → TFLite (生态优先)
│
├── 非NVIDIA GPU (AMD/Qualcomm)
│   └── → ONNX Runtime
│       └── → TVM (自动调优)
│
└── FPGA/自定义硬件
    └── → TVM (代码生成)
```

## 2. 典型机器人配置

| 机器人平台 | 推理框架 | 理由 |
|-----------|---------|------|
| Jetson Orin AGX | TensorRT | 极致GPU加速 |
| Intel NUC | OpenVINO | Intel优化 |
| 树莓派5 | NCNN | ARM极致优化 |
| RK3588 | MNN/RKNN | 国产支持 |
| Qualcomm RB5 | QNN/SNPE | Qualcomm NPU |

## 3. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Inference Engines Comparison for Robotics" | IEEE Access 2023 | 推理引擎对比 |
| "Edge AI Platforms Benchmarking" | IROS 2022 | 边缘平台基准 |
