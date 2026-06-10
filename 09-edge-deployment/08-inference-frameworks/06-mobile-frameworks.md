# 8.6 移动端框架 (MNN/NCNN/TFLite)

## 1. 框架对比

| 框架 | 开发者 | 语言 | 特点 | 机器人应用 |
|------|-------|------|------|-----------|
| **MNN** | 阿里巴巴 | C++ | CPU/GPU/OpenCL | 树莓派/ARM |
| **NCNN** | 腾讯 | C++ | 极致ARM优化, 无依赖 | 所有ARM设备 |
| **TFLite** | Google | C++ | 生态完善, Delegate | 树莓派 |

## 2. 性能对比 (ARM Cortex-A78)

| 框架 | MobileNetV3 | YOLOv5s |
|------|------------|---------|
| MNN | 5ms | 50ms |
| NCNN | **4ms** | **45ms** |
| TFLite | 6ms | 55ms |

## 3. 在机器人中的应用

- **树莓派5**: 低成本机器人视觉
- **RK3588**: MNN/NCNN优化推理
- **传感器端推理**: 处理器上运行轻量模型

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "MNN Inference Engine" | Alibaba 2019 | MNN框架 |
| "NCNN for Mobile" | Tencent 2018 | NCNN |
| "TFLite On-Device ML" | Google 2017 | TFLite |
