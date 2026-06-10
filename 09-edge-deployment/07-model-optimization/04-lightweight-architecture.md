# 7.4 轻量化模型架构

## 1. 主流轻量架构

| 架构 | 参数量 | 设计理念 | 机器人应用 |
|------|-------|---------|-----------|
| MobileNetV3 | 2.5-5.4M | Depthwise Conv + SE | 移动机器人视觉 |
| EfficientNet-Lite | 4-30M | NAS搜索 | 分类/检测 |
| YOLOv8n | 3.2M | CSPDarknet + PAN | 目标检测 |
| TinyViT | 5-21M | 蒸馏+Transformer | 轻量ViT |
| FastViT | 3.6-8.5M | 结构重参数化 | 高速推理 |

## 2. 结构重参数化

```
训练时 (复杂结构):           推理时 (等效简单结构):
3×3 Conv ──┐               
1×1 Conv ──┤──→ Add → BN     →    3×3 Conv (单分支)
BN ────────┘               

好处: 训练期高精度, 推理期零额外开销
```

## 3. 在Jetson上的推理速度

| 模型 | Jetson Orin(FP16) | Jetson Nano |
|------|------------------|-------------|
| YOLOv8n | 2ms | 15ms |
| MobileNetV3 | 1ms | 8ms |
| TinyViT-5M | 3ms | 25ms |
| FastViT-S | 2ms | 18ms |

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "MobileNets" | Howard 2017 | 深度可分离卷积 |
| "EfficientNet" | Tan & Le, ICML 2019 | 复合缩放 |
| "RepVGG" | Ding et al., CVPR 2021 | 结构重参数化 |
| "FastViT" | ICCV 2023 | 快速混合ViT |
