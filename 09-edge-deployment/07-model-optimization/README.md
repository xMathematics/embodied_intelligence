# 第7章：模型压缩、量化与轻量化

## 章节概览

模型优化是将大模型部署到边缘设备的关键。本章6个文件：

| 文件 | 内容 | 机器人中的价值 |
|------|------|--------------|
| [01-model-quantization.md](01-model-quantization.md) | 模型量化 (INT8/FP16/BF16) | 模型缩小4×, 加速3-4× |
| [02-model-pruning.md](02-model-pruning.md) | 模型剪枝与稀疏化 | 减少30-50%计算量 |
| [03-knowledge-distillation.md](03-knowledge-distillation.md) | 知识蒸馏 | VLA大模型→轻量策略 |
| [04-lightweight-architecture.md](04-lightweight-architecture.md) | 轻量化模型架构 | 原生高效结构 |
| [05-neural-architecture-search.md](05-neural-architecture-search.md) | 神经架构搜索(NAS) | 自动搜索最优架构 |
| [06-end-to-end-pipeline.md](06-end-to-end-pipeline.md) | 综合优化管线 | 案例: 344MB→5.5MB |
