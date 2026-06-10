# 5.4 分布式训练技术

## 1. 并行策略

| 策略 | 原理 | 通信量 | 适用模型 | 机器人场景 |
|------|------|-------|---------|-----------|
| **数据并行** | 各GPU不同batch, 梯度同步 | 高 | 任何 | YOLO/ViT训练 |
| **模型并行** | 模型分层放不同GPU | 中 | 超大模型 | VLA训练 |
| **流水线并行** | 分层流水线处理 | 低 | 大模型 | 长序列模型 |
| **张量并行** | 单算子划分多GPU | 高 | 超大层 | 超大Transformer |

## 2. 混合精度训练

```python
# PyTorch混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for data in dataloader:
    with autocast(dtype=torch.bfloat16):  # BF16
        loss = model(data)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 3. 在VLA训练中的应用

| VLA模型 | 参数量 | 分布式策略 | GPU | 时间 |
|---------|-------|-----------|-----|------|
| RT-2 | 55B | TP+PP+DP | 256×A100 | 数周 |
| Octo-B | 1.3B | DP | 32×A100 | 数天 |
| Mobile ALOHA | 10M | 单卡 | 1×A100 | 数小时 |

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Efficient Large-Scale Training" | SC 2021 | 大模型训练 |
| "Mixed Precision Training" | ICLR 2018 | 混合精度训练 |
| "ZeRO: Memory Optimization" | SC 2020 | ZeRO显存优化 |
