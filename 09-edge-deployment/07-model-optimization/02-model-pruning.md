# 7.2 模型剪枝与稀疏化

## 1. 剪枝类型

| 类型 | 粒度 | 实际加速 | 硬件友好 | 精度恢复 |
|------|------|---------|---------|---------|
| 非结构化 | 单个权重 | ❌ 稀疏矩阵慢 | ❌ | 容易 |
| 结构化(通道) | 整个通道 | ✅ | ✅ | 较难 |
| 头部剪枝 | Attention头 | ✅ | ✅ | 容易 |
| 2:4结构化稀疏 | 4中取2 | ✅ (Tensor Core) | ✅ | 易 |

## 2. 结构化通道剪枝

```python
import torch.nn.utils.prune as prune

# L1范数通道剪枝
prune.ln_structured(conv_layer, name='weight', 
                    amount=0.3, n=1, dim=0)
# 移除30%输出通道 → 实际加速30%
```

## 3. NVIDIA 2:4结构化稀疏

```
Ampere GPU原生支持: 每4个权重中2个为零
[1, 2, 3, 4] → [1, 0, 3, 0]  (2:4模式)
Tensor Core计算时自动跳过零 → 2×加速
精度损失 < 0.5%
```

## 4. 在机器人中的应用

| 模型 | 剪枝比例 | 加速 | 精度恢复 |
|------|---------|------|---------|
| YOLOv8m | 30%通道 | 1.3× | Fine-tune后<0.1% |
| ViT-B | 25% Attention头 | 1.2× | 几乎无损 |
| ACT策略 | 40%权重 | 1.5× | <1%成功率 |

## 5. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Learning both Weights and Connections" | Han et al., NeurIPS 2015 | 三阶段剪枝 |
| "Pruning Filters for Efficient ConvNets" | Li et al., ICLR 2017 | 通道剪枝 |
| "Lottery Ticket Hypothesis" | Frankle & Carbin, ICLR 2019 | 彩票假设 |
