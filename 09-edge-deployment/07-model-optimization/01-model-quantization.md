# 7.1 模型量化

## 1. 精度格式对比

| 格式 | 位宽 | 模型大小 | 精度(相对FP32) | 加速比 | 适用 |
|------|------|---------|---------------|-------|------|
| FP32 | 32-bit | 100% | 基准 | 1× | 训练/基线 |
| FP16 | 16-bit | 50% | ~100% | 2× | 推理/训练 |
| BF16 | 16-bit | 50% | ~100% | 2× | 训练(大模型) |
| INT8 | 8-bit | 25% | 98-99.5% | 3-4× | **边缘推理** |
| INT4 | 4-bit | 12.5% | 95-98% | 4-6× | 超低精度 |
| NF4 | 4-bit | 12.5% | 96-99% | 4-6× | QLoRA微调 |

## 2. 后训练量化(PTQ)

```python
import torch

# PyTorch动态量化
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 优势: 无需重新训练
# 劣势: 大模型精度损失可能较大
```

## 3. 量化感知训练(QAT)

```python
# 模拟量化效应
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
model = torch.ao.quantization.prepare_qat(model, inplace=True)
# → 正常训练 ...
model = torch.ao.quantization.convert(model, inplace=True)
```

## 4. 在Jetson上的效果

| 模型 | FP32 | INT8 | 精度损失 | 加速 |
|------|------|------|---------|------|
| YOLOv8m | 30ms | 5ms | <0.5% mAP | **6×** |
| ViT-B | 50ms | 12ms | <1% top-1 | **4×** |
| DINOv2 | 80ms | 20ms | <0.5% | **4×** |

## 5. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Quantization and Training of NN" | Jacob et al., CVPR 2018 | QAT经典 |
| "LSQ: Learned Step Size Quantization" | ICLR 2020 | 可学习步长 |
| "GPTQ: Accurate PTQ for Transformers" | ICLR 2023 | 大模型量化 |
| "QLoRA: 4-bit Finetuning" | NeurIPS 2023 | 4-bit微调 |
