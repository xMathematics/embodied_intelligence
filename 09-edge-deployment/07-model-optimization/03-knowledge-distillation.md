# 7.3 知识蒸馏

## 1. 基本原理

```
Teacher (大模型)                      Student (小模型)
┌──────────────┐                    ┌──────────────┐
│  ViT-L (300M) │                    │ TinyViT (5M) │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       ▼                                   ▼
  softmax(T=4)                        softmax(T=4)
       │                                   │
       └────────── KL Loss ─────────────────┘
                   +
             Hard Label Loss (CE)
```

## 2. 蒸馏变体

| 方法 | 原理 | 机器人应用 |
|------|------|-----------|
| Logit蒸馏 | 匹配输出logits | 感知模型压缩 |
| 特征蒸馏 | 匹配中间层特征 | 注意力迁移 |
| 关系蒸馏 | 匹配样本间关系 | 场景理解 |
| SeqKD | Teacher生成伪标签 | VLM蒸馏 |

## 3. 具身智能中的蒸馏案例

| Teacher | Student | 场景 | 效果 |
|---------|---------|------|------|
| DINOv2 (300M) | ResNet-18 (11M) | 分类 | 精度提升5% |
| RT-2 (55B) | RT-1 (35M) | 操作 | 保留80%性能 |
| LLaVA (7B) | TinyLLaVA (2.7B) | VLM | 保留85% |

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Distilling the Knowledge in a NN" | Hinton, NeurIPS 2015 | 蒸馏开山作 |
| "Knowledge Distillation: A Survey" | Gou et al., IJCV 2021 | 综述 |
| "Distilling VLM for Embodied Tasks" | CoRL 2023 | VLM蒸馏 |
