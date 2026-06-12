# 11.2 参数高效微调（PEFT）

## 1. 为什么需要PEFT

**问题**：全参数微调大模型需要大量GPU资源，且会遗忘预训练知识。

**PEFT方案**：只更新少量参数，保持大部分参数冻结。

## 2. LoRA（Low-Rank Adaptation）

**论文**：Hu et al., 2021 — ICLR 2022

$$\mathbf{W}' = \mathbf{W}_0 + \mathbf{BA}, \quad \mathbf{B} \in \mathbb{R}^{d \times r}, \mathbf{A} \in \mathbb{R}^{r \times k}$$

**为什么有效**：预训练权重的更新呈低秩结构。

**典型配置**：$r=8, 16, 32$，可训练参数 < 1%。

**QLoRA**（Dettmers 2023）：4-bit量化 + LoRA，65B模型可在单张48G GPU微调。

## 3. PEFT方法对比

| 方法 | 可训练参数 | 特点 |
|------|-----------|------|
| Adapter | 1-5% | 插入式小网络 |
| Prefix Tuning | 0.1% | 前缀token |
| LoRA | 0.1-1% | 低秩矩阵 |
| DoRA | 0.1-1% | 权重分解LoRA |

## 4. 在具身智能中的应用

- **机器人策略微调**：用LoRA快速适配新机器人或新任务
- **多任务适配**：不同任务使用不同LoRA模块
- **Sim-to-Real微调**：仿真预训练 + LoRA真实环境适配

## 5. 参考文献

1. Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language models. *ICLR*.
2. Dettmers, T., et al. (2023). QLoRA: Efficient finetuning of quantized language models. *NeurIPS*.
3. Liu, S., et al. (2024). DoRA: Weight-decomposed low-rank adaptation. *arXiv*.
