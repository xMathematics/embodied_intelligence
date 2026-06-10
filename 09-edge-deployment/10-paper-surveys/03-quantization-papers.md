# 10.3 模型量化论文

## 1. "Quantization and Training of NN" (QAT)

| 项目 | 内容 |
|------|------|
| 作者 | Jacob et al. |
| 发表 | CVPR 2018 |
| 核心 | 量化感知训练, INT8无损推理 |
| 引用 | >4000次 |
| 机器人价值 | 部署量化模型到Jetson |

## 2. "LSQ: Learned Step Size Quantization"

| 项目 | 内容 |
|------|------|
| 作者 | Esser et al. |
| 发表 | ICLR 2020 |
| 核心 | 可学习的量化步长 |
| 机器人价值 | 感知模型中敏感层更优量化 |

## 3. "GPTQ: Accurate PTQ for Generative Pre-trained Transformers"

| 项目 | 内容 |
|------|------|
| 作者 | Frantar et al. |
| 发表 | ICLR 2023 |
| 核心 | 大模型3-4bit后训练量化 |
| 机器人价值 | VLA大模型量化部署到Jetson |

## 4. "QLoRA: Efficient Finetuning of Quantized Language Models"

| 项目 | 内容 |
|------|------|
| 作者 | Dettmers et al. |
| 发表 | NeurIPS 2023 |
| 核心 | NF4 + 双重量化 + 分页优化 |
| 机器人价值 | 边缘设备微调VLA模型 |
