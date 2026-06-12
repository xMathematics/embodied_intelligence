# 4.3 GPT解码器家族

## 1. 为什么需要GPT

### 1.1 问题

- 有监督学习需要大量标注数据
- 单向（从左到右）生成是自然形式
- 语言模型可以无监督学习

### 1.2 GPT系列演进

| 版本 | 年份 | 参数量 | 数据量 | 关键突破 |
|------|------|--------|--------|----------|
| GPT-1 | 2018 | 117M | BookCorpus | 生成式预训练 |
| GPT-2 | 2019 | 1.5B | WebText | Zero-shot能力 |
| GPT-3 | 2020 | 175B | CommonCrawl | In-context learning |
| GPT-3.5 | 2022 | 175B | +指令 | InstructGPT/RLHF |
| GPT-4 | 2023 | ~1.8T(MoE) | 多模态 | 多模态+长上下文 |
| GPT-4o | 2024 | - | 多模态 | 原生多模态 |

## 2. GPT架构特点

### 2.1 解码器-only架构

**为什么只有解码器**：生成任务只需要解码器，简化架构。

**因果掩码**（Causal Masking）：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

其中 $M_{ij} = 0$ 如果 $i \geq j$（可以看到自己），$M_{ij} = -\infty$ 如果 $i < j$。

### 2.2 训练目标

$$\mathcal{L} = -\sum_{t} \log P(x_t \mid x_{<t})$$

### 2.3 In-Context Learning

**发现**：GPT-3可以在不更新参数的情况下，通过提示（prompt）中的示例学习新任务。

**为什么有效**：大量训练数据中隐式包含了"任务模式"。

## 3. Scaling Laws

### 3.1 Kaplan et al. (2020)

模型性能随参数量、数据量、计算量呈幂律关系：

$$L(N) \propto N^{-\alpha}, \quad L(D) \propto D^{-\beta}$$

### 3.2 Chinchilla (Hoffmann et al., 2022)

**关键发现**：GPT-3训练不足。对于175B模型，需要3.1T tokens而非300B tokens。

**计算最优**：模型参数和训练数据应等比例扩展。

## 4. 局限性与改进

| 局限 | 描述 | 改进方案 |
|------|------|----------|
| 事实错误 | 内容可能不准确 | RAG检索增强 |
| 幻觉 | 生成看似合理但错误的内容 | 验证链, 工具使用 |
| 推理不足 | 复杂推理易出错 | CoT, ToT, PoT |
| 长上下文 | 4K/8K限制 | 扩展到128K+ |

## 5. 在具身智能中的应用

- **RT-2**：基于PaLM/GPT的VLA模型
- **SayCan**：用LLM进行机器人任务规划
- **代码生成控制**：LLM生成机器人控制代码
- **自然语言界面**：用户用自然语言指挥机器人

## 6. 参考文献

1. Radford, A., et al. (2018). Improving language understanding by generative pre-training. *OpenAI*.
2. Radford, A., et al. (2019). Language models are unsupervised multitask learners. *OpenAI*.
3. Brown, T. B., et al. (2020). Language models are few-shot learners. *NeurIPS*.
4. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*.
5. Hoffmann, J., et al. (2022). Training compute-optimal large language models. *NeurIPS*.
