# 4.2 BERT编码器家族

## 1. 为什么需要BERT

### 1.1 问题

- GPT是单向的（从左到右），无法利用双向上下文
- ELMo是双向LSTM拼接，不是真正的联合双向

### 1.2 BERT的提出

**论文**：Devlin et al., 2018 — NAACL 2019

**核心创新**：
- **Masked Language Model（MLM）**：随机遮盖15%的token并预测
- **Next Sentence Prediction（NSP）**：预测两个句子是否连续
- 真正双向的上下文理解

## 2. BERT架构

### 2.1 输入表示

$$\text{Input} = \text{Token Embedding} + \text{Segment Embedding} + \text{Position Embedding}$$

### 2.2 预训练目标

**MLM**：遮盖15% token → 预测遮盖的词（并非所有token都要预测，只预测被mask的80%+随机替换10%+不变10%）

**NSP**：二分类判断句子关系。

### 2.3 为什么设计如此

- MLM让模型从双向上下文理解语义
- NSP让模型理解句子间关系
- 两个任务互补

## 3. BERT的局限

| 局限 | 原因 | 后续改进 |
|------|------|----------|
| 预训练-微调不一致 | 预训练有[MASK]，微调没有 | 不做MASK的预训练 |
| NSP任务太简单 | 单句和下一句容易区分 | RoBERTa移除NSP |
| 模型容量有限 | 只有编码器 | 编码器-解码器如T5 |
| 长序列受限 | 位置编码512 | 相对位置编码 |

## 4. 重要变体

| 模型 | 改进点 | 效果 |
|------|--------|------|
| **RoBERTa** | 动态MLM、移除NSP、更大batch | 全面超越BERT |
| **ALBERT** | 跨层参数共享、分解嵌入矩阵 | 参数量减少18x |
| **ELECTRA** | 判别式RTD替代生成式MLM | 效率更高 |
| **DeBERTa** | 解耦位置-内容注意力、增强掩码解码 | 更强 |
| **DistilBERT** | 知识蒸馏 | 速度提升60% |

## 5. 在具身智能中的应用

- **语言理解**：BERT理解自然语言指令
- **语义导航**：BERT将自然语言编码为导航目标
- **人机交互**：BERT理解操作请求并生成响应
- **任务规划**：BERT将复杂指令分解为子任务

## 6. 参考文献

1. Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
2. Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*.
3. Lan, Z., et al. (2019). ALBERT: A lite BERT for self-supervised learning of language representations. *ICLR*.
4. Clark, K., et al. (2020). ELECTRA: Pre-training text encoders as discriminators rather than generators. *ICLR*.
5. He, P., et al. (2020). DeBERTa: Decoding-enhanced BERT with disentangled attention. *ICLR*.
