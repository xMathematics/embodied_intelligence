# 1.2 预训练策略

## 目录

- [1. 预训练概述](#1-预训练概述)
- [2. 掩码语言模型（MLM）](#2-掩码语言模型mlm)
- [3. 因果语言模型（CLM）](#3-因果语言模型clm)
- [4. 对比学习](#4-对比学习)
- [5. 预训练数据](#5-预训练数据)
- [6. 预训练策略对比](#6-预训练策略对比)
- [7. 未解决的问题与未来方向](#7-未解决的问题与未来方向)
- [8. 实践练习](#8-实践练习)

---

## 1. 预训练概述

### 1.1 预训练-微调范式

预训练语言模型的核心思想是：
1. **预训练阶段**：在大规模无标签文本上训练通用语言模型
2. **微调阶段**：在特定任务的标注数据上进行微调

```
预训练数据 → [预训练] → 通用语言模型 → [微调] → 任务特定模型
```

### 1.2 预训练的动机

| 问题 | 传统方法 | 预训练方法 |
|------|---------|-----------|
| **数据效率** | 需要大量标注数据 | 利用海量无标签数据 |
| **泛化能力** | 任务特定，泛化差 | 通用知识，泛化好 |
| **迁移能力** | 难以跨任务迁移 | 容易迁移到新任务 |

### 1.3 预训练目标的演变

| 阶段 | 目标类型 | 代表模型 |
|------|---------|---------|
| 第一代 | 词级目标 | Word2Vec, GloVe |
| 第二代 | 句子级目标 | ELMo, ULMFiT |
| 第三代 | 上下文目标 | BERT, GPT |

---

## 2. 掩码语言模型（MLM）

### 2.1 论文背景

**论文**：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., NAACL 2019)

**问题提出**：
- 传统语言模型（如ELMo）是单向的，只能利用左上下文或右上下文
- 双向上下文对于理解词义消歧、指代消解等任务至关重要

### 2.2 MLM的核心思想

随机掩盖输入序列中的部分token，让模型预测被掩盖的token：

$$\text{Loss} = -\sum_{i \in \text{masked}} \log p(x_i | x_{\backslash i})$$

### 2.3 掩码策略

| 策略 | 比例 | 说明 |
|------|------|------|
| **[MASK]替换** | 80% | 用特殊token [MASK]替换 |
| **随机替换** | 10% | 用随机词替换 |
| **保持不变** | 10% | 保持原词不变 |

**为什么需要随机替换和保持不变？**
- 防止模型过度依赖[MASK]符号
- 增强模型的鲁棒性
- 避免预训练和微调阶段的差异

### 2.4 MLM的数学表达

给定输入序列 $x = [x_1, x_2, ..., x_n]$：

1. 随机选择15%的token进行掩码
2. 对于被选中的token $x_i$：
   - 80%概率：$x_i \leftarrow [MASK]$
   - 10%概率：$x_i \leftarrow \text{random token}$
   - 10%概率：$x_i \leftarrow x_i$（保持不变）
3. 模型预测原始token：
   $$p(x_i | x_1, ..., x_{i-1}, x_{i+1}, ..., x_n)$$

### 2.5 MLM解决的问题

| 问题 | 传统方法的局限 | MLM的解决方案 |
|------|--------------|-------------|
| **双向依赖** | 单向语言模型只能利用单方向上下文 | 双向Transformer + 掩码机制 |
| **上下文理解** | 无法处理歧义词 | 结合左右上下文进行预测 |
| **预训练-微调差异** | 训练目标与下游任务不匹配 | 通用的填空任务适配多种任务 |

### 2.6 MLM的优缺点

| 优点 | 说明 |
|------|------|
| **双向上下文** | 充分利用左右上下文信息 |
| **任务适配性** | 填空任务天然适配多种NLP任务 |
| **模型鲁棒性** | 随机替换策略增强泛化能力 |

| 缺点 | 说明 |
|------|------|
| **预训练-微调差异** | [MASK]符号只在预训练中出现 |
| **计算效率** | 只预测15%的token，效率较低 |
| **位置偏差** | 模型可能学会利用位置信息而非上下文 |

### 2.7 改进方向

| 改进方法 | 说明 | 代表工作 |
|----------|------|---------|
| **ERNIE** | 实体级掩码，增强实体理解 | ERNIE 1.0/2.0 |
| **SpanBERT** | 连续片段掩码，增强上下文建模 | SpanBERT |
| **ELECTRA** | 替换检测任务，提高训练效率 | ELECTRA |

---

## 3. 因果语言模型（CLM）

### 3.1 论文背景

**论文**：Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)

**问题提出**：
- 传统预训练方法主要关注理解任务（如分类、问答）
- 生成任务（如文本生成、摘要）需要不同的训练目标
- 自回归生成是许多NLP任务的核心

### 3.2 CLM的核心思想

给定前序token，预测下一个token：

$$p(x) = \prod_{i=1}^n p(x_i | x_1, ..., x_{i-1})$$

### 3.3 CLM的数学表达

$$\text{Loss} = -\sum_{i=1}^n \log p(x_i | x_1, ..., x_{i-1})$$

### 3.4 Transformer在CLM中的应用

CLM使用Transformer解码器结构，带有因果掩码：

```
因果掩码矩阵（3x3示例）:
[[0, -inf, -inf]
 [0,    0, -inf]
 [0,    0,    0]]
```

掩码确保预测第i个token时只能看到前i-1个token。

### 3.5 CLM解决的问题

| 问题 | 传统方法的局限 | CLM的解决方案 |
|------|--------------|-------------|
| **生成能力** | MLM擅长理解但不擅长生成 | 自回归目标天然适合生成 |
| **长文本生成** | 难以建模长距离依赖 | Transformer架构支持长序列 |
| **自然语言流畅性** | 生成文本可能不连贯 | 逐词生成保证流畅性 |

### 3.6 CLM的优缺点

| 优点 | 说明 |
|------|------|
| **生成能力强** | 天然支持自回归生成 |
| **预训练-微调一致** | 微调时不需要特殊处理 |
| **效率高** | 每个token都参与损失计算 |

| 缺点 | 说明 |
|------|------|
| **单向上下文** | 只能利用左上下文 |
| **推理速度慢** | 自回归解码无法并行 |
| **曝光偏差** | 训练时看到真实token，推理时看到生成的token |

### 3.7 GPT系列的演进

| 模型 | 关键改进 |
|------|---------|
| **GPT-1** | 首次将Transformer用于CLM预训练 |
| **GPT-2** | 更大规模、更多数据、无监督微调 |
| **GPT-3** | 千亿参数、few-shot学习能力 |
| **GPT-4** | 多模态、更强的推理能力 |

---

## 4. 对比学习

### 4.1 论文背景

**论文**：SimCLR: A Simple Framework for Contrastive Learning of Visual Representations (Chen et al., ICML 2020)

虽然SimCLR是为视觉领域设计的，但其思想被广泛应用于NLP领域。

### 4.2 对比学习的核心思想

通过比较相似和不相似的样本对来学习有用的表示：

$$\text{Loss} = -\log \frac{e^{\text{sim}(x_i, x_i^+) / \tau}}{\sum_{j=1}^N e^{\text{sim}(x_i, x_j^-) / \tau}}$$

其中：
- $x_i^+$：$x_i$的正样本（增强后的同一样本）
- $x_j^-$：$x_i$的负样本（其他样本）
- $\tau$：温度参数
- $\text{sim}$：相似度度量（通常是余弦相似度）

### 4.3 NLP中的对比学习应用

| 应用场景 | 方法 | 代表工作 |
|----------|------|---------|
| **句嵌入** | 对比句子级表示 | SimCSE, Sentence-BERT |
| **词嵌入** | 对比词级表示 | ConSERT, InfoNCE |
| **跨模态学习** | 对比不同模态的表示 | CLIP, ALIGN |

### 4.4 SimCSE示例

**SimCSE**（Simple Contrastive Learning of Sentence Embeddings）：

1. **正样本对**：同一句子的两次不同dropout增强
2. **负样本对**：batch中的其他句子
3. **损失函数**：InfoNCE

```python
# SimCSE的核心思想
def simcse_loss(embeddings, temperature=0.05):
    # embeddings: (batch_size, hidden_size)
    batch_size = embeddings.size(0)
    
    # 计算相似度矩阵
    sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), 
                                     embeddings.unsqueeze(0), dim=2)
    
    # 掩码：对角线是自己，正样本是batch_size+i
    mask = torch.eye(batch_size * 2, device=embeddings.device)
    
    # InfoNCE损失
    logits = sim_matrix / temperature
    logits = logits - mask * 1e9  # 排除自己
    
    labels = torch.arange(batch_size, device=embeddings.device)
    loss = F.cross_entropy(logits[:batch_size], labels)
    loss += F.cross_entropy(logits[batch_size:], labels)
    
    return loss / 2
```

### 4.5 对比学习解决的问题

| 问题 | 传统方法的局限 | 对比学习的解决方案 |
|------|--------------|-----------------|
| **表示质量** | 依赖标注数据 | 无监督学习高质量表示 |
| **样本效率** | 需要大量标注 | 利用无标签数据 |
| **泛化能力** | 容易过拟合 | 对比学习增强鲁棒性 |

### 4.6 对比学习的优缺点

| 优点 | 说明 |
|------|------|
| **无监督学习** | 不需要标注数据 |
| **表示质量高** | 学习到有意义的特征 |
| **通用性强** | 适用于多种任务 |

| 缺点 | 说明 |
|------|------|
| **计算成本高** | 需要大batch和负样本 |
| **对比难度** | 设计好的正负样本对很关键 |
| **理论理解不足** | 为什么有效还不完全清楚 |

---

## 5. 预训练数据

### 5.1 数据来源

| 来源类型 | 示例 | 特点 |
|---------|------|------|
| **网页文本** | Common Crawl, Wikipedia | 规模大、噪声多 |
| **书籍** | BookCorpus, Project Gutenberg | 质量高、语言规范 |
| **新闻** | News Crawl | 时效性强 |
| **代码** | GitHub | 特定领域数据 |
| **多语言** | XLM-R数据 | 支持多种语言 |

### 5.2 数据预处理

| 步骤 | 说明 | 工具 |
|------|------|------|
| **清洗** | 去除噪声、重复内容 | Python脚本 |
| **分词** | 子词切分 | BPE, WordPiece |
| **过滤** | 去除低质量内容 | 质量评分 |
| **去重** | 去除重复文档 | MinHash, SimHash |

### 5.3 数据规模与模型性能

| 模型 | 数据量 | 参数 | 性能 |
|------|-------|------|------|
| BERT-base | 16GB | 110M | GLUE 82.1 |
| GPT-2 | 40GB | 1.5B | 强生成能力 |
| GPT-3 | 500GB | 175B | 强few-shot能力 |
| LLaMA-2 | 2T tokens | 70B | 开源SOTA |

### 5.4 数据质量的重要性

| 数据问题 | 影响 | 解决方案 |
|---------|------|---------|
| **噪声** | 模型学习错误信息 | 数据清洗、过滤 |
| **偏见** | 模型产生偏见输出 | 偏见检测、去偏 |
| **重复** | 过拟合、泛化差 | 去重处理 |
| **覆盖不全** | 领域适应性差 | 多样化数据来源 |

---

## 6. 预训练策略对比

### 6.1 MLM vs CLM

| 维度 | MLM | CLM |
|------|-----|-----|
| **上下文方向** | 双向 | 单向 |
| **核心能力** | 理解 | 生成 |
| **预训练-微调差异** | 有差异（[MASK]） | 无差异 |
| **计算效率** | 较低（只预测15%） | 较高（全部预测） |
| **代表模型** | BERT, RoBERTa | GPT, LLaMA |

### 6.2 混合目标

许多模型结合了多种预训练目标：

| 模型 | 目标组合 |
|------|---------|
| **T5** | Span Corruption（类似MLM）+ CLM |
| **UniLM** | MLM + CLM（根据任务切换） |
| **BERT-GPT** | 双向编码 + 单向解码 |
| **XLNet** | Permutation Language Modeling（排列语言建模） |

**XLNet深度解析**：[XLNet论文阅读笔记](papers/xlnet.md)

---

## 7. 未解决的问题与未来方向

### 7.1 当前挑战

| 问题 | 描述 |
|------|------|
| **数据效率** | 预训练需要海量数据 |
| **计算成本** | 训练大模型成本极高 |
| **预训练-微调差距** | 预训练目标与下游任务不完全匹配 |
| **长序列处理** | Transformer对长序列效率低 |
| **可解释性** | 预训练知识的存储和调用机制不透明 |
| **偏见与公平** | 模型可能学习数据中的偏见 |

### 7.2 未来研究方向

| 方向 | 说明 |
|------|------|
| **高效预训练** | 减少数据和计算需求 |
| **结构化预训练** | 结合知识图谱等结构化信息 |
| **多模态预训练** | 融合文本、图像、语音等 |
| **持续预训练** | 模型上线后继续学习 |
| **可控预训练** | 控制模型学习的知识 |

---

## 8. 实践练习

### 练习1：理解MLM掩码机制

```python
import random

def mlm_mask(tokens, mask_token='[MASK]', vocab_size=30522):
    """
    实现MLM掩码策略
    
    参数:
        tokens: 原始token序列
        mask_token: 掩码token
        vocab_size: 词汇表大小
    
    返回:
        masked_tokens: 掩码后的token序列
        labels: 原始token（用于计算损失）
    """
    masked_tokens = tokens.copy()
    labels = [-100] * len(tokens)  # -100表示不计算损失
    
    for i in range(len(tokens)):
        # 15%概率进行掩码
        if random.random() < 0.15:
            labels[i] = tokens[i]  # 记录原始token
            
            rand = random.random()
            if rand < 0.8:
                # 80%概率替换为[MASK]
                masked_tokens[i] = mask_token
            elif rand < 0.9:
                # 10%概率替换为随机token
                masked_tokens[i] = random.randint(0, vocab_size - 1)
            # 10%概率保持不变
        
    return masked_tokens, labels

# 测试
tokens = [101, 2023, 2003, 1037, 1010, 2026, 1029, 102]  # 示例token序列
masked, labels = mlm_mask(tokens)
print(f"原始tokens: {tokens}")
print(f"掩码后tokens: {masked}")
print(f"标签: {labels}")
```

### 练习2：理解CLM自回归预测

```python
import torch
import torch.nn.functional as F

def clm_loss(logits, labels):
    """
    计算CLM损失
    
    参数:
        logits: 模型输出logits, shape: (batch, seq_len, vocab_size)
        labels: 真实标签, shape: (batch, seq_len)
    
    返回:
        loss: 平均损失
    """
    # CLM使用前一个token预测后一个token
    # 输入: x_1, x_2, ..., x_{n-1}
    # 目标: x_2, x_3, ..., x_n
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))
    
    return loss

# 测试
batch_size = 2
seq_len = 5
vocab_size = 10000

logits = torch.randn(batch_size, seq_len, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))

loss = clm_loss(logits, labels)
print(f"CLM损失: {loss.item():.4f}")
```

### 练习3：对比学习损失实现

```python
import torch
import torch.nn.functional as F

def info_nce_loss(embeddings, temperature=0.05):
    """
    实现InfoNCE对比损失
    
    参数:
        embeddings: 样本嵌入, shape: (batch_size * 2, hidden_size)
                    前batch_size是原始样本，后batch_size是增强样本
        temperature: 温度参数
    
    返回:
        loss: InfoNCE损失
    """
    batch_size = embeddings.size(0) // 2
    
    # 计算余弦相似度矩阵
    sim = F.cosine_similarity(embeddings.unsqueeze(1), 
                              embeddings.unsqueeze(0), dim=2)
    
    # 正样本对：(i, i+batch_size) 和 (i+batch_size, i)
    positive_mask = torch.zeros_like(sim)
    for i in range(batch_size):
        positive_mask[i, i + batch_size] = 1
        positive_mask[i + batch_size, i] = 1
    
    # 负样本：除了自己和正样本之外的所有样本
    negative_mask = 1 - positive_mask - torch.eye(embeddings.size(0))
    
    # 计算logits
    logits = sim / temperature
    
    # 只保留正样本和负样本的logits
    pos_logits = logits * positive_mask
    neg_logits = logits * negative_mask
    
    # InfoNCE损失
    pos_sum = torch.logsumexp(pos_logits, dim=1)
    neg_sum = torch.logsumexp(neg_logits, dim=1)
    loss = -(pos_sum - neg_sum).mean()
    
    return loss

# 测试
batch_size = 32
hidden_size = 768

# 模拟嵌入：前32是原始，后32是增强
embeddings = torch.randn(batch_size * 2, hidden_size)

loss = info_nce_loss(embeddings)
print(f"InfoNCE损失: {loss.item():.4f}")
```

---

**论文引用**：
1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
2. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.
3. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. In International conference on machine learning (pp. 1597-1607). PMLR.

**下一节**：[著名LLM模型](03-famous-models.md)
