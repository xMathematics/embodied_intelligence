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

---

## 9. 高级预训练策略

### 9.1 ELECTRA：替换检测

**问题提出：**
MLM只预测15%的token，训练效率低。能否让所有token都参与训练？

**解决方案：**
ELECTRA（Efficiently Learning an Encoder that Classifies Token Replacements Accurately）使用替换检测任务。

**核心创新：**
1. **生成器**：小型MLM模型生成替换token
2. **判别器**：判断每个token是否被替换
3. **全token训练**：所有token都参与损失计算

**论文核心思想（Clark et al., 2020）：**
将MLM任务转换为二元分类任务，大幅提升训练效率。

**优势：**
- 训练效率高（所有token参与）
- 参数效率高（判别器可以更大）
- 性能优于MLM

**挑战：**
- 需要训练两个模型
- 生成器和判别器的平衡

**代码实现：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ELECTRAGenerator(nn.Module):
    """
    ELECTRA生成器
    
    使用MLM目标生成替换token。
    """
    
    def __init__(self, vocab_size, d_model=256, num_heads=4, 
                 num_layers=4, dim_feedforward=1024, max_len=512, dropout=0.1):
        super().__init__()
        
        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLM头
        self.mlm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + position_embeds)
        
        # 编码器
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        hidden_states = self.encoder(x, src_key_padding_mask=key_padding_mask)
        
        # MLM预测
        logits = self.mlm_head(hidden_states)
        
        return logits


class ELECTRADiscriminator(nn.Module):
    """
    ELECTRA判别器
    
    判断每个token是否被替换。
    """
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, 
                 num_layers=12, dim_feedforward=3072, max_len=512, dropout=0.1):
        super().__init__()
        
        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 判别头
        self.discriminator_head = nn.Linear(d_model, 1)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + position_embeds)
        
        # 编码器
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        hidden_states = self.encoder(x, src_key_padding_mask=key_padding_mask)
        
        # 判别预测
        logits = self.discriminator_head(hidden_states).squeeze(-1)
        
        return logits


class ELECTRA(nn.Module):
    """
    ELECTRA模型
    
    论文核心思想（Clark et al., 2020）：
    使用替换检测任务进行预训练，提升训练效率。
    
    训练流程：
    1. 生成器生成替换token
    2. 判别器判断token是否被替换
    3. 两个模型联合训练
    """
    
    def __init__(self, vocab_size, generator_config, discriminator_config):
        super().__init__()
        
        self.generator = ELECTRAGenerator(vocab_size, **generator_config)
        self.discriminator = ELECTRADiscriminator(vocab_size, **discriminator_config)
        self.vocab_size = vocab_size
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len) 原始token（用于生成器MLM损失）
        
        Returns:
            loss: 总损失
            generator_loss: 生成器损失
            discriminator_loss: 判别器损失
        """
        batch_size, seq_len = input_ids.shape
        
        # 生成器前向传播
        generator_logits = self.generator(input_ids, attention_mask)
        
        # 生成替换token
        with torch.no_grad():
            # 采样替换token
            probs = F.softmax(generator_logits, dim=-1)
            replaced_ids = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1)
            replaced_ids = replaced_ids.view(batch_size, seq_len)
            
            # 随机选择15%的位置进行替换
            mask_ratio = 0.15
            mask = torch.rand(batch_size, seq_len, device=input_ids.device) < mask_ratio
            
            # 创建替换后的输入
            corrupted_input_ids = input_ids.clone()
            corrupted_input_ids[mask] = replaced_ids[mask]
            
            # 创建判别器标签（1表示被替换，0表示原始）
            discriminator_labels = mask.long()
        
        # 判别器前向传播
        discriminator_logits = self.discriminator(corrupted_input_ids, attention_mask)
        
        # 计算损失
        if labels is not None:
            # 生成器MLM损失（只在掩码位置计算）
            masked_positions = mask
            generator_loss = F.cross_entropy(
                generator_logits[masked_positions],
                input_ids[masked_positions]
            )
        else:
            generator_loss = 0.0
        
        # 判别器二元分类损失
        discriminator_loss = F.binary_cross_entropy_with_logits(
            discriminator_logits,
            discriminator_labels.float()
        )
        
        # 总损失
        loss = generator_loss + 50 * discriminator_loss  # 判别器损失权重更高
        
        return loss, generator_loss, discriminator_loss
```

### 9.2 SpanBERT：片段掩码

**问题提出：**
MLM随机掩码单个token，能否更好地建模连续片段？

**解决方案：**
SpanBERT掩码连续的token片段，增强上下文建模能力。

**核心创新：**
1. **片段掩码**：随机选择连续片段进行掩码
2. **边界目标**：预测片段的边界token
3. **随机起始**：从片段中随机选择起始位置

**论文核心思想（Joshi et al., 2020）：**
通过掩码连续片段，模型学习更好的上下文表示。

**代码实现：**

```python
import random
import torch

class SpanMasking:
    """
    Span掩码策略
    
    随机选择连续片段进行掩码。
    """
    
    def __init__(self, mask_ratio=0.15, max_span_length=10, vocab_size=30522):
        self.mask_ratio = mask_ratio
        self.max_span_length = max_span_length
        self.vocab_size = vocab_size
    
    def mask_spans(self, tokens):
        """
        掩码连续片段
        
        Args:
            tokens: 原始token序列
        
        Returns:
            masked_tokens: 掩码后的token序列
            labels: 原始token
            span_boundaries: 片段边界
        """
        masked_tokens = tokens.copy()
        labels = [-100] * len(tokens)
        span_boundaries = []
        
        num_tokens_to_mask = int(len(tokens) * self.mask_ratio)
        num_masked = 0
        
        while num_masked < num_tokens_to_mask:
            # 随机选择起始位置
            start = random.randint(0, len(tokens) - 1)
            
            # 随机选择片段长度
            span_length = random.randint(1, min(self.max_span_length, len(tokens) - start))
            
            # 掩码片段
            for i in range(start, min(start + span_length, len(tokens))):
                if masked_tokens[i] != -100:
                    labels[i] = tokens[i]
                    masked_tokens[i] = 103  # [MASK] token
                    num_masked += 1
            
            span_boundaries.append((start, min(start + span_length, len(tokens))))
        
        return masked_tokens, labels, span_boundaries

# 测试
tokens = [101, 2023, 2003, 1037, 1010, 2026, 1029, 102, 0, 0, 0, 0]
span_masker = SpanMasking()
masked, labels, boundaries = span_masker.mask_spans(tokens)
print(f"原始tokens: {tokens}")
print(f"掩码后tokens: {masked}")
print(f"片段边界: {boundaries}")
```

### 9.3 DeBERTa：解耦注意力

**问题提出：**
标准注意力将内容和位置混合在一起，能否解耦它们？

**解决方案：**
DeBERTa（Decoding-enhanced BERT with disentangled attention）将内容和位置解耦。

**核心创新：**
1. **解耦注意力**：分别计算内容注意力和位置注意力
2. **增强解码器掩码**：相对位置编码的改进
3. **尺度不变性**：归一化注意力权重

**论文核心思想（He et al., 2020）：**
通过解耦内容和位置，提升模型性能。

**代码实现：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DisentangledAttention(nn.Module):
    """
    解耦注意力
    
    分别计算内容注意力和位置注意力。
    """
    
    def __init__(self, d_model, num_heads, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 内容投影
        self.W_q_content = nn.Linear(d_model, d_model)
        self.W_k_content = nn.Linear(d_model, d_model)
        self.W_v_content = nn.Linear(d_model, d_model)
        
        # 位置投影
        self.W_q_position = nn.Linear(d_model, d_model)
        self.W_k_position = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
        
        # 相对位置嵌入
        self.relative_position_embedding = nn.Embedding(2 * max_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, content, position):
        """
        Args:
            content: (batch_size, seq_len, d_model) 内容嵌入
            position: (batch_size, seq_len, d_model) 位置嵌入
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = content.shape
        
        # 内容投影
        Q_content = self.W_q_content(content).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K_content = self.W_k_content(content).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V_content = self.W_v_content(content).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 位置投影
        Q_position = self.W_q_position(position).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K_position = self.W_k_position(position).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 内容注意力分数
        content_scores = torch.matmul(Q_content, K_content.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # 位置注意力分数
        position_scores = torch.matmul(Q_position, K_position.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # 相对位置分数
        relative_position_ids = torch.arange(seq_len, device=content.device).unsqueeze(0) - torch.arange(seq_len, device=content.device).unsqueeze(1)
        relative_position_ids = relative_position_ids + seq_len  # 偏移到正数
        relative_position_embeds = self.relative_position_embedding(relative_position_ids)
        relative_position_embeds = relative_position_embeds.view(seq_len, seq_len, self.num_heads, self.d_k).permute(2, 0, 1, 3)
        
        relative_scores = torch.einsum('bhqd,qkhd->bhqk', Q_position, relative_position_embeds) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # 总注意力分数
        scores = content_scores + position_scores + relative_scores
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, V_content)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.W_o(output)
        
        return output
```

### 9.4 T5：Span Corruption

**问题提出：**
能否用统一的文本到文本框架进行预训练？

**解决方案：**
T5使用Span Corruption任务，将所有任务都转换为文本生成。

**核心创新：**
1. **Span Corruption**：掩码连续片段并生成
2. **统一框架**：所有任务都是文本到文本
3. **大规模预训练**：在C4数据集上训练

**论文核心思想（Raffel et al., 2020）：**
将预训练和下游任务统一为文本生成任务。

**代码实现：**

```python
import random
import torch

class SpanCorruption:
    """
    Span Corruption
    
    类似MLM，但掩码连续片段并生成。
    """
    
    def __init__(self, mask_ratio=0.15, max_span_length=10, sentinel_token_id=32000):
        self.mask_ratio = mask_ratio
        self.max_span_length = max_span_length
        self.sentinel_token_id = sentinel_token_id
        self.next_sentinel_id = sentinel_token_id
    
    def corrupt_spans(self, tokens):
        """
        腐蚀连续片段
        
        Args:
            tokens: 原始token序列
        
        Returns:
            corrupted_tokens: 腐蚀后的token序列
            target_tokens: 目标token序列（被腐蚀的片段）
        """
        corrupted_tokens = tokens.copy()
        target_tokens = []
        
        num_tokens_to_mask = int(len(tokens) * self.mask_ratio)
        num_masked = 0
        
        while num_masked < num_tokens_to_mask:
            # 随机选择起始位置
            start = random.randint(0, len(tokens) - 1)
            
            # 随机选择片段长度
            span_length = random.randint(1, min(self.max_span_length, len(tokens) - start))
            
            # 腐蚀片段
            span = tokens[start:min(start + span_length, len(tokens))]
            target_tokens.extend(span)
            target_tokens.append(self.next_sentinel_id)
            
            # 替换为sentinel token
            for i in range(start, min(start + span_length, len(tokens))):
                if corrupted_tokens[i] != -100:
                    corrupted_tokens[i] = self.next_sentinel_id
                    num_masked += 1
            
            self.next_sentinel_id += 1
        
        return corrupted_tokens, target_tokens

# 测试
tokens = [101, 2023, 2003, 1037, 1010, 2026, 1029, 102, 0, 0, 0, 0]
span_corruptor = SpanCorruption()
corrupted, target = span_corruptor.corrupt_spans(tokens)
print(f"原始tokens: {tokens}")
print(f"腐蚀后tokens: {corrupted}")
print(f"目标tokens: {target}")
```

### 9.5 BART：去噪自编码器

**问题提出：**
能否结合编码器-解码器进行预训练？

**解决方案：**
BART（Bidirectional and Auto-Regressive Transformers）使用去噪自编码器目标。

**核心创新：**
1. **编码器-解码器**：双向编码器 + 自回归解码器
2. **多种噪声**：token掩码、token删除、句子打乱等
3. **生成能力**：强大的文本生成能力

**论文核心思想（Lewis et al., 2019）：**
使用去噪自编码器进行预训练，结合理解和生成能力。

**代码实现：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BART(nn.Module):
    """
    BART模型
    
    论文核心思想（Lewis et al., 2019）：
    使用去噪自编码器进行预训练。
    
    噪声类型：
    - Token掩码
    - Token删除
    - 文本填充
    - 句子打乱
    """
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, 
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=3072, max_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 共享嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 语言模型头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
    
    def add_noise(self, input_ids, noise_type='mask'):
        """
        添加噪声
        
        Args:
            input_ids: (batch_size, seq_len)
            noise_type: 噪声类型
        
        Returns:
            noisy_input_ids: 带噪声的输入
            labels: 原始token
        """
        batch_size, seq_len = input_ids.shape
        noisy_input_ids = input_ids.clone()
        labels = input_ids.clone()
        
        if noise_type == 'mask':
            # Token掩码
            mask_ratio = 0.15
            mask = torch.rand(batch_size, seq_len, device=input_ids.device) < mask_ratio
            noisy_input_ids[mask] = 0  # [PAD] token
        
        elif noise_type == 'delete':
            # Token删除
            delete_ratio = 0.15
            delete_mask = torch.rand(batch_size, seq_len, device=input_ids.device) < delete_ratio
            noisy_input_ids[delete_mask] = 0
        
        elif noise_type == 'permute':
            # 句子打乱
            for i in range(batch_size):
                seq = input_ids[i].tolist()
                random.shuffle(seq)
                noisy_input_ids[i] = torch.tensor(seq, device=input_ids.device)
        
        return noisy_input_ids, labels
    
    def forward(self, noisy_input_ids, target_ids, attention_mask=None):
        """
        Args:
            noisy_input_ids: (batch_size, seq_len) 带噪声的输入
            target_ids: (batch_size, seq_len) 目标序列
            attention_mask: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # 编码器
        encoder_embeds = self.embedding(noisy_input_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        if attention_mask is not None:
            encoder_key_padding_mask = (attention_mask == 0)
        else:
            encoder_key_padding_mask = None
        
        memory = self.encoder(encoder_embeds, src_key_padding_mask=encoder_key_padding_mask)
        
        # 解码器
        decoder_embeds = self.embedding(target_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # 因果掩码
        tgt_seq_len = target_ids.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=target_ids.device), diagonal=1).bool()
        
        output = self.decoder(decoder_embeds, memory, tgt_mask=tgt_mask)
        
        # 语言模型头
        logits = self.lm_head(output)
        
        return logits
```

---

## 10. 预训练数据增强

### 10.1 数据增强策略

| 策略 | 说明 | 效果 |
|------|------|------|
| **回译** | 翻译后翻译回原语言 | 增加多样性 |
| **同义词替换** | 用同义词替换词 | 增强鲁棒性 |
| **随机删除** | 随机删除词 | 增强泛化 |
| **随机交换** | 交换相邻词 | 增强鲁棒性 |

**代码实现：**

```python
import random

class DataAugmentation:
    """
    数据增强
    
    通过多种策略增强训练数据。
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def synonym_replacement(self, text, n=1):
        """
        同义词替换
        
        Args:
            text: 原始文本
            n: 替换次数
        
        Returns:
            augmented_text: 增强后的文本
        """
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            idx = random.randint(0, len(words) - 1)
            # 这里应该使用同义词库，简化起见随机替换
            new_words[idx] = words[random.randint(0, len(words) - 1)]
        
        return ' '.join(new_words)
    
    def random_deletion(self, text, p=0.1):
        """
        随机删除
        
        Args:
            text: 原始文本
            p: 删除概率
        
        Returns:
            augmented_text: 增强后的文本
        """
        words = text.split()
        new_words = [word for word in words if random.random() > p]
        
        if len(new_words) == 0:
            return text
        
        return ' '.join(new_words)
    
    def random_swap(self, text, n=1):
        """
        随机交换
        
        Args:
            text: 原始文本
            n: 交换次数
        
        Returns:
            augmented_text: 增强后的文本
        """
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) >= 2:
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
```

### 10.2 数据质量评估

**代码实现：**

```python
import re
from collections import Counter

class DataQualityAssessment:
    """
    数据质量评估
    
    评估预训练数据的质量。
    """
    
    def __init__(self):
        self.english_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i'])
    
    def assess_quality(self, text):
        """
        评估文本质量
        
        Args:
            text: 待评估文本
        
        Returns:
            quality_score: 质量分数（0-1）
        """
        score = 0.0
        
        # 1. 长度检查
        if 10 <= len(text.split()) <= 1000:
            score += 0.2
        
        # 2. 英语单词比例
        words = text.lower().split()
        english_ratio = sum(1 for word in words if word in self.english_words) / len(words)
        score += english_ratio * 0.3
        
        # 3. 特殊字符比例
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', text))
        special_ratio = special_chars / len(text)
        if special_ratio < 0.1:
            score += 0.2
        
        # 4. 重复字符检查
        max_repeat = max(len(list(g)) for k, g in itertools.groupby(text))
        if max_repeat < 5:
            score += 0.2
        
        # 5. 词频多样性
        word_freq = Counter(words)
        unique_ratio = len(word_freq) / len(words)
        score += unique_ratio * 0.1
        
        return score
```

---

## 11. 预训练优化

### 11.1 学习率调度

**代码实现：**

```python
import math

class PretrainingScheduler:
    """
    预训练学习率调度器
    
    结合warmup和余弦衰减。
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, peak_lr=1e-4):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.peak_lr = peak_lr
        self.current_step = 0
    
    def step(self):
        """更新学习率"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Warmup阶段：线性增加
            lr = self.peak_lr * self.current_step / self.warmup_steps
        else:
            # 衰减阶段：余弦衰减
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
```

### 11.2 梯度累积

**代码实现：**

```python
class GradientAccumulator:
    """
    梯度累积器
    
    通过累积多个小批量的梯度来模拟大批量训练。
    """
    
    def __init__(self, model, optimizer, accumulation_steps):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def step(self, loss):
        """执行训练步骤"""
        # 计算梯度
        loss = loss / self.accumulation_steps
        loss.backward()
        
        self.current_step += 1
        
        # 累积足够后更新参数
        if self.current_step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
```

---

## 12. 总结

预训练是现代大语言模型的核心技术。通过在大规模无标签数据上预训练，模型学习到丰富的语言知识和世界知识。

**关键要点：**

1. **预训练目标**：MLM、CLM、对比学习等
2. **预训练数据**：数据来源、预处理、质量评估
3. **预训练策略**：ELECTRA、SpanBERT、DeBERTa等
4. **预训练优化**：学习率调度、梯度累积等

**未来方向：**

- 更高效的预训练方法
- 更好的数据利用
- 更强的跨任务泛化
- 更低的计算成本

---

## 参考文献

### 核心论文

1. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers". NAACL.
2. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training". OpenAI.
3. Clark, K., et al. (2020). "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators". ICLR.
4. Joshi, M., et al. (2020). "SpanBERT: Improving Pre-training by Representing and Predicting Spans". TACL.
5. He, J., et al. (2020). "DeBERTa: Decoding-enhanced BERT with Disentangled Attention". ICLR.
6. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". JMLR.
7. Lewis, M., et al. (2019). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension". EMNLP.

### 预训练技术

1. Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach". arXiv.
2. Lan, Z., et al. (2019). "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations". ICLR.
3. Yang, Z., et al. (2019). "XLNet: Generalized Autoregressive Pretraining for Language Understanding". NeurIPS.
