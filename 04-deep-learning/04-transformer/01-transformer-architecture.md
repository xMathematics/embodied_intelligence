# 3.4 Transformer架构

## 目录

- [1. Transformer概述](#1-transformer概述)
- [2. 自注意力机制](#2-自注意力机制)
- [3. Transformer架构详解](#3-transformer架构详解)
- [4. BERT](#4-bert)
- [5. GPT](#5-gpt)
- [6. 重要论文详解](#6-重要论文详解)
- [7. 实践练习](#7-实践练习)

---

## 1. Transformer概述

### 1.1 什么是Transformer？

**Transformer**：基于自注意力机制的序列建模架构。

**核心特点**：
- 完全依赖注意力机制
- 并行计算
- 长距离依赖建模

### 1.2 Transformer的优势

| 方面 | RNN | Transformer |
|------|-----|------------|
| 并行性 | 差（顺序依赖） | 好（无顺序依赖） |
| 长距离依赖 | 差（梯度消失） | 好（直接连接） |
| 计算效率 | 低 | 高 |
| 建模能力 | 受限 | 强 |

---

## 2. 自注意力机制

### 2.1 Scaled Dot-Product Attention

**公式**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

**缩放因子**：
- $\sqrt{d_k}$ 防止内积过大
- 保持梯度稳定

### 2.2 多头注意力

**机制**：
- 使用多个注意力头
- 每个头学习不同的表示
- 拼接输出

**公式**：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h) W^O$$
$$\text{其中 } head_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### 2.3 位置编码

**问题**：Transformer没有顺序信息

**解决方案**：
- 加入位置编码
- 使用正弦/余弦函数

**公式**：
$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

---

## 3. Transformer架构详解

### 3.1 整体架构

```
输入 → Embedding + Positional Encoding → Encoder → Decoder → Linear → Softmax → 输出
```

### 3.2 Encoder

**组成**：
- N个相同的层
- 每层包含：
  - Multi-Head Self-Attention
  - Feed-Forward Network
  - Layer Normalization
  - Residual Connection

### 3.3 Decoder

**组成**：
- N个相同的层
- 每层包含：
  - Masked Multi-Head Self-Attention（防止未来信息泄露）
  - Multi-Head Attention（Encoder-Decoder Attention）
  - Feed-Forward Network
  - Layer Normalization
  - Residual Connection

### 3.4 架构参数

| 参数 | 说明 |
|------|------|
| $d_{\text{model}}$ | 模型维度 |
| $n_{\text{head}}$ | 注意力头数 |
| $d_{\text{ff}}$ | 前馈网络维度 |
| $n_{\text{layers}}$ | 层数 |

---

## 4. BERT

### 4.1 Bidirectional Encoder Representations from Transformers

**特点**：
- 双向预训练
- Masked Language Model（MLM）
- Next Sentence Prediction（NSP）

**预训练任务**：
1. **MLM**：随机掩盖部分token，预测被掩盖的token
2. **NSP**：判断两个句子是否连续

### 4.2 BERT变体

| 变体 | 参数 |
|------|------|
| BERT-Base | 12层，768维，12头，110M参数 |
| BERT-Large | 24层，1024维，16头，340M参数 |
| BERT-Small | 4层，512维，8头，12M参数 |

---

## 5. GPT

### 5.1 Generative Pre-trained Transformer

**特点**：
- 单向语言模型
- 因果语言建模
- 自回归生成

**预训练任务**：
- 预测下一个token

### 5.2 GPT系列

| 模型 | 参数 | 特点 |
|------|------|------|
| GPT-1 | 117M | 首次提出 |
| GPT-2 | 1.5B | 零样本学习 |
| GPT-3 | 175B | 大规模语言模型 |
| GPT-3.5 | 约175B | 代码能力 |
| GPT-4 | 未知 | 多模态 |

---

## 6. 重要论文详解

### 6.1 Attention is All You Need (2017)

**作者**：Ashish Vaswani et al.

**发表会议**：NIPS

**详细分析**：[查看论文详解](../papers/transformer.md)

**核心贡献**：
- 提出Transformer架构
- 自注意力机制
- 位置编码

**为什么重要**：
- 改变了NLP领域
- 成为大语言模型的基础

### 6.2 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)

**作者**：Jacob Devlin et al.

**发表会议**：NAACL

**详细分析**：[查看论文详解](../papers/bert.md)

**核心贡献**：
- 双向预训练
- MLM和NSP任务
- 迁移学习成功

### 6.3 GPT-3: Language Models are Few-Shot Learners (2020)

**作者**：Tom B. Brown et al.

**发表会议**：NeurIPS

**详细分析**：[查看论文详解](../papers/gpt3.md)

**核心贡献**：
- 大规模模型
- 少样本学习
- 上下文学习

---

## 7. 实践练习

### 练习1：使用PyTorch实现Transformer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn
    
    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        output, attn = self.scaled_dot_product_attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous().view(-1, self.n_heads * self.d_k)
        output = self.W_o(output)
        return output, attn

# 测试
mha = MultiHeadAttention(512, 8)
Q = K = V = torch.randn(32, 10, 512)
output, attn = mha(Q, K, V)
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {attn.shape}")
```

### 练习2：使用Hugging Face Transformers

```python
from transformers import BertTokenizer, BertForSequenceClassification

# 加载预训练模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 准备输入
text = "I love using BERT for NLP tasks!"
inputs = tokenizer(text, return_tensors='pt')

# 推理
outputs = model(**inputs)
logits = outputs.logits
prediction = torch.argmax(logits, dim=1)

print(f"输入文本: {text}")
print(f"预测结果: {'正面' if prediction.item() == 1 else '负面'}")
```

---

**下一步**：[生成模型](../04-advanced-topics/01-generative-models.md)