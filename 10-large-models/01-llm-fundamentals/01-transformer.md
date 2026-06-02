# 1.1 Transformer架构详解

## 目录

- [1. 引言](#1-引言)
- [2. Transformer论文概述](#2-transformer论文概述)
- [3. 自注意力机制](#3-自注意力机制)
- [4. Multi-Head Attention](#4-multi-head-attention)
- [5. Transformer架构](#5-transformer架构)
- [6. 位置编码](#6-位置编码)
- [7. 论文创新点分析](#7-论文创新点分析)
- [8. 未解决的问题](#8-未解决的问题)
- [9. 未来研究方向](#9-未来研究方向)
- [10. 实践练习](#10-实践练习)

---

## 1. 引言

### 1.1 背景

在Transformer之前，序列建模主要依赖于：
- **循环神经网络（RNN）**：存在梯度消失问题，难以处理长序列
- **长短时记忆网络（LSTM）**：缓解了梯度消失，但仍存在顺序依赖
- **卷积神经网络（CNN）**：并行计算能力强，但捕捉长距离依赖能力有限

### 1.2 Transformer的革命性贡献

Transformer完全抛弃了递归结构，仅依靠**自注意力机制**实现序列建模，带来了：
- **并行计算**：所有位置可以同时处理
- **长距离依赖**：直接建模任意两个位置的关系
- **统一架构**：编码器和解码器共享相似结构

---

## 2. Transformer论文概述

### 2.1 论文信息

| 项目 | 内容 |
|------|------|
| **标题** | Attention Is All You Need |
| **作者** | Vaswani et al. |
| **发表会议** | NeurIPS 2017 |
| **引用数** | >100,000（截至2024年） |

**论文深度解析**：[Attention Is All You Need 论文阅读笔记](papers/attention-is-all-you-need.md)

### 2.2 论文核心结论

> "我们提出了一种新的简单网络架构Transformer，仅基于注意力机制，完全摒弃了递归和卷积。"

### 2.3 主要贡献

1. 提出Transformer架构，仅使用注意力机制
2. 引入Multi-Head Attention机制
3. 在WMT 2014英德翻译任务上达到当时最优
4. 训练效率远超RNN模型

---

## 3. 自注意力机制

### 3.1 注意力机制的本质

注意力机制允许模型在处理某个位置时，关注输入序列中其他位置的信息：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q$（Query）：查询矩阵
- $K$（Key）：键矩阵  
- $V$（Value）：值矩阵
- $d_k$：键向量的维度

### 3.2 公式推导

**步骤1：计算相似度分数**
$$\text{scores} = QK^T$$

每个查询向量与所有键向量进行点积，得到相似度分数。

**步骤2：缩放**
$$\text{scaled\_scores} = \frac{QK^T}{\sqrt{d_k}}$$

缩放的目的是防止点积结果过大，导致softmax梯度消失。

**步骤3：Softmax归一化**
$$\text{weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

将相似度分数转换为注意力权重，权重之和为1。

**步骤4：加权求和**
$$\text{output} = \text{weights} \cdot V$$

根据注意力权重对值向量进行加权求和。

### 3.3 自注意力的特殊之处

在自注意力中，Q、K、V都来自同一输入序列：
$$Q = K = V = \text{Input}$$

这使得模型能够学习序列内部的依赖关系。

### 3.4 示例说明

假设有输入序列 `[I, love, NLP]`：

```
输入序列: [I, love, NLP]
维度: d_model = 512

Q = K = V = Input @ W_q, Input @ W_k, Input @ W_v

注意力权重矩阵:
           I    love    NLP
I        0.6     0.3    0.1
love     0.2     0.5    0.3
NLP      0.1     0.4    0.5

输出: weighted sum of values
```

---

## 4. Multi-Head Attention

### 4.1 动机

单一注意力头只能学习一种类型的依赖关系，Multi-Head Attention允许模型同时学习多种不同的注意力模式。

### 4.2 数学定义

$$\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, head_2, ..., head_h)W^O$$

其中每个注意力头：
$$head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### 4.3 关键参数

| 参数 | 符号 | 典型值 |
|------|------|--------|
| 注意力头数 | $h$ | 8 |
| 模型维度 | $d_{\text{model}}$ | 512 |
| 单头维度 | $d_k = d_v = d_{\text{model}}/h$ | 64 |

### 4.4 优势

| 优势 | 说明 |
|------|------|
| **多头多样性** | 不同头学习不同类型的依赖 |
| **参数效率** | 总参数数量不变，计算更高效 |
| **表达能力** | 捕捉更丰富的语言结构 |

---

## 5. Transformer架构

### 5.1 整体架构

```
输入序列 → [编码器] → [解码器] → 输出序列
              ↑              ↑
              │              │
         位置编码        位置编码
```

### 5.2 编码器结构

编码器由N个相同的层堆叠而成（论文中N=6）：

```
编码器层:
┌─────────────────────────────────────┐
│  Multi-Head Self-Attention         │
│  (残差连接 + LayerNorm)             │
├─────────────────────────────────────┤
│  Feed Forward Network               │
│  (残差连接 + LayerNorm)             │
└─────────────────────────────────────┘
```

### 5.3 解码器结构

解码器也由N个相同的层堆叠而成：

```
解码器层:
┌─────────────────────────────────────┐
│  Masked Multi-Head Self-Attention  │
│  (残差连接 + LayerNorm)             │
├─────────────────────────────────────┤
│  Multi-Head Encoder-Decoder Attention│
│  (残差连接 + LayerNorm)             │
├─────────────────────────────────────┤
│  Feed Forward Network               │
│  (残差连接 + LayerNorm)             │
└─────────────────────────────────────┘
```

### 5.4 关键组件对比

| 组件 | 作用 | 输入来源 |
|------|------|---------|
| **Masked Self-Attention** | 防止未来信息泄露 | 解码器输入 |
| **Encoder-Decoder Attention** | 建立源-目标对齐 | Q来自解码器，K/V来自编码器 |
| **Feed Forward** | 特征变换 | 注意力输出 |

### 5.5 Feed Forward Network

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

这是一个两层全连接网络，使用ReLU激活函数。

---

## 6. 位置编码

### 6.1 必要性

Transformer没有递归结构，无法直接获取序列的顺序信息，需要显式添加位置编码。

### 6.2 正弦位置编码

论文提出的位置编码方法：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

其中：
- $pos$：位置索引
- $i$：维度索引
- $d_{\text{model}}$：模型维度

### 6.3 特性

| 特性 | 说明 |
|------|------|
| **周期性** | 相同周期模式重复出现 |
| **相对位置** | 位置差固定的token具有固定的相对编码 |
| **外推性** | 可以处理训练时未见过的长度 |

### 6.4 位置编码的替代方案

| 方案 | 说明 |
|------|------|
| **学习型位置编码** | 随机初始化，随模型一起训练 |
| **RoPE** | Rotary Position Embedding（旋转位置编码） |
| **ALiBi** | Attention with Linear Biases |

---

## 7. 论文创新点分析

### 7.1 解决的核心问题

| 问题 | RNN的局限 | Transformer的解决方案 |
|------|----------|---------------------|
| **并行计算** | 顺序依赖，无法并行 | 全并行处理所有位置 |
| **长距离依赖** | 梯度消失/爆炸 | 直接建模任意位置关系 |
| **计算效率** | O(n)复杂度 | O(n²)但常数小 |
| **模型容量** | 难以扩展 | 易于扩展到深层 |

### 7.2 关键创新

1. **纯注意力架构**：完全抛弃递归结构
2. **Multi-Head Attention**：多头并行学习不同依赖
3. **残差连接 + LayerNorm**：稳定训练深层网络
4. **位置编码**：注入序列顺序信息

### 7.3 实验验证

论文在WMT 2014英德翻译任务上的结果：

| 模型 | 参数 | BLEU分数 |
|------|------|---------|
| Transformer (base) | 65M | 28.4 |
| Transformer (big) | 213M | 29.9 |
| Google's GNMT | 380M | 26.3 |

---

## 8. 未解决的问题

### 8.1 计算复杂度问题

- **自注意力复杂度**：O(n²·d)，长序列计算成本高
- **内存占用**：存储注意力矩阵需要O(n²)空间

### 8.2 长距离依赖的局限性

虽然Transformer理论上可以建模任意距离依赖，但实际中：
- 注意力权重分布不均匀
- 远距离token的注意力权重通常很小
- 缺乏对层次结构的显式建模

### 8.3 推理效率

- 自回归解码是顺序的，无法并行
- 每个时间步都需要完整的注意力计算

### 8.4 可解释性差

- 注意力权重的语义不明确
- 难以理解模型的决策过程
- 缺乏理论保证

---

## 9. 未来研究方向

### 9.1 高效Transformer

| 方向 | 方法 | 代表工作 |
|------|------|---------|
| **稀疏注意力** | 只计算部分注意力 | Longformer, BigBird |
| **线性注意力** | 将注意力复杂度降为O(n) | Linformer, Performer |
| **结构化注意力** | 引入归纳偏置 | Transformer-XL, Compressive Transformer |

### 9.2 改进位置编码

| 方向 | 方法 | 代表工作 |
|------|------|---------|
| **旋转位置编码** | 旋转矩阵编码位置 | RoPE (GPT-NeoX, LLaMA) |
| **线性偏置** | 添加线性位置偏置 | ALiBi |
| **动态位置编码** | 位置编码随上下文变化 | T5 |

### 9.3 增强推理能力

| 方向 | 方法 | 代表工作 |
|------|------|---------|
| **思维链** | 引导模型逐步推理 | Chain-of-Thought |
| **工具使用** | 允许调用外部工具 | ReAct, Toolformer |
| **检索增强** | 结合外部知识库 | RAG |

### 9.4 多模态扩展

| 方向 | 方法 | 代表工作 |
|------|------|---------|
| **视觉-语言** | 融合视觉和语言 | CLIP, Flamingo |
| **视频-语言** | 处理时序视觉信息 | TimeSformer |
| **3D-语言** | 理解三维空间 | PointNet + Transformer |

### 9.5 理论研究

- 注意力机制的数学性质
- 涌现能力的理论解释
- 泛化能力的边界分析

---

## 10. 实践练习

### 练习1：实现自注意力机制

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    实现缩放点积注意力
    
    参数:
        Q: 查询矩阵, shape: (batch_size, seq_len_q, d_k)
        K: 键矩阵, shape: (batch_size, seq_len_k, d_k)
        V: 值矩阵, shape: (batch_size, seq_len_v, d_v)
        mask: 掩码矩阵, shape: (batch_size, seq_len_q, seq_len_k)
    
    返回:
        output: 注意力输出
        attn_weights: 注意力权重
    """
    d_k = Q.size(-1)
    
    # 计算相似度分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax归一化
    attn_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights

# 测试
batch_size = 2
seq_len = 4
d_k = d_v = 64

Q = torch.randn(batch_size, seq_len, d_k)
K = torch.randn(batch_size, seq_len, d_k)
V = torch.randn(batch_size, seq_len, d_v)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")
print(f"注意力权重行和: {weights.sum(dim=-1)}")
```

### 练习2：实现Multi-Head Attention

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        """将输入拆分为多个注意力头"""
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换
        Q = self.W_q(Q)  # (batch, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 拆分多头
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出线性变换
        output = self.W_o(output)
        
        return output, attn_weights

# 测试
d_model = 512
num_heads = 8
batch_size = 2
seq_len = 10

mha = MultiHeadAttention(d_model, num_heads)
x = torch.randn(batch_size, seq_len, d_model)
output, weights = mha(x, x, x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")
```

### 练习3：实现Transformer编码器层

```python
import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Multi-Head Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

# 测试
d_model = 512
num_heads = 8
batch_size = 2
seq_len = 10

encoder_layer = TransformerEncoderLayer(d_model, num_heads)
x = torch.randn(batch_size, seq_len, d_model)
output = encoder_layer(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
```

---

**论文引用**：Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

**下一节**：[预训练策略](02-pretraining.md)

---

## 11. Transformer变体与改进

### 11.1 BERT架构

**问题提出：**
Transformer编码器能否用于理解任务而非生成任务？

**解决方案：**
BERT（Bidirectional Encoder Representations from Transformers）仅使用Transformer编码器，通过双向上下文学习语言表示。

**核心创新：**
1. **双向编码**：同时利用左右上下文
2. **掩码语言模型（MLM）**：随机掩码部分token并预测
3. **下一句预测（NSP）**：判断两句话是否连续

**代码实现：**

```python
import torch
import torch.nn as nn

class BERTEmbedding(nn.Module):
    """
    BERT嵌入层
    
    组成：
    - Token嵌入
    - 位置嵌入
    - 段嵌入
    """
    
    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
    
    def forward(self, input_ids, segment_ids):
        """
        Args:
            input_ids: (batch_size, seq_len)
            segment_ids: (batch_size, seq_len)
        
        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token嵌入
        token_embeds = self.token_embedding(input_ids)
        
        # 位置嵌入
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        
        # 段嵌入
        segment_embeds = self.segment_embedding(segment_ids)
        
        # 组合并缩放
        embeddings = token_embeds + position_embeds + segment_embeds
        embeddings = embeddings * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BERT(nn.Module):
    """
    BERT模型
    
    论文核心思想（Devlin et al., 2018）：
    使用双向Transformer编码器进行预训练，
    通过MLM和NSP任务学习语言表示。
    
    优势：
    - 双向上下文理解
    - 强大的迁移学习能力
    - 适用于多种NLP任务
    
    挑战：
    - 预训练成本高
    - 掩码策略影响性能
    - [MASK] token在微调时不存在
    """
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, 
                 num_layers=12, dim_feedforward=3072, max_len=512, dropout=0.1):
        super().__init__()
        
        # 嵌入层
        self.embedding = BERTEmbedding(vocab_size, d_model, max_len, dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLM头
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size, bias=False)
        )
        self.mlm_head[3].weight = self.embedding.token_embedding.weight
        
        # NSP头
        self.nsp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2)
        )
    
    def forward(self, input_ids, segment_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            segment_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            mlm_logits: (batch_size, seq_len, vocab_size)
            nsp_logits: (batch_size, 2)
            hidden_states: (batch_size, seq_len, d_model)
        """
        # 嵌入
        x = self.embedding(input_ids, segment_ids)
        
        # 编码器
        if attention_mask is not None:
            # 转换为key_padding_mask
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        hidden_states = self.encoder(x, src_key_padding_mask=key_padding_mask)
        
        # MLM预测
        mlm_logits = self.mlm_head(hidden_states)
        
        # NSP预测（使用[CLS] token）
        cls_hidden = hidden_states[:, 0, :]
        nsp_logits = self.nsp_head(cls_hidden)
        
        return mlm_logits, nsp_logits, hidden_states
```

### 11.2 GPT架构

**问题提出：**
如何使用Transformer进行自回归文本生成？

**解决方案：**
GPT（Generative Pre-trained Transformer）仅使用Transformer解码器，通过因果掩码实现自回归生成。

**核心创新：**
1. **自回归建模**：预测下一个token
2. **因果掩码**：防止未来信息泄露
3. **大规模预训练**：在海量文本上训练

**代码实现：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):
    """
    因果多头注意力
    
    使用下三角掩码确保自回归性质
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性变换
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # 因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # 额外掩码
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出变换
        output = self.W_o(output)
        
        return output, attn_weights


class GPTDecoderLayer(nn.Module):
    """
    GPT解码器层
    
    包含：
    - 因果自注意力
    - 前馈网络
    - 残差连接和层归一化
    """
    
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        
        self.self_attn = CausalMultiHeadAttention(d_model, num_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力
        attn_output, _ = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class GPT(nn.Module):
    """
    GPT模型
    
    论文核心思想（Radford et al., 2018）：
    使用Transformer解码器进行自回归语言建模。
    
    优势：
    - 强大的生成能力
    - 适合文本生成任务
    - 可扩展到大规模
    
    挑战：
    - 单向上下文
    - 推理速度慢
    - 需要大量训练数据
    """
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, 
                 num_layers=12, dim_feedforward=3072, max_len=1024, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 解码器层
        self.layers = nn.ModuleList([
            GPTDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(d_model)
        
        # 语言模型头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + position_embeds)
        
        # 解码器层
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # 最终归一化
        x = self.ln_f(x)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50):
        """
        自回归生成
        
        Args:
            input_ids: (batch_size, seq_len)
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_k: top-k采样
        
        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 前向传播
                logits = self.forward(generated_ids)
                
                # 取最后一个token的logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k采样
                if top_k > 0:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, indices, values)
                
                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 添加到生成序列
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        return generated_ids
```

### 11.3 T5架构

**问题提出：**
能否用统一的文本到文本框架处理所有NLP任务？

**解决方案：**
T5（Text-to-Text Transfer Transformer）将所有任务都转换为文本生成任务。

**核心创新：**
1. **统一框架**：所有任务都是文本到文本
2. **任务前缀**：用前缀描述任务类型
3. **大规模预训练**：在C4数据集上训练

**代码实现：**

```python
import torch
import torch.nn as nn

class T5Block(nn.Module):
    """
    T5 Transformer块
    
    特点：
    - 使用层归一化的变体（RMSNorm）
    - 前馈网络先扩展后压缩
    - 相对位置编码
    """
    
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, self_attn_mask=None, cross_attn_mask=None):
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 交叉注意力
        cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, cross_attn_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class T5(nn.Module):
    """
    T5模型
    
    论文核心思想（Raffel et al., 2020）：
    将所有NLP任务统一为文本到文本格式。
    
    示例：
    - 翻译: "translate English to German: That is good." → "Das ist gut."
    - 摘要: "summarize: [长文本]" → "[摘要]"
    - 分类: "sentiment: I love it!" → "positive"
    
    优势：
    - 统一框架简化设计
    - 强大的迁移学习
    - 易于扩展新任务
    
    挑战：
    - 需要设计任务前缀
    - 生成速度慢
    - 需要大量训练数据
    """
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, max_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_len = max_len
        
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
        self.decoder_layers = nn.ModuleList([
            T5Block(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(d_model)
        
        # 语言模型头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, encoder_input_ids, decoder_input_ids, 
                encoder_attention_mask=None, decoder_attention_mask=None):
        """
        Args:
            encoder_input_ids: (batch_size, encoder_seq_len)
            decoder_input_ids: (batch_size, decoder_seq_len)
            encoder_attention_mask: (batch_size, encoder_seq_len)
            decoder_attention_mask: (batch_size, decoder_seq_len)
        
        Returns:
            logits: (batch_size, decoder_seq_len, vocab_size)
        """
        # 编码器嵌入
        encoder_embeds = self.embedding(encoder_input_ids)
        encoder_embeds = encoder_embeds * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # 编码器
        if encoder_attention_mask is not None:
            encoder_key_padding_mask = (encoder_attention_mask == 0)
        else:
            encoder_key_padding_mask = None
        
        encoder_output = self.encoder(encoder_embeds, src_key_padding_mask=encoder_key_padding_mask)
        
        # 解码器嵌入
        decoder_embeds = self.embedding(decoder_input_ids)
        decoder_embeds = decoder_embeds * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        # 解码器
        x = decoder_embeds
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, decoder_attention_mask, encoder_attention_mask)
        
        # 最终归一化
        x = self.ln_f(x)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, encoder_input_ids, max_new_tokens=100, temperature=1.0, 
                 top_k=50, encoder_attention_mask=None):
        """
        生成文本
        
        Args:
            encoder_input_ids: (batch_size, encoder_seq_len)
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_k: top-k采样
            encoder_attention_mask: (batch_size, encoder_seq_len)
        
        Returns:
            generated_ids: (batch_size, max_new_tokens)
        """
        self.eval()
        batch_size = encoder_input_ids.size(0)
        device = encoder_input_ids.device
        
        # 编码器前向传播
        encoder_embeds = self.embedding(encoder_input_ids)
        encoder_embeds = encoder_embeds * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
        if encoder_attention_mask is not None:
            encoder_key_padding_mask = (encoder_attention_mask == 0)
        else:
            encoder_key_padding_mask = None
        
        encoder_output = self.encoder(encoder_embeds, src_key_padding_mask=encoder_key_padding_mask)
        
        # 初始化解码器输入（从[EOS]开始）
        decoder_input_ids = torch.full((batch_size, 1), 0, device=device)
        generated_ids = decoder_input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 解码器前向传播
                decoder_embeds = self.embedding(decoder_input_ids)
                decoder_embeds = decoder_embeds * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
                
                x = decoder_embeds
                for layer in self.decoder_layers:
                    x = layer(x, encoder_output)
                
                x = self.ln_f(x)
                logits = self.lm_head(x)
                
                # 取最后一个token的logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k采样
                if top_k > 0:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, indices, values)
                
                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 添加到生成序列
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                decoder_input_ids = generated_ids
        
        return generated_ids
```

### 11.4 ViT架构

**问题提出：**
Transformer能否用于计算机视觉任务？

**解决方案：**
ViT（Vision Transformer）将图像分割成patch，将其视为token序列。

**核心创新：**
1. **Patch分割**：将图像分割成固定大小的patch
2. **线性投影**：将patch映射到向量空间
3. **类别token**：添加可学习的[CLS] token用于分类

**代码实现：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Patch嵌入层
    
    将图像分割成patch并线性投影
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # 使用卷积进行patch嵌入
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        
        Returns:
            patches: (batch_size, n_patches, d_model)
        """
        x = self.proj(x)  # (batch_size, d_model, n_patches_h, n_patches_w)
        x = x.flatten(2)  # (batch_size, d_model, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, d_model)
        return x


class ViT(nn.Module):
    """
    Vision Transformer
    
    论文核心思想（Dosovitskiy et al., 2020）：
    将Transformer应用于计算机视觉任务。
    
    优势：
    - 统一视觉和语言架构
    - 强大的全局建模能力
    - 易于扩展
    
    挑战：
    - 需要大量训练数据
    - 缺乏归纳偏置
    - 计算成本高
    """
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 num_classes=1000, d_model=768, num_heads=12, 
                 num_layers=12, dim_feedforward=3072, dropout=0.1):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        self.n_patches = self.patch_embedding.n_patches
        
        # 类别token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 位置嵌入
        self.position_embedding = nn.Parameter(torch.zeros(1, self.n_patches + 1, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )
        
        # 初始化
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Patch嵌入
        patches = self.patch_embedding(x)  # (batch_size, n_patches, d_model)
        
        # 添加类别token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)  # (batch_size, n_patches+1, d_model)
        
        # 添加位置嵌入
        x = x + self.position_embedding
        
        # Transformer编码器
        x = self.encoder(x)
        
        # 使用[CLS] token进行分类
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits
```

---

## 12. 高效Transformer变体

### 12.1 Longformer

**问题提出：**
标准Transformer的自注意力复杂度为O(n²)，如何处理长序列？

**解决方案：**
Longformer使用稀疏注意力模式，将复杂度降为O(n)。

**核心创新：**
1. **滑动窗口注意力**：每个token只关注附近的token
2. **全局注意力**：部分token（如[CLS]）可以关注所有token
3. **膨胀注意力**：扩大感受野

**代码实现：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LongformerSelfAttention(nn.Module):
    """
    Longformer自注意力
    
    使用稀疏注意力模式：
    - 滑动窗口注意力
    - 全局注意力
    """
    
    def __init__(self, d_model, num_heads, window_size=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, global_attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性变换
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 滑动窗口注意力
        output = self._sliding_window_attention(Q, K, V)
        
        # 全局注意力
        if global_attention_mask is not None:
            output = self._global_attention(output, Q, K, V, global_attention_mask)
        
        # 输出变换
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output
    
    def _sliding_window_attention(self, Q, K, V):
        """滑动窗口注意力"""
        batch_size, num_heads, seq_len, d_k = Q.shape
        
        # 创建滑动窗口掩码
        window_mask = self._create_window_mask(seq_len)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # 应用窗口掩码
        scores = scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, V)
        
        return output
    
    def _create_window_mask(self, seq_len):
        """创建滑动窗口掩码"""
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask
    
    def _global_attention(self, output, Q, K, V, global_attention_mask):
        """全局注意力"""
        batch_size, num_heads, seq_len, d_k = Q.shape
        
        # 找到全局token
        global_indices = torch.where(global_attention_mask[0] == 1)[0]
        
        if len(global_indices) == 0:
            return output
        
        # 全局token关注所有位置
        for idx in global_indices:
            global_q = Q[:, :, idx:idx+1, :]
            scores = torch.matmul(global_q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
            attn_weights = F.softmax(scores, dim=-1)
            global_output = torch.matmul(attn_weights, V)
            output[:, :, idx:idx+1, :] = global_output
        
        return output
```

### 12.2 Performer

**问题提出：**
能否将注意力复杂度降为线性？

**解决方案：**
Performer使用随机特征近似softmax注意力。

**核心创新：**
1. **随机特征**：使用随机特征近似softmax
2. **线性复杂度**：注意力计算复杂度降为O(n)
3. **无偏估计**：保持注意力的无偏性

**数学原理：**

标准softmax注意力：
$$\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d})V$$

使用随机特征近似：
$$\text{softmax}(x) \approx \phi(x)^T\phi(x)$$

其中$\phi(x)$是随机特征映射。

**代码实现：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PerformerAttention(nn.Module):
    """
    Performer注意力
    
    使用随机特征近似softmax注意力，实现线性复杂度。
    
    论文核心思想（Choromanski et al., 2020）：
    通过随机特征方法将softmax注意力近似为线性计算。
    
    优势：
    - 线性复杂度
    - 适用于长序列
    - 理论保证
    
    挑战：
    - 近似误差
    - 需要大量随机特征
    """
    
    def __init__(self, d_model, num_heads, num_random_features=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_random_features = num_random_features
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 随机特征投影矩阵
        self.register_buffer(
            'projection_matrix',
            self._get_projection_matrix(self.d_k, num_random_features)
        )
    
    def _get_projection_matrix(self, d_k, num_features):
        """生成随机特征投影矩阵"""
        # 使用高斯随机特征
        projection_matrix = torch.randn(d_k, num_features)
        return projection_matrix
    
    def _softmax_kernel(self, x, is_query=True):
        """
        Softmax核函数近似
        
        使用随机特征近似softmax
        """
        # 线性投影
        x = torch.matmul(x, self.projection_matrix)
        
        # 应用核函数
        if is_query:
            # 查询向量使用exp
            x = torch.exp(x - torch.amax(x, dim=-1, keepdim=True))
        else:
            # 键向量使用exp
            x = torch.exp(x - torch.amax(x, dim=-1, keepdim=True))
        
        return x
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性变换
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 应用随机特征
        Q_prime = self._softmax_kernel(Q, is_query=True)
        K_prime = self._softmax_kernel(K, is_query=False)
        
        # 线性注意力计算
        # Attention(Q, K, V) = Q_prime @ (K_prime^T @ V)
        KV = torch.einsum('bhnd,bhnv->bhvd', K_prime, V)
        output = torch.einsum('bhnd,bhvd->bhnv', Q_prime, KV)
        
        # 归一化
        Q_sum = torch.einsum('bhnd,bhnd->bhn', Q_prime, K_prime)
        output = output / (Q_sum.unsqueeze(-1) + 1e-6)
        
        # 输出变换
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output
```

### 12.3 Linformer

**问题提出：**
能否通过降维减少注意力复杂度？

**解决方案：**
Linformer将键值矩阵投影到低维空间。

**核心创新：**
1. **低秩投影**：将K和V投影到低维
2. **线性复杂度**：注意力计算复杂度降为O(n)
3. **可学习投影**：投影矩阵可学习

**数学原理：**

标准注意力：
$$\text{Attention}(Q, K, V) = \text{softmax}(QK^T)V$$

Linformer近似：
$$\text{Attention}(Q, K, V) \approx \text{softmax}(QE^T)(FV)$$

其中E和F是低维投影矩阵。

**代码实现：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinformerAttention(nn.Module):
    """
    Linformer注意力
    
    通过低秩投影将注意力复杂度降为线性。
    
    论文核心思想（Wang et al., 2020）：
    将键值矩阵投影到低维空间，减少计算复杂度。
    
    优势：
    - 线性复杂度
    - 适用于长序列
    - 简单有效
    
    挑战：
    - 信息损失
    - 需要调优投影维度
    """
    
    def __init__(self, d_model, num_heads, seq_len, k=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.seq_len = seq_len
        self.k = k
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # 低秩投影矩阵
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 线性变换
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 低秩投影
        K_low = torch.matmul(K.transpose(1, 2), self.E)  # (batch, num_heads, d_k, k)
        V_low = torch.matmul(V.transpose(1, 2), self.F)  # (batch, num_heads, d_v, k)
        
        # 计算注意力
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        scores = torch.matmul(Q, K_low) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, V_low.transpose(-2, -1))
        
        # 输出变换
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        
        return output
```

---

## 13. Transformer训练技巧

### 13.1 学习率调度

**Warmup策略：**

```python
import torch
import torch.optim as optim
import math

class WarmupScheduler:
    """
    Warmup学习率调度器
    
    先线性增加学习率，然后按余弦衰减。
    
    论文核心思想（Ginsburg et al., 2019）：
    在训练初期使用较小的学习率，逐渐增加到目标值，
    然后按余弦衰减。
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        """更新学习率"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Warmup阶段：线性增加
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # 衰减阶段：余弦衰减
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # 更新优化器学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
```

### 13.2 梯度累积

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

### 13.3 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    """
    混合精度训练器
    
    使用FP16进行计算，FP32进行权重更新。
    """
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
    
    def train_step(self, batch):
        """训练步骤"""
        self.model.train()
        
        # 自动混合精度
        with autocast():
            outputs = self.model(**batch)
            loss = outputs.loss
        
        # 缩放梯度
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 更新参数
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss.item()
```

---

## 14. Transformer应用

### 14.1 机器翻译

```python
class TranslationModel(nn.Module):
    """
    翻译模型
    
    使用Transformer进行机器翻译。
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, dim_feedforward=2048, 
                 max_len=512, dropout=0.1):
        super().__init__()
        
        # 编码器
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 解码器
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        # 编码器
        src_embeds = self.encoder_embedding(src_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src_embeds = self.dropout(src_embeds)
        memory = self.encoder(src_embeds, src_key_padding_mask=src_mask)
        
        # 解码器
        tgt_embeds = self.decoder_embedding(tgt_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt_embeds = self.dropout(tgt_embeds)
        output = self.decoder(tgt_embeds, memory, tgt_mask=tgt_mask)
        
        # 输出投影
        logits = self.output_projection(output)
        
        return logits
    
    def translate(self, src_ids, max_len=100, temperature=1.0):
        """翻译"""
        self.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device
        
        # 编码
        src_embeds = self.encoder_embedding(src_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        memory = self.encoder(src_embeds)
        
        # 初始化输出
        tgt_ids = torch.full((batch_size, 1), 0, device=device)  # [SOS] token
        
        with torch.no_grad():
            for _ in range(max_len):
                # 解码
                tgt_embeds = self.decoder_embedding(tgt_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
                output = self.decoder(tgt_embeds, memory)
                logits = self.output_projection(output)
                
                # 采样
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 添加到输出
                tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
        
        return tgt_ids
```

### 14.2 文本摘要

```python
class SummarizationModel(nn.Module):
    """
    摘要模型
    
    使用Transformer进行文本摘要。
    """
    
    def __init__(self, vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, dim_feedforward=2048, max_len=512, dropout=0.1):
        super().__init__()
        
        # 编码器（处理源文本）
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 解码器（生成摘要）
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        # 编码器
        src_embeds = self.encoder_embedding(src_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src_embeds = self.dropout(src_embeds)
        memory = self.encoder(src_embeds, src_key_padding_mask=src_mask)
        
        # 解码器
        tgt_embeds = self.decoder_embedding(tgt_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt_embeds = self.dropout(tgt_embeds)
        output = self.decoder(tgt_embeds, memory, tgt_mask=tgt_mask)
        
        # 输出投影
        logits = self.output_projection(output)
        
        return logits
    
    def summarize(self, src_ids, max_len=150, temperature=1.0):
        """生成摘要"""
        self.eval()
        batch_size = src_ids.size(0)
        device = src_ids.device
        
        # 编码
        src_embeds = self.encoder_embedding(src_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        memory = self.encoder(src_embeds)
        
        # 初始化输出
        tgt_ids = torch.full((batch_size, 1), 0, device=device)
        
        with torch.no_grad():
            for _ in range(max_len):
                # 解码
                tgt_embeds = self.decoder_embedding(tgt_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
                output = self.decoder(tgt_embeds, memory)
                logits = self.output_projection(output)
                
                # 采样
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 检查是否到达[EOS]
                if (next_token == 2).all():  # 假设2是[EOS] token
                    break
                
                # 添加到输出
                tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
        
        return tgt_ids
```

---

## 15. 总结

Transformer是深度学习领域最重要的架构之一，它彻底改变了自然语言处理、计算机视觉等多个领域。

**关键要点：**

1. **自注意力机制**：Transformer的核心，直接建模序列中任意两个位置的关系
2. **多头注意力**：学习多种不同类型的依赖关系
3. **位置编码**：注入序列顺序信息
4. **编码器-解码器架构**：适用于序列到序列任务
5. **纯注意力架构**：完全摒弃递归和卷积

**主要变体：**

1. **BERT**：仅使用编码器，适用于理解任务
2. **GPT**：仅使用解码器，适用于生成任务
3. **T5**：文本到文本统一框架
4. **ViT**：应用于计算机视觉

**高效变体：**

1. **Longformer**：稀疏注意力
2. **Performer**：随机特征近似
3. **Linformer**：低秩投影

**未来方向：**

- 更高效的注意力机制
- 更长的上下文窗口
- 更好的多模态融合
- 更强的推理能力

---

## 参考文献

### 核心论文

1. Vaswani, A., et al. (2017). "Attention Is All You Need". NeurIPS.
2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers". NAACL.
3. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training". OpenAI.
4. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners". OpenAI.
5. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". JMLR.
6. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". ICLR.

### 高效Transformer

1. Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer". arXiv.
2. Choromanski, K., et al. (2020). "Rethinking Attention with Performers". ICLR.
3. Wang, S., et al. (2020). "Linformer: Self-Attention with Linear Complexity". ICLR.

### 训练技巧

1. Popel, M., & Bojar, O. (2018). "Training Tips for the Transformer Model". ACL.
2. Liu, Y., et al. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach". arXiv.
