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
