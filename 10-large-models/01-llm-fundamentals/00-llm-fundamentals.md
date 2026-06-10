# 大语言模型基础 (LLM Fundamentals)

## 概述

大语言模型（Large Language Models, LLMs）是当前人工智能领域最重要的突破之一。它们通过在大规模文本数据上进行预训练，学习语言的统计规律和世界知识，展现出令人惊叹的语言理解和生成能力。本模块将系统介绍大语言模型的基础知识，包括架构设计、预训练策略、经典模型、能力分析和上下文学习等核心内容。

## 1.1 什么是大语言模型

### 1.1.1 定义与特征

**大语言模型**是指参数规模达到数十亿甚至数千亿的深度神经网络模型，通常基于Transformer架构，通过自监督学习在超大规模文本语料上进行训练。

**核心特征：**

1. **大规模参数**：参数量从数十亿到万亿级别
   - GPT-3: 175B参数
   - LLaMA 2: 7B-70B参数
   - PaLM 2: 540B参数
   - GPT-4: 估计1.7T+参数

2. **海量训练数据**：使用万亿级别的token进行训练
   - Common Crawl: 数万亿网页文本
   - Books: 数百万本书籍
   - Wikipedia: 全部百科内容
   - 代码库: GitHub等开源代码

3. **通用能力**：无需特定任务训练即可处理多种任务
   - 文本生成与理解
   - 翻译与摘要
   - 问答与对话
   - 代码生成与调试

4. **涌现能力**：规模达到临界点后出现的新能力
   - 上下文学习
   - 思维链推理
   - 算术运算
   - 多语言理解

### 1.1.2 发展历程

**早期语言模型（1950s-2000s）：**
- N-gram模型：基于统计的马尔可夫假设
- 隐马尔可夫模型（HMM）：序列标注
- 条件随机场（CRF）：结构化预测

**问题：**
- 数据稀疏性问题
- 长距离依赖难以建模
- 无法捕捉语义关系

**神经网络语言模型（2003-2017）：**
- Bengio et al. (2003): 神经概率语言模型
- Mikolov et al. (2013): Word2Vec词向量
- Cho et al. (2014): RNN/LSTM/GRU
- Sutskever et al. (2014): Seq2Seq模型

**突破：**
- 词嵌入表示学习
- 序列建模能力提升
- 端到端训练

**问题：**
- 串行计算效率低
- 长序列梯度消失
- 并行化困难

**Transformer时代（2017-至今）：**
- Vaswani et al. (2017): "Attention Is All You Need"
- BERT (2018): 双向编码器
- GPT (2018): 自回归解码器
- GPT-3 (2020): 175B参数，涌现能力
- ChatGPT (2022): 对话式AI
- GPT-4 (2023): 多模态能力

**核心创新：**
- 自注意力机制
- 并行化训练
- 可扩展性

### 1.1.3 为什么需要大模型

**Scaling Laws（缩放定律）：**

Kaplan et al. (2020) 通过大规模实验发现：

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + C$$

其中：
- $L$: 损失函数
- $N$: 模型参数量
- $D$: 训练数据量
- $A, B, C, \alpha, \beta$: 经验常数

**关键发现：**
1. 损失随模型规模呈幂律下降
2. 性能提升需要同时增加模型和数据
3. 计算预算最优分配：模型参数与数据各占50%

**涌现能力（Emergent Abilities）：**

Wei et al. (2022) 发现某些能力只在模型规模超过阈值后才出现：

```
能力涌现曲线：
        ___________
       /
      /
_____/  临界点
```

**涌现能力示例：**
- 上下文学习（In-Context Learning）
- 思维链推理（Chain-of-Thought）
- 算术运算
- 多语言理解
- 代码生成

**理论解释：**
- 模型容量足够大才能学习复杂模式
- 大规模数据提供足够多样性
- 预训练学习通用表示而非特定任务

### 1.1.4 LLM与传统NLP方法对比

| 维度 | 传统NLP | 大语言模型 |
|------|---------|------------|
| **架构** | 特征工程+浅层模型 | 深度神经网络 |
| **训练方式** | 监督学习 | 自监督预训练 |
| **数据需求** | 标注数据（小规模） | 无标注数据（大规模） |
| **泛化能力** | 任务特定 | 通用能力强 |
| **任务适配** | 需要重新训练 | 提示工程/微调 |
| **推理能力** | 规则或浅层推理 | 深度推理 |
| **可解释性** | 相对可解释 | 黑盒模型 |

**传统NLP优势：**
- 计算资源需求低
- 可解释性强
- 适合小数据场景

**LLM优势：**
- 通用能力强
- 性能上限高
- 易于使用（提示工程）

## 1.2 LLM核心架构

### 1.2.1 Transformer架构概述

**问题提出：**
RNN/LSTM等序列模型存在以下问题：
1. 串行计算，无法充分利用GPU并行能力
2. 长距离依赖难以捕捉（梯度消失/爆炸）
3. 固定长度上下文限制

**解决方案：**
Vaswani et al. (2017) 提出Transformer架构，完全基于注意力机制。

**核心思想：**
- 抛弃循环结构，使用自注意力处理序列
- 完全并行化训练
- 理论上可处理任意长度序列

**架构组成：**
```
输入 → 词嵌入 → 位置编码 → [编码器层 × N] → 输出
                                   ↓
                            [解码器层 × N]
```

**编码器层：**
- 多头自注意力（Multi-Head Self-Attention）
- 前馈神经网络（Feed-Forward Network）
- 残差连接与层归一化

**解码器层：**
- 掩码多头自注意力
- 编码器-解码器注意力
- 前馈神经网络
- 残差连接与层归一化

### 1.2.2 自注意力机制

**问题提出：**
如何让模型关注输入序列中的相关信息？

**核心思想：**
通过查询（Query）、键（Key）、值（Value）三个向量计算注意力权重。

**数学公式：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q \in \mathbb{R}^{n \times d_k}$: 查询矩阵
- $K \in \mathbb{R}^{m \times d_k}$: 键矩阵
- $V \in \mathbb{R}^{m \times d_v}$: 值矩阵
- $d_k$: 键的维度
- $\sqrt{d_k}$: 缩放因子，防止梯度消失

**计算步骤：**

1. **相似度计算**：$QK^T$ 计算查询与键的相似度
2. **缩放**：除以 $\sqrt{d_k}$ 稳定梯度
3. **归一化**：softmax得到注意力权重
4. **加权求和**：用权重对值进行加权

**代码实现：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    自注意力机制实现
    
    论文核心思想（Vaswani et al., 2017）：
    通过Q、K、V三组向量计算注意力权重，让模型关注相关信息
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim必须能被num_heads整除"
        
        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, embed_dim)
            mask: 注意力掩码 (batch_size, seq_len) 或 (batch_size, seq_len, seq_len)
        
        Returns:
            output: 注意力输出 (batch_size, seq_len, embed_dim)
            attn_weights: 注意力权重 (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # 线性变换得到Q、K、V
        Q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # 多头分割
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置以便矩阵乘法
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = attn_scores * self.scale
        
        # 应用掩码
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            
            # 将掩码位置设为负无穷
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        output = output.view(batch_size, seq_len, self.embed_dim)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    问题提出：
    单个注意力头可能难以捕捉不同类型的关系。
    
    解决方案：
    使用多个注意力头并行计算，每个头学习不同的表示子空间。
    
    优点：
    - 捕捉多种类型的依赖关系
    - 增强模型表达能力
    - 提高训练稳定性
    
    论文核心思想（Vaswani et al., 2017）：
    "Multi-head attention allows the model to jointly attend to information
     from different representation subspaces at different positions."
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        残差连接 + 层归一化 + 注意力
        """
        residual = x
        attn_output, attn_weights = self.attention(x, mask)
        output = self.norm(residual + self.dropout(attn_output))
        return output, attn_weights


class CrossAttention(nn.Module):
    """
    交叉注意力（编码器-解码器注意力）
    
    问题提出：
    解码器如何利用编码器的输出？
    
    解决方案：
    查询来自解码器，键和值来自编码器。
    
    应用：
    - 机器翻译
    - 文本摘要
    - 图像描述生成
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: 查询（来自解码器） (batch_size, target_len, embed_dim)
            key_value: 键值（来自编码器） (batch_size, source_len, embed_dim)
            mask: 注意力掩码
        
        Returns:
            output: 交叉注意力输出
        """
        batch_size, target_len, _ = query.shape
        source_len = key_value.size(1)
        
        # 线性变换
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)
        
        # 多头分割
        Q = Q.view(batch_size, target_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, source_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, source_len, self.num_heads, self.head_dim)
        
        # 转置
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, target_len, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, source_len, head_dim)
        V = V.transpose(1, 2)
        
        # 计算注意力
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, V)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, target_len, self.embed_dim)
        
        output = self.out_proj(output)
        return output
```

### 1.2.3 前馈神经网络

**结构：**

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

其中：
- $W_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$
- $W_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$
- $d_{ff} = 4d_{model}$（通常）

**代码实现：**

```python
class FeedForwardNetwork(nn.Module):
    """
    前馈神经网络
    
    结构：Linear -> ReLU -> Linear -> Dropout
    """
    
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        residual = x
        output = self.net(x)
        output = self.norm(residual + output)
        return output
```

### 1.2.4 位置编码

**问题提出：**
自注意力机制本身不包含位置信息，如何让模型知道词序？

**解决方案：**

1. **正弦位置编码**（原始Transformer）
2. **可学习位置编码**
3. **相对位置编码**
4. **RoPE（旋转位置编码）**

**正弦位置编码公式：**

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**代码实现：**

```python
class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    
    论文核心思想（Vaswani et al., 2017）：
    使用不同频率的正弦和余弦函数编码位置信息，
    允许模型外推到训练时未见过的序列长度。
    
    优点：
    - 固定编码，无需学习
    - 可外推到任意长度
    - 相对位置信息可通过线性变换获得
    """
    
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            (-math.log(10000.0) / embed_dim))
        
        # 偶数维度用sin，奇数维度用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, embed_dim)
        
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码
    
    优点：
    - 灵活性更高
    - 可以学习到更好的位置表示
    
    缺点：
    - 无法外推到训练时未见过的长度
    - 增加模型参数
    """
    
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        x = x + position_embeddings
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码（RoPE）
    
    论文核心思想（Su et al., 2021）：
    通过旋转矩阵编码相对位置信息，
    在注意力计算中自然融入位置信息。
    
    优点：
    - 显式建模相对位置
    - 保持外推能力
    - 计算效率高
    
    应用：
    - LLaMA
    - PaLM
    - ChatGLM
    """
    
    def __init__(self, embed_dim, max_len=2048, base=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.base = base
        
        # 计算旋转角度
        inv_freq = 1.0 / (base ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x, seq_len=None):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, embed_dim)
        
        Returns:
            旋转后的张量
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        # 生成位置索引
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        
        # 计算旋转矩阵
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        # 应用旋转
        x_rotated = self.rotate_half(x) * sin.unsqueeze(0) + x * cos.unsqueeze(0)
        return x_rotated
    
    @staticmethod
    def rotate_half(x):
        """将输入张量旋转一半"""
        x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
        return torch.cat((-x2, x1), dim=-1)
```

### 1.2.5 完整Transformer层

```python
class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    
    结构：
    - 多头自注意力
    - 前馈神经网络
    - 残差连接与层归一化
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, embed_dim)
            mask: 注意力掩码
        
        Returns:
            output: 编码器输出
            attn_weights: 注意力权重
        """
        # 自注意力
        x, attn_weights = self.self_attn(x, mask)
        
        # 前馈网络
        x = self.ffn(x)
        
        return x, attn_weights


class TransformerDecoderLayer(nn.Module):
    """
    Transformer解码器层
    
    结构：
    - 掩码多头自注意力
    - 交叉注意力
    - 前馈神经网络
    - 残差连接与层归一化
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout)
    
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        """
        Args:
            x: 解码器输入 (batch_size, target_len, embed_dim)
            encoder_output: 编码器输出 (batch_size, source_len, embed_dim)
            self_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
        
        Returns:
            output: 解码器输出
            self_attn_weights: 自注意力权重
            cross_attn_weights: 交叉注意力权重
        """
        # 掩码自注意力
        x, self_attn_weights = self.self_attn(x, self_mask)
        
        # 交叉注意力
        x = self.cross_attn(x, encoder_output, cross_mask)
        x = x + x  # 残差连接
        x = nn.LayerNorm(x.size(-1))(x)
        
        # 前馈网络
        x = self.ffn(x)
        
        return x, self_attn_weights, cross_attn_weights
```

## 1.3 预训练策略

### 1.3.1 自监督学习

**问题提出：**
标注数据稀缺，如何利用海量无标注文本？

**解决方案：**
自监督学习（Self-Supervised Learning），从数据本身生成监督信号。

**核心思想：**
设计预训练任务，让模型从文本中学习语言知识。

### 1.3.2 掩码语言模型（MLM）

**问题提出：**
如何让模型理解上下文？

**解决方案：**
随机掩码部分token，让模型预测被掩码的token。

**代表模型：**
- BERT (Devlin et al., 2018)
- RoBERTa (Liu et al., 2019)
- ALBERT (Lan et al., 2019)

**训练目标：**

$$\mathcal{L}_{MLM} = -\sum_{i \in M} \log p(x_i | x_{\setminus M})$$

其中：
- $M$: 被掩码的位置集合
- $x_{\setminus M}$: 未被掩码的上下文

**代码实现：**

```python
class MaskedLanguageModel(nn.Module):
    """
    掩码语言模型
    
    论文核心思想（Devlin et al., 2018）：
    随机掩码输入序列中的部分token，让模型根据上下文预测被掩码的token。
    
    优点：
    - 双向上下文理解
    - 适合分类、NER等任务
    
    缺点：
    - 训练与推理不一致（推理时无掩码）
    - 预训练与微调差距大
    """
    
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        self.mlm_head = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: 输入token ID (batch_size, seq_len)
            attention_mask: 注意力掩码
        
        Returns:
            logits: MLM预测logits (batch_size, seq_len, vocab_size)
        """
        # 词嵌入
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 编码器层
        all_attn_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x, attention_mask)
            all_attn_weights.append(attn_weights)
        
        # MLM预测头
        logits = self.mlm_head(x)
        
        return logits, all_attn_weights
    
    def generate_masked_input(self, input_ids, mask_prob=0.15):
        """
        生成掩码输入
        
        策略：
        - 80%替换为[MASK]
        - 10%替换为随机token
        - 10%保持不变
        """
        batch_size, seq_len = input_ids.shape
        mask = torch.rand(batch_size, seq_len) < mask_prob
        
        # 创建掩码后的输入
        masked_input = input_ids.clone()
        
        # 80% [MASK]
        mask_indices = mask & (torch.rand(batch_size, seq_len) < 0.8)
        masked_input[mask_indices] = self.token_embedding.weight.shape[0] - 1  # [MASK] token
        
        # 10% 随机token
        random_indices = mask & (~mask_indices) & (torch.rand(batch_size, seq_len) < 0.5)
        masked_input[random_indices] = torch.randint(
            0, self.token_embedding.weight.shape[0], 
            (random_indices.sum(),), 
            device=input_ids.device
        )
        
        # 10% 保持不变
        
        # 标签（只计算被掩码位置的损失）
        labels = input_ids.clone()
        labels[~mask] = -100  # 忽略未掩码位置
        
        return masked_input, labels, mask


def mlm_loss_function(logits, labels):
    """
    MLM损失函数
    
    Args:
        logits: 模型输出 (batch_size, seq_len, vocab_size)
        labels: 真实标签 (batch_size, seq_len)，-100表示忽略
    
    Returns:
        loss: 交叉熵损失
    """
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss
```

### 1.3.3 因果语言模型（CLM）

**问题提出：**
如何让模型生成连贯的文本？

**解决方案：**
自回归建模，预测下一个token。

**代表模型：**
- GPT系列 (Radford et al., 2018, 2019, 2020)
- LLaMA (Touvron et al., 2023)
- PaLM (Chowdhery et al., 2022)

**训练目标：**

$$\mathcal{L}_{CLM} = -\sum_{t=1}^{T} \log p(x_t | x_{<t})$$

**代码实现：**

```python
class CausalLanguageModel(nn.Module):
    """
    因果语言模型（自回归）
    
    论文核心思想（Radford et al., 2018）：
    通过预测下一个token进行训练，使用因果掩码确保只能看到历史信息。
    
    优点：
    - 训练与推理一致
    - 适合文本生成
    - 可扩展性强
    
    缺点：
    - 单向上下文
    - 无法双向理解
    """
    
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_len=2048):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: 输入token ID (batch_size, seq_len)
            attention_mask: 注意力掩码
        
        Returns:
            logits: LM预测logits (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 因果掩码（下三角矩阵）
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
        
        # 解码器层
        all_attn_weights = []
        for layer in self.decoder_layers:
            x, self_attn_weights, _ = layer(x, x, causal_mask, attention_mask)
            all_attn_weights.append(self_attn_weights)
        
        # LM预测头
        logits = self.lm_head(x)
        
        return logits, all_attn_weights
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50):
        """
        文本生成
        
        Args:
            input_ids: 输入token ID (batch_size, seq_len)
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_k: Top-K采样
        
        Returns:
            generated_ids: 生成的token ID
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 前向传播
                logits, _ = self.forward(input_ids)
                
                # 取最后一个token的logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-K采样
                if top_k > 0:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, indices, values)
                
                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 拼接
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def clm_loss_function(logits, labels):
    """
    CLM损失函数
    
    Args:
        logits: 模型输出 (batch_size, seq_len, vocab_size)
        labels: 真实标签 (batch_size, seq_len)
    
    Returns:
        loss: 交叉熵损失
    """
    # 移位：logits预测位置t，labels是位置t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss
```

### 1.3.4 对比学习

**问题提出：**
如何学习更好的句子表示？

**解决方案：**
对比学习，拉近相似样本，推远不相似样本。

**代表模型：**
- SimCSE (Gao et al., 2021)
- ConSERT (Yan et al., 2021)
- E5 (Wang et al., 2022)

**训练目标：**

$$\mathcal{L}_{contrastive} = -\log \frac{\exp(\text{sim}(h_i, h_j)/\tau)}{\sum_{k=1}^{N} \exp(\text{sim}(h_i, h_k)/\tau)}$$

**代码实现：**

```python
class ContrastiveLearning(nn.Module):
    """
    对比学习
    
    论文核心思想（Gao et al., 2021）：
    通过对比正样本对和负样本对，学习更好的句子表示。
    
    策略：
    - SimCSE: Dropout作为数据增强
    - 硬负样本挖掘
    - 温度缩放
    """
    
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, temperature=0.05):
        super().__init__()
        self.encoder = TransformerEncoder(embed_dim, num_layers, num_heads, ff_dim)
        self.temperature = temperature
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: 输入token ID (batch_size, seq_len)
        
        Returns:
            embeddings: 句子嵌入 (batch_size, embed_dim)
        """
        # 编码
        embeddings, _ = self.encoder(input_ids, attention_mask)
        
        # 取[CLS] token的嵌入作为句子表示
        sentence_embeddings = embeddings[:, 0, :]
        
        # L2归一化
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings
    
    def contrastive_loss(self, embeddings, labels=None):
        """
        对比损失（InfoNCE）
        
        Args:
            embeddings: 句子嵌入 (batch_size, embed_dim)
            labels: 标签（可选）
        
        Returns:
            loss: 对比损失
        """
        batch_size = embeddings.size(0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # 如果没有标签，假设每个样本的正样本是它自己（SimCSE策略）
        if labels is None:
            labels = torch.arange(batch_size, device=embeddings.device)
        
        # 创建标签掩码
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        
        # 计算损失
        exp_sim = torch.exp(similarity_matrix / self.temperature)
        
        # 正样本相似度
        pos_sim = exp_sim[mask].view(batch_size, 1)
        
        # 负样本相似度
        neg_sim = exp_sim[~mask].view(batch_size, -1).sum(dim=1, keepdim=True)
        
        # InfoNCE损失
        loss = -torch.log(pos_sim / (pos_sim + neg_sim))
        
        return loss.mean()
```

## 1.4 著名LLM模型

### 1.4.1 GPT系列

**GPT-1 (2018):**
- 论文: "Improving Language Understanding by Generative Pre-Training"
- 参数量: 117M
- 架构: 12层Transformer解码器
- 训练数据: BooksCorpus
- 创新: 首次大规模预训练+微调范式

**GPT-2 (2019):**
- 论文: "Language Models are Unsupervised Multitask Learners"
- 参数量: 1.5B
- 训练数据: WebText (40GB网页文本)
- 创新: 零样本学习能力

**GPT-3 (2020):**
- 论文: "Language Models are Few-Shot Learners"
- 参数量: 175B
- 训练数据: Common Crawl + WebText + Books + Wikipedia
- 创新: 上下文学习、涌现能力

**GPT-4 (2023):**
- 参数量: 估计1.7T+
- 创新: 多模态能力、更强的推理能力
- 应用: ChatGPT、API服务

**代码实现（简化版GPT-2）：**

```python
class GPT2(nn.Module):
    """
    GPT-2模型实现
    
    论文核心思想（Radford et al., 2019）：
    通过大规模自监督预训练，实现零样本和少样本学习能力。
    
    架构特点：
    - 因果掩码自注意力
    - 层归一化在注意力之前
    - 无偏置的线性层
    """
    
    def __init__(self, vocab_size, embed_dim=768, num_layers=12, num_heads=12, 
                 ff_dim=3072, max_len=1024, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            GPT2Block(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # 语言模型头
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: 输入token ID (batch_size, seq_len)
            attention_mask: 注意力掩码
        
        Returns:
            logits: 预测logits (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入 + 位置嵌入
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        
        x = self.dropout(token_embeds + position_embeds)
        
        # Transformer块
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # 最终层归一化
        x = self.ln_f(x)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        return logits


class GPT2Block(nn.Module):
    """
    GPT-2 Transformer块
    
    架构：
    - 层归一化
    - 因果自注意力
    - 残差连接
    - 层归一化
    - 前馈网络
    - 残差连接
    """
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = GPT2MLP(embed_dim, ff_dim, dropout)
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, embed_dim)
            attention_mask: 注意力掩码
        
        Returns:
            output: 输出张量
        """
        # 自注意力 + 残差
        x = x + self.attn(self.ln_1(x), attention_mask)
        
        # 前馈网络 + 残差
        x = x + self.mlp(self.ln_2(x))
        
        return x


class CausalSelfAttention(nn.Module):
    """
    因果自注意力
    
    特点：
    - 因果掩码（只能看到历史）
    - 无偏置的投影层
    """
    
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q、K、V投影（无偏置）
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        
        # 输出投影（无偏置）
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # 注意力掩码缓冲区
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(2048, 2048)).view(1, 1, 2048, 2048)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, embed_dim)
            attention_mask: 注意力掩码
        
        Returns:
            output: 注意力输出
        """
        batch_size, seq_len, _ = x.shape
        
        # Q、K、V投影
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        
        # 多头分割
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 因果掩码
        causal_mask = self.bias[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        
        # 额外掩码
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        output = attn @ v
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 输出投影
        output = self.c_proj(output)
        
        return output


class GPT2MLP(nn.Module):
    """
    GPT-2前馈网络
    
    结构：Linear -> GELU -> Linear
    """
    
    def __init__(self, embed_dim, ff_dim, dropout):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, ff_dim, bias=False)
        self.c_proj = nn.Linear(ff_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, embed_dim)
        
        Returns:
            output: 输出张量
        """
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

### 1.4.2 LLaMA系列

**LLaMA (2023):**
- 论文: "LLaMA: Open and Efficient Foundation Language Models"
- 参数量: 7B, 13B, 33B, 65B
- 训练数据: 1.4T tokens
- 创新:
  - 预归一化（Pre-normalization）
  - SwiGLU激活函数
  - 旋转位置编码（RoPE）
  - 分组查询注意力（GQA）

**LLaMA 2 (2023):**
- 参数量: 7B, 13B, 70B
- 训练数据: 2T tokens
- 创新:
  - 上下文长度扩展到4096
  - 分组查询注意力（GQA）
  - 更好的对齐训练

**代码实现（简化版LLaMA）：**

```python
class LLaMA(nn.Module):
    """
    LLaMA模型实现
    
    论文核心思想（Touvron et al., 2023）：
    通过架构改进和大规模训练，实现高效的开源大语言模型。
    
    架构改进：
    - 预归一化（RMSNorm在注意力之前）
    - SwiGLU激活函数
    - 旋转位置编码（RoPE）
    - 分组查询注意力（GQA）
    """
    
    def __init__(self, vocab_size, embed_dim=4096, num_layers=32, num_heads=32, 
                 ff_dim=11008, max_len=2048, num_kv_heads=None, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # 分组查询注意力
        if num_kv_heads is None:
            num_kv_heads = num_heads
        self.num_kv_heads = num_kv_heads
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Transformer块
        self.layers = nn.ModuleList([
            LLaMABlock(embed_dim, num_heads, num_kv_heads, ff_dim, max_len, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.norm = RMSNorm(embed_dim)
        
        # 语言模型头
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.token_embedding.weight
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: 输入token ID (batch_size, seq_len)
            attention_mask: 注意力掩码
        
        Returns:
            logits: 预测logits
        """
        # 词嵌入
        x = self.token_embedding(input_ids)
        
        # Transformer块
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # 最终层归一化
        x = self.norm(x)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        return logits


class LLaMABlock(nn.Module):
    """
    LLaMA Transformer块
    
    架构：
    - RMSNorm
    - 分组查询自注意力
    - 残差连接
    - RMSNorm
    - SwiGLU前馈网络
    - 残差连接
    """
    
    def __init__(self, embed_dim, num_heads, num_kv_heads, ff_dim, max_len, dropout):
        super().__init__()
        self.attention = GroupedQueryAttention(
            embed_dim, num_heads, num_kv_heads, max_len, dropout
        )
        self.feed_forward = SwiGLUFFN(embed_dim, ff_dim, dropout)
        self.attention_norm = RMSNorm(embed_dim)
        self.ffn_norm = RMSNorm(embed_dim)
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, embed_dim)
            attention_mask: 注意力掩码
        
        Returns:
            output: 输出张量
        """
        # 自注意力 + 残差
        x = x + self.attention(self.attention_norm(x), attention_mask)
        
        # 前馈网络 + 残差
        x = x + self.feed_forward(self.ffn_norm(x))
        
        return x


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（GQA）
    
    论文核心思想（Ainslie et al., 2023）：
    将查询头分组，每组共享键值头，减少计算量和内存占用。
    
    优点：
    - 计算效率高
    - 内存占用低
    - 性能损失小
    
    应用：
    - LLaMA 2
    - PaLM 2
    - Mistral
    """
    
    def __init__(self, embed_dim, num_heads, num_kv_heads, max_len, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.num_groups = num_heads // num_kv_heads
        
        # Q、K、V投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # 旋转位置编码
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_len)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, embed_dim)
            attention_mask: 注意力掩码
        
        Returns:
            output: 注意力输出
        """
        batch_size, seq_len, _ = x.shape
        
        # Q、K、V投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 多头分割
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # 应用旋转位置编码
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)
        
        # 转置
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_kv_heads, seq_len, head_dim)
        v = v.transpose(1, 2)
        
        # 重复K、V以匹配Q的头数
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 因果掩码
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))
        
        # 额外掩码
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 加权求和
        output = attn @ v
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 输出投影
        output = self.o_proj(output)
        
        return output


class SwiGLUFFN(nn.Module):
    """
    SwiGLU前馈网络
    
    论文核心思想（Shazeer, 2020）：
    使用Swish激活函数的门控线性单元，提升模型性能。
    
    公式：
    FFN(x) = (Swish(xW_g) ⊙ xW_1)W_2
    
    其中：
    - Swish(x) = x * sigmoid(x)
    - ⊙ 表示逐元素乘法
    """
    
    def __init__(self, embed_dim, ff_dim, dropout):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, ff_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ff_dim, bias=False)
        self.down_proj = nn.Linear(ff_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()  # Swish
    
    def forward(self, x):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, embed_dim)
        
        Returns:
            output: 输出张量
        """
        # 门控
        gate = self.activation(self.gate_proj(x))
        
        # 上投影
        up = self.up_proj(x)
        
        # 逐元素乘法
        x = gate * up
        
        # 下投影
        x = self.down_proj(x)
        x = self.dropout(x)
        
        return x


class RMSNorm(nn.Module):
    """
    RMSNorm（Root Mean Square Layer Normalization）
    
    论文核心思想（Zhang & Sennrich, 2019）：
    简化的层归一化，移除均值计算，保留缩放。
    
    公式：
    RMSNorm(x) = x / sqrt(mean(x^2) + ε) * γ
    
    优点：
    - 计算效率高
    - 性能相当
    - 稳定性好
    """
    
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embed_dim))
    
    def forward(self, x):
        """
        Args:
            x: 输入张量 (batch_size, seq_len, embed_dim)
        
        Returns:
            output: 归一化后的张量
        """
        # 计算RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # 归一化并缩放
        output = x / rms * self.weight
        
        return output
```

### 1.4.3 其他重要模型

**PaLM (2022):**
- 论文: "PaLM: Scaling Language Modeling with Pathways"
- 参数量: 540B
- 创新:
  - Pathways系统（稀疏激活）
  - 多语言训练
  - 思维链推理

**Gemini (2023):**
- 参数量: 估计1T+
- 创新:
  - 原生多模态
  - 长上下文（1M tokens）
  - 工具使用能力

**Qwen (2023):**
- 参数量: 7B, 14B, 72B
- 创新:
  - 双语（中英）能力
  - 工具调用
  - 代码生成

## 1.5 LLM能力分析

### 1.5.1 涌现能力

**定义：**
模型规模达到临界点后突然出现的能力，小模型不具备。

**涌现能力示例：**

1. **上下文学习（In-Context Learning）**
   - GPT-3 (2020): 175B参数首次展示
   - 无需梯度更新即可学习新任务

2. **思维链推理（Chain-of-Thought）**
   - Wei et al. (2022): 通过分步推理提升性能
   - 需要模型规模 > 100B

3. **算术运算**
   - 简单加减乘除
   - 需要模型规模 > 10B

4. **多语言理解**
   - 零样本跨语言迁移
   - 需要多语言预训练

**理论解释：**

- Grokking (Power et al., 2022): 长期训练后的泛化
- Phase Transitions: 能力随规模非线性变化
- Criticality: 系统处于临界点时涌现新能力

### 1.5.2 推理能力

**推理类型：**

1. **演绎推理（Deductive Reasoning）**
   - 从一般到特殊
   - 逻辑推理
   - 数学证明

2. **归纳推理（Inductive Reasoning）**
   - 从特殊到一般
   - 模式识别
   - 规则学习

3. **溯因推理（Abductive Reasoning）**
   - 最佳解释推理
   - 诊断推理
   - 常识推理

**思维链（Chain-of-Thought）：**

```python
def chain_of_thought_prompt(question):
    """
    思维链提示
    
    问题提出：
    直接给出答案，模型可能无法正确推理。
    
    解决方案：
    要求模型展示推理步骤，逐步得出答案。
    
    论文核心思想（Wei et al., 2022）：
    "Let's think step by step" 可以显著提升推理性能。
    """
    prompt = f"""
Question: {question}

Let's think step by step:
1. 
2. 
3. 

Answer:
"""
    return prompt


def zero_shot_cot(question):
    """
    零样本思维链
    
    示例：
    Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
    Each can has 3 tennis balls. How many tennis balls does he have now?
    
    Reasoning:
    1. Roger started with 5 balls.
    2. He buys 2 cans, each with 3 balls, so he gets 2 * 3 = 6 balls.
    3. Total = 5 + 6 = 11 balls.
    
    Answer: 11
    """
    prompt = f"""
Question: {question}

Let's think step by step.
"""
    return prompt


def few_shot_cot(question, examples):
    """
    少样本思维链
    
    提供带推理步骤的示例，让模型模仿。
    """
    prompt = "Q: Roger has 5 tennis balls...\nA: Let's think step by step.\n1. Roger started with 5 balls.\n2. He buys 2 cans...\nAnswer: 11\n\n"
    
    for ex in examples:
        prompt += f"Q: {ex['question']}\nA: Let's think step by step.\n{ex['reasoning']}\nAnswer: {ex['answer']}\n\n"
    
    prompt += f"Q: {question}\nA: Let's think step by step.\n"
    return prompt
```

### 1.5.3 上下文学习

**定义：**
无需梯度更新，通过在上下文中提供示例来学习新任务。

**机制：**

1. **任务描述**：说明任务要求
2. **示例提供**：给出输入-输出对
3. **测试输入**：新的查询
4. **模型生成**：基于模式生成输出

**代码实现：**

```python
class InContextLearning:
    """
    上下文学习
    
    论文核心思想（Brown et al., 2020）：
    大语言模型可以通过上下文中的示例学习新任务，
    无需参数更新。
    
    优势：
    - 快速适应新任务
    - 无需训练数据
    - 易于使用
    
    挑战：
    - 上下文长度限制
    - 示例选择敏感
    - 泛化能力有限
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def construct_prompt(self, task_description, examples, test_input):
        """
        构建上下文学习提示
        
        Args:
            task_description: 任务描述
            examples: 示例列表 [{"input": ..., "output": ...}]
            test_input: 测试输入
        
        Returns:
            prompt: 完整提示
        """
        prompt = f"Task: {task_description}\n\n"
        
        # 添加示例
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {ex['input']}\n"
            prompt += f"Output: {ex['output']}\n\n"
        
        # 添加测试输入
        prompt += f"Input: {test_input}\n"
        prompt += "Output:"
        
        return prompt
    
    def select_examples(self, dataset, k=5, strategy='random'):
        """
        选择示例
        
        策略：
        - random: 随机选择
        - diverse: 多样性选择
        - similar: 相似性选择
        """
        if strategy == 'random':
            indices = np.random.choice(len(dataset), k, replace=False)
            return [dataset[i] for i in indices]
        
        elif strategy == 'diverse':
            # 基于嵌入的多样性选择
            pass
        
        elif strategy == 'similar':
            # 基于相似性的选择
            pass
    
    def generate(self, prompt, max_new_tokens=100, temperature=0.7):
        """
        生成输出
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成token数
            temperature: 采样温度
        
        Returns:
            output: 生成文本
        """
        # 编码
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
        
        # 解码
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成部分
        output = output[len(prompt):]
        
        return output
```

### 1.5.4 少样本/零样本学习

**零样本学习（Zero-Shot Learning）：**

```python
def zero_shot_prompt(task, input_text):
    """
    零样本提示
    
    问题提出：
    如何让模型在没有示例的情况下完成任务？
    
    解决方案：
    通过清晰的指令描述任务要求。
    
    示例：
    Task: Translate the following English text to Chinese.
    Input: Hello, how are you?
    Output: 你好，你好吗？
    """
    prompt = f"""
Task: {task}

Input: {input_text}

Output:
"""
    return prompt
```

**少样本学习（Few-Shot Learning）：**

```python
def few_shot_prompt(task, examples, input_text):
    """
    少样本提示
    
    问题提出：
    零样本性能不足，如何提升？
    
    解决方案：
    提供少量示例，让模型学习模式。
    
    示例：
    Task: Sentiment analysis
    
    Example 1:
    Input: I love this movie!
    Output: Positive
    
    Example 2:
    Input: This is terrible.
    Output: Negative
    
    Input: It's okay, not great.
    Output:
    """
    prompt = f"Task: {task}\n\n"
    
    for i, ex in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Input: {ex['input']}\n"
        prompt += f"Output: {ex['output']}\n\n"
    
    prompt += f"Input: {input_text}\n"
    prompt += "Output:"
    
    return prompt
```

## 1.6 上下文学习

### 1.6.1 指令跟随

**问题提出：**
如何让模型理解并执行自然语言指令？

**解决方案：**
指令微调（Instruction Tuning），在指令-响应对上微调模型。

**代表工作：**
- InstructGPT (Ouyang et al., 2022)
- FLAN (Wei et al., 2022)
- T0 (Sanh et al., 2022)

**代码实现：**

```python
class InstructionDataset(torch.utils.data.Dataset):
    """
    指令数据集
    
    格式：
    {
        "instruction": "任务描述",
        "input": "输入文本",
        "output": "期望输出"
    }
    """
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构建提示
        prompt = self.format_prompt(item['instruction'], item['input'])
        
        # 编码
        inputs = self.tokenizer(
            prompt + item['output'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 标签（只计算输出部分的损失）
        labels = inputs['input_ids'].clone()
        prompt_length = len(self.tokenizer(prompt)['input_ids'])
        labels[:, :prompt_length] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
    
    def format_prompt(self, instruction, input_text):
        """
        格式化提示
        
        模板：
        ### Instruction:
        {instruction}
        
        ### Input:
        {input}
        
        ### Response:
        """
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        return prompt


class InstructionTuning(nn.Module):
    """
    指令微调
    
    论文核心思想（Wei et al., 2022）：
    通过在指令-响应对上微调预训练模型，
    提升模型遵循指令的能力。
    
    优势：
    - 提升零样本性能
    - 改善指令理解
    - 增强可控性
    
    挑战：
    - 需要高质量指令数据
    - 可能损害预训练知识
    - 对齐-性能权衡
    """
    
    def __init__(self, base_model, tokenizer, learning_rate=5e-5):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
    
    def train(self, train_dataset, num_epochs=3, batch_size=8):
        """
        训练
        
        Args:
            train_dataset: 训练数据集
            num_epochs: 训练轮数
            batch_size: 批次大小
        """
        # 数据加载器
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # 优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * len(dataloader)
        )
        
        # 训练循环
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                # 前向传播
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    
    def generate(self, instruction, input_text, max_new_tokens=256):
        """
        生成响应
        
        Args:
            instruction: 指令
            input_text: 输入文本
            max_new_tokens: 最大生成token数
        
        Returns:
            response: 生成响应
        """
        # 构建提示
        prompt = self.format_prompt(instruction, input_text)
        
        # 编码
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        # 生成
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取响应部分
        response = response[len(prompt):]
        
        return response
```

### 1.6.2 提示工程

**定义：**
通过精心设计提示来引导模型生成期望的输出。

**提示工程技巧：**

1. **清晰指令**
   - 明确任务要求
   - 提供示例
   - 指定输出格式

2. **思维链**
   - "Let's think step by step"
   - 展示推理过程

3. **角色扮演**
   - "You are a helpful assistant"
   - "You are an expert in..."

4. **少样本学习**
   - 提供相关示例
   - 展示期望模式

**代码实现：**

```python
class PromptEngineering:
    """
    提示工程工具
    
    论文核心思想：
    通过精心设计提示，可以显著提升模型性能。
    
    技巧：
    - 清晰指令
    - 少样本学习
    - 思维链
    - 角色扮演
    """
    
    @staticmethod
    def clear_instruction(task, input_text):
        """
        清晰指令
        
        示例：
        Task: Summarize the following text in 3 sentences.
        Input: [长文本]
        """
        prompt = f"""
Task: {task}

Input: {input_text}

Please provide your response below:
"""
        return prompt
    
    @staticmethod
    def role_playing(role, task, input_text):
        """
        角色扮演
        
        示例：
        You are a helpful assistant.
        Task: Answer the following question.
        Input: What is the capital of France?
        """
        prompt = f"""
You are a {role}.

Task: {task}

Input: {input_text}

Response:
"""
        return prompt
    
    @staticmethod
    def chain_of_thought(question):
        """
        思维链
        
        示例：
        Question: Roger has 5 tennis balls...
        Let's think step by step:
        1. Roger started with 5 balls.
        2. He buys 2 cans...
        """
        prompt = f"""
Question: {question}

Let's think step by step:
1. 
2. 
3. 

Answer:
"""
        return prompt
    
    @staticmethod
    def few_shot_prompt(task, examples, test_input):
        """
        少样本提示
        
        示例：
        Task: Sentiment analysis
        
        Example 1:
        Input: I love this movie!
        Output: Positive
        
        Example 2:
        Input: This is terrible.
        Output: Negative
        
        Input: It's okay, not great.
        Output:
        """
        prompt = f"Task: {task}\n\n"
        
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {ex['input']}\n"
            prompt += f"Output: {ex['output']}\n\n"
        
        prompt += f"Input: {test_input}\n"
        prompt += "Output:"
        
        return prompt
```

### 1.6.3 示例学习

**问题提出：**
如何让模型通过示例学习任务模式？

**解决方案：**
在提示中提供输入-输出示例，让模型模仿。

**代码实现：**

```python
class ExampleLearning:
    """
    示例学习
    
    论文核心思想（Brown et al., 2020）：
    通过在上下文中提供示例，让模型学习任务模式。
    
    优势：
    - 无需训练
    - 快速适应
    - 易于使用
    
    挑战：
    - 示例选择
    - 上下文长度
    - 泛化能力
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def select_examples(self, dataset, test_input, k=5, strategy='random'):
        """
        选择示例
        
        策略：
        - random: 随机选择
        - similar: 相似性选择
        - diverse: 多样性选择
        """
        if strategy == 'random':
            indices = np.random.choice(len(dataset), k, replace=False)
            return [dataset[i] for i in indices]
        
        elif strategy == 'similar':
            # 基于相似性的选择
            test_embedding = self.get_embedding(test_input)
            similarities = []
            for item in dataset:
                item_embedding = self.get_embedding(item['input'])
                sim = np.dot(test_embedding, item_embedding)
                similarities.append((sim, item))
            
            similarities.sort(reverse=True)
            return [item for _, item in similarities[:k]]
        
        elif strategy == 'diverse':
            # 多样性选择
            selected = []
            remaining = dataset.copy()
            
            while len(selected) < k and remaining:
                # 选择与已选示例最不相似的
                if not selected:
                    # 随机选择第一个
                    idx = np.random.choice(len(remaining))
                    selected.append(remaining.pop(idx))
                else:
                    selected_embeddings = [self.get_embedding(s['input']) for s in selected]
                    min_sim = float('inf')
                    best_idx = 0
                    
                    for i, item in enumerate(remaining):
                        item_embedding = self.get_embedding(item['input'])
                        max_sim = max(np.dot(item_embedding, se) for se in selected_embeddings)
                        if max_sim < min_sim:
                            min_sim = max_sim
                            best_idx = i
                    
                    selected.append(remaining.pop(best_idx))
            
            return selected
    
    def get_embedding(self, text):
        """获取文本嵌入"""
        inputs = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # 使用最后一层的[CLS] token
            embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
        return embedding[0]
    
    def construct_prompt(self, task_description, examples, test_input):
        """构建提示"""
        prompt = f"Task: {task_description}\n\n"
        
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {ex['input']}\n"
            prompt += f"Output: {ex['output']}\n\n"
        
        prompt += f"Input: {test_input}\n"
        prompt += "Output:"
        
        return prompt
    
    def generate(self, prompt, max_new_tokens=100):
        """生成输出"""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
        
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = output[len(prompt):]
        
        return output
```

### 1.6.4 思维链

**问题提出：**
复杂推理任务需要分步思考，如何引导模型展示推理过程？

**解决方案：**
要求模型逐步推理，展示中间步骤。

**代码实现：**

```python
class ChainOfThought:
    """
    思维链推理
    
    论文核心思想（Wei et al., 2022）：
    通过要求模型展示推理步骤，可以显著提升复杂推理任务的性能。
    
    优势：
    - 提升推理准确性
    - 可解释性强
    - 适用于复杂任务
    
    挑战：
    - 增加生成成本
    - 可能产生错误推理
    - 需要更多token
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def zero_shot_cot(self, question):
        """
        零样本思维链
        
        示例：
        Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
        Each can has 3 tennis balls. How many tennis balls does he have now?
        
        Let's think step by step:
        1. Roger started with 5 balls.
        2. He buys 2 cans, each with 3 balls, so he gets 2 * 3 = 6 balls.
        3. Total = 5 + 6 = 11 balls.
        
        Answer: 11
        """
        prompt = f"""
Question: {question}

Let's think step by step.
"""
        return self.generate(prompt)
    
    def few_shot_cot(self, question, examples):
        """
        少样本思维链
        
        提供带推理步骤的示例，让模型模仿。
        """
        prompt = ""
        
        for ex in examples:
            prompt += f"Question: {ex['question']}\n"
            prompt += f"Let's think step by step.\n"
            prompt += f"{ex['reasoning']}\n"
            prompt += f"Answer: {ex['answer']}\n\n"
        
        prompt += f"Question: {question}\n"
        prompt += "Let's think step by step.\n"
        
        return self.generate(prompt)
    
    def auto_cot(self, question):
        """
        自动思维链
        
        论文核心思想（Zhang et al., 2022）：
        自动生成推理示例，无需人工标注。
        
        步骤：
        1. 生成多个推理路径
        2. 选择最一致的答案
        """
        # 生成多个推理路径
        reasoning_paths = []
        for _ in range(5):
            prompt = f"""
Question: {question}

Let's think step by step.
"""
            reasoning = self.generate(prompt)
            reasoning_paths.append(reasoning)
        
        # 选择最一致的答案
        answers = self.extract_answers(reasoning_paths)
        most_common = max(set(answers), key=answers.count)
        
        return most_common
    
    def extract_answers(self, reasoning_paths):
        """从推理路径中提取答案"""
        answers = []
        for path in reasoning_paths:
            # 简单的答案提取（实际需要更复杂的解析）
            if "Answer:" in path:
                answer = path.split("Answer:")[-1].strip()
                answers.append(answer)
        return answers
    
    def generate(self, prompt, max_new_tokens=256):
        """生成输出"""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
        
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output = output[len(prompt):]
        
        return output
```

## 1.7 训练与优化

### 1.7.1 分布式训练

**问题提出：**
大模型训练需要大量计算资源，如何高效利用多GPU/多节点？

**解决方案：**
分布式训练策略。

**代码实现：**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedTrainer:
    """
    分布式训练器
    
    策略：
    - 数据并行（Data Parallel）
    - 模型并行（Model Parallel）
    - 流水线并行（Pipeline Parallel）
    - 混合并行（Hybrid Parallel）
    """
    
    def __init__(self, model, rank, world_size):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        
        # 初始化进程组
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        # 包装模型
        self.model = DDP(model, device_ids=[rank])
    
    def train(self, train_loader, optimizer, epoch):
        """训练循环"""
        self.model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 移动到当前GPU
            data, target = data.to(self.rank), target.to(self.rank)
            
            # 前向传播
            optimizer.zero_grad()
            output = self.model(data)
            
            # 计算损失
            loss = F.cross_entropy(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0 and self.rank == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}")
```

### 1.7.2 混合精度训练

**问题提出：**
FP32训练内存占用大，如何减少内存使用？

**解决方案：**
混合精度训练（FP16 + FP32）。

**代码实现：**

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    """
    混合精度训练
    
    优势：
    - 减少内存占用
    - 加速训练
    - 保持精度
    
    论文核心思想（Micikevicius et al., 2018）：
    使用FP16进行计算，FP32进行权重更新。
    """
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
    
    def train_step(self, batch):
        """训练步骤"""
        self.model.train()
        
        # 前向传播（自动混合精度）
        with autocast():
            outputs = self.model(**batch)
            loss = outputs.loss
        
        # 反向传播（缩放梯度）
        self.scaler.scale(loss).backward()
        
        # 梯度裁剪
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 优化器步骤
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss.item()
```

### 1.7.3 梯度检查点

**问题提出：**
大模型激活值占用大量内存，如何减少？

**解决方案：**
梯度检查点（Gradient Checkpointing）。

**代码实现：**

```python
from torch.utils.checkpoint import checkpoint

class GradientCheckpointing:
    """
    梯度检查点
    
    论文核心思想（Chen et al., 2016）：
    在前向传播时只保存部分激活值，
    反向传播时重新计算未保存的激活值。
    
    优势：
    - 大幅减少内存占用
    - 以计算换内存
    
    缺点：
    - 增加计算时间
    """
    
    def __init__(self, model):
        self.model = model
    
    def enable_checkpointing(self):
        """启用梯度检查点"""
        def checkpointed_forward(module, *args, **kwargs):
            return checkpoint(module.forward, *args, **kwargs)
        
        # 对Transformer层应用检查点
        for layer in self.model.layers:
            layer.forward = checkpointed_forward.__get__(layer, type(layer))
```

## 1.8 评估与基准

### 1.8.1 语言建模评估

**困惑度（Perplexity）：**

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{t=1}^{N}\log p(x_t|x_{<t})\right)$$

**代码实现：**

```python
def compute_perplexity(model, dataloader, device):
    """
    计算困惑度
    
    Args:
        model: 语言模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        perplexity: 困惑度
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    # 计算困惑度
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity
```

### 1.8.2 下游任务评估

**代码实现：**

```python
class TaskEvaluator:
    """
    任务评估器
    
    支持的任务：
    - 文本分类
    - 问答
    - 摘要
    - 翻译
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate_classification(self, dataset, prompt_template):
        """
        评估分类任务
        
        Args:
            dataset: 数据集 [{"text": ..., "label": ...}]
            prompt_template: 提示模板
        
        Returns:
            accuracy: 准确率
        """
        correct = 0
        total = 0
        
        for item in dataset:
            # 构建提示
            prompt = prompt_template.format(text=item['text'])
            
            # 生成
            inputs = self.tokenizer(prompt, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=10)
            
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = prediction[len(prompt):].strip()
            
            # 比较预测
            if prediction == item['label']:
                correct += 1
            total += 1
        
        accuracy = correct / total
        return accuracy
    
    def evaluate_qa(self, dataset, prompt_template):
        """
        评估问答任务
        
        Args:
            dataset: 数据集 [{"question": ..., "answer": ...}]
            prompt_template: 提示模板
        
        Returns:
            exact_match: 精确匹配率
            f1_score: F1分数
        """
        exact_match = 0
        total_f1 = 0
        
        for item in dataset:
            # 构建提示
            prompt = prompt_template.format(question=item['question'])
            
            # 生成
            inputs = self.tokenizer(prompt, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=50)
            
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = prediction[len(prompt):].strip()
            
            # 精确匹配
            if prediction == item['answer']:
                exact_match += 1
            
            # F1分数
            f1 = self.compute_f1(prediction, item['answer'])
            total_f1 += f1
        
        exact_match = exact_match / len(dataset)
        f1_score = total_f1 / len(dataset)
        
        return exact_match, f1_score
    
    def compute_f1(self, prediction, reference):
        """计算F1分数"""
        prediction_tokens = set(prediction.lower().split())
        reference_tokens = set(reference.lower().split())
        
        common_tokens = prediction_tokens & reference_tokens
        
        if len(common_tokens) == 0:
            return 0.0
        
        precision = len(common_tokens) / len(prediction_tokens)
        recall = len(common_tokens) / len(reference_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
```

## 1.9 挑战与前沿

### 1.9.1 当前挑战

1. **计算资源**
   - 训练成本高昂
   - 推理延迟高
   - 能耗问题

2. **数据质量**
   - 训练数据偏见
   - 有毒内容
   - 版权问题

3. **模型可解释性**
   - 黑盒模型
   - 决策过程不透明
   - 难以调试

4. **安全性**
   - 对抗攻击
   - 提示注入
   - 隐私泄露

### 1.9.2 前沿方向

1. **高效训练**
   - 稀疏激活
   - 参数高效微调
   - 知识蒸馏

2. **长上下文**
   - 线性注意力
   - 分层注意力
   - 记忆增强

3. **多模态融合**
   - 视觉-语言模型
   - 音频-语言模型
   - 原生多模态

4. **神经符号**
   - 结合符号推理
   - 可解释AI
   - 知识增强

## 1.10 总结

大语言模型是当前AI领域最重要的突破之一。通过大规模预训练，模型学习到了丰富的语言知识和世界知识，展现出惊人的泛化能力。

**关键要点：**

1. **架构**：Transformer是现代LLM的核心架构
2. **预训练**：自监督学习是训练LLM的关键
3. **规模**：模型规模和训练数据量决定性能上限
4. **能力**：涌现能力包括上下文学习、思维链等
5. **应用**：通过提示工程和微调适应各种任务

**未来展望：**

- 更高效的训练方法
- 更长的上下文窗口
- 更强的推理能力
- 更好的多模态融合
- 更高的安全性和可解释性

---

## 参考文献

### 核心论文

1. Vaswani, A., et al. (2017). "Attention Is All You Need". NeurIPS.
2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers". NAACL.
3. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training". OpenAI.
4. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners". OpenAI.
5. Brown, T., et al. (2020). "Language Models are Few-Shot Learners". NeurIPS.
6. Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models". NeurIPS.
7. Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models".
8. Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback". NeurIPS.

### 综述论文

1. Zhao, W. X., et al. (2023). "A Survey of Large Language Models". arXiv.
2. Chang, Y., et al. (2024). "A Survey on Evaluation of Large Language Models". arXiv.
3. Yang, J., et al. (2024). "Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond". arXiv.

### 技术论文

1. Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models". arXiv.
2. Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models". arXiv.
3. Wei, J., et al. (2022). "Emergent Abilities of Large Language Models". TMLR.
4. Liu, H., et al. (2023). "Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing". ACM Computing Surveys.
