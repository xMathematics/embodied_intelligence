# 4.1 Transformer起源与原理

## 1. 为什么提出Transformer

### 1.1 历史背景

**2017年**：Google团队在"Attention is All You Need"（Vaswani et al.）中提出Transformer。

### 1.2 要解决的根本问题

| 问题 | RNN的局限 | Transformer的解决方案 |
|------|----------|---------------------|
| 顺序计算 | 逐时间步处理，无法并行 | 一次性处理所有位置 |
| 长距离依赖 | 梯度消失，有效距离~200 | 任意位置直接连接，$O(1)$路径长度 |
| 长序列注意力 | 线性计算 | 仍然是问题(二次复杂度) |
| 固定输入尺寸 | RNN可处理变长 | Transformer通过padding处理变长 |

### 1.3 核心突破

完全**删除循环**，仅使用**注意力机制**+**前馈网络**。

## 2. 自注意力机制

### 2.1 Scaled Dot-Product Attention

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{QK}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

**为什么需要缩放**：当 $d_k$ 较大时，点积值变大，Softmax进入梯度饱和区。

### 2.2 Multi-Head Attention

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

**为什么需要多头**：不同head关注不同位置的不同语义特征。

### 2.3 位置编码

由于没有循环结构，Transformer需要显式编码位置信息。

**正弦编码**：

$$\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad \text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

**为什么用正弦**：可以让模型学到相对位置（$PE_{pos+k}$ 可表示为 $PE_{pos}$ 的线性函数）。

## 3. 整体架构

### 3.1 编码器（N=6层）

每个子层：**Multi-Head Self-Attention** → **Add & LayerNorm** → **FFN** → **Add & LayerNorm**

### 3.2 解码器（N=6层）

**Masked Multi-Head Self-Attention** → **Cross-Attention** → **FFN**

- **Masked**：防止未来信息泄露（look-ahead mask）
- **Cross-Attention**：Q来自解码器，K/V来自编码器

### 3.3 FFN

$$\text{FFN}(\mathbf{x}) = \max(0, \mathbf{xW}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

## 4. 局限性

| 局限 | 原因 | 影响 |
|------|------|------|
| 二次复杂度 | 自注意力 $O(n^2)$ | 长序列无法处理 |
| 位置编码有限 | 正弦编码固定 | RoPE等改进 |
| 缺乏归纳偏置 | 对比CNN | 需要更多数据 |
| 推理时显存大 | KV Cache占用 | 长文本推理贵 |

## 5. 在具身智能中的应用

- **RT-2**：使用Transformer架构的视觉-语言-动作模型
- **Decision Transformer**：将决策建模为序列问题
- **GATO**：同一Transformer处理文本/图像/动作
- **Octo**：通用机器人策略的Transformer骨干

## 6. 参考文献

1. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
2. Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
3. Brown, T. B., et al. (2020). Language models are few-shot learners. *NeurIPS*.
4. Brohan, A., et al. (2023). RT-2: Vision-language-action models transfer web knowledge to robotic control. *arXiv*.
