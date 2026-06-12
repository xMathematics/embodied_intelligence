# 3.3 Seq2Seq与注意力机制

## 1. 为什么需要Seq2Seq

### 1.1 问题

前面RNN/LSTM的输入和输出长度必须相同。但实际任务需要变长到变长的映射。

### 1.2 需要Seq2Seq的场景

| 任务 | 输入长度 | 输出长度 | 关系 |
|------|---------|---------|------|
| 机器翻译 | 源语言句子 | 目标语言句子 | 长度不同 |
| 语音识别 | 音频帧序列 | 文本序列 | 压缩 |
| 文本摘要 | 长文 | 短摘要 | 压缩 |
| 对话系统 | 用户输入 | 回复 | 可变 |
| 机器人指令 | 自然语言 | 动作序列 | 可变 |

## 2. 编码器-解码器框架

**论文**：Sutskever et al., 2014 — NeurIPS

### 2.1 架构

**编码器**：将输入序列编码为固定长度的上下文向量 $\mathbf{c}$。

$$\mathbf{h}_t = \text{RNN}(\mathbf{x}_t, \mathbf{h}_{t-1}), \quad \mathbf{c} = \mathbf{h}_T$$

**解码器**：从上下文向量生成输出序列。

$$\mathbf{s}_t = \text{RNN}(\mathbf{y}_{t-1}, \mathbf{s}_{t-1}, \mathbf{c})$$

### 2.2 局限：信息瓶颈

$\mathbf{c}$ 是编码器和解码器的唯一连接。对于长句子，一个固定向量无法包含所有信息——这是**信息瓶颈问题**。

## 3. 注意力机制的提出

**论文**：Bahdanau et al., 2014 — ICLR 2015

### 3.1 为什么需要注意力

- 人类翻译时关注源句子的不同部分
- 固定上下文向量对长句失效
- 解码器每一步需要使用源句子的不同信息

### 3.2 Bahdanau注意力（加法注意力）

$$\mathbf{e}_{tj} = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_j)$$
$$\alpha_{tj} = \frac{\exp(\mathbf{e}_{tj})}{\sum_{k=1}^{T} \exp(\mathbf{e}_{tk})}$$
$$\mathbf{c}_t = \sum_{j=1}^{T} \alpha_{tj} \mathbf{h}_j$$

### 3.3 Luong注意力（乘法注意力，2015）

**dot**：$\text{score}(\mathbf{s}_t, \mathbf{h}_j) = \mathbf{s}_t^T \mathbf{h}_j$

**general**：$\text{score}(\mathbf{s}_t, \mathbf{h}_j) = \mathbf{s}_t^T \mathbf{W}_a \mathbf{h}_j$

**concat**：$\text{score}(\mathbf{s}_t, \mathbf{h}_j) = \mathbf{v}_a^T \tanh(\mathbf{W}_a[\mathbf{s}_t; \mathbf{h}_j])$

### 3.4 注意力解决了什么

- **信息瓶颈**：解码器可以访问源序列的所有位置
- **可解释性**：注意力权重显示了"翻译时关注源句子的哪些词"
- **梯度流动**：短路径连接改善了梯度传播

## 4. 局限性与演进

**局限**：
- 仍然基于RNN的顺序计算
- 二次复杂度（$O(T^2)$）
- 长序列时注意力矩阵过大

**演进**：注意力机制从辅助工具→成为主角（Transformer）。

## 5. 在具身智能中的应用

- **语言条件操作**：将语言指令映射到机器人操作序列
- **视觉-语言导航**：注意力地图显示机器人关注场景的哪部分
- **模仿学习**：从演示序列中学习"关注哪些关键帧"
- **抓取规划**：注意力机制定位物体上的最佳抓取点

## 6. 参考文献

1. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. *NeurIPS*.
2. Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *ICLR*.
3. Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. *EMNLP*.
