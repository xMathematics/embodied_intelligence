# 3.2 LSTM与GRU

## 1. 为什么需要门控RNN

### 1.1 问题

经典RNN无法学习长期依赖——语言中可能需要依赖20步之前的词，而RNN有效记忆只有5-10步。

### 1.2 LSTM的提出

**论文**：Hochreater & Schmidhuber, 1997 — *Neural Computation*

**核心洞见**：引入**细胞状态**（cell state）和**门控机制**，使信息可以选择性地写入、读取和遗忘。

## 2. LSTM原理

### 2.1 细胞状态：信息高速公路

细胞状态 $c_t$ 在图中的最上方水平线上流动，仅通过线性加法更新，梯度可以无损传播。

### 2.2 三个门

| 门 | 公式 | 作用 |
|------|------|------|
| 遗忘门 | $f_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$ | 决定丢弃多少旧信息 |
| 输入门 | $i_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$ | 决定写入多少新信息 |
| 输出门 | $o_t = \sigma(\mathbf{W}_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$ | 决定输出多少信息 |

### 2.3 更新流程

$$\tilde{c}_t = \tanh(\mathbf{W}_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

**为什么有效**：$c_t = f_t \odot c_{t-1} + ...$ 是加法运算，梯度可通过恒等映射传播。

## 3. GRU

**论文**：Cho et al., 2014 — EMNLP

**为什么提出**：简化LSTM，减少参数。

### 3.1 两个门

| 门 | 公式 | 作用 |
|------|------|------|
| 重置门 | $r_t = \sigma(\mathbf{W}_r[\mathbf{h}_{t-1}, \mathbf{x}_t])$ | 控制忽略过去信息的程度 |
| 更新门 | $z_t = \sigma(\mathbf{W}_z[\mathbf{h}_{t-1}, \mathbf{x}_t])$ | 合并遗忘门+输入门 |

### 3.2 更新流程

$$\tilde{h}_t = \tanh(\mathbf{W}[r_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### 3.3 LSTM vs GRU对比

| 维度 | LSTM | GRU |
|------|------|-----|
| 参数量 | 4组权重 | 3组权重 |
| 门数量 | 3 | 2 |
| 细胞状态 | 显式 | 隐式 |
| 计算量 | 更大 | 更小 |
| 大数据表现 | 略好 | 无异 |
| 小数据表现 | 可能过拟合 | 更好 |

## 4. 双向RNN

$$\mathbf{h}_t = [\overrightarrow{\text{RNN}}(\mathbf{x}_t); \overleftarrow{\text{RNN}}(\mathbf{x}_t)]$$

**为什么需要**：许多任务中，当前输出依赖整个序列的上下文。

## 5. 局限性与后续

**局限**：
- 仍是顺序计算，无法并行
- 长距离依赖仍有上限（200步左右）
- 对于极长序列（书籍等）仍然困难

**后续改进**：
- Transformer完全替代了RNN序列建模
- Mamba等状态空间模型挑战Transformer

## 6. 在具身智能中的应用

- **机器人轨迹预测**：LSTM预测未来轨迹
- **视觉-语言导航**：GRU处理指令序列
- **传感器融合**：多模态传感器时序数据建模
- **行为识别**：从机器人关节序列识别操作意图

## 7. 参考文献

1. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.
2. Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*.
3. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. *Neural Computation*, 12(10), 2451-2471.
