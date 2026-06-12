# 3.4 RNN的局限与现代替代方案

## 1. RNN的核心局限

| 问题 | 技术原因 | 实际影响 |
|------|----------|----------|
| **顺序计算** | 每个时间步依赖前一步的状态 | 无法并行化，训练慢 |
| **梯度消失** | 链式法则连乘 | 无法捕获长距离依赖 |
| **长序列计算** | 随着序列长度线性增长计算量 | 长文档、长视频难以处理 |
| **单方向** | 经典RNN仅单向 | 双向RNN增加2倍计算 |

## 2. Transformer的革命

**论文**：Vaswani et al., 2017 — NeurIPS

**关键改进**：
- **完全并行**：自注意力一次性处理所有位置
- **长距离依赖**：任意位置间直接通路（$O(1)$步）
- **全局上下文**：每个位置关注所有位置

**代价**：$O(n^2)$ 复杂度（改进后可通过稀疏注意力降到 $O(n \log n)$）

## 3. 状态空间模型（SSM）

**Mamba**（Gu & Dao, 2023）：挑战Transformer的新范式。

### 3.1 核心思想

将序列建模为连续系统的离散化：

$$h'(t) = \mathbf{A} h(t) + \mathbf{B} x(t)$$
$$y(t) = \mathbf{C} h(t)$$

**优势**：
- $O(n)$ 复杂度（线性）
- 不需要注意力
- 在长序列上表现优异

**局限**：
- 对于密集交互任务不如Transformer
- 生态不如Transformer成熟

## 4. 选择指南

| 特性 | RNN/LSTM | Transformer | Mamba(SSM) |
|------|----------|-------------|------------|
| 并行化 | ❌ | ✅ | ✅ |
| 长距离 | ❌(200步) | ✅(16K+) | ✅(1M+) |
| 复杂度 | $O(n)$ | $O(n^2)$ | $O(n)$ |
| 推理速度 | 快 | 慢(长序列) | 快 |
| 适合任务 | 短序列 | 通用 | 极长序列 |

## 5. 在具身智能中的选择

- **实时控制**：轻量GRU仍用于Jetson等边缘设备
- **复杂感知**：Transformer用于RT-2等大模型
- **长期操作**：Mamba可能适合长期任务规划
- **混合方案**：Hybrid RNN-Transformer结合两者优势

## 6. 参考文献

1. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
2. Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv:2312.00752*.
3. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML*.
