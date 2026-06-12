# 3.1 序列建模与RNN起源

## 1. 为什么需要序列模型

### 1.1 问题：前馈网络无法处理序列

- **固定输入输出**：MLP/CNN要求固定尺寸输入，无法处理变长序列
- **无时间概念**：每个输入独立处理，不知道时间顺序
- **上下文缺失**：无法利用过去的信息理解当前输入

### 1.2 需要通过序列建模解决的问题

| 任务 | 输入 | 输出 | 序列特性 |
|------|------|------|----------|
| 语言建模 | 单词序列 | 下一个单词 | 长期依赖 |
| 语音识别 | 音频帧 | 文本 | 变长对齐 |
| 视频理解 | 帧序列 | 行为标签 | 时序推理 |
| 机器人轨迹 | 状态序列 | 动作序列 | 时间连续性 |

## 2. RNN的提出

**Elman网络**（Elman, 1990）：第一个现代RNN。

**核心思想**：在隐藏层引入自循环连接，使网络拥有"记忆"：

$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

### 2.1 解决了什么问题

- 处理变长序列：RNN可以处理任意长度的序列
- 时间依赖建模：通过隐藏状态传递信息
- 参数共享：相同RNN单元在不同时间步使用相同权重

### 2.2 BPTT（随时间反向传播）

将RNN展开为深度前馈网络，然后应用标准BP：

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}$$

## 3. 梯度消失/爆炸问题

### 3.1 数学分析

长期依赖的梯度：

$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_1} = \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \prod_{t=2}^{T} \mathbf{W}_{hh}^T \text{diag}(f'(\mathbf{h}_t))$$

- 权重矩阵的谱半径 > 1 → 梯度爆炸
- 权重矩阵的谱半径 < 1 → 梯度消失

### 3.2 影响

| 问题 | 现象 | 后果 |
|------|------|------|
| 梯度消失 | 长距离梯度→0 | 只能学习5-10步依赖 |
| 梯度爆炸 | 梯度指数增长 | 训练不稳定NaN |

## 4. 局限性与改进

**局限性**：
- 长期记忆不足（梯度消失）
- 顺序计算无法并行
- 单方向信息流

**改进**：
- LSTM/GRU门控机制
- 双向RNN
- 注意力机制
- Transformer（完全替代）

## 5. 在具身智能中的应用

- **机器人状态估计**：从传感器序列估计当前状态
- **轨迹预测**：预测未来运动轨迹
- **动作序列生成**：生成连续动作指令
- **异常检测**：检测传感器序列中的异常

## 6. 参考文献

1. Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2), 179-211.
2. Rumelhart, D. E., et al. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
3. Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML*.
