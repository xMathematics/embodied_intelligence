# 1.1 神经网络起源与神经元模型

## 1. 为什么提出神经网络

### 1.1 历史背景

**1943年**：McCulloch和Pitts提出第一个神经元数学模型（M-P模型），试图用数学模拟生物神经元的信息处理方式。

**1958年**：Frank Rosenblatt提出感知机（Perceptron），这是第一个可学习的神经网络，曾在当时引起巨大轰动。

**1969年**：Minsky和Papert在《Perceptrons》一书中指出感知机无法解决XOR问题，导致神经网络进入第一次"AI寒冬"。

### 1.2 解决了什么问题

- 提供了一种从数据中自动学习特征的方法，无需手工设计特征
- 通过非线性激活函数，理论上可以逼近任意函数（万能近似定理）

### 1.3 核心原理

**M-P神经元模型**：

$$ y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) $$

其中 $x_i$ 是输入信号，$w_i$ 是突触权重，$b$ 是偏置阈值，$f$ 是激活函数。

**感知机算法**：

$$ \mathbf{w}_{t+1} = \mathbf{w}_t + \eta(y_i - \hat{y}_i)\mathbf{x}_i $$

感知机收敛定理证明：对于线性可分数据，感知机在有限步内收敛。

## 2. 局限性

| 局限 | 原因 | 影响 |
|------|------|------|
| 线性不可分 | 单层结构无法学习非线性决策边界 | 无法解决XOR问题 |
| 表示能力有限 | 单层感知机只能表示线性函数 | 无法处理复杂模式 |
| 离散输出 | 阶跃函数不可微 | 无法使用梯度下降 |

## 3. 改进方向

**多层感知机（MLP）**：引入隐藏层和非线性激活函数，解决了XOR问题。

**反向传播算法**（Rumelhart et al., 1986）：使多层网络可以训练，引发了神经网络的复兴。

## 4. 在具身智能中的应用

- 机器人控制中用作策略网络的基础组件
- 传感器数据处理中的特征提取层
- 在早期机器人学习（如ALVINN自动驾驶）中直接使用

## 5. 参考文献

1. McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *The bulletin of mathematical biophysics*, 5(4), 115-133.
2. Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386.
3. Minsky, M., & Papert, S. (1969). *Perceptrons*. MIT Press.
4. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
