# 1.2 多层感知机与反向传播

## 1. 为什么提出MLP与BP

### 1.1 历史背景

**1986年**：Rumelhart、Hinton和Williams发表反向传播算法（Backpropagation），解决了多层网络的训练问题。这项工作重新点燃了神经网络的研究热情。

**1989年**：Cybenko证明万能近似定理——单隐层MLP可以逼近任意连续函数。

### 1.2 解决了什么问题

- **XOR问题**：单层感知机无法解决，MLP通过隐藏层实现非线性变换解决了该问题
- **训练难题**：反向传播提供了有效的梯度计算方法，使多层网络变得可训练
- **表示学习**：多层结构使网络能自动学习层次化特征表示

### 1.3 核心原理

**MLP前向传播**：

$$ \mathbf{h}^{(l)} = f^{(l)}(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}) $$

**反向传播的链式法则**：

$$ \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{h}^{(l-1)})^T $$

其中误差信号：

$$ \boldsymbol{\delta}^{(l)} = ((\mathbf{W}^{(l+1)})^T \boldsymbol{\delta}^{(l+1)}) \odot f'(\mathbf{z}^{(l)}) $$

## 2. 局限性

| 问题 | 原因 | 表现 |
|------|------|------|
| 梯度消失 | 链式法则连乘导致梯度指数级衰减 | 深层网络无法训练 |
| 梯度爆炸 | 权重过大导致梯度指数级增长 | 训练不稳定 |
| 过拟合 | 参数量远大于样本量 | 泛化性能差 |
| 局部最优 | 非凸优化存在大量局部极小 | 无法保证找到全局最优 |
| 训练缓慢 | 全连接层参数量大 | 计算开销高 |
| 缺乏空间结构利用 | 全连接忽略输入的空间关系 | 图像任务效率低 |

## 3. 改进方向

| 改进 | 解决的问题 | 代表工作 |
|------|-----------|----------|
| ReLU激活函数 | 梯度消失 | Nair & Hinton, 2010 |
| Batch Normalization | 内部协变量偏移 | Ioffe & Szegedy, 2015 |
| Dropout | 过拟合 | Srivastava et al., 2014 |
| 残差连接 | 深层网络退化 | ResNet, He et al., 2016 |
| Adam优化器 | 学习率调参困难 | Kingma & Ba, 2014 |
| 卷积/自注意力 | 空间结构利用 | LeCun, Vaswani |

## 4. 在具身智能中的应用

- **策略网络**：MLP作为机器人策略网络的基础架构，将传感器观测映射到动作
- **逆运动学求解**：学习关节角度到末端执行器位置的映射
- **动力学模型**：学习机器人系统的状态转移函数
- **抓取质量评估**：分类抓取点的稳定性

## 5. 参考文献

1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
2. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303-314.
3. Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359-366.
