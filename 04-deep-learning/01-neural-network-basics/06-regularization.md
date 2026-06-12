# 1.6 正则化方法

## 1. 为什么需要正则化

### 1.1 过拟合问题

深度学习模型参数量远多于样本量，不加约束就会"死记硬背"训练数据，导致泛化性能差。

### 1.2 偏差-方差权衡

正则化增加偏差以减少方差，使总期望风险最小化。

## 2. 参数正则化

### 2.1 L2正则化（权重衰减）

$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2} \|\mathbf{W}\|_2^2$$

**为什么**：大权重导致决策边界过于复杂。L2惩罚大权重，迫使权重分散在所有特征上。

**局限**：无法产生稀疏解。

### 2.2 L1正则化（Lasso）

$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{\text{data}} + \lambda \|\mathbf{W}\|_1$$

**改进**：产生稀疏解，自动特征选择。

**局限**：不可导点处需要特殊处理。

## 3. Dropout（Hinton团队, 2012）

**为什么提出**：防止特征检测器的共适应（co-adaptation）。

**训练时**：以概率 $p$ 随机丢弃神经元：

$$r_j \sim \text{Bernoulli}(p), \quad \mathbf{h}' = \mathbf{r} \odot \mathbf{h}$$

**推理时**：使用全部神经元，输出乘 $p$。

**为什么有效**：相当于训练了指数级数量的子网络集合（ensemble）。

**局限**：增加训练时间1.5-2倍，不适合RNN。

**改进**：DropConnect, Spatial Dropout, Monte Carlo Dropout。

## 4. 数据增强

**为什么提出**：过拟合的本质是训练数据不足，数据增强可以生成无限量的"新"训练数据。

**图像增强方法对比**：
| 方法 | 适用 | 是否保留标签 |
|------|------|-------------|
| 翻转/旋转 | 通用 | 是 |
| 颜色抖动 | 通用 | 是 |
| Random Crop | 目标检测 | 部分 |
| MixUp | 通用 | 线性混合 |
| CutMix | 通用 | 块状混合 |
| RandAugment | 通用 | 自动搜索 |

**在具身智能中的应用**：
- 机器人数据采集成本高，数据增强至关重要
- 视点增强：模拟不同相机视角
- 域随机化：调整光照、纹理、颜色，帮助Sim-to-Real迁移

## 5. Early Stopping

**原理**：监控验证集性能，在验证损失不再下降时停止训练。

**为什么有效**：参数更新过程中，有效自由度逐渐增加，早停相当于限制了模型复杂度。

## 6. Label Smoothing

$$\hat{y}_i = (1 - \epsilon) y_i + \frac{\epsilon}{K}$$

**为什么提出**：硬标签（0或1）使模型过于自信，导致过拟合和校准失败。

**效果**：提高模型校准性，提升泛化能力。

## 7. 在具身智能中的应用

- **Sim-to-Real迁移**：域随机化是机器人学习从仿真到现实的关键技术
- **行为克隆正则化**：在从专家演示学习策略时，正则化防止过拟合到特定演示
- **模型增强**：在机器人强化学习中，数据增强提高样本效率

## 8. 参考文献

1. Srivastava, N., et al. (2014). Dropout: a simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929-1958.
2. Zhang, H., et al. (2017). mixup: Beyond empirical risk minimization. *ICLR*.
3. Cubuk, E. D., et al. (2020). RandAugment: Practical automated data augmentation with a reduced search space. *NeurIPS*.
4. Tobin, J., et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IROS*.
