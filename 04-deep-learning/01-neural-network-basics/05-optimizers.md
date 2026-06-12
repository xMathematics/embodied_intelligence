# 1.5 优化器演进

## 1. 为什么需要优化器

### 1.1 问题

神经网络的训练本质上是求解非凸优化问题：$\min_{\theta} \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(f_\theta(x_i), y_i)$。优化器决定如何更新参数以最小化损失。

### 1.2 核心挑战

- 高维非凸空间存在大量鞍点
- 不同参数的尺度差异大
- 梯度噪声来自小批量采样

## 2. SGD及其变体

### 2.1 批量梯度下降（BGD）

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta)$$

**局限**：全数据集计算梯度，速度极慢，无法在线学习。

### 2.2 随机梯度下降（SGD）

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta; x^{(i)}, y^{(i)})$$

**改进**：每个样本更新一次，速度快，能跳出局部极小。

**新问题**：梯度方差大，收敛不稳定。

### 2.3 小批量SGD（Mini-batch SGD）

每次从数据集中随机采样 $m$ 个样本：

$$\theta_{t+1} = \theta_t - \eta \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta \mathcal{L}(\theta; x^{(i)}, y^{(i)})$$

## 3. 动量方法

### 3.1 Momentum

**为什么提出**：加速收敛，抑制震荡。

$$v_{t+1} = \gamma v_t + \eta \nabla_\theta \mathcal{L}(\theta)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

**作用**：在梯度方向一致时加速，在方向变化时减速。

### 3.2 Nesterov Accelerated Gradient（NAG）

**改进**：在更新前先"看一步"：

$$v_{t+1} = \gamma v_t + \eta \nabla_\theta \mathcal{L}(\theta_t - \gamma v_t)$$

## 4. 自适应学习率方法

### 4.1 AdaGrad

**为什么提出**：为每个参数分配不同的学习率。

$$G_t = G_{t-1} + g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} g_t$$

**局限**：学习率单调递减，训练后期停滞。

### 4.2 RMSProp

**改进**：使用指数移动平均替代累积：

$$\mathbb{E}[g^2]_t = \beta \mathbb{E}[g^2]_{t-1} + (1-\beta) g_t^2$$

### 4.3 Adam（2014）

**论文**：Kingma & Ba

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(一阶矩)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(二阶矩)}$$
$$\hat{m}_t = m_t / (1-\beta_1^t), \quad \hat{v}_t = v_t / (1-\beta_2^t)$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**优势**：结合动量和自适应学习率。

**局限**：在部分任务中泛化不如SGD（后来发现是权重衰减问题）。

### 4.4 AdamW（2017）

**改进**：将权重衰减与自适应学习率解耦：

$$\theta_{t+1} = \theta_t - \eta (\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t)$$

**影响**：成为ViT和LLM训练的首选优化器。

## 5. 优化器对比

| 优化器 | 动量 | 自适应LR | 权重衰减 | 适用场景 |
|--------|------|----------|----------|----------|
| SGD | 需手动加 | 否 | 隐含 | CV, 小模型 |
| Adam | 内置 | 是 | 标准 | NLP, ViT |
| AdamW | 内置 | 是 | **解耦** | LLM, ViT |
| Lion | 签名动量 | 是 | 解耦 | 高效, 大模型 |
| Sophia | 二阶信息 | 是 | 标准 | LLM预训练 |

## 6. 在具身智能中的应用

- **强化学习策略训练**：AdamW训练策略网络和价值网络
- **机器人模仿学习**：行为克隆使用Adam优化
- **端到端驾驶**：大规模驾驶模型使用AdamW
- **高效微调**：LoRA中使用AdamW优化低秩矩阵

## 7. 参考文献

1. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv:1412.6980*.
2. Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *arXiv:1711.05101*.
3. Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. *JMLR*.
4. Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. *COURSERA*.
