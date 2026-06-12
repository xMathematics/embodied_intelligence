# 1.3 激活函数演进

## 1. 为什么需要激活函数

### 1.1 问题

没有激活函数的神经网络只是线性变换的堆叠，无论多少层都等价于单层线性变换。非线性激活函数使网络能够学习复杂的非线性映射。

### 1.2 解决了什么问题

- 引入非线性，使多层网络有意义
- 提供稀疏性、梯度流动等有利训练特性
- 控制神经元输出范围

## 2. 各激活函数对比

### 2.1 Sigmoid

**提出**：1950年代，最早使用的平滑激活函数。

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**解决**：输出在(0,1)之间，适合作为概率输出。

**局限**：
- 梯度饱和：两端导数接近0，导致梯度消失
- 非零中心：输出全为正，导致参数更新呈锯齿状
- 指数运算：计算开销大

### 2.2 Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**改进**：零中心输出，梯度强度更大。

**局限**：仍然存在梯度饱和问题。

### 2.3 ReLU

**提出**：Nair & Hinton, 2010

$$\text{ReLU}(x) = \max(0, x)$$

**解决**：
- 梯度饱和→正区间梯度恒为1，缓解梯度消失
- 稀疏激活→负区间输出0，增加表示稀疏性
- 计算效率→仅需比较运算

**局限**：
- 神经元死亡：负区间梯度为0，一旦进入永不恢复
- 非零中心
- 无上界，可能激活值过大

### 2.4 ReLU改进家族

| 激活函数 | 改进点 | 公式 |
|----------|--------|------|
| Leaky ReLU | 负区间给小斜率 | $\max(\alpha x, x)$ |
| PReLU | $\alpha$ 可学习 | 同上 |
| ELU | 负区间指数平滑 | $\begin{cases} x & x>0 \\ \alpha(e^x-1) & x \leq 0 \end{cases}$ |
| SELU | 自归一化 | $\lambda \cdot \text{ELU}(x)$ |
| GELU | 概率门控 | $x \cdot \Phi(x)$ |
| Swish | 自动门控 | $x \cdot \sigma(x)$ |

### 2.5 GELU（GPT使用）

**提出**：Hendrycks & Gimpel, 2016

$$\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$$

**优势**：比ReLU更平滑，比ELU更稳定，是Transformer模型的首选。

## 3. Softmax

**提出**：用于多分类输出的概率归一化

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}$$

**作用**：将logits转换为和为1的概率分布。

## 4. 在具身智能中的应用

- 机器人控制中的连续动作输出通常使用Tanh或线性激活
- 离散动作选择使用Softmax
- GELU广泛应用于现代具身策略网络（如RT-2中的Transformer）
- 深层策略网络使用ReLU/Leaky ReLU避免梯度消失

## 5. 参考文献

1. Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted Boltzmann machines. *ICML*.
2. Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). *arXiv:1606.08415*.
3. Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for activation functions. *arXiv:1710.05941*.
4. Clevert, D. A., Unterthiner, T., & Hochreiter, S. (2015). Fast and accurate deep network learning by exponential linear units (ELUs). *ICLR*.
