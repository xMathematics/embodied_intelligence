# 5.4 自回归生成模型

## 1. 为什么需要自回归生成

### 1.1 核心思想

将生成问题统一为"下一个token预测"——通过链式法则分解联合概率：

$$p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_{<t})$$

### 1.2 为什么重要

- **统一框架**：文本、图像、代码、动作都可以token化后统一建模
- **简单有效**：训练目标简单（交叉熵）
- **可扩展**：GPT系列证明了scaling的力量

## 2. 因果掩码与KV Cache

### 2.1 因果掩码

确保每个位置只能关注到之前的位置，是实现自回归的关键。

### 2.2 KV Cache

**为什么需要**：生成第 $t$ 个token时，前 $t-1$ 个token的K/V已经计算过。

**缓存复用**：只计算新token的K/V，与缓存拼接。

**加速比**：$O(n^2) \rightarrow O(n)$（生成阶段）。

## 3. 采样策略

| 策略 | 特点 | 适用场景 |
|------|------|----------|
| Greedy | 确定性，易重复 | 精度优先 |
| Temperature | $p_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$ | 控制随机性 |
| Top-K | 从K个最高概率中采样 | 稳定输出 |
| Top-P (Nucleus) | 累计概率达P的token集采样 | 动态K |
| Mirostat | 自适应目标困惑度 | 避免重复 |

## 4. 局限性

**缺点**：
- **逐token生成** → 推理速度慢
- **暴露偏差**：训练时看到的都是真实token，推理时看到自己的预测
- **无法并行**：解码严格串行
- **长序列误差累积**：一步错后面步步错

**改进**：Speculative Decoding（小模型提议+大模型验证）。

## 5. 在具身智能中的应用

- **Decision Transformer**：将回报+状态+动作作为序列预测
- **轨迹生成**：以过去轨迹条件预测未来航点
- **技能库**：动作序列token化为技能
- **任务规划**：高层任务分解为子任务序列

## 6. 参考文献

1. Brown, T. B., et al. (2020). Language models are few-shot learners. *NeurIPS*.
2. van den Oord, A., et al. (2016). Pixel recurrent neural networks. *ICML*.
3. Chen, L., et al. (2021). Decision transformer: Reinforcement learning via sequence modeling. *NeurIPS*.
4. Leviathan, Y., et al. (2023). Fast inference from transformers via speculative decoding. *ICML*.
