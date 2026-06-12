# 5.1 变分自编码器（VAE）

## 1. 为什么提出VAE

### 1.1 问题：自编码器无法生成

标准自编码器（AE）将输入压缩到潜在空间再重建，但潜在空间不连续——两个接近的编码点之间可能有语义鸿沟，随机采样无法生成合理数据。

### 1.2 VAE的目标

- **可生成**：能从潜在空间随机采样生成新数据
- **连续性**：潜在空间中接近的点对应语义相似的输出
- **表示学习**：学到有意义的解耦表示

## 2. VAE核心原理

**论文**：Kingma & Welling, 2013 — ICLR 2014

### 2.1 从AE到VAE

AE：$x \rightarrow \text{Encoder} \rightarrow z \rightarrow \text{Decoder} \rightarrow \hat{x}$

VAE：编码器输出分布 $q_\phi(z|x)$ ，解码器生成分布 $p_\theta(x|z)$。

### 2.2 ELBO推导

$$\log p_\theta(x) = \text{KL}(q_\phi(z|x) \| p_\theta(z|x)) + \mathcal{L}_{\text{ELBO}}$$

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$$

- **重建项**：解码器能重建输入
- **KL正则项**：后验接近先验 $\mathcal{N}(0, I)$

### 2.3 重参数化技巧

使采样操作可微：

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

## 3. 局限性

| 问题 | 原因 | 影响 |
|------|------|------|
| 生成模糊 | 高斯先验强制所有维度独立 | 细节丢失 |
| 后验崩塌 | 解码器太强，忽略潜在变量 | KL=0 |
| 先验限制 | 高斯分布表达能力有限 | 无法复杂分布 |

## 4. 改进方向

| 改进 | 方法 | 效果 |
|------|------|------|
| β-VAE | 增大β权重 | 解耦表示 |
| VQ-VAE | 离散码本编码 | 高质量生成，DALL·E基础 |
| NVAE | 层次化VAE | 高质量图像生成 |
| Very Deep VAE | 极深层次 | 超越自回归模型 |

## 5. 在具身智能中的应用

- **场景表示压缩**：将多视角观测编码为紧凑潜在场景状态
- **潜在空间规划**：在潜空间中搜索轨迹或子目标
- **VQ-VAE用于动作离散化**：将连续动作量化为离散技能
- **隐状态建模**：处理部分可观测环境

## 6. 参考文献

1. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational Bayes. *ICLR*.
2. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning. *NeurIPS*.
3. Higgins, I., et al. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework. *ICLR*.
