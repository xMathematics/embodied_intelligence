# 5.3 扩散模型

## 1. 为什么提出扩散模型

### 1.1 问题

GAN训练不稳定且可能模式崩塌，VAE生成模糊。需要一种**训练稳定、生成质量高、模式覆盖完整**的生成方法。

### 1.2 扩散模型的优势

- **训练稳定**：固定目标函数（MSE去噪）
- **高质量**：当前最先进的图像生成
- **全覆盖**：不存在模式崩塌
- **灵活编辑**：支持多种条件控制

## 2. DDPM原理

**论文**：Ho, Jain & Abbeel, 2020 — NeurIPS

### 2.1 前向扩散过程

逐步添加高斯噪声：

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

可以直接一步到任意步：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

### 2.2 反向去噪过程

学习逆转噪声添加：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

**简化训练目标**：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$$

### 2.3 DDPM与得分匹配的联系

扩散模型等价于学习数据分布梯度的得分函数。

## 3. 加速采样

| 方法 | 步数 | 原理 |
|------|------|------|
| DDIM | 50-100 | 确定性非马尔可夫采样 |
| DPM-Solver | 10-25 | ODE求解器 |
| LCM | 1-4 | 一致性模型 |
| Progressive Distillation | 2-8 | 逐步蒸馏 |

## 4. 文本到图像扩散

### 4.1 潜在扩散模型（LDM）

**Stable Diffusion**（Rombach et al., 2022）：

```
文本 → CLIP文本编码 → Cross-Attention → U-Net去噪 → VAE解码 → 图像
```

**为什么在潜在空间做**：计算效率高（64×64 vs 512×512）。

### 4.2 CFG（无分类器引导）

$$\tilde{\epsilon} = \epsilon_\theta(x_t, \emptyset) + w(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

### 4.3 Stable Diffusion 3

**MMDiT**：双流Transformer架构，文本和图像各自处理再融合。

| 模型 | 参数 | 特点 |
|------|------|------|
| SD 1.5 | 0.9B | 开源标准 |
| SD XL | 2.6B | 高质量 |
| SD 3 | 8B | MMDiT |
| FLUX | 12B | 当前最强开源 |

## 5. 在具身智能中的应用

### 5.1 Diffusion Policy（Chi et al., 2023）

**核心创新**：将机器人动作序列建模为去噪过程：

```
噪声 → [条件去噪(观测)] → 平滑动作序列
```

**优势**：
- 多模态动作分布建模
- 平滑的运动轨迹
- 对高维动作空间有效

### 5.2 其他应用

- 世界状态预测：生成未来观测
- 轨迹规划：扩散生成可行轨迹
- 抓取检测：生成抓取姿态分布
- Sim-to-Real：将仿真图像转换为真实风格

## 6. 参考文献

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
2. Song, J., Meng, C., & Ermon, S. (2020). Denoising diffusion implicit models. *arXiv*.
3. Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models. *CVPR*.
4. Saharia, C., et al. (2022). Photorealistic text-to-image diffusion models with deep language understanding. *NeurIPS*.
5. Chi, C., et al. (2023). Diffusion policy: Visuomotor policy learning via action diffusion. *RSS*.
