# 5.2 生成对抗网络（GAN）

## 1. 为什么提出GAN

### 1.1 问题

VAE生成的图像偏模糊，缺乏高频细节。需要一个能生成清晰、锐利图像的方法。

### 1.2 GAN的核心思想

**论文**：Goodfellow et al., 2014 — NeurIPS

通过**对抗训练**让生成器（Generator）和判别器（Discriminator）互相博弈：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

- **生成器**：生成逼真样本以欺骗判别器
- **判别器**：区分真实样本和生成样本

## 2. 训练挑战

| 问题 | 原因 | 表现 |
|------|------|------|
| 模式崩塌 | 生成器只生成少数模式 | 缺乏多样性 |
| 训练不稳定 | D和G平衡难 | Loss震荡 |
| 梯度消失 | D太强 | G停止学习 |
| 超参数敏感 | 需要精细调参 | 重现困难 |

## 3. 重要改进

| 变体 | 改进点 | 效果 |
|------|--------|------|
| **DCGAN** | 全卷积架构 | 稳定训练 |
| **WGAN** | Wasserstein距离替代JS | 解决梯度消失 |
| **WGAN-GP** | 梯度惩罚 | 更稳定 |
| **LSGAN** | 最小二乘损失 | 更好质量 |
| **StyleGAN系列** | 风格调制 | 最先进的图像生成 |
| **Conditional GAN** | 条件约束 | 可控生成 |

### 3.1 WGAN

用Earth Mover距离替代JS散度：

$$W(p_r, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]$$

### 3.2 StyleGAN

**StyleGAN v1**：映射网络+AdaIN → 解耦风格
**StyleGAN v2**：去除水滴伪影 → 更高质量
**StyleGAN v3**：等变架构 → 消除纹理粘连

## 4. 局限性与替代

**局限性**：
- 训练困难，需要大量调参
- 模式崩塌仍是问题
- 缺乏似然评估能力
- 生成多样性有限

**被扩散模型替代**：扩散模型在图像质量上已全面超越GAN。

## 5. 在具身智能中的应用

- **Sim-to-Real**：GAN将仿真图像转换为真实风格
- **数据增强**：生成新的训练样本（遮挡、光照变化）
- **域适应**：缩小不同机器人/环境的视觉差异
- **视频预测**：GAN预测未来帧（已被扩散替代）

## 6. 参考文献

1. Goodfellow, I. J., et al. (2014). Generative adversarial nets. *NeurIPS*.
2. Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. *ICML*.
3. Gulrajani, I., et al. (2017). Improved training of Wasserstein GANs. *NeurIPS*.
4. Karras, T., et al. (2019). A style-based generator architecture for generative adversarial networks. *CVPR*.
5. Karras, T., et al. (2021). Alias-free generative adversarial networks. *NeurIPS*.
