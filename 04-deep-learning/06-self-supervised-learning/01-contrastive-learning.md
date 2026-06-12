# 6.1 对比学习

## 1. 为什么需要对比学习

### 1.1 问题

监督学习需要大量标注数据，而标注成本高昂。对比学习从数据自身构造监督信号，无需标注。

### 1.2 核心思想

**拉近正样本（augmented views），推远负样本（不同样本）**。

## 2. SimCLR

**论文**：Chen et al., 2020 — ICML

**框架**：
1. 图像增强 → 两个视图
2. 编码器提取特征
3. 投影头映射到对比空间
4. InfoNCE损失

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

**关键发现**：数据增强组合（RandomResizedCrop + ColorDistortion）至关重要。

## 3. MoCo系列

**MoCo v1**：动量编码器 + 动态队列 → 解耦batch size与负样本数量

**MoCo v3**：ViT编码器 + 发现ViT不稳定训练问题

## 4. BYOL——无需负样本

**为什么可以**：通过不对称网络结构（在线+目标网络）防止模式崩塌。

## 5. 在具身智能中的应用

- **视觉表示学习**：从机器人采集的大量未标注数据学习
- **跨视角匹配**：对比学习让不同视角的同一物体表示接近
- **域适应**：对比学习缩小仿真与真实的差距

## 6. 参考文献

1. Chen, T., et al. (2020). SimCLR: A simple framework for contrastive learning. *ICML*.
2. He, K., et al. (2020). Momentum contrast for unsupervised visual representation learning. *CVPR*.
3. Grill, J. B., et al. (2020). BYOL: Bootstrap your own latent. *NeurIPS*.
