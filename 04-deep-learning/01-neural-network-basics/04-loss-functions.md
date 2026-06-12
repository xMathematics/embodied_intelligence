# 1.4 损失函数设计

## 1. 为什么需要损失函数

### 1.1 问题

神经网络需要通过量化预测与真实值的差距来指导参数更新。损失函数定义了"好"与"坏"的度量标准。

### 1.2 不同任务需要不同损失

不同任务（回归、分类、生成、度量学习）的优化目标不同，需要针对性的损失函数设计。

## 2. 回归损失

### 2.1 MSE（L2 Loss）

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**为什么提出**：最小化高斯噪声假设下的负对数似然。

**局限**：对异常值敏感，大误差主导梯度。

### 2.2 MAE（L1 Loss）

$$\mathcal{L}_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

**改进**：对异常值鲁棒。

**局限**：在0处不可导，优化效率低。

### 2.3 Huber Loss

**提出**：结合MSE和MAE的优点

$$\mathcal{L}_{\text{Huber}} = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & |y-\hat{y}| \leq \delta \\ \delta|y-\hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

## 3. 分类损失

### 3.1 交叉熵损失

**为什么提出**：最大化似然估计在分类任务中的自然选择。

**二分类**：$\mathcal{L}_{\text{BCE}} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]$

**多分类**：$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{C} y_c \log \hat{y}_c$

### 3.2 Focal Loss

**为什么提出**：解决类别不平衡（如目标检测中背景远多于前景）。

$$\mathcal{L}_{\text{Focal}} = -(1 - \hat{y}_t)^\gamma \log(\hat{y}_t)$$

**改进**：$\gamma$ 越大，对易分类样本的抑制越强。

## 4. 度量学习损失

### 4.1 Triplet Loss

**为什么提出**：学习嵌入空间，使同类相近、异类远离。

$$\mathcal{L} = \max(0, d(a,p) - d(a,n) + \text{margin})$$

**局限**：需要精心选择三元组（hard negative mining）。

### 4.2 InfoNCE Loss（对比学习）

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

**在具身智能中的应用**：用于表示学习，使机器人在不同视角/环境下识别同一物体。

## 5. 生成模型损失

| 模型 | 损失函数 | 为什么 |
|------|----------|--------|
| VAE | ELBO = KL + 重建 | 变分推断的自然下界 |
| GAN | 对抗损失 | 生成器与判别器的博弈 |
| Diffusion | 噪声预测MSE | 去噪得分匹配 |
| VQ-VAE | 编码+码本+承诺损失 | 离散表示学习 |

## 6. 在具身智能中的应用

- 机器人抓取：使用交叉熵损失分类抓取成功率
- 操作任务：MSE/Huber损失用于动作回归
- 模仿学习：行为克隆使用交叉熵（离散动作）或MSE（连续动作）
- 逆强化学习：通过比较专家与策略的性能差异学习奖励函数

## 7. 参考文献

1. Lin, T. Y., et al. (2017). Focal loss for dense object detection. *ICCV*.
2. Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *CVPR*.
3. van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv:1807.03748*.
4. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv:1312.6114*.
