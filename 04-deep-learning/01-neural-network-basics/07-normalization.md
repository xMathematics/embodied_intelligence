# 1.7 归一化技术

## 1. 为什么需要归一化

### 1.1 Internal Covariate Shift

**问题**：每层输入的分布在训练过程中不断变化，导致：
- 后续层需要持续适应新的分布
- 需要更小的学习率和更谨慎的参数初始化
- 饱和激活函数容易卡在饱和区

### 1.2 归一化解决的核心问题

- 加速收敛（可支持更高学习率）
- 缓解梯度消失/爆炸
- 减少对初始化的依赖
- 提供正则化效果

## 2. Batch Normalization（Ioffe & Szegedy, 2015）

$$\hat{x}_i = \frac{x_i - \mu_{\text{batch}}}{\sqrt{\sigma_{\text{batch}}^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

**为什么重要**：让每层输入的均值和方差稳定，允许使用更高学习率。

**局限**：
- 依赖batch size（小batch时不稳定）
- 训练和推理行为不同（使用运行统计量）
- 不适合RNN（不同时间步统计量不同）

## 3. Layer Normalization（Ba et al., 2016）

**为什么提出**：解决BN在RNN和变长序列中的问题。

$$\hat{x}_i = \frac{x_i - \mu_{\text{layer}}}{\sqrt{\sigma_{\text{layer}}^2 + \epsilon}}$$

**优势**：
- 不受batch size影响
- 训练和推理行为一致
- 适合序列模型

**成为Transformer的标准归一化**。

## 4. Instance Normalization（Ulyanov et al., 2016）

**为什么提出**：风格迁移中，图像的对比度和亮度信息应该被归一化掉。

**归一化维度**：对每个样本的每个通道独立归一化。

## 5. Group Normalization（Wu & He, 2018）

**为什么提出**：BN在小batch（如Mask R-CNN中使用batch=2）时性能严重下降。

**方案**：将通道分组，每组内归一化。

## 6. RMSNorm（Zhang & Sennrich, 2019）

$$\hat{x}_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2}} \cdot \gamma$$

**为什么提出**：Layer Normalization中减均值的操作在Transformer中不是必需的。

**优势**：减少约5-10%的计算量，被LLaMA采用。

## 7. 归一化方法对比

| 方法 | 归一化维度 | 小batch表现 | 序列模型 | 推荐场景 |
|------|-----------|------------|----------|----------|
| BatchNorm | batch×通道 | ❌ | ❌ | CNN |
| LayerNorm | 特征维度 | ✅ | ✅ | Transformer |
| InstanceNorm | 单样本单通道 | ✅ | ❌ | 风格迁移 |
| GroupNorm | 分组 | ✅ | ❌ | 小batch CNN |
| RMSNorm | 特征维度 | ✅ | ✅ | LLM(LLaMA) |

## 8. 在具身智能中的应用

- **机器人策略网络**常用LayerNorm（Transformer架构）
- **视觉感知模块**使用BatchNorm或GroupNorm
- **仿真到现实迁移**中，归一化层有助于消除域差异
- **训练大规模VLA模型**时RMSNorm降低计算开销

## 9. 参考文献

1. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. *ICML*.
2. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv:1607.06450*.
3. Wu, Y., & He, K. (2018). Group normalization. *ECCV*.
4. Zhang, B., & Sennrich, R. (2019). Root mean square layer normalization. *NeurIPS*.
5. Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). Instance normalization: The missing ingredient for fast stylization. *arXiv:1607.08022*.
