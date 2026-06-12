# 5.5 视频生成模型

## 1. 为什么需要视频生成

### 1.1 问题

图像生成只在空间域操作，视频生成需要同时处理**空间和时间**维度，保证帧间连续性和运动自然性。

### 1.2 核心挑战

| 挑战 | 描述 | 难度 |
|------|------|------|
| 时序一致性 | 物体在不同帧中外观一致 | 高 |
| 运动自然 | 物理运动规律 | 高 |
| 长视频生成 | 生成长时间不漂移 | 极高 |
| 计算成本 | 视频数据是图像的T倍 | 极高 |

## 2. 视频扩散模型

**Video Diffusion Models**（Ho et al., 2022）：将图像扩散扩展到时空维度。

**Sora**（OpenAI, 2024）：
- **Spacetime Patches**：将视频压缩为时空patch
- **DiT**（Diffusion Transformer）架构
- **涌现能力**：3D一致性、物体持久性、物理交互

## 3. 在具身智能中的应用

- **世界模型**：Dreamer系列使用视频预测进行规划
- **动作后果预测**：预测"如果我这样动，场景会怎么变化"
- **仿真数据生成**：生成机器人操作视频用于训练
- **Robot Dreaming**：在想象中练习技能

## 4. 参考文献

1. Ho, J., et al. (2022). Video diffusion models. *NeurIPS*.
2. Brooks, T., et al. (2024). Video generation models as world simulators. *OpenAI Technical Report*.
3. Hafner, D., et al. (2020). Dream to control: Learning behaviors by latent imagination. *ICLR*.
