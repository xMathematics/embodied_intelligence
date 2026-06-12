# 深度学习模块

## 模块概述

深度学习是机器学习的重要分支，通过多层神经网络学习数据的层次化特征表示。本模块全面覆盖深度学习**从起源到前沿**的完整知识体系，包括神经网络基础、CNN/RNN/Transformer核心架构、生成式AI（VAE/GAN/扩散/自回归/3D/视频）、自监督学习、多模态学习、感知与规划、大语言模型、模型优化部署以及具身基础模型等所有关键方向。

**特色**：每个知识点均包含"为什么提出→解决了什么问题→核心原理→局限性→后续改进→在具身智能中的应用"的完整链条。

---

## 模块结构

```
04-deep-learning/
├── 01-neural-network-basics/           # 神经网络基础（7章）
├── 02-cnn/                             # 卷积神经网络（3章）
├── 03-rnn/                             # 循环神经网络（4章）
├── 04-transformer/                     # Transformer架构（4章）
├── 05-generative-models/               # 生成模型（6章）
├── 06-self-supervised-learning/        # 自监督学习（2章）
├── 07-multimodal-learning/             # 多模态学习（1章）
├── 08-perception/                      # 深度学习感知（2章）
├── 09-planning-decision/               # 规划与决策（2章）
├── 10-large-language-models/           # 大语言模型（2章）
├── 11-model-optimization/              # 模型优化与部署（2章）
├── 12-foundation-models-embodied/      # 基础模型与具身智能（1章）
├── 13-applications/                    # 应用与前沿
├── 14-paper-surveys/                   # 论文详解
└── README.md                           # 本文件
```

---

## 模块目录

### 01 — 神经网络基础

| 文件 | 核心内容 | 提出背景 | 具身联系 |
|------|----------|----------|----------|
| 01-neuron-model.md | M-P神经元、感知机、XOR问题、第一次AI寒冬 | 1943-1969，模拟生物神经元 | 早期机器人控制 |
| 02-mlp-backpropagation.md | MLP、反向传播、万能近似定理 | 1986，解决非线性学习 | 策略网络、动力学模型 |
| 03-activation-functions.md | Sigmoid/Tanh/ReLU/GELU/Swish、梯度饱和 | 解决神经元死亡、梯度消失 | 策略网络设计 |
| 04-loss-functions.md | MSE/交叉熵/Focal/Triplet/InfoNCE | 不同任务的不同优化目标 | 模仿学习损失 |
| 05-optimizers.md | SGD/Adam/AdamW/Lion、学习率调度 | 非凸优化挑战 | 策略训练 |
| 06-regularization.md | L1/L2/Dropout/数据增强 | 解决过拟合 | Sim-to-Real域随机化 |
| 07-normalization.md | BatchNorm/LayerNorm/RMSNorm | 内部协变量偏移 | VLA模型训练 |

### 02 — 卷积神经网络

| 文件 | 核心内容 | 提出背景 | 具身联系 |
|------|----------|----------|----------|
| 01-cnn-origin.md | 感受野、局部连接、权重共享、LeNet | 1998，解决MLP参数爆炸 | 视觉感知骨干 |
| 02-classic-cnn.md | AlexNet/VGG/ResNet/EfficientNet演进 | ImageNet竞赛 | 机器人视觉 |
| 03-modern-cnn.md | 深度可分离卷积、ConvNeXt、注意力增强CNN | 移动端/效率需求 | 边缘部署 |

### 03 — 循环神经网络

| 文件 | 核心内容 | 提出背景 | 具身联系 |
|------|----------|----------|----------|
| 01-seq-modeling-origin.md | 序列建模、Elman网络、BPTT、梯度消失 | 1990，处理变长序列 | 状态估计、轨迹预测 |
| 02-lstm-gru.md | LSTM门控机制、GRU简化、细胞状态 | 1997，解决长期依赖 | 行为识别、传感器融合 |
| 03-seq2seq-attention.md | 编码器-解码器、Bahdanau/Luong注意力 | 2014，信息瓶颈问题 | 语言条件操作 |
| 04-rnn-limitations-alternatives.md | 顺序计算、Transformer替代、Mamba | RNN的固有限制 | 实时控制选择 |

### 04 — Transformer架构

| 文件 | 核心内容 | 提出背景 | 具身联系 |
|------|----------|----------|----------|
| 01-transformer-origin.md | 自注意力、多头、位置编码、编码器-解码器 | 2017，RNN不可并行 | RT-2, Decision Transformer |
| 02-bert-family.md | BERT MLM、RoBERTa/ALBERT/ELECTRA | 2018，双向上下文理解 | 语言指令理解 |
| 03-gpt-family.md | GPT演进、因果掩码、Scaling Laws、ICL | 2018-2024，生成式预训练 | SayCan任务规划 |
| 04-efficient-transformers.md | FlashAttention、稀疏/线性注意力、KV Cache | 解决二次复杂度 | 边缘部署、实时推理 |

### 05 — 生成模型

| 文件 | 核心内容 | 提出背景 | 具身联系 |
|------|----------|----------|----------|
| 01-vae.md | VAE原理、ELBO、重参数化、VQ-VAE | 2013，AE无法生成 | 潜在规划、场景压缩 |
| 02-gan.md | 对抗训练、WGAN、StyleGAN、模式崩塌 | 2014，模糊→清晰 | Sim-to-Real数据增强 |
| 03-diffusion-models.md | DDPM/DDIM、潜在扩散、Stable Diffusion、CFG | 2020，稳定高质量生成 | Diffusion Policy操作 |
| 04-autoregressive-models.md | 因果掩码、KV Cache、采样策略、Speculative Decoding | 统一序列生成 | Decision Transformer |
| 05-video-generation.md | 视频扩散、Sora时空Patch、世界模型 | 2022-2024，时序生成 | Dreamer世界模型 |
| 06-3d-generation.md | DreamFusion SDS、Zero123、TripoSR | 2022，文本/图像→3D | 仿真场景生成 |

### 06~14章内容详见各子模块文件

---

## 内容覆盖说明

| 覆盖维度 | 详情 |
|----------|------|
| **时间跨度** | 1943年M-P神经元 → 2024年π0/FLUX/Sora等最新模型 |
| **论文覆盖** | 100+篇经典论文，均有原文出处 |
| **公式密度** | 每章核心公式完整推导 |
| **具身AI链接** | 每章末尾专门讨论具身智能应用 |
| **局限性分析** | 每项技术讨论局限和改进方向 |

---

## 学习路径

**路径A：快速入门**（1-2周）
01-neural-network-basics → 02-cnn → 04-transformer → 05-generative-models/03-diffusion

**路径B：具身智能方向**（2-3周）
01 → 02 → 04 → 08-perception → 09-planning-decision → 12-foundation-models-embodied

**路径C：生成式AI方向**（2-3周）
01 → 04 → 05-generative-models全6章 → 06 → 07

**路径D：大模型方向**（2-3周）
01 → 04 → 05/04-autoregressive → 10-LLM → 11-model-optimization

---

## 推荐学习资源

### 经典书籍
1. Goodfellow, Bengio, Courville. *Deep Learning*. MIT Press, 2016.
2. Zhang, Lipton, Li, Smola. *Dive into Deep Learning*. 2023.
3. Sutton & Barto. *Reinforcement Learning: An Introduction*. MIT Press.

### 开源框架
1. **PyTorch** - 研究首选
2. **Hugging Face Transformers** - 模型库
3. **DeepSpeed / FSDP** - 分布式训练
4. **vLLM** - 高效推理
