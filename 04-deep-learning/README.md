# 深度学习模块

## 模块概述

深度学习是机器学习的重要分支，通过多层神经网络学习数据的层次特征表示。本模块从神经网络基础开始，系统讲解CNN、RNN、Transformer等核心架构，以及生成模型、自监督学习、多模态学习等前沿内容，为后续的强化学习和具身智能打下坚实基础。

## 学习目标

完成本模块学习后，您将能够：
- 理解神经网络的基本原理和训练方法
- 掌握CNN、RNN、Transformer等核心架构
- 理解生成模型的原理和应用
- 了解自监督学习和多模态学习
- 能够使用PyTorch/TensorFlow实现深度学习模型
- 理解大语言模型的发展和应用

---

## 模块结构

```
04-deep-learning/
├── 01-neural-network-basics/    # 神经网络基础
├── 02-cnn/                      # 卷积神经网络
├── 03-rnn/                      # 循环神经网络
├── 04-transformer/              # Transformer架构
├── 05-generative-models/        # 生成模型
├── 06-self-supervised-learning/ # 自监督学习
├── 07-multimodal-learning/      # 多模态学习
├── 08-model-training/           # 模型训练与优化
├── 09-applications/             # 应用领域
└── 10-paper-surveys/            # 论文详解
```

---

## 学习路径

### 第一部分：神经网络基础（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 1.1 | 神经网络概述 | ⭐⭐⭐ |
| 1.2 | 前向传播与反向传播 | ⭐⭐⭐⭐ |
| 1.3 | 激活函数 | ⭐⭐⭐ |
| 1.4 | 损失函数 | ⭐⭐⭐ |
| 1.5 | 优化器 | ⭐⭐⭐⭐ |
| 1.6 | 正则化 | ⭐⭐⭐⭐ |

**核心内容**：
- 感知机、多层感知机
- BP算法、梯度下降
- Sigmoid、ReLU、GELU
- MSE、Cross-Entropy
- SGD、Adam、AdamW
- Dropout、BatchNorm、Weight Decay

---

### 第二部分：卷积神经网络（2周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 2.1 | CNN基础 | ⭐⭐⭐⭐ |
| 2.2 | 经典CNN架构 | ⭐⭐⭐⭐ |
| 2.3 | 高级CNN技术 | ⭐⭐⭐⭐⭐ |
| 2.4 | 视觉Transformer | ⭐⭐⭐⭐⭐ |
| 2.5 | 迁移学习 | ⭐⭐⭐⭐ |

**核心内容**：
- 卷积层、池化层、全连接层
- LeNet、AlexNet、VGG、ResNet、GoogLeNet
- 注意力机制、可变形卷积
- ViT、Swin Transformer、MAE
- Fine-tuning、Feature Extraction

---

### 第三部分：循环神经网络（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 3.1 | RNN基础 | ⭐⭐⭐⭐ |
| 3.2 | LSTM | ⭐⭐⭐⭐ |
| 3.3 | GRU | ⭐⭐⭐⭐ |
| 3.4 | 序列建模应用 | ⭐⭐⭐⭐ |
| 3.5 | 注意力机制引入 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- RNN结构、梯度消失问题
- LSTM门控机制、记忆单元
- GRU简化结构
- 文本分类、机器翻译
- Seq2Seq、Attention机制

---

### 第四部分：Transformer架构（2周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 4.1 | 自注意力机制 | ⭐⭐⭐⭐⭐ |
| 4.2 | Transformer架构 | ⭐⭐⭐⭐⭐ |
| 4.3 | BERT | ⭐⭐⭐⭐⭐ |
| 4.4 | GPT | ⭐⭐⭐⭐⭐ |
| 4.5 | Transformer变体 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- Scaled Dot-Product Attention
- Multi-Head Attention
- Encoder-Decoder架构
- Masked Self-Attention
- BERT预训练、GPT生成
- T5、BART、XLNet

---

### 第五部分：生成模型（2周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 5.1 | VAE | ⭐⭐⭐⭐⭐ |
| 5.2 | GAN | ⭐⭐⭐⭐⭐ |
| 5.3 | 扩散模型 | ⭐⭐⭐⭐⭐ |
| 5.4 | 流模型 | ⭐⭐⭐⭐⭐ |
| 5.5 | 生成模型应用 | ⭐⭐⭐⭐ |

**核心内容**：
- 变分推断、潜在变量模型
- GAN训练技巧、模式崩溃
- Diffusion、DDPM、Stable Diffusion
- Normalizing Flows
- 图像生成、文本生成

---

### 第六部分：自监督学习（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 6.1 | 自监督学习概述 | ⭐⭐⭐⭐ |
| 6.2 | 对比学习 | ⭐⭐⭐⭐⭐ |
| 6.3 | 掩码建模 | ⭐⭐⭐⭐⭐ |
| 6.4 | 预测学习 | ⭐⭐⭐⭐ |
| 6.5 | 预训练范式 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 自监督学习定义与分类
- SimCLR、MoCo、BYOL
- BERT MLM、MAE
- 上下文预测、时序预测
- 预训练-微调范式

---

### 第七部分：多模态学习（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 7.1 | 多模态学习概述 | ⭐⭐⭐⭐ |
| 7.2 | 视觉-语言模型 | ⭐⭐⭐⭐⭐ |
| 7.3 | 多模态融合 | ⭐⭐⭐⭐⭐ |
| 7.4 | 跨模态生成 | ⭐⭐⭐⭐⭐ |
| 7.5 | 通用多模态模型 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 多模态表示学习
- CLIP、ALIGN、FLAVA
- 模态融合策略
- 图像描述、文本到图像
- GPT-4V、Gemini

---

### 第八部分：模型训练与优化（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 8.1 | 训练技巧 | ⭐⭐⭐⭐ |
| 8.2 | 正则化方法 | ⭐⭐⭐⭐ |
| 8.3 | 优化策略 | ⭐⭐⭐⭐⭐ |
| 8.4 | 分布式训练 | ⭐⭐⭐⭐⭐ |
| 8.5 | 模型压缩 | ⭐⭐⭐⭐ |

**核心内容**：
- 学习率调度、梯度裁剪
- 正则化、数据增强
- 混合精度训练
- 数据并行、模型并行
- 量化、剪枝、知识蒸馏

---

### 第九部分：应用领域（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 9.1 | 计算机视觉 | ⭐⭐⭐⭐ |
| 9.2 | 自然语言处理 | ⭐⭐⭐⭐ |
| 9.3 | 语音处理 | ⭐⭐⭐⭐ |
| 9.4 | 推荐系统 | ⭐⭐⭐⭐ |
| 9.5 | 具身智能应用 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 图像分类、目标检测、分割
- 文本分类、机器翻译、问答
- 语音识别、语音合成
- 个性化推荐、点击率预测
- 机器人视觉、具身感知

---

### 第十部分：论文详解（3周）

| 章节 | 论文 | 发表年份 |
|------|------|----------|
| 10.1 | "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet) | 2012 |
| 10.2 | "Deep Residual Learning for Image Recognition" (ResNet) | 2015 |
| 10.3 | "Attention Is All You Need" (Transformer) | 2017 |
| 10.4 | "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" | 2018 |
| 10.5 | "Language Models are Few-Shot Learners" (GPT-3) | 2020 |
| 10.6 | "Generative Adversarial Networks" (GAN) | 2014 |
| 10.7 | "Auto-Encoding Variational Bayes" (VAE) | 2014 |
| 10.8 | "Diffusion Models Beat GANs on Image Synthesis" | 2021 |
| 10.9 | "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT) | 2021 |
| 10.10 | "Masked Autoencoders Are Scalable Vision Learners" (MAE) | 2021 |

---

## 推荐学习资源

### 书籍
1. 《Deep Learning》- Goodfellow, Bengio, Courville
2. 《动手学深度学习》- 李沐
3. 《Python深度学习》- François Chollet
4. 《Transformers for Natural Language Processing》- Thomas Wolf
5. 《Generative Deep Learning》- David Foster

### 在线课程
1. **CS231n** - Stanford Computer Vision
2. **CS224n** - Stanford NLP with Deep Learning
3. **Fast.ai** - Practical Deep Learning
4. **MIT 6.S191** - Deep Learning
5. **DeepLearning.AI** - Deep Learning Specialization

### 工具与框架
1. **PyTorch** - 深度学习框架
2. **TensorFlow/Keras** - 深度学习框架
3. **Hugging Face Transformers** - 预训练模型库
4. **Detectron2** - 计算机视觉
5. **OpenAI CLIP** - 视觉-语言模型

---

## 学习评估

### 自测题
1. 解释反向传播算法的原理
2. 比较CNN、RNN和Transformer的适用场景
3. 什么是注意力机制？为什么它在Transformer中如此重要？
4. 比较GAN和VAE的优缺点
5. 解释扩散模型的工作原理
6. 什么是自监督学习？它与监督学习有什么区别？
7. 什么是迁移学习？为什么它很重要？
8. 解释BERT和GPT的区别

### 实践项目
1. 使用PyTorch实现CNN图像分类
2. 实现Transformer模型
3. 训练一个GAN生成图像
4. 使用扩散模型生成图像
5. 微调预训练语言模型
6. 使用CLIP进行图像检索
7. 实现简单的图像描述模型

---

## 模块关系

```
具身智能
├── 03-machine-learning (机器学习) ← 前置模块
├── 04-deep-learning (深度学习) ← 当前模块
├── 05-reinforcement-learning (强化学习) ← 后续模块
└── 10-large-models (大模型) ← 相关模块
```

**学习顺序建议**：
1. 先学习机器学习模块（03-machine-learning）
2. 再学习本模块（深度学习）
3. 最后学习强化学习模块（05-reinforcement-learning）

**前置知识**：
- 机器学习基础（监督学习、无监督学习）
- 线性代数、概率论
- Python编程基础

---

## 重要论文列表

| 论文标题 | 作者 | 发表期刊/会议 | 年份 | 核心贡献 |
|----------|------|---------------|------|----------|
| AlexNet | Krizhevsky et al. | NIPS | 2012 | 深度学习复兴 |
| ResNet | He et al. | CVPR | 2015 | 残差学习 |
| Transformer | Vaswani et al. | NIPS | 2017 | 自注意力机制 |
| BERT | Devlin et al. | NAACL | 2018 | 双向预训练 |
| GPT-3 | Brown et al. | NeurIPS | 2020 | 大规模语言模型 |
| GAN | Goodfellow et al. | NIPS | 2014 | 生成对抗网络 |
| VAE | Kingma & Welling | ICLR | 2014 | 变分自编码器 |
| Diffusion Models | Dhariwal & Nichol | NeurIPS | 2021 | 扩散模型 |
| ViT | Dosovitskiy et al. | ICLR | 2021 | 视觉Transformer |
| MAE | He et al. | CVPR | 2021 | 掩码自编码器 |

---

**上一个模块**：[机器学习](../03-machine-learning/README.md)
**下一个模块**：[强化学习](../05-reinforcement-learning/README.md)