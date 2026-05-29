# 论文与综述

## 概述

本模块介绍大模型领域的重要论文和综述。

## 内容列表

### 1. 大语言模型论文
- Transformer架构
- GPT系列
- LLaMA系列

### 2. 视觉-语言模型论文
- CLIP
- FLAVA
- GPT-4V

### 3. 多模态模型论文
- 多模态Transformer
- 视频-语言模型
- 3D-语言模型

### 4. 世界模型论文
- World Models
- Dreamer
- 视频预测

### 5. 具身AI模型论文
- PaLM-E
- OpenVLA
- RT-X

## 重要论文详解

### 1. Attention Is All You Need (2017)

**作者**: Vaswani et al.

**发表会议**: NeurIPS

**核心贡献**:
- 提出Transformer架构
- 自注意力机制
- 多头注意力

**架构**:
- 编码器-解码器结构
- 位置编码
- 残差连接

**解决的问题**:
- 序列建模
- 长距离依赖

---

### 2. Language Models are Few-Shot Learners (2020)

**作者**: Brown et al.

**发表会议**: NeurIPS

**核心贡献**:
- GPT-3模型
- 上下文学习能力
- 少样本学习

**架构**:
- 175B参数模型
- 自回归语言建模
- 多样化预训练数据

**解决的问题**:
- 少样本泛化
- 零样本学习

---

### 3. CLIP: Contrastive Language-Image Pretraining (2021)

**作者**: Radford et al.

**发表会议**: ICML

**核心贡献**:
- 对比学习预训练
- 图文匹配
- 零样本迁移

**架构**:
- 双编码器结构
- 对比损失
- 跨模态对齐

**解决的问题**:
- 多模态理解
- 零样本分类

---

### 4. PaLM-E: An Embodied Multimodal Language Model (2023)

**作者**: Driess et al.

**发表会议**: arXiv

**核心贡献**:
- 具身多模态语言模型
- 视觉-语言-行动统一
- 端到端具身推理

**架构**:
- 大语言模型作为核心
- 多模态输入嵌入
- 行动输出

**解决的问题**:
- 具身推理
- 机器人指令跟随

---

### 5. World Models (2018)

**作者**: Ha & Schmidhuber

**发表会议**: NeurIPS

**核心贡献**:
- 世界模型概念
- 视觉预测模型
- 模型预测控制

**架构**:
- VAE视觉编码器
- MDN-RNN动态模型
- CMA-ES控制器

**解决的问题**:
- 强化学习样本效率
- 长期规划

---

### 6. DreamerV3: Mastering Atari with Discrete World Models (2023)

**作者**: Hafner et al.

**发表会议**: ICLR

**核心贡献**:
- 离散世界模型
- 单样本学习
- 高性能强化学习

**架构**:
- 离散潜在空间
- 世界模型训练
- 想象增强学习

**解决的问题**:
- 样本高效学习
- 复杂环境决策

---

### 7. OpenVLA: Open Vocabulary Vision-Language-Action Models (2023)

**作者**: Brohan et al.

**发表会议**: arXiv

**核心贡献**:
- 开放词汇VLA模型
- 视觉语言行动统一
- 机器人操控

**架构**:
- 视觉编码器
- LLM作为核心
- 行动头

**解决的问题**:
- 开放词汇操控
- 泛化能力

---

### 8. RLHF: Reinforcement Learning from Human Feedback (2017)

**作者**: Christiano et al.

**发表会议**: NeurIPS

**核心贡献**:
- 人类反馈强化学习
- 对齐技术
- 偏好建模

**架构**:
- 奖励模型训练
- 策略优化
- 迭代改进

**解决的问题**:
- 模型对齐
- 安全性

---

### 9. vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention (2023)

**作者**: Kwon et al.

**发表会议**: arXiv

**核心贡献**:
- 高效LLM推理
- PagedAttention机制
- 高吞吐量

**架构**:
- 注意力优化
- 动态批处理
- 显存管理

**解决的问题**:
- LLM部署成本
- 服务性能

---

### 10. QLoRA: Efficient Finetuning of Quantized LLMs (2023)

**作者**: Dettmers et al.

**发表会议**: arXiv

**核心贡献**:
- 高效量化微调
- 低显存训练
- 保持性能

**架构**:
- 4-bit量化
- LoRA适配器
- 梯度优化

**解决的问题**:
- 微调成本
- 资源受限