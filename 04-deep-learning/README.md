
# 深度学习模块

## 模块概述

深度学习是当前人工智能的核心技术，本模块涵盖CNN、Transformer、PyTorch实践等内容。

## 学习目标

完成本模块学习后，您将能够：
- 理解深度学习的基本原理
- 掌握CNN和Transformer架构
- 能够使用PyTorch构建和训练模型

## 学习路径

### 第一部分：神经网络基础（2周）

#### 1.1 前向传播与反向传播

**核心概念：**
- 神经网络结构
- 激活函数
- 链式法则与反向传播

**推荐资源：**
- 《Deep Learning》- Goodfellow et al.
- 《Neural Networks and Deep Learning》- Nielsen

**实践练习：**
- 实现简单的全连接神经网络
- 手动推导反向传播

#### 1.2 训练技巧

**核心概念：**
- 损失函数
- 优化器（SGD、Adam）
- 正则化（Dropout、权重衰减）

**实践练习：**
- 实现不同的优化器
- 比较正则化效果

---

### 第二部分：卷积神经网络（2周）

#### 2.1 CNN基础

**核心概念：**
- 卷积操作
- 池化层
- 经典架构（LeNet、AlexNet）

**推荐论文：**
- "Gradient-Based Learning Applied to Document Recognition" (LeNet)
- "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet)

**实践练习：**
- 实现简单的CNN
- 训练图像分类模型

#### 2.2 高级CNN架构

**核心概念：**
- ResNet残差连接
- Inception模块
- 注意力机制

**推荐论文：**
- "Deep Residual Learning for Image Recognition" (ResNet)
- "Going Deeper with Convolutions" (Inception)

**实践练习：**
- 使用ResNet进行图像分类
- 理解残差连接的作用

---

### 第三部分：Transformer（2周）

#### 3.1 自注意力机制

**核心概念：**
- 注意力机制
- 多头注意力
- 位置编码

**推荐论文：**
- "Attention Is All You Need" (Transformer)

**实践练习：**
- 实现自注意力机制
- 理解Transformer架构

#### 3.2 视觉Transformer

**核心概念：**
- ViT架构
- 视觉注意力
- CLIP模型

**推荐论文：**
- "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT)
- "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)

**实践练习：**
- 使用CLIP进行图像检索
- 理解VLM的原理

---

## 推荐学习资源

### 书籍
1. 《Deep Learning》- Goodfellow et al.
2. 《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》- Geron
3. 《Transformers for Natural Language Processing》- Chopra

### 在线课程
1. Deep Learning Specialization (Coursera)
2. CS231n - Convolutional Neural Networks for Visual Recognition

### 工具
- PyTorch - 深度学习框架
- torchvision - 计算机视觉工具
- Hugging Face Transformers - Transformer库

## 学习评估

### 自测题
1. 解释反向传播的原理
2. 比较CNN与Transformer的优缺点
3. 解释CLIP的训练方式

### 实践项目
1. 使用PyTorch实现CNN图像分类
2. 实现简单的Transformer
3. 使用CLIP进行零样本分类

---

**下一个模块：** [强化学习](../05-reinforcement-learning/README.md)
