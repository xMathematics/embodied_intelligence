# 2.3 跨模态对齐

## 目录

- [1. 引言](#1-引言)
- [2. 跨模态对齐概述](#2-跨模态对齐概述)
- [3. 对比学习方法](#3-对比学习方法)
- [4. 注意力机制方法](#4-注意力机制方法)
- [5. 联合嵌入方法](#5-联合嵌入方法)
- [6. 生成式方法](#6-生成式方法)
- [7. 跨模态对齐评估](#7-跨模态对齐评估)
- [8. 实践练习](#8-实践练习)

---

## 1. 引言

### 1.1 跨模态对齐的重要性

**跨模态对齐**是建立不同模态（视觉、语言）之间语义对应关系的过程。在VLM中，有效的跨模态对齐是实现视觉-语言理解和生成的关键。

### 1.2 核心挑战

| 挑战 | 描述 |
|------|------|
| **模态差异** | 视觉和语言数据的本质差异 |
| **语义鸿沟** | 像素和文字之间的语义差距 |
| **数据稀疏** | 高质量图文对标注成本高 |
| **多义性** | 同一视觉内容可能有多种语言描述 |

---

## 2. 跨模态对齐概述

### 2.1 对齐目标

**目标**：找到视觉特征和语言特征之间的映射关系，使得语义相似的内容在特征空间中接近。

### 2.2 对齐层次

| 层次 | 描述 | 示例 |
|------|------|------|
| **单词级对齐** | 图像区域与单词的对应 | "猫"对应图像中的猫 |
| **短语级对齐** | 图像区域与短语的对应 | "一只黑猫"对应特定区域 |
| **句子级对齐** | 整个图像与句子的对应 | 图像描述 |
| **篇章级对齐** | 图像序列与文本的对应 | 视频描述 |

### 2.3 对齐方法分类

| 方法类型 | 描述 | 代表工作 |
|---------|------|---------|
| **对比学习** | 通过对比正负样本学习对齐 | CLIP、ALIGN |
| **注意力机制** | 使用注意力建立对应关系 | ViLBERT、LXMERT |
| **联合嵌入** | 将模态映射到同一空间 | MIL-NCE、CLIP |
| **生成式** | 通过生成任务学习对齐 | BLIP、Flamingo |

---

## 3. 对比学习方法

### 3.1 核心思想

通过最大化正样本对的相似度，最小化负样本对的相似度来学习对齐。

### 3.2 CLIP的对比学习

**对比损失函数**：
```
L = -log(exp(sim(I,T+)) / sum(exp(sim(I,T))) for all T)
```

**架构**：
```
图像 → ViT → 图像特征 (归一化)
文本 → BERT → 文本特征 (归一化)
相似度 = 图像特征 · 文本特征
对比学习：匹配对相似度最大化，非匹配对最小化
```

### 3.3 代码示例

```python
import torch
import torch.nn.functional as F

def contrastive_loss(image_features, text_features, temperature=0.07):
    """
    计算对比损失（CLIP风格）
    
    参数:
        image_features: [batch_size, dim]
        text_features: [batch_size, dim]
        temperature: 温度系数
    
    返回:
        对比损失
    """
    # 归一化特征
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # 计算相似度矩阵
    logits = image_features @ text_features.t() / temperature
    
    # 标签：对角线是正样本
    labels = torch.arange(logits.shape[0]).to(logits.device)
    
    # 双向对比损失
    loss_img = F.cross_entropy(logits, labels)
    loss_txt = F.cross_entropy(logits.t(), labels)
    
    return (loss_img + loss_txt) / 2

# 测试
image_feat = torch.randn(8, 512)
text_feat = torch.randn(8, 512)
loss = contrastive_loss(image_feat, text_feat)
print(f"对比损失: {loss.item()}")
```

### 3.4 ALIGN的改进

**特点**：
- 使用更大规模的数据（18亿图文对）
- 处理噪声数据的鲁棒性
- 多语言支持

---

## 4. 注意力机制方法

### 4.1 视觉-语言注意力

**核心思想**：通过注意力机制动态地建立视觉区域和语言token之间的对应关系。

### 4.2 ViLBERT的双流注意力

**架构**：
```
图像特征 → 视觉Transformer
文本特征 → 语言Transformer
交叉注意力：视觉→语言，语言→视觉
```

**交叉注意力层**：
```python
class CrossAttention(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(dim, num_heads)
    
    def forward(self, query, key, value):
        # query: 目标模态特征
        # key, value: 源模态特征
        output, _ = self.multihead_attn(query, key, value)
        return output
```

### 4.3 LXMERT的跨模态注意力

**三个预训练任务**：
1. **Masked Language Modeling (MLM)**：预测被掩盖的词
2. **Masked Region Modeling (MRM)**：预测被掩盖的图像区域
3. **Visual Question Answering (VQA)**：回答关于图像的问题

---

## 5. 联合嵌入方法

### 5.1 核心思想

将不同模态的特征映射到同一个嵌入空间，使得语义相似的内容在空间中距离相近。

### 5.2 嵌入空间设计

| 设计策略 | 描述 |
|---------|------|
| **共享空间** | 视觉和语言特征投影到同一空间 |
| **双向映射** | 视觉→语言空间，语言→视觉空间 |
| **模态特定空间** | 每个模态有独立空间，学习映射 |

### 5.3 度量学习

**常用距离度量**：
| 度量 | 公式 | 特点 |
|------|------|------|
| **余弦相似度** | cos(θ) = A·B/(||A||·||B||) | 方向相似性 |
| **欧氏距离** | ||A-B|| | 空间距离 |
| **曼哈顿距离** | Σ|A_i - B_i| | L1距离 |

### 5.4 代码示例：联合嵌入

```python
class JointEmbedding(torch.nn.Module):
    def __init__(self, vision_dim, text_dim, joint_dim=512):
        super().__init__()
        self.vision_proj = torch.nn.Linear(vision_dim, joint_dim)
        self.text_proj = torch.nn.Linear(text_dim, joint_dim)
        self.norm = torch.nn.LayerNorm(joint_dim)
    
    def forward(self, vision_feat, text_feat):
        # 投影到联合空间
        vision_proj = self.norm(self.vision_proj(vision_feat))
        text_proj = self.norm(self.text_proj(text_feat))
        
        # 归一化
        vision_proj = F.normalize(vision_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)
        
        return vision_proj, text_proj

# 测试
model = JointEmbedding(768, 768)
vision_feat = torch.randn(8, 768)
text_feat = torch.randn(8, 768)
vision_joint, text_joint = model(vision_feat, text_feat)

# 计算相似度
similarity = (vision_joint * text_joint).sum(dim=-1)
print(f"平均相似度: {similarity.mean().item()}")
```

---

## 6. 生成式方法

### 6.1 核心思想

通过生成任务（如图像描述、视觉问答）来学习跨模态对齐。

### 6.2 图像描述任务

**目标**：根据图像生成自然语言描述

**架构**：
```
图像 → 视觉编码器 → 图像特征
图像特征 → 解码器 → 文本序列
```

### 6.3 BLIP的统一框架

**三种预训练任务**：
1. **图像描述生成**：生成图像的文字描述
2. **图像-文本匹配**：判断图文是否匹配
3. **对比学习**：CLIP风格的对比损失

**代码示例**：
```python
from transformers import BlipProcessor, BlipForConditionalGeneration

# 加载模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 图像描述生成
image = Image.open("example.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt")
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(f"图像描述: {caption}")
```

### 6.4 Flamingo的视觉特征注入

**核心思想**：将视觉特征作为特殊token注入语言模型

**架构**：
```
图像 → 视觉编码器 → 视觉token
视觉token + 文本token → 统一序列
统一序列 → 语言模型 → 输出
```

---

## 7. 跨模态对齐评估

### 7.1 评估指标

| 任务 | 指标 | 描述 |
|------|------|------|
| **图文检索** | Recall@k | 检索准确率 |
| **图像描述** | CIDEr, BLEU | 生成质量 |
| **视觉问答** | Accuracy | 回答正确率 |
| **零样本分类** | Top-k Accuracy | 分类准确率 |

### 7.2 评估数据集

| 数据集 | 任务 | 特点 |
|--------|------|------|
| **Flickr30k** | 图文检索 | 31K图像 |
| **MSCOCO** | 图像描述 | 123K图像 |
| **VQA v2** | 视觉问答 | 1.1M问题 |
| **ImageNet** | 零样本分类 | 14M图像 |

### 7.3 对齐质量分析

**定性分析**：
- 可视化注意力权重
- 检查生成描述的合理性
- 人工评估问答质量

**定量分析**：
- 计算特征空间的对齐程度
- 使用互信息度量模态相关性

---

## 8. 实践练习

### 练习1：实现对比学习训练

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 模拟数据
num_samples = 1000
vision_dim = 512
text_dim = 512

vision_features = torch.randn(num_samples, vision_dim)
text_features = torch.randn(num_samples, text_dim)

# 创建数据集（假设每个样本的视觉和文本是匹配的）
dataset = TensorDataset(vision_features, text_features)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义投影层
vision_proj = nn.Linear(vision_dim, 512)
text_proj = nn.Linear(text_dim, 512)

# 优化器
optimizer = optim.Adam(list(vision_proj.parameters()) + list(text_proj.parameters()), lr=1e-4)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch_vision, batch_text in dataloader:
        optimizer.zero_grad()
        
        # 投影到联合空间
        vision_emb = F.normalize(vision_proj(batch_vision), dim=-1)
        text_emb = F.normalize(text_proj(batch_text), dim=-1)
        
        # 计算对比损失
        logits = vision_emb @ text_emb.t() / 0.07
        labels = torch.arange(logits.shape[0])
        
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
```

### 练习2：可视化跨模态注意力

```python
import torch
import matplotlib.pyplot as plt
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 加载模型（需要修改模型以输出注意力）
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 加载图像和文本
image = Image.open("example.jpg").convert("RGB")
text = "a cat sitting on a couch"

# 获取注意力权重（需要修改模型配置）
inputs = processor(image, text, return_tensors="pt")
outputs = model(**inputs, output_attentions=True)

# 获取跨模态注意力（假设模型返回注意力）
# 注意：实际实现需要修改模型以返回视觉-文本注意力
cross_attention = outputs.cross_attentions[-1][0, 0].detach().numpy()

# 可视化
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cross_attention, cmap='viridis')
plt.title("Cross-Modal Attention")
plt.xlabel("Text Tokens")
plt.ylabel("Image Patches")
plt.show()
```

### 练习3：跨模态检索系统

```python
import torch
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 加载CLIP模型
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# 索引图像数据库
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
image_features = []

for path in image_paths:
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    image_features.append(features)

# 构建FAISS索引
index = faiss.IndexFlatL2(512)
image_features_np = torch.cat(image_features).numpy()
index.add(image_features_np)

# 文本检索图像
query_text = "a cat"
text_inputs = processor(text=query_text, return_tensors="pt")
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)

# 检索
D, I = index.search(text_features.numpy(), k=2)
print(f"查询: {query_text}")
print(f"检索结果: {[image_paths[i] for i in I[0]]}")
```

---

**下一节**：[视觉问答](04-visual-question-answering.md)

---

## 参考文献

1. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision.
2. Lu, J., Batra, D., Parikh, D., & Lee, S. (2019). Vilbert: Pretraining task-agnostic visiolinguistic representations.
3. Tan, H., & Bansal, M. (2019). Lxmert: Learning cross-modality encoder representations.
4. Li, J., Li, D., Xiong, C., & Hoi, S. C. (2022). Blip: Bootstrapping language-image pre-training.
