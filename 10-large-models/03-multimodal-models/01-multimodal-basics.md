# 3.1 多模态基础

## 目录

- [1. 引言](#1-引言)
- [2. 多模态学习概述](#2-多模态学习概述)
- [3. 模态类型与特征](#3-模态类型与特征)
- [4. 多模态学习挑战](#4-多模态学习挑战)
- [5. 多模态融合方法](#5-多模态融合方法)
- [6. 多模态预训练策略](#6-多模态预训练策略)
- [7. 多模态评估基准](#7-多模态评估基准)
- [8. 实践练习](#8-实践练习)

---

## 1. 引言

### 1.1 什么是多模态学习

**多模态学习**（Multimodal Learning）是指同时处理和理解多种模态数据的机器学习方法。模态可以包括文本、图像、音频、视频、3D数据等。

### 1.2 多模态学习的重要性

| 方面 | 说明 |
|------|------|
| **更丰富的信息** | 多种模态提供互补的信息 |
| **更好的理解** | 综合多种信息源提高理解能力 |
| **鲁棒性** | 多种模态可以互相补充，提高系统鲁棒性 |
| **贴近人类认知** | 人类通常同时使用多种感官感知世界 |

---

## 2. 多模态学习概述

### 2.1 多模态数据类型

| 模态 | 数据类型 | 特点 |
|------|---------|------|
| **文本** | 自然语言 | 抽象、符号化 |
| **图像** | 静态视觉信息 | 直观、丰富 |
| **音频** | 声音信息 | 时序、连续 |
| **视频** | 动态视觉信息 | 包含时间维度 |
| **3D** | 三维结构信息 | 空间感知 |
| **传感器** | 各类传感器数据 | 实时、精确 |

### 2.2 多模态学习任务

| 任务类型 | 描述 | 示例 |
|---------|------|------|
| **多模态理解** | 理解多种模态的内容 | 视频字幕生成 |
| **多模态生成** | 根据多种模态生成内容 | 文本+图像生成新图像 |
| **跨模态检索** | 在不同模态间检索 | 文本检索视频 |
| **多模态问答** | 根据多种模态回答问题 | 视频问答 |
| **多模态翻译** | 在不同模态间转换 | 语音转文字 |

### 2.3 多模态学习架构

```
┌─────────────────────────────────────────────────────────────┐
│                   多模态学习架构                            │
├─────────────────────────────────────────────────────────────┤
│  模态1编码器  │  模态2编码器  │  模态3编码器  │      │      │
│  (文本)      │  (图像)      │  (音频)      │ ...  │      │
│      │       │       │       │       │       │      │      │
│      ▼       │       ▼       │       ▼       │      │      │
│  ┌──────────────────────────────────────────────────┐      │
│  │              多模态融合模块                       │      │
│  │  (融合策略: 早期融合/晚期融合/混合融合)           │      │
│  └──────────────────────────────────────────────────┘      │
│                          │                                 │
│                          ▼                                 │
│              ┌─────────────────┐                           │
│              │    输出层        │                           │
│              │  (分类/生成/检索)│                           │
│              └─────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 模态类型与特征

### 3.1 文本模态

**特征**：
- 离散符号序列
- 具有语法和语义结构
- 可以表示抽象概念

**常用编码器**：
- BERT、GPT等Transformer模型
- LSTM、GRU等序列模型

### 3.2 图像模态

**特征**：
- 连续像素值
- 空间结构信息
- 颜色、纹理、形状

**常用编码器**：
- CNN（ResNet、EfficientNet）
- Vision Transformer（ViT、Swin）

### 3.3 音频模态

**特征**：
- 连续波形
- 频率信息
- 时间序列

**常用编码器**：
- CNN（音频谱图）
- RNN、Transformer
- 专门模型（Whisper、AudioLM）

### 3.4 视频模态

**特征**：
- 图像序列
- 时间动态信息
- 空间+时间结构

**常用编码器**：
- 3D CNN
- 2D CNN + 时序模型
- Video Transformer（TimeSformer）

### 3.5 3D模态

**特征**：
- 点云、网格、体素
- 三维空间结构
- 几何信息

**常用编码器**：
- PointNet、PointTransformer
- 3D CNN
- 神经辐射场（NeRF）

---

## 4. 多模态学习挑战

### 4.1 模态异质性

**问题**：不同模态的数据格式和特征差异很大

| 挑战 | 描述 | 示例 |
|------|------|------|
| **表示差异** | 文本是离散符号，图像是连续像素 | 难以直接比较 |
| **尺度差异** | 不同模态的数据量和维度差异大 | 文本长度vs图像像素数 |
| **噪声特性** | 不同模态的噪声类型不同 | 图像噪声vs语音噪声 |

### 4.2 模态缺失

**问题**：在实际应用中，某些模态可能缺失

| 场景 | 描述 |
|------|------|
| **部分缺失** | 某些样本缺少特定模态 | 视频没有音频 |
| **完全缺失** | 某些模态完全不可用 | 没有摄像头 |

### 4.3 跨模态语义鸿沟

**问题**：不同模态之间的语义映射困难

| 鸿沟类型 | 描述 |
|---------|------|
| **词汇-视觉鸿沟** | 词语和视觉概念之间的对应 | "猫"对应图像中的猫 |
| **抽象-具体鸿沟** | 抽象概念和具体感知之间的对应 | "快乐"的视觉表现 |

### 4.4 计算复杂度

**问题**：处理多种模态会显著增加计算负担

| 挑战 | 描述 |
|------|------|
| **模型规模** | 多模态模型通常更大 | 参数更多 |
| **训练数据** | 需要更多样化的数据 | 数据收集成本高 |
| **推理速度** | 多模态推理通常更慢 | 实时应用受限 |

---

## 5. 多模态融合方法

### 5.1 融合层次

| 层次 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **早期融合** | 在特征提取前融合 | 保留细粒度信息 | 计算量大 |
| **中期融合** | 在特征级别融合 | 平衡信息与计算 | 需要对齐 |
| **晚期融合** | 在决策级别融合 | 简单灵活 | 丢失交互信息 |

### 5.2 融合策略

| 策略 | 描述 | 代表方法 |
|------|------|---------|
| **拼接** | 简单拼接特征 | Concatenation |
| **注意力** | 使用注意力加权融合 | Cross-Attention |
| **门控机制** | 使用门控控制信息流 | Gated Fusion |
| **张量融合** | 使用外积建模交互 | Tensor Fusion Network |
| **Transformer** | 使用Transformer处理多模态 | MMT、FLAVA |

### 5.3 代码示例：多模态融合

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusion(nn.Module):
    def __init__(self, text_dim=768, vision_dim=512, audio_dim=128, hidden_dim=512):
        super().__init__()
        # 投影层
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 分类头
        self.classifier = nn.Linear(hidden_dim, 10)
    
    def forward(self, text_feat, vision_feat, audio_feat):
        # 投影到统一维度
        text_proj = F.relu(self.text_proj(text_feat))
        vision_proj = F.relu(self.vision_proj(vision_feat))
        audio_proj = F.relu(self.audio_proj(audio_feat))
        
        # 拼接融合
        fused = torch.cat([text_proj, vision_proj, audio_proj], dim=-1)
        fused = self.fusion(fused)
        
        # 分类
        logits = self.classifier(fused)
        return logits

# 测试
model = MultimodalFusion()
text_feat = torch.randn(8, 768)
vision_feat = torch.randn(8, 512)
audio_feat = torch.randn(8, 128)
output = model(text_feat, vision_feat, audio_feat)
print(f"输出形状: {output.shape}")  # [8, 10]
```

---

## 6. 多模态预训练策略

### 6.1 预训练目标

| 目标类型 | 描述 | 示例 |
|---------|------|------|
| **对比学习** | 对齐不同模态的特征 | CLIP、ALIGN |
| **掩码建模** | 预测被掩盖的模态内容 | MMM、FLAVA |
| **生成任务** | 生成缺失的模态内容 | Flamingo、GPT-4V |
| **匹配任务** | 判断模态是否匹配 | 图文匹配 |

### 6.2 统一预训练框架

| 框架 | 特点 | 支持的模态 |
|------|------|-----------|
| **FLAVA** | 统一多模态理解和生成 | 文本、图像 |
| **BEiT-3** | 统一视觉-语言预训练 | 文本、图像 |
| **MM-LLM** | 大语言模型+多模态 | 多种模态 |
| **GPT-4** | 原生多模态支持 | 文本、图像 |

### 6.3 跨模态预训练

**流程**：
```
多模态数据 → 分别编码 → 跨模态交互 → 预训练任务 → 微调下游任务
```

**优势**：
- 学习通用的跨模态表示
- 提高下游任务性能
- 支持零样本迁移

---

## 7. 多模态评估基准

### 7.1 评估数据集

| 数据集 | 模态 | 任务 | 规模 |
|--------|------|------|------|
| **MSRVTT** | 视频+文本 | 视频字幕 | 10K视频 |
| **LSMDC** | 视频+文本 | 视频字幕 | 220K视频 |
| **ActivityNet** | 视频+文本 | 动作识别 | 200K视频 |
| **VGGSound** | 音频+视频 | 音频-视觉匹配 | 130K视频 |
| **Flickr30k** | 图像+文本 | 图文检索 | 31K图像 |

### 7.2 评估指标

| 任务 | 指标 | 描述 |
|------|------|------|
| **检索** | Recall@k | 检索准确率 |
| **字幕生成** | BLEU、METEOR、CIDEr | 生成质量 |
| **分类** | Accuracy、F1 | 分类性能 |
| **问答** | Accuracy | 回答正确率 |

---

## 8. 实践练习

### 练习1：多模态特征融合

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads)
    
    def forward(self, query, key, value):
        # query: 目标模态
        # key, value: 源模态
        output, attn_weights = self.multihead_attn(query, key, value)
        return output, attn_weights

# 测试跨模态注意力
attention = CrossModalAttention()
text_feat = torch.randn(10, 8, 512)  # [seq_len, batch, dim]
vision_feat = torch.randn(196, 8, 512)  # [patch_num, batch, dim]

# 文本引导的视觉注意力
output, weights = attention(text_feat, vision_feat, vision_feat)
print(f"输出形状: {output.shape}")  # [10, 8, 512]
print(f"注意力权重形状: {weights.shape}")  # [8, 10, 196]
```

### 练习2：多模态对比学习

```python
import torch
import torch.nn.functional as F

def multimodal_contrastive_loss(modalities, temperature=0.07):
    """
    多模态对比损失
    
    参数:
        modalities: 各模态特征列表 [mod1_feat, mod2_feat, ...]
        temperature: 温度系数
    
    返回:
        对比损失
    """
    # 归一化所有特征
    normalized = [F.normalize(feat, dim=-1) for feat in modalities]
    
    # 计算所有模态对的相似度
    loss = 0
    num_modalities = len(normalized)
    
    for i in range(num_modalities):
        for j in range(i+1, num_modalities):
            # 计算相似度矩阵
            sim = normalized[i] @ normalized[j].t() / temperature
            batch_size = sim.shape[0]
            labels = torch.arange(batch_size).to(sim.device)
            
            # 双向损失
            loss += (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
    
    return loss / (num_modalities * (num_modalities - 1) / 2)

# 测试
text_feat = torch.randn(8, 512)
vision_feat = torch.randn(8, 512)
audio_feat = torch.randn(8, 512)

loss = multimodal_contrastive_loss([text_feat, vision_feat, audio_feat])
print(f"多模态对比损失: {loss.item()}")
```

### 练习3：多模态数据加载

```python
import json
import os
from PIL import Image
import torchaudio

class MultimodalDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.metadata = self.load_metadata()
    
    def load_metadata(self):
        """加载元数据"""
        metadata_file = os.path.join(self.data_dir, "metadata.json")
        with open(metadata_file, 'r') as f:
            return json.load(f)
    
    def __getitem__(self, idx):
        """获取样本"""
        item = self.metadata[idx]
        
        # 加载文本
        text = item['text']
        
        # 加载图像
        image_path = os.path.join(self.data_dir, item['image_path'])
        image = Image.open(image_path).convert("RGB")
        
        # 加载音频
        audio_path = os.path.join(self.data_dir, item['audio_path'])
        waveform, sample_rate = torchaudio.load(audio_path)
        
        return {
            'text': text,
            'image': image,
            'audio': waveform,
            'sample_rate': sample_rate,
            'label': item['label']
        }
    
    def __len__(self):
        return len(self.metadata)

# 使用示例
# dataset = MultimodalDataset("path/to/dataset")
# sample = dataset[0]
# print(f"文本: {sample['text']}")
# print(f"图像形状: {sample['image'].size}")
# print(f"音频形状: {sample['audio'].shape}")
# print(f"标签: {sample['label']}")
```

---

**下一节**：[音频-语言模型](02-audio-language.md)

---

## 参考文献

1. Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2018). Multimodal machine learning: A survey and taxonomy.
2. Wang, W., Liu, Y., Wu, J., & Wang, L. (2021). FLAVA: A Foundational Language and Vision Alignment Model.
3. Bao, H., Dong, L., Wei, F., & Keutzer, K. (2021). BEiT: BERT Pre-Training of Image Transformers.
4. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision.
