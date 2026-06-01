# 3.3 视频-语言模型

## 目录

- [1. 引言](#1-引言)
- [2. 视频-语言学习概述](#2-视频-语言学习概述)
- [3. 视频特征提取](#3-视频特征提取)
- [4. 视频-语言模型架构](#4-视频-语言模型架构)
- [5. 代表性模型](#5-代表性模型)
- [6. 视频-语言任务](#6-视频-语言任务)
- [7. 实践练习](#7-实践练习)

---

## 1. 引言

### 1.1 视频-语言模型的重要性

**视频-语言模型**是能够理解视频内容并生成相关语言描述的AI模型，在视频理解、视频检索、视频问答等领域有广泛应用。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **视频字幕** | 为视频生成字幕 | 自动字幕生成 |
| **视频检索** | 根据文本检索视频 | 查找相关视频 |
| **视频问答** | 根据视频回答问题 | 理解视频内容 |
| **视频摘要** | 生成视频摘要 | 浓缩视频内容 |

---

## 2. 视频-语言学习概述

### 2.1 视频数据特点

| 特点 | 描述 |
|------|------|
| **时空结构** | 包含时间和空间信息 |
| **长序列** | 视频通常较长 |
| **高维度** | 每帧都是高维图像 |
| **动态信息** | 包含运动信息 |

### 2.2 视频-语言任务类型

| 任务类型 | 描述 | 示例 |
|---------|------|------|
| **视频字幕生成** | 为视频生成描述 | 视频内容描述 |
| **视频问答** | 根据视频回答问题 | "视频中的人在做什么？" |
| **文本到视频检索** | 根据文本查找视频 | 检索相关视频 |
| **视频到文本检索** | 根据视频查找文本 | 视频内容匹配 |
| **视频摘要** | 生成视频摘要 | 浓缩视频 |

---

## 3. 视频特征提取

### 3.1 帧级特征提取

**使用预训练视觉模型**：
```python
import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image

# 加载ViT模型
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# 提取单帧特征
frame = Image.open("frame.jpg").convert("RGB")
inputs = processor(images=frame, return_tensors="pt")
outputs = model(**inputs)
frame_feature = outputs.last_hidden_state[:, 0, :]  # [CLS] token
print(f"单帧特征形状: {frame_feature.shape}")  # [1, 768]
```

### 3.2 时序特征建模

**使用Transformer处理时序**：
```python
import torch
import torch.nn as nn

class TemporalModel(nn.Module):
    def __init__(self, frame_dim=768, hidden_dim=512, num_frames=32):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(num_frames, frame_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=frame_dim, nhead=8),
            num_layers=6
        )
    
    def forward(self, frame_features):
        # frame_features: [batch, num_frames, frame_dim]
        batch_size = frame_features.shape[0]
        num_frames = frame_features.shape[1]
        
        # 添加位置编码
        pos_embed = self.pos_embedding[:num_frames, :].unsqueeze(0).repeat(batch_size, 1, 1)
        frame_features = frame_features + pos_embed
        
        # Transformer处理
        output = self.transformer(frame_features.transpose(0, 1))
        return output.transpose(0, 1)
```

### 3.3 视频编码器类型

| 类型 | 描述 | 代表方法 |
|------|------|---------|
| **3D CNN** | 直接处理3D数据 | C3D, I3D |
| **2D CNN + RNN** | 先提取帧特征，再建模时序 | CNN+LSTM |
| **2D CNN + Transformer** | 使用Transformer处理时序 | TimeSformer |
| **视频Transformer** | 专门设计的视频Transformer | VideoMAE |

---

## 4. 视频-语言模型架构

### 4.1 双流架构

**视觉和语言分开处理，然后融合**：
```python
import torch
import torch.nn as nn

class VideoLanguageModel(nn.Module):
    def __init__(self, video_dim=768, text_dim=768, hidden_dim=512):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, video_feat, text_feat):
        # 投影
        video_proj = torch.tanh(self.video_proj(video_feat))
        text_proj = torch.tanh(self.text_proj(text_feat))
        
        # 融合
        fused = torch.cat([video_proj, text_proj], dim=-1)
        fused = torch.tanh(self.fusion(fused))
        
        return fused
```

### 4.2 跨模态注意力

**视频和语言之间的注意力交互**：
```python
class CrossModalVideoAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads)
    
    def forward(self, video_feat, text_feat):
        # video_feat: [num_frames, batch, dim]
        # text_feat: [seq_len, batch, dim]
        
        # 文本引导的视频注意力
        output, weights = self.multihead_attn(
            query=text_feat,
            key=video_feat,
            value=video_feat
        )
        return output, weights
```

---

## 5. 代表性模型

### 5.1 TimeSformer

**论文**：Is Space-Time Attention All You Need for Video Understanding? (Bertasius et al., 2021)

**核心特点**：
- 纯Transformer架构
- 不同的时空注意力模式
- 高效的视频理解

**注意力模式**：
| 模式 | 描述 |
|------|------|
| **Spatio-Temporal Attention** | 同时关注空间和时间 |
| **Divided Space-Time Attention** | 分开处理空间和时间 |
| **Factorized Attention** | 分解时空注意力 |

### 5.2 VideoMAE

**论文**：VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training (Xu et al., 2022)

**核心特点**：
- 掩码自编码器
- 高效的自监督学习
- 数据效率高

### 5.3 Flamingo

**论文**：Flamingo: a Visual Language Model for Few-Shot Learning (Alayrac et al., 2022)

**核心特点**：
- 冻结预训练模型
- 注入视觉特征
- 支持视频理解

---

## 6. 视频-语言任务

### 6.1 视频字幕生成

**定义**：为视频生成文字描述

**评估指标**：
- BLEU、METEOR、CIDEr

**代码示例**：
```python
from transformers import VideoMAEProcessor, VideoMAEModel
import torch

# 加载模型
processor = VideoMAEProcessor.from_pretrained("facebook/videomae-base")
model = VideoMAEModel.from_pretrained("facebook/videomae-base")

# 假设有视频帧
video_frames = [Image.open(f"frame_{i}.jpg") for i in range(16)]

# 预处理
inputs = processor(video_frames, return_tensors="pt")

# 提取特征
outputs = model(**inputs)
video_feature = outputs.last_hidden_state[:, 0, :]  # [CLS] token
print(f"视频特征形状: {video_feature.shape}")
```

### 6.2 视频问答

**定义**：根据视频内容回答问题

**代码示例**：
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 加载模型（需要视频理解模型）
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 对于视频，我们可以采样关键帧
key_frames = [Image.open(f"keyframe_{i}.jpg") for i in range(4)]

# 可以对每个关键帧进行描述，然后汇总
descriptions = []
for frame in key_frames:
    inputs = processor(frame, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    descriptions.append(description)

print(f"视频帧描述: {descriptions}")
```

### 6.3 视频检索

**定义**：根据文本检索视频

**流程**：
```
文本查询 → 文本特征 → 与视频特征库匹配 → 返回匹配视频
```

---

## 7. 实践练习

### 练习1：视频帧特征提取

```python
import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import os

# 加载ViT模型
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")

# 提取视频帧特征
def extract_video_features(frame_dir):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    features = []
    
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = Image.open(frame_path).convert("RGB")
        inputs = processor(images=frame, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_feature = outputs.last_hidden_state[:, 0, :]
            features.append(cls_feature)
    
    # 堆叠成视频特征
    video_feature = torch.stack(features).squeeze(1)  # [num_frames, dim]
    return video_feature

# 使用示例
# video_feature = extract_video_features("path/to/frames")
# print(f"视频特征形状: {video_feature.shape}")
```

### 练习2：视频-文本匹配

```python
import torch
import torch.nn.functional as F

def video_text_matching(video_features, text_features, temperature=0.07):
    """
    计算视频和文本的匹配度
    
    参数:
        video_features: [batch, num_frames, dim]
        text_features: [batch, dim]
    
    返回:
        匹配分数
    """
    # 平均池化视频帧特征
    video_avg = video_features.mean(dim=1)  # [batch, dim]
    
    # 归一化
    video_norm = F.normalize(video_avg, dim=-1)
    text_norm = F.normalize(text_features, dim=-1)
    
    # 计算相似度
    similarity = (video_norm * text_norm).sum(dim=-1) / temperature
    return similarity

# 测试
video_feat = torch.randn(8, 16, 768)  # [batch, num_frames, dim]
text_feat = torch.randn(8, 768)       # [batch, dim]

similarity = video_text_matching(video_feat, text_feat)
print(f"视频-文本匹配分数: {similarity}")
```

### 练习3：简单的视频问答系统

```python
class VideoQA(nn.Module):
    def __init__(self, video_dim=768, text_dim=768, hidden_dim=512, num_answers=1000):
        super().__init__()
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(hidden_dim, num_answers)
    
    def forward(self, video_feat, question_feat):
        # 视频特征平均池化
        video_avg = video_feat.mean(dim=1)
        video_proj = F.relu(self.video_proj(video_avg))
        
        # 问题特征（使用CLS token）
        question_proj = F.relu(self.text_proj(question_feat[:, 0, :]))
        
        # 融合
        fused = torch.cat([video_proj, question_proj], dim=-1)
        fused = self.fusion(fused)
        
        # 分类
        logits = self.classifier(fused)
        return logits

# 测试
model = VideoQA()
video_feat = torch.randn(8, 16, 768)    # [batch, num_frames, dim]
question_feat = torch.randn(8, 10, 768) # [batch, seq_len, dim]

logits = model(video_feat, question_feat)
print(f"输出形状: {logits.shape}")  # [8, 1000]
```

---

**下一节**：[3D-语言模型](04-3d-language.md)

---

## 参考文献

1. Bertasius, G., Wang, H., & Torresani, L. (2021). Is space-time attention all you need for video understanding?
2. Xu, M., Li, J., Ye, H., Xiong, C., & Hoi, S. C. (2022). VideoMAE: Masked autoencoders are data-efficient learners for self-supervised video pre-training.
3. Alayrac, J. B., Donahue, J., Lucic, M., Miech, A., Barr, I., Hasson, Y., ... & Botvinick, M. (2022). Flamingo: a visual language model for few-shot learning.
