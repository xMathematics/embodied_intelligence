# 通用多模态模型

## 目录

- [1. 引言](#1-引言)
- [2. 通用多模态模型概述](#2-通用多模态模型概述)
- [3. 统一架构设计](#3-统一架构设计)
- [4. 代表性通用多模态模型详解](#4-代表性通用多模态模型详解)
- [5. 通用多模态预训练策略](#5-通用多模态预训练策略)
- [6. 通用多模态应用](#6-通用多模态应用)
- [7. 进阶话题](#7-进阶话题)
- [8. 实战项目案例](#8-实战项目案例)
- [9. 模型优化与部署](#9-模型优化与部署)
- [10. 挑战与展望](#10-挑战与展望)
- [11. 实践练习](#11-实践练习)

---

## 1. 引言

### 1.1 什么是通用多模态模型

**通用多模态模型**是指能够处理多种模态数据（文本、图像、音频、视频、3D等）并执行多种任务的统一AI模型。

### 1.2 发展趋势

| 阶段 | 特点 | 代表模型 |
|------|------|---------|
| **单模态模型** | 专门处理一种模态 | BERT、ViT |
| **双模态模型** | 处理两种模态 | CLIP、BLIP |
| **多模态模型** | 处理多种模态 | Flamingo、PaLM-E |
| **通用模型** | 统一处理所有模态 | GPT-4、Gemini |

---

## 2. 通用多模态模型概述

### 2.1 设计目标

| 目标 | 描述 |
|------|------|
| **通用性** | 支持多种模态和任务 |
| **统一接口** | 统一的输入输出格式 |
| **零样本迁移** | 无需微调即可执行新任务 |
| **高效学习** | 从多种数据中学习 |

### 2.2 核心能力

| 能力 | 描述 |
|------|------|
| **多模态理解** | 理解多种模态的内容 |
| **多模态生成** | 生成多种模态的内容 |
| **跨模态转换** | 在不同模态间转换 |
| **多模态推理** | 基于多种模态进行推理 |

---

## 3. 统一架构设计

### 3.1 架构原则

| 原则 | 描述 |
|------|------|
| **统一编码器** | 所有模态使用统一的编码器结构 |
| **共享表示空间** | 所有模态映射到同一特征空间 |
| **灵活融合** | 支持不同模态组合的融合 |
| **任务无关** | 架构与具体任务无关 |

### 3.2 通用架构示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalMultimodalModel(nn.Module):
    def __init__(self, modalities, hidden_dim=512, num_layers=6, num_heads=8):
        super().__init__()
        self.modalities = modalities
        
        # 模态特定编码器
        self.encoders = nn.ModuleDict()
        for modality in modalities:
            if modality == 'text':
                self.encoders[modality] = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(hidden_dim, num_heads),
                    num_layers=num_layers
                )
            elif modality in ['image', 'audio', 'video']:
                self.encoders[modality] = nn.Sequential(
                    nn.Conv2d(3 if modality == 'image' else 1, hidden_dim, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1)
                )
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads)
        
        # 任务头
        self.task_heads = nn.ModuleDict({
            'classification': nn.Linear(hidden_dim, 1000),
            'generation': nn.Linear(hidden_dim, 50000),
            'qa': nn.Linear(hidden_dim, 1000)
        })
    
    def forward(self, inputs, task_type='classification'):
        # 编码各模态
        features = []
        for modality, data in inputs.items():
            if modality in self.encoders:
                if modality == 'text':
                    feat = self.encoders[modality](data.transpose(0, 1)).transpose(0, 1)
                    features.append(feat[:, 0, :])  # CLS token
                else:
                    feat = self.encoders[modality](data).flatten(1)
                    features.append(feat)
        
        # 融合特征
        if len(features) > 1:
            fused = torch.stack(features, dim=1)  # [batch, num_modalities, dim]
            fused = fused.mean(dim=1)  # 简单平均融合
        else:
            fused = features[0]
        
        # 任务输出
        if task_type in self.task_heads:
            output = self.task_heads[task_type](fused)
        else:
            output = fused
        
        return output
```

---

## 4. 代表性通用多模态模型

### 4.1 GPT-4

**特点**：
- 原生多模态支持（文本、图像）
- 强大的推理能力
- 上下文窗口大
- 支持多种任务

**架构**：
```
图像/文本 → 统一编码器 → Transformer → 文本输出
```

### 4.2 Gemini

**论文**：Gemini: A Family of Highly Capable Multimodal Models (Google, 2023)

**特点**：
- 原生多模态（文本、图像、音频、视频）
- 多种型号（Ultra、Pro、Nano）
- 强大的推理能力
- 多语言支持

**模型系列**：
| 型号 | 特点 | 适用场景 |
|------|------|---------|
| **Gemini Ultra** | 最强能力 | 复杂推理、研究 |
| **Gemini Pro** | 平衡性能 | 通用场景、API |
| **Gemini Nano** | 轻量级 | 移动端、边缘设备 |

### 4.3 PaLM-E

**论文**：PaLM-E: An Embodied Multimodal Language Model (Driess et al., 2023)

**特点**：
- 具身多模态语言模型
- 支持机器人控制
- 结合视觉和语言
- 支持多种传感器输入

**架构**：
```
传感器输入 → 编码器 → PaLM → 行动输出
```

### 4.4 Qwen-VL

**特点**：
- 开源多模态模型
- 支持图像理解
- 中文支持好
- 可商用

---

## 5. 通用多模态预训练

### 5.1 预训练任务

| 任务类型 | 描述 | 示例 |
|---------|------|------|
| **对比学习** | 对齐不同模态 | CLIP风格对比 |
| **掩码建模** | 预测被掩盖的内容 | 掩码文本、掩码图像区域 |
| **生成任务** | 生成缺失模态 | 图像描述、文本生成 |
| **匹配任务** | 判断模态匹配 | 图文匹配、音视频匹配 |

### 5.2 训练策略

| 策略 | 描述 |
|------|------|
| **联合训练** | 所有模态一起训练 |
| **增量训练** | 逐步添加新模态 |
| **迁移学习** | 从单模态迁移到多模态 |
| **持续学习** | 不断学习新能力 |

### 5.3 数据挑战

| 挑战 | 描述 |
|------|------|
| **数据多样性** | 需要多种模态的数据 |
| **数据质量** | 多模态数据标注成本高 |
| **数据规模** | 需要大规模训练数据 |
| **数据对齐** | 不同模态需要对齐 |

---

## 6. 通用多模态应用

### 6.1 智能助手

**功能**：
- 理解多模态输入
- 生成多模态输出
- 提供智能回答

**示例**：
```
用户输入：显示一张猫的图片并问"这是什么动物？"
助手输出："这是一只猫，它看起来很可爱！"
```

### 6.2 内容创作

**功能**：
- 文本生成
- 图像生成
- 视频生成
- 多模态内容创作

### 6.3 教育领域

**功能**：
- 交互式学习
- 多模态教学材料
- 智能辅导

### 6.4 机器人控制

**功能**：
- 自然语言指令
- 视觉感知
- 动作规划

---

## 7. 挑战与展望

### 7.1 当前挑战

| 挑战 | 描述 |
|------|------|
| **模态差距** | 不同模态间的语义差距 |
| **计算成本** | 多模态模型计算量大 |
| **数据稀缺** | 高质量多模态数据稀缺 |
| **可解释性** | 多模态决策难以解释 |
| **鲁棒性** | 对噪声和对抗攻击敏感 |

### 7.2 未来方向

| 方向 | 描述 |
|------|------|
| **更强大的推理** | 复杂推理能力 |
| **更多模态支持** | 支持更多数据类型 |
| **更高效率** | 更高效的训练和推理 |
| **更好的对齐** | 更好的跨模态对齐 |
| **通用人工智能** | 迈向AGI |

---

## 8. 实践练习

### 练习1：多模态特征融合

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusion(nn.Module):
    def __init__(self, modalities, hidden_dim=512):
        super().__init__()
        self.modalities = modalities
        
        # 投影层
        self.projections = nn.ModuleDict()
        for modality in modalities:
            if modality == 'text':
                self.projections[modality] = nn.Linear(768, hidden_dim)
            elif modality == 'image':
                self.projections[modality] = nn.Linear(512, hidden_dim)
            elif modality == 'audio':
                self.projections[modality] = nn.Linear(128, hidden_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(modalities), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, inputs):
        projected = []
        for modality in self.modalities:
            if modality in inputs and modality in self.projections:
                feat = inputs[modality]
                if modality == 'text':
                    feat = feat[:, 0, :]  # CLS token
                proj = F.relu(self.projections[modality](feat))
                projected.append(proj)
        
        if len(projected) == 0:
            return torch.zeros(inputs[list(inputs.keys())[0]].shape[0], 512)
        
        fused = torch.cat(projected, dim=-1)
        fused = self.fusion(fused)
        
        return fused

# 测试
fusion = MultimodalFusion(['text', 'image', 'audio'])
inputs = {
    'text': torch.randn(8, 10, 768),   # [batch, seq_len, dim]
    'image': torch.randn(8, 512),       # [batch, dim]
    'audio': torch.randn(8, 100, 128)   # [batch, time, dim]
}
inputs['audio'] = inputs['audio'].mean(dim=1)  # 平均池化

output = fusion(inputs)
print(f"融合特征形状: {output.shape}")  # [8, 512]
```

### 练习2：多模态对比学习

```python
def multimodal_contrastive_loss(features, temperature=0.07):
    """
    多模态对比损失
    
    参数:
        features: 各模态特征字典 {modality: [batch, dim]}
        temperature: 温度系数
    
    返回:
        对比损失
    """
    modalities = list(features.keys())
    num_modalities = len(modalities)
    loss = 0
    
    for i in range(num_modalities):
        for j in range(i + 1, num_modalities):
            mod1_feat = F.normalize(features[modalities[i]], dim=-1)
            mod2_feat = F.normalize(features[modalities[j]], dim=-1)
            
            # 计算相似度矩阵
            sim = mod1_feat @ mod2_feat.t() / temperature
            batch_size = sim.shape[0]
            labels = torch.arange(batch_size).to(sim.device)
            
            # 双向损失
            loss += (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
    
    return loss / (num_modalities * (num_modalities - 1) / 2)

# 测试
features = {
    'text': torch.randn(8, 512),
    'image': torch.randn(8, 512),
    'audio': torch.randn(8, 512)
}

loss = multimodal_contrastive_loss(features)
print(f"多模态对比损失: {loss.item()}")
```

### 练习3：通用多模态模型框架

```python
class UniversalModel(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, num_layers=6):
        super().__init__()
        
        # 模态编码器
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads),
            num_layers=num_layers
        )
        
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        
        # 任务头
        self.classifier = nn.Linear(hidden_dim, 100)
        self.generator = nn.Linear(hidden_dim, 50000)
    
    def encode_text(self, text):
        # text: [batch, seq_len, dim]
        return self.text_encoder(text.transpose(0, 1)).transpose(0, 1)[:, 0, :]
    
    def encode_image(self, image):
        # image: [batch, channels, height, width]
        return self.image_encoder(image).flatten(1)
    
    def encode_audio(self, audio):
        # audio: [batch, channels, time]
        return self.audio_encoder(audio).flatten(1)
    
    def forward(self, inputs, task='classification'):
        features = []
        
        if 'text' in inputs:
            features.append(self.encode_text(inputs['text']))
        if 'image' in inputs:
            features.append(self.encode_image(inputs['image']))
        if 'audio' in inputs:
            features.append(self.encode_audio(inputs['audio']))
        
        # 融合
        if len(features) > 1:
            fused = torch.stack(features, dim=1).mean(dim=1)
        else:
            fused = features[0]
        
        # 任务输出
        if task == 'classification':
            return self.classifier(fused)
        elif task == 'generation':
            return self.generator(fused)
        else:
            return fused

# 测试
model = UniversalModel()
inputs = {
    'text': torch.randn(8, 10, 512),
    'image': torch.randn(8, 3, 224, 224),
    'audio': torch.randn(8, 1, 1000)
}

class_output = model(inputs, task='classification')
gen_output = model(inputs, task='generation')

print(f"分类输出形状: {class_output.shape}")  # [8, 100]
print(f"生成输出形状: {gen_output.shape}")    # [8, 50000]
```

---

## 4. 代表性通用多模态模型详解

### 4.1 GPT-4

**技术报告**：GPT-4 Technical Report (OpenAI, 2023)

**核心架构**：
```python
class GPT4Multimodal(nn.Module):
    """GPT-4多模态模型简化版"""
    
    def __init__(self, vocab_size=50257, hidden_dim=8192, num_layers=96, num_heads=128):
        super().__init__()
        
        # 文本嵌入
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_pos_encoding = nn.Parameter(torch.randn(8192, hidden_dim))
        
        # 图像处理器
        self.image_processor = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 4, kernel_size=16, stride=16),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 4, hidden_dim, kernel_size=1)
        )
        
        # 图像位置编码
        self.image_pos_encoding = nn.Parameter(torch.randn(16*16, hidden_dim))
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, vocab_size)
    
    def encode_image(self, image):
        """编码图像"""
        # image: [batch, channels, height, width]
        image_feat = self.image_processor(image)  # [batch, hidden_dim, 16, 16]
        image_feat = image_feat.flatten(2).transpose(1, 2)  # [batch, 256, hidden_dim]
        image_feat = image_feat + self.image_pos_encoding.unsqueeze(0)
        return image_feat
    
    def forward(self, text, image=None):
        """前向传播"""
        # text: [batch, seq_len]
        
        # 文本嵌入
        text_emb = self.text_embedding(text)  # [batch, seq_len, hidden_dim]
        text_emb = text_emb + self.text_pos_encoding[:text.shape[1]].unsqueeze(0)
        
        # 准备memory（图像特征）
        if image is not None:
            memory = self.encode_image(image)  # [batch, 256, hidden_dim]
            memory = memory.transpose(0, 1)  # [256, batch, hidden_dim]
        else:
            memory = torch.zeros(1, text.shape[0], self.hidden_dim)
        
        # Transformer解码
        tgt = text_emb.transpose(0, 1)  # [seq_len, batch, hidden_dim]
        output = self.transformer(tgt, memory).transpose(0, 1)  # [batch, seq_len, hidden_dim]
        
        # 预测
        logits = self.classifier(output)  # [batch, seq_len, vocab_size]
        
        return logits
```

**训练策略**：
```python
def train_gpt4_multimodal(model, dataloader, optimizer, num_epochs=10):
    """训练GPT-4风格多模态模型"""
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            text_input = batch['text_input']  # [batch, seq_len]
            text_target = batch['text_target']  # [batch, seq_len]
            images = batch.get('images', None)  # [batch, channels, height, width]
            
            logits = model(text_input, images)
            
            # 计算损失（只计算target部分）
            loss = criterion(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                text_target[:, 1:].reshape(-1)
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

### 4.2 Gemini

**论文**：Gemini: A Family of Highly Capable Multimodal Models (Google, 2023)

**架构实现**：
```python
class GeminiModel(nn.Module):
    """Gemini模型简化版"""
    
    def __init__(self, model_size='pro'):
        super().__init__()
        
        # 根据模型大小配置
        configs = {
            'nano': {'hidden_dim': 512, 'num_layers': 8, 'num_heads': 8},
            'pro': {'hidden_dim': 2048, 'num_layers': 28, 'num_heads': 32},
            'ultra': {'hidden_dim': 8192, 'num_layers': 96, 'num_heads': 128}
        }
        
        config = configs[model_size]
        self.hidden_dim = config['hidden_dim']
        
        # 模态编码器
        self.text_encoder = TextEncoder(config)
        self.image_encoder = VisionEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.video_encoder = VideoEncoder(config)
        
        # 统一Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config['num_heads'],
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        
        # 任务头
        self.classifier = nn.Linear(self.hidden_dim, 1000)
        self.generator = nn.Linear(self.hidden_dim, 50000)
    
    def forward(self, inputs, task='generation'):
        """前向传播"""
        # inputs: {'text': ..., 'image': ..., 'audio': ..., 'video': ...}
        
        # 编码各模态
        features = []
        
        if 'text' in inputs:
            text_feat = self.text_encoder(inputs['text'])  # [batch, seq_len, dim]
            features.append(text_feat)
        
        if 'image' in inputs:
            image_feat = self.image_encoder(inputs['image'])  # [batch, num_patches, dim]
            features.append(image_feat)
        
        if 'audio' in inputs:
            audio_feat = self.audio_encoder(inputs['audio'])  # [batch, time_steps, dim]
            features.append(audio_feat)
        
        if 'video' in inputs:
            video_feat = self.video_encoder(inputs['video'])  # [batch, num_frames*num_patches, dim]
            features.append(video_feat)
        
        # 合并特征
        if len(features) > 0:
            max_len = max(f.shape[1] for f in features)
            padded_features = []
            
            for f in features:
                if f.shape[1] < max_len:
                    pad = torch.zeros(f.shape[0], max_len - f.shape[1], f.shape[2])
                    f = torch.cat([f, pad], dim=1)
                padded_features.append(f)
            
            # 简单拼接
            combined = torch.cat(padded_features, dim=1)  # [batch, total_len, dim]
        else:
            raise ValueError("至少需要一种模态输入")
        
        # Transformer编码
        output = self.transformer(combined.transpose(0, 1)).transpose(0, 1)
        
        # 任务输出
        if task == 'classification':
            return self.classifier(output[:, 0, :])
        elif task == 'generation':
            return self.generator(output)
        
        return output
```

**模型配置**：
```python
class TextEncoder(nn.Module):
    """文本编码器"""
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(50000, config['hidden_dim'])
        self.pos_encoding = nn.Parameter(torch.randn(8192, config['hidden_dim']))
    
    def forward(self, text):
        emb = self.embedding(text)
        emb = emb + self.pos_encoding[:text.shape[1]].unsqueeze(0)
        return emb

class VisionEncoder(nn.Module):
    """视觉编码器"""
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, config['hidden_dim'], 16, 16)
        self.pos_encoding = nn.Parameter(torch.randn(14*14, config['hidden_dim']))
    
    def forward(self, image):
        patches = self.patch_embedding(image)  # [batch, dim, 14, 14]
        patches = patches.flatten(2).transpose(1, 2)  # [batch, 196, dim]
        patches = patches + self.pos_encoding.unsqueeze(0)
        return patches

class AudioEncoder(nn.Module):
    """音频编码器"""
    def __init__(self, config):
        super().__init__()
        self.conv = nn.Conv1d(1, config['hidden_dim'], 3, padding=1)
        self.pos_encoding = nn.Parameter(torch.randn(1000, config['hidden_dim']))
    
    def forward(self, audio):
        feat = self.conv(audio)  # [batch, dim, time]
        feat = feat.transpose(1, 2)  # [batch, time, dim]
        feat = feat + self.pos_encoding[:feat.shape[1]].unsqueeze(0)
        return feat

class VideoEncoder(nn.Module):
    """视频编码器"""
    def __init__(self, config):
        super().__init__()
        self.vision_encoder = VisionEncoder(config)
    
    def forward(self, video):
        # video: [batch, num_frames, channels, height, width]
        batch_size, num_frames = video.shape[:2]
        video = video.view(-1, 3, 224, 224)  # [batch*num_frames, 3, 224, 224]
        
        frame_feats = self.vision_encoder(video)  # [batch*num_frames, 196, dim]
        frame_feats = frame_feats.view(batch_size, num_frames * 196, -1)  # [batch, total_patches, dim]
        
        return frame_feats
```

### 4.3 PaLM-E

**论文**：PaLM-E: An Embodied Multimodal Language Model (Driess et al., 2023)

**架构实现**：
```python
class PaLME(nn.Module):
    """PaLM-E具身多模态语言模型"""
    
    def __init__(self, hidden_dim=2048, num_layers=12, num_heads=16):
        super().__init__()
        
        # 传感器编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.lidar_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 投影层
        self.sensor_proj = nn.Linear(128, hidden_dim)
        
        # PaLM语言模型
        self.text_embedding = nn.Embedding(50000, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(2048, hidden_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 动作输出头
        self.action_head = nn.Linear(hidden_dim, 7)  # 7自由度动作
    
    def encode_sensors(self, sensors):
        """编码传感器输入"""
        features = []
        
        if 'image' in sensors:
            img_feat = self.image_encoder(sensors['image']).flatten(1)
            features.append(img_feat)
        
        if 'depth' in sensors:
            depth_feat = self.depth_encoder(sensors['depth']).flatten(1)
            features.append(depth_feat)
        
        if 'lidar' in sensors:
            lidar_feat = self.lidar_encoder(sensors['lidar']).flatten(1)
            features.append(lidar_feat)
        
        if len(features) > 0:
            combined = torch.cat(features, dim=-1)  # [batch, 384]
            projected = F.relu(self.sensor_proj(combined))  # [batch, hidden_dim]
            return projected.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        return None
    
    def forward(self, text, sensors=None):
        """前向传播"""
        # text: [batch, seq_len]
        
        # 文本嵌入
        text_emb = self.text_embedding(text)  # [batch, seq_len, hidden_dim]
        text_emb = text_emb + self.pos_encoding[:text.shape[1]].unsqueeze(0)
        
        # 准备memory（传感器特征）
        if sensors is not None:
            sensor_feat = self.encode_sensors(sensors)
            memory = sensor_feat.transpose(0, 1)  # [1, batch, hidden_dim]
        else:
            memory = torch.zeros(1, text.shape[0], self.hidden_dim)
        
        # Transformer解码
        tgt = text_emb.transpose(0, 1)  # [seq_len, batch, hidden_dim]
        output = self.transformer(tgt, memory).transpose(0, 1)
        
        # 动作预测
        action_logits = self.action_head(output[:, -1, :])  # [batch, 7]
        
        return action_logits
```

---

## 5. 通用多模态预训练策略

### 5.1 多模态对比学习

**实现**：
```python
class MultimodalContrastiveLearning(nn.Module):
    """多模态对比学习框架"""
    
    def __init__(self, encoders, temperature=0.07):
        super().__init__()
        self.encoders = encoders
        self.temperature = temperature
        
        # 投影层
        self.projections = nn.ModuleDict()
        for modality, encoder in encoders.items():
            if hasattr(encoder, 'output_dim'):
                self.projections[modality] = nn.Linear(encoder.output_dim, 256)
    
    def forward(self, inputs):
        """计算对比损失"""
        # inputs: {modality: data}
        
        # 编码各模态
        features = {}
        for modality, data in inputs.items():
            if modality in self.encoders:
                feat = self.encoders[modality](data)
                if modality in self.projections:
                    feat = F.normalize(self.projections[modality](feat), dim=-1)
                features[modality] = feat
        
        # 计算跨模态对比损失
        modalities = list(features.keys())
        total_loss = 0
        num_pairs = 0
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1 = modalities[i]
                mod2 = modalities[j]
                
                sim = features[mod1] @ features[mod2].t() / self.temperature
                batch_size = sim.shape[0]
                labels = torch.arange(batch_size)
                
                loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
                total_loss += loss
                num_pairs += 1
        
        return total_loss / num_pairs if num_pairs > 0 else 0
```

### 5.2 掩码建模

**多模态掩码建模**：
```python
class MultimodalMaskedModeling(nn.Module):
    """多模态掩码建模"""
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def create_mask(self, data, mask_ratio=0.15):
        """创建掩码"""
        batch_size, seq_len = data.shape[:2]
        num_masked = int(seq_len * mask_ratio)
        
        mask = torch.zeros(batch_size, seq_len)
        for i in range(batch_size):
            mask_indices = torch.randperm(seq_len)[:num_masked]
            mask[i, mask_indices] = 1
        
        return mask.bool()
    
    def forward(self, text, image):
        """前向传播"""
        # 掩码文本
        text_mask = self.create_mask(text)
        masked_text = text.clone()
        masked_text[text_mask] = 0  # 特殊掩码token
        
        # 编码
        text_feat = self.encoder(masked_text)
        image_feat = self.encoder(image)
        
        # 融合
        fused = torch.cat([text_feat[:, 0, :], image_feat[:, 0, :]], dim=-1)
        
        # 解码掩码位置
        decoder_input = text_feat
        decoder_input[text_mask] = 0
        
        output = self.decoder(decoder_input, fused.unsqueeze(1))
        
        # 计算损失
        loss = F.cross_entropy(
            output[text_mask].reshape(-1, output.shape[-1]),
            text[text_mask].reshape(-1)
        )
        
        return loss
```

### 5.3 生成式预训练

**实现**：
```python
class MultimodalGenerativePretraining(nn.Module):
    """多模态生成式预训练"""
    
    def __init__(self, encoder, decoder, vocab_size=50000):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
    
    def forward(self, inputs, targets):
        """前向传播"""
        # inputs: 多模态输入
        # targets: 文本目标
        
        # 编码输入
        features = []
        for modality, data in inputs.items():
            if modality == 'text':
                feat = self.encoder.encode_text(data)
            elif modality == 'image':
                feat = self.encoder.encode_image(data)
            elif modality == 'audio':
                feat = self.encoder.encode_audio(data)
            features.append(feat)
        
        # 融合特征
        fused = torch.stack(features, dim=1).mean(dim=1)  # [batch, hidden_dim]
        
        # 解码生成文本
        decoder_input = targets[:, :-1]  # 去掉最后一个token
        decoder_output = self.decoder(decoder_input, fused.unsqueeze(1))
        
        # 计算损失
        loss = F.cross_entropy(
            decoder_output.reshape(-1, self.vocab_size),
            targets[:, 1:].reshape(-1)
        )
        
        return loss
```

---

## 6. 通用多模态应用

### 6.1 智能助手

**完整实现**：
```python
class MultimodalAssistant:
    """多模态智能助手"""
    
    def __init__(self, model_path='assistant_model.pth'):
        # 加载模型
        self.model = UniversalModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 加载处理器
        self.text_processor = BertTokenizer.from_pretrained("bert-base-uncased")
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    
    def process_inputs(self, inputs):
        """处理多模态输入"""
        processed = {}
        
        if 'text' in inputs:
            text = self.text_processor(inputs['text'], return_tensors="pt")
            processed['text'] = text['input_ids']
        
        if 'image' in inputs:
            image = self.image_processor(inputs['image'], return_tensors="pt")
            processed['image'] = image['pixel_values']
        
        if 'audio' in inputs:
            audio = self.audio_processor(inputs['audio'], return_tensors="pt")
            processed['audio'] = audio['input_features']
        
        return processed
    
    def generate_response(self, inputs, max_length=100):
        """生成响应"""
        processed = self.process_inputs(inputs)
        
        with torch.no_grad():
            response = self.model.generate(processed, max_length=max_length)
        
        return self.text_processor.decode(response[0], skip_special_tokens=True)
```

### 6.2 多模态内容创作

**实现**：
```python
class MultimodalContentCreator:
    """多模态内容创作系统"""
    
    def __init__(self):
        # 加载模型
        self.text_generator = GPT2LMHeadModel.from_pretrained("gpt2")
        self.image_generator = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    
    def create_text(self, prompt):
        """生成文本"""
        inputs = self.text_generator.tokenizer(prompt, return_tensors="pt")
        outputs = self.text_generator.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            no_repeat_ngram_size=2
        )
        return self.text_generator.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def create_image(self, prompt):
        """生成图像"""
        image = self.image_generator(prompt).images[0]
        return image
    
    def create_multimodal_content(self, prompt):
        """生成多模态内容"""
        # 先生成文本描述
        text_content = self.create_text(prompt)
        
        # 再根据文本生成图像
        image_content = self.create_image(text_content)
        
        return {
            'text': text_content,
            'image': image_content
        }
```

---

## 7. 进阶话题

### 7.1 模态缺失处理

**实现**：
```python
class ModalityDropout(nn.Module):
    """模态缺失处理"""
    
    def __init__(self, modalities, dropout_prob=0.1):
        super().__init__()
        self.modalities = modalities
        self.dropout_prob = dropout_prob
    
    def forward(self, inputs):
        """随机丢弃模态"""
        if not self.training:
            return inputs
        
        dropped_inputs = {}
        
        for modality, data in inputs.items():
            if torch.rand(1) > self.dropout_prob:
                dropped_inputs[modality] = data
        
        return dropped_inputs

class AdaptiveFusion(nn.Module):
    """自适应融合"""
    
    def __init__(self, hidden_dim=512, num_modalities=3):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.gating = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, num_modalities),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features):
        """自适应融合特征"""
        # features: [num_modalities, batch, dim]
        
        # 计算注意力权重
        query = features[0:1]  # 使用第一个模态作为query
        attn_out, weights = self.attention(query, features, features)
        
        # 门控融合
        features_concat = torch.cat([f.transpose(0, 1) for f in features], dim=-1)  # [batch, num_modalities*dim]
        gates = self.gating(features_concat).unsqueeze(-1)  # [batch, num_modalities, 1]
        
        # 加权融合
        features_stack = torch.stack([f.transpose(0, 1) for f in features], dim=1)  # [batch, num_modalities, dim]
        fused = (features_stack * gates).sum(dim=1)  # [batch, dim]
        
        return fused
```

### 7.2 多模态推理增强

**实现**：
```python
class MultimodalReasoning(nn.Module):
    """多模态推理模型"""
    
    def __init__(self, hidden_dim=512):
        super().__init__()
        
        # 推理步骤编码器
        self.step_encoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        
        # 推理控制器
        self.controller = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 停止、继续、回溯
        )
        
        # 推理记忆
        self.memory = nn.Parameter(torch.randn(10, hidden_dim))
    
    def forward(self, inputs, max_steps=10):
        """执行多步推理"""
        # inputs: 多模态特征 [batch, dim]
        
        batch_size = inputs.shape[0]
        memory = self.memory.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, 10, dim]
        
        reasoning_steps = []
        current_step = 0
        
        while current_step < max_steps:
            # 编码当前状态
            state = torch.cat([inputs, memory[:, current_step, :]], dim=-1)  # [batch, dim * 2]
            
            # 推理步骤
            step_enc, _ = self.step_encoder(state.unsqueeze(1))
            step_out = step_enc[:, -1, :]  # [batch, dim * 2]
            
            # 控制决策
            control_input = torch.cat([inputs, step_out], dim=-1)  # [batch, dim * 3]
            control_logits = self.controller(control_input)  # [batch, 3]
            control_action = torch.argmax(control_logits, dim=-1)
            
            # 更新记忆
            memory[:, (current_step + 1) % 10, :] = step_out[:, :hidden_dim]
            
            reasoning_steps.append(step_out)
            
            # 检查是否停止
            if (control_action == 0).all():
                break
            
            current_step += 1
        
        # 最终推理结果
        final_result = torch.stack(reasoning_steps, dim=1).mean(dim=1)
        
        return final_result
```

### 7.3 多模态对齐评估

**实现**：
```python
def evaluate_multimodal_alignment(model, dataloader):
    """评估多模态对齐质量"""
    model.eval()
    
    metrics = {
        'text-image-similarity': [],
        'text-audio-similarity': [],
        'image-audio-similarity': [],
        'retrieval-accuracy': []
    }
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'text': batch['text'],
                'image': batch['image'],
                'audio': batch['audio']
            }
            
            # 获取特征
            features = model.encode(inputs)
            
            # 计算相似度
            text_feat = F.normalize(features['text'], dim=-1)
            image_feat = F.normalize(features['image'], dim=-1)
            audio_feat = F.normalize(features['audio'], dim=-1)
            
            text_image_sim = (text_feat * image_feat).sum(dim=-1).mean().item()
            text_audio_sim = (text_feat * audio_feat).sum(dim=-1).mean().item()
            image_audio_sim = (image_feat * audio_feat).sum(dim=-1).mean().item()
            
            metrics['text-image-similarity'].append(text_image_sim)
            metrics['text-audio-similarity'].append(text_audio_sim)
            metrics['image-audio-similarity'].append(image_audio_sim)
            
            # 检索准确性
            similarity_matrix = text_feat @ image_feat.t()
            ranks = torch.argsort(similarity_matrix, dim=1, descending=True)
            retrieval_acc = (ranks[:, 0] == torch.arange(len(ranks))).float().mean().item()
            metrics['retrieval-accuracy'].append(retrieval_acc)
    
    # 计算平均值
    for key in metrics:
        metrics[key] = sum(metrics[key]) / len(metrics[key])
    
    return metrics
```

---

## 8. 实战项目案例

### 8.1 多模态智能问答系统

```python
class MultimodalQA System:
    """多模态智能问答系统"""
    
    def __init__(self):
        # 加载模型
        self.model = UniversalModel()
        self.model.load_state_dict(torch.load('multimodal_qa.pth'))
        self.model.eval()
        
        # 加载处理器
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.audio_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    
    def process_inputs(self, question, image=None, audio=None):
        """处理输入"""
        inputs = {}
        
        # 处理问题
        question_tokens = self.tokenizer(question, return_tensors="pt")
        inputs['text'] = question_tokens['input_ids']
        
        # 处理图像
        if image is not None:
            image_tensor = self.image_processor(image, return_tensors="pt")
            inputs['image'] = image_tensor['pixel_values']
        
        # 处理音频
        if audio is not None:
            audio_tensor = self.audio_processor(audio, return_tensors="pt")
            inputs['audio'] = audio_tensor['input_features']
        
        return inputs
    
    def answer(self, question, image=None, audio=None):
        """回答问题"""
        inputs = self.process_inputs(question, image, audio)
        
        with torch.no_grad():
            logits = self.model(inputs, task='qa')
        
        # 获取答案
        answer_idx = torch.argmax(logits, dim=-1).item()
        
        # 返回答案（简化版）
        return f"答案类别: {answer_idx}"
```

### 8.2 多模态内容检索平台

```python
class MultimodalSearchEngine:
    """多模态内容检索平台"""
    
    def __init__(self):
        # 加载模型
        self.model = UniversalModel()
        self.model.load_state_dict(torch.load('search_model.pth'))
        self.model.eval()
        
        # 特征数据库
        self.feature_database = {
            'text': [],
            'image': [],
            'audio': []
        }
        
        # 索引映射
        self.index_mapping = []
    
    def add_content(self, content_id, text=None, image=None, audio=None):
        """添加内容到数据库"""
        features = {}
        
        if text is not None:
            text_feat = self.model.encode_text(text)
            features['text'] = text_feat
        
        if image is not None:
            image_feat = self.model.encode_image(image)
            features['image'] = image_feat
        
        if audio is not None:
            audio_feat = self.model.encode_audio(audio)
            features['audio'] = audio_feat
        
        # 保存特征
        for modality, feat in features.items():
            self.feature_database[modality].append(feat)
        
        # 添加索引
        self.index_mapping.append(content_id)
    
    def search(self, query, modality='text', top_k=5):
        """搜索内容"""
        # 获取查询特征
        if modality == 'text':
            query_feat = self.model.encode_text(query)
            db_features = torch.cat(self.feature_database['text'], dim=0)
        elif modality == 'image':
            query_feat = self.model.encode_image(query)
            db_features = torch.cat(self.feature_database['image'], dim=0)
        else:
            query_feat = self.model.encode_audio(query)
            db_features = torch.cat(self.feature_database['audio'], dim=0)
        
        # 计算相似度
        query_feat = F.normalize(query_feat, dim=-1)
        db_features = F.normalize(db_features, dim=-1)
        
        similarities = query_feat @ db_features.t()
        top_indices = torch.topk(similarities, k=top_k).indices
        
        # 返回结果
        results = [self.index_mapping[i] for i in top_indices]
        
        return results
```

---

## 9. 模型优化与部署

### 9.1 模型压缩

**量化**：
```python
def quantize_universal_model(model):
    """量化通用多模态模型"""
    # 准备模型
    model.eval()
    
    # 动态量化
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.Conv2d, nn.Conv1d},
        dtype=torch.qint8
    )
    
    return quantized_model
```

**知识蒸馏**：
```python
class KnowledgeDistillation(nn.Module):
    """知识蒸馏"""
    
    def __init__(self, teacher_model, student_model, temperature=2.0):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
    
    def forward(self, inputs):
        """前向传播"""
        # 教师输出
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        # 学生输出
        student_logits = self.student(inputs)
        
        # 蒸馏损失
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        distillation_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        
        return distillation_loss
```

### 9.2 ONNX导出

```python
def export_universal_model_to_onnx(model, output_path):
    """导出通用多模态模型到ONNX"""
    
    # 准备dummy输入
    dummy_inputs = {
        'text': torch.randint(0, 50000, (1, 10)),
        'image': torch.randn(1, 3, 224, 224),
        'audio': torch.randn(1, 1, 1000)
    }
    
    # 导出
    torch.onnx.export(
        model,
        (dummy_inputs,),
        output_path,
        opset_version=13,
        input_names=['text', 'image', 'audio'],
        output_names=['logits'],
        dynamic_axes={
            'text': {0: 'batch_size', 1: 'seq_len'},
            'image': {0: 'batch_size'},
            'audio': {0: 'batch_size', 2: 'time_steps'},
            'logits': {0: 'batch_size'}
        }
    )
```

---

## 10. 挑战与展望

### 10.1 当前挑战

| 挑战 | 描述 | 影响 |
|------|------|------|
| **模态差距** | 不同模态间的语义和表示差距 | 影响跨模态理解质量 |
| **计算成本** | 多模态模型计算量大 | 部署困难，推理慢 |
| **数据稀缺** | 高质量多模态标注数据稀缺 | 训练效果受限 |
| **可解释性** | 多模态决策难以解释 | 信任度低 |
| **鲁棒性** | 对噪声和对抗攻击敏感 | 实际应用风险 |

### 10.2 未来方向

| 方向 | 描述 | 关键技术 |
|------|------|---------|
| **通用人工智能** | 迈向AGI | 统一模型、持续学习 |
| **具身智能** | 结合机器人感知与行动 | 传感器融合、强化学习 |
| **高效推理** | 更高效的模型设计 | MoE、稀疏激活 |
| **多模态推理** | 复杂推理能力 | 逻辑推理、符号推理 |
| **个性化** | 适应不同用户需求 | 用户建模、自适应 |

### 10.3 推荐阅读

1. GPT-4 Technical Report. (2023). OpenAI.
2. Gemini: A Family of Highly Capable Multimodal Models. (2023). Google.
3. Driess, D., Xia, F., Bucher, B., et al. (2023). PaLM-E: An Embodied Multimodal Language Model.
4. Wang, W., Liu, Y., Wu, J., & Wang, L. (2021). FLAVA: A Foundational Language and Vision Alignment Model.
5. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision.

---

**返回**：[多模态基础](01-multimodal-basics.md)
