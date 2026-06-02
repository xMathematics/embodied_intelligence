# 多模态基础

## 目录

- [1. 模态类型与特征](#1-模态类型与特征)
- [2. 多模态融合层次](#2-多模态融合层次)
- [3. 融合策略详解](#3-融合策略详解)
- [4. 预训练目标与策略](#4-预训练目标与策略)
- [5. 跨模态对齐技术](#5-跨模态对齐技术)
- [6. 进阶话题](#6-进阶话题)
- [7. 实战项目案例](#7-实战项目案例)
- [8. 模型优化与部署](#8-模型优化与部署)
- [9. 未来方向](#9-未来方向)

---

## 1. 模态类型与特征

### 1.1 常见模态类型

**文本模态**：
- 数据形式：自然语言文字序列
- 特征特点：离散符号、语义丰富、上下文依赖
- 常用编码器：Transformer、LSTM、BERT

**图像模态**：
- 数据形式：像素矩阵（RGB）
- 特征特点：连续信号、空间结构、视觉语义
- 常用编码器：CNN、ViT、CLIP

**音频模态**：
- 数据形式：波形信号
- 特征特点：时序连续、频率信息、情感表达
- 常用编码器：CNN、RNN、Wav2Vec

**视频模态**：
- 数据形式：图像序列
- 特征特点：时空信息、运动模式、动态语义
- 常用编码器：3D CNN、TimeSformer、VideoMAE

**3D模态**：
- 数据形式：点云、网格、体素
- 特征特点：三维结构、空间关系、几何信息
- 常用编码器：PointNet、PointNet++、3D CNN

**传感器模态**：
- 数据形式：各类传感器读数
- 特征特点：实时、精确、多维度
- 常用编码器：时序模型、特征工程

### 1.2 模态特征提取方法

**文本特征提取**：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    """文本特征编码器"""
    
    def __init__(self, vocab_size=30522, embed_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(512, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, text):
        # text: [batch, seq_len]
        
        # 嵌入
        embedded = self.embedding(text)  # [batch, seq_len, embed_dim]
        
        # 位置编码
        seq_len = text.shape[1]
        pos_embed = self.pos_encoding[:seq_len, :].unsqueeze(0).repeat(text.shape[0], 1, 1)
        embedded = embedded + pos_embed
        
        # Transformer编码
        output = self.transformer(embedded.transpose(0, 1)).transpose(0, 1)
        
        # 返回CLS token作为文本特征
        return output[:, 0, :]  # [batch, embed_dim]
```

**图像特征提取**：
```python
class ImageEncoder(nn.Module):
    """图像特征编码器"""
    
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        
        # 分块嵌入
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 位置编码
        num_patches = (img_size // patch_size) ** 2
        self.pos_encoding = nn.Parameter(torch.randn(num_patches + 1, embed_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, image):
        # image: [batch, channels, height, width]
        
        # 分块
        patches = self.patch_embedding(image)  # [batch, embed_dim, num_patches_h, num_patches_w]
        patches = patches.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        
        # 添加CLS token
        batch_size = patches.shape[0]
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)  # [batch, 1, embed_dim]
        patches = torch.cat([cls_tokens, patches], dim=1)  # [batch, num_patches+1, embed_dim]
        
        # 位置编码
        patches = patches + self.pos_encoding.unsqueeze(0)
        
        # Transformer编码
        output = self.transformer(patches.transpose(0, 1)).transpose(0, 1)
        
        # 返回CLS token作为图像特征
        return output[:, 0, :]  # [batch, embed_dim]
```

**音频特征提取**：
```python
class AudioEncoder(nn.Module):
    """音频特征编码器"""
    
    def __init__(self, input_dim=80, hidden_dim=512, num_layers=6, num_heads=8):
        super().__init__()
        
        # 卷积层提取局部特征
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, audio):
        # audio: [batch, freq_bins, time_steps]
        
        # 卷积提取特征
        conv_out = self.conv_layers(audio)  # [batch, hidden_dim, time_steps/2]
        conv_out = conv_out.transpose(1, 2)  # [batch, time_steps/2, hidden_dim]
        
        # Transformer编码
        output = self.transformer(conv_out.transpose(0, 1)).transpose(0, 1)
        
        # 全局池化
        global_feat = torch.mean(output, dim=1)  # [batch, hidden_dim]
        
        return global_feat
```

### 1.3 模态特征对比

| 模态 | 数据类型 | 特征维度 | 主要特征 | 典型编码器 |
|------|---------|---------|---------|-----------|
| 文本 | 离散 | 768-1024 | 语义、上下文 | BERT、GPT |
| 图像 | 连续 | 512-2048 | 视觉结构、纹理 | ResNet、ViT |
| 音频 | 连续 | 128-512 | 频率、时序 | Wav2Vec、CNN |
| 视频 | 连续 | 512-1024 | 时空、运动 | TimeSformer、3D CNN |
| 3D | 连续 | 512-1024 | 几何、空间 | PointNet、3D CNN |

---

## 2. 多模态融合层次

### 2.1 早期融合

**定义**：在特征提取之前融合原始数据

**架构**：
```
原始数据1 + 原始数据2 → 融合 → 特征提取 → 任务输出
```

**实现方式**：
```python
class EarlyFusion(nn.Module):
    """早期融合"""
    
    def __init__(self, text_vocab=30522, text_dim=768, img_channels=3, img_dim=512):
        super().__init__()
        
        # 文本嵌入
        self.text_embedding = nn.Embedding(text_vocab, text_dim)
        
        # 图像卷积
        self.image_conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + 128 * 56 * 56, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512)
        )
    
    def forward(self, text, image):
        # text: [batch, seq_len]
        # image: [batch, 3, 224, 224]
        
        # 文本嵌入（取第一个token）
        text_emb = self.text_embedding(text)[:, 0, :]  # [batch, text_dim]
        
        # 图像特征
        img_feat = self.image_conv(image)  # [batch, 128, 56, 56]
        img_feat = img_feat.flatten(1)  # [batch, 128*56*56]
        
        # 早期融合
        fused = torch.cat([text_emb, img_feat], dim=-1)  # [batch, text_dim + 128*56*56]
        fused = self.fusion(fused)  # [batch, 512]
        
        return fused
```

**优缺点分析**：

| 优点 | 缺点 |
|------|------|
| 保留细粒度信息 | 计算量大 |
| 早期交互学习 | 需要对齐 |
| 端到端训练 | 过拟合风险 |

### 2.2 中期融合

**定义**：在特征级别进行融合

**架构**：
```
原始数据1 → 特征提取1 →
                        → 特征融合 → 任务输出
原始数据2 → 特征提取2 →
```

**实现方式**：
```python
class IntermediateFusion(nn.Module):
    """中期融合"""
    
    def __init__(self, text_dim=768, img_dim=2048, audio_dim=512, hidden_dim=512):
        super().__init__()
        
        # 投影层
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, text_feat, img_feat, audio_feat):
        # 投影到统一空间
        text_proj = F.relu(self.text_proj(text_feat))  # [batch, hidden_dim]
        img_proj = F.relu(self.img_proj(img_feat))      # [batch, hidden_dim]
        audio_proj = F.relu(self.audio_proj(audio_feat))  # [batch, hidden_dim]
        
        # 融合
        fused = torch.cat([text_proj, img_proj, audio_proj], dim=-1)  # [batch, hidden_dim*3]
        fused = self.fusion(fused)  # [batch, hidden_dim]
        
        return fused
```

**优缺点分析**：

| 优点 | 缺点 |
|------|------|
| 平衡信息与计算 | 需要统一特征维度 |
| 灵活的融合策略 | 丢失部分细粒度信息 |
| 模块化设计 | 需要对齐机制 |

### 2.3 晚期融合

**定义**：在决策级别进行融合

**架构**：
```
原始数据1 → 特征提取1 → 决策1 →
                              → 决策融合 → 最终输出
原始数据2 → 特征提取2 → 决策2 →
```

**实现方式**：
```python
class LateFusion(nn.Module):
    """晚期融合"""
    
    def __init__(self, text_dim=768, img_dim=2048, num_classes=100):
        super().__init__()
        
        # 文本分类器
        self.text_classifier = nn.Linear(text_dim, num_classes)
        
        # 图像分类器
        self.img_classifier = nn.Linear(img_dim, num_classes)
        
        # 决策融合权重
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
    
    def forward(self, text_feat, img_feat):
        # 分别决策
        text_logits = self.text_classifier(text_feat)  # [batch, num_classes]
        img_logits = self.img_classifier(img_feat)      # [batch, num_classes]
        
        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_logits = weights[0] * text_logits + weights[1] * img_logits
        
        return fused_logits
```

**优缺点分析**：

| 优点 | 缺点 |
|------|------|
| 简单灵活 | 丢失交互信息 |
| 模块独立 | 无法捕捉跨模态关系 |
| 易于扩展 | 次优决策 |

### 2.4 融合层次对比

| 层次 | 融合位置 | 信息保留 | 计算复杂度 | 适用场景 |
|------|---------|---------|-----------|---------|
| 早期融合 | 原始数据 | 高 | 高 | 小规模模型 |
| 中期融合 | 特征级别 | 中 | 中 | 中等规模模型 |
| 晚期融合 | 决策级别 | 低 | 低 | 大规模模型 |

---

## 3. 融合策略详解

### 3.1 拼接融合

**原理**：将各模态特征直接拼接

**实现**：
```python
class ConcatFusion(nn.Module):
    """拼接融合"""
    
    def __init__(self, dims, hidden_dim=512):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in dims
        ])
        self.fusion = nn.Linear(hidden_dim * len(dims), hidden_dim)
    
    def forward(self, features):
        projected = [F.relu(proj(feat)) for proj, feat in zip(self.projections, features)]
        concatenated = torch.cat(projected, dim=-1)
        fused = F.relu(self.fusion(concatenated))
        return fused
```

**优缺点**：

| 优点 | 缺点 |
|------|------|
| 简单高效 | 维度爆炸 |
| 保留全部信息 | 缺乏交互建模 |
| 易于实现 | 计算量大 |

### 3.2 元素级融合

**原理**：对各模态特征进行元素级运算

**实现**：
```python
class ElementwiseFusion(nn.Module):
    """元素级融合"""
    
    def __init__(self, dim=512):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
    
    def forward(self, features):
        # features: list of [batch, dim]
        
        # 加权求和
        fused = features[0] * self.weight
        for i in range(1, len(features)):
            fused += features[i] * self.weight
        
        return fused
```

**变体**：
```python
class ElementwiseMaxFusion(nn.Module):
    """元素级最大值融合"""
    
    def forward(self, features):
        stacked = torch.stack(features, dim=0)  # [num_modalities, batch, dim]
        fused = torch.max(stacked, dim=0)[0]    # [batch, dim]
        return fused

class ElementwiseMeanFusion(nn.Module):
    """元素级均值融合"""
    
    def forward(self, features):
        stacked = torch.stack(features, dim=0)  # [num_modalities, batch, dim]
        fused = torch.mean(stacked, dim=0)      # [batch, dim]
        return fused
```

### 3.3 注意力融合

**原理**：使用注意力机制动态加权各模态

**实现**：
```python
class AttentionFusion(nn.Module):
    """注意力融合"""
    
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads)
    
    def forward(self, features):
        # features: [num_modalities, batch, dim]
        features = features.transpose(0, 1)  # [batch, num_modalities, dim]
        
        # 使用第一个模态作为query
        query = features[:, 0:1, :].transpose(0, 1)  # [1, batch, dim]
        key = features.transpose(0, 1)  # [num_modalities, batch, dim]
        value = features.transpose(0, 1)  # [num_modalities, batch, dim]
        
        output, weights = self.multihead_attn(query, key, value)
        return output.squeeze(0), weights
```

**多头注意力变体**：
```python
class CrossModalityAttention(nn.Module):
    """跨模态注意力融合"""
    
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads)
    
    def forward(self, text_feat, img_feat):
        # text_feat: [batch, seq_len, dim]
        # img_feat: [batch, num_patches, dim]
        
        # 投影
        query = self.query_proj(text_feat)
        key = self.key_proj(img_feat)
        value = self.value_proj(img_feat)
        
        # 跨模态注意力
        output, weights = self.multihead_attn(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1)
        )
        
        return output.transpose(0, 1), weights
```

### 3.4 门控融合

**原理**：使用门控机制动态控制各模态的贡献

**实现**：
```python
class GatedFusion(nn.Module):
    """门控融合"""
    
    def __init__(self, dim=512, num_modalities=3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * num_modalities, num_modalities),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features):
        # features: [batch, num_modalities, dim]
        flattened = features.flatten(1)  # [batch, num_modalities * dim]
        weights = self.gate(flattened).unsqueeze(-1)  # [batch, num_modalities, 1]
        fused = (features * weights).sum(dim=1)  # [batch, dim]
        return fused, weights
```

**自适应门控**：
```python
class AdaptiveGatedFusion(nn.Module):
    """自适应门控融合"""
    
    def __init__(self, dim=512, num_modalities=3, hidden_dim=256):
        super().__init__()
        
        # 门控网络
        self.gating_network = nn.Sequential(
            nn.Linear(dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, features):
        # features: list of [batch, dim]
        
        # 拼接特征
        concatenated = torch.cat(features, dim=-1)  # [batch, num_modalities * dim]
        
        # 计算门控权重
        weights = self.gating_network(concatenated).unsqueeze(-1)  # [batch, num_modalities, 1]
        
        # 堆叠特征
        stacked = torch.stack(features, dim=1)  # [batch, num_modalities, dim]
        
        # 加权融合
        fused = (stacked * weights).sum(dim=1)  # [batch, dim]
        
        return fused, weights
```

### 3.5 张量融合

**原理**：使用外积建模模态间交互

**实现**：
```python
class TensorFusion(nn.Module):
    """张量融合"""
    
    def __init__(self, dim=512, output_dim=512):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim * dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, features):
        # features: [batch, num_modalities, dim]
        batch_size = features.shape[0]
        num_modalities = features.shape[1]
        
        # 计算外积
        fusion_tensor = []
        for i in range(num_modalities):
            for j in range(i, num_modalities):
                outer = torch.bmm(
                    features[:, i:i+1, :].transpose(1, 2),
                    features[:, j:j+1, :]
                ).flatten(1)
                fusion_tensor.append(outer)
        
        fused = torch.cat(fusion_tensor, dim=-1)
        fused = self.fusion(fused)
        return fused
```

**高阶张量融合**：
```python
class HighOrderTensorFusion(nn.Module):
    """高阶张量融合"""
    
    def __init__(self, dim=512, order=3, output_dim=512):
        super().__init__()
        self.order = order
        self.output_dim = output_dim
        
        # 张量分解层
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn(dim, output_dim)) for _ in range(order)
        ])
    
    def forward(self, features):
        # features: [batch, num_modalities, dim]
        
        # 选择前order个模态
        selected = features[:, :self.order, :]  # [batch, order, dim]
        
        # 计算高阶张量乘积
        result = selected[:, 0, :]  # [batch, dim]
        
        for i in range(1, self.order):
            result = result.unsqueeze(-1) * selected[:, i:i+1, :].transpose(1, 2)  # [batch, dim, dim]
            result = result.flatten(1)  # [batch, dim*dim]
        
        # 投影到输出维度
        for i, factor in enumerate(self.factors):
            result = result @ factor
        
        return result
```

### 3.6 融合策略对比

| 策略 | 复杂度 | 表达能力 | 计算成本 | 适用场景 |
|------|-------|---------|---------|---------|
| 拼接 | O(d) | 低 | 低 | 简单任务 |
| 元素级 | O(d) | 低 | 低 | 小规模模型 |
| 注意力 | O(d^2) | 高 | 中 | 中等规模 |
| 门控 | O(d) | 中 | 低 | 需要自适应 |
| 张量融合 | O(d^k) | 很高 | 高 | 需要细粒度交互 |

---

## 4. 预训练目标与策略

### 4.1 对比学习

**原理**：通过对比正样本对和负样本对学习对齐

**实现**：
```python
def contrastive_loss(modalities, temperature=0.07):
    """跨模态对比损失"""
    loss = 0
    num_modalities = len(modalities)
    
    for i in range(num_modalities):
        for j in range(i + 1, num_modalities):
            # 归一化特征
            mod1 = F.normalize(modalities[i], dim=-1)
            mod2 = F.normalize(modalities[j], dim=-1)
            
            # 计算相似度
            sim = mod1 @ mod2.t() / temperature
            batch_size = sim.shape[0]
            labels = torch.arange(batch_size)
            
            # 双向对比损失
            loss += (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
    
    return loss / (num_modalities * (num_modalities - 1) / 2)
```

**对比学习框架**：
```python
class ContrastivePretraining(nn.Module):
    """对比学习预训练框架"""
    
    def __init__(self, encoders, temperature=0.07):
        super().__init__()
        self.encoders = encoders
        self.temperature = temperature
    
    def forward(self, inputs):
        # 编码各模态
        features = []
        for modality, encoder in self.encoders.items():
            if modality in inputs:
                features.append(encoder(inputs[modality]))
        
        # 计算对比损失
        loss = contrastive_loss(features, self.temperature)
        return loss
```

### 4.2 掩码建模

**原理**：随机掩码部分输入，预测掩码内容

**实现**：
```python
class MaskedModeling(nn.Module):
    """掩码建模"""
    
    def __init__(self, encoders, decoder):
        super().__init__()
        self.encoders = encoders
        self.decoder = decoder
    
    def forward(self, inputs, masks):
        # 掩码输入
        masked_inputs = {}
        for modality, data in inputs.items():
            if modality in masks:
                masked_inputs[modality] = data * (1 - masks[modality])
            else:
                masked_inputs[modality] = data
        
        # 编码
        features = {}
        for modality, encoder in self.encoders.items():
            if modality in masked_inputs:
                features[modality] = encoder(masked_inputs[modality])
        
        # 解码预测掩码内容
        predictions = {}
        total_loss = 0
        
        for modality, mask in masks.items():
            if mask.sum() > 0:
                pred = self.decoder(features[modality], modality)
                loss = F.mse_loss(pred[mask.bool()], inputs[modality][mask.bool()])
                total_loss += loss
                predictions[modality] = pred
        
        return total_loss, predictions
```

**掩码策略**：
```python
def create_mask(inputs, mask_ratio=0.15):
    """创建掩码"""
    masks = {}
    
    for modality, data in inputs.items():
        if modality == 'text':
            # 文本掩码
            batch_size, seq_len = data.shape
            mask = torch.rand(batch_size, seq_len) < mask_ratio
            mask[:, 0] = False  # CLS token不掩码
            masks[modality] = mask
        
        elif modality == 'image':
            # 图像掩码（patch级别）
            batch_size, _, height, width = data.shape
            num_patches = (height // 16) * (width // 16)
            mask = torch.rand(batch_size, num_patches) < mask_ratio
            masks[modality] = mask
        
        elif modality == 'audio':
            # 音频掩码
            batch_size, freq_bins, time_steps = data.shape
            mask = torch.rand(batch_size, time_steps) < mask_ratio
            masks[modality] = mask
    
    return masks
```

### 4.3 生成任务

**原理**：从一种模态生成另一种模态

**实现**：
```python
class GenerativePretraining(nn.Module):
    """生成式预训练"""
    
    def __init__(self, encoders, generator):
        super().__init__()
        self.encoders = encoders
        self.generator = generator
    
    def forward(self, inputs, target_modality):
        # 编码输入模态
        features = []
        for modality, encoder in self.encoders.items():
            if modality in inputs and modality != target_modality:
                features.append(encoder(inputs[modality]))
        
        # 融合特征
        fused = torch.cat(features, dim=-1) if len(features) > 1 else features[0]
        
        # 生成目标模态
        generated = self.generator(fused)
        
        # 计算损失
        target = inputs[target_modality]
        loss = F.cross_entropy(generated, target)
        
        return loss, generated
```

### 4.4 匹配任务

**原理**：判断多模态输入是否匹配

**实现**：
```python
class MatchingPretraining(nn.Module):
    """匹配任务预训练"""
    
    def __init__(self, text_encoder, image_encoder, hidden_dim=512):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        
        # 匹配分类器
        self.matching_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, text, image, labels):
        # 编码
        text_feat = self.text_encoder(text)
        image_feat = self.image_encoder(image)
        
        # 拼接特征
        combined = torch.cat([text_feat, image_feat], dim=-1)
        
        # 分类
        logits = self.matching_head(combined)
        
        # 计算损失
        loss = F.cross_entropy(logits, labels)
        
        return loss, logits
```

### 4.5 混合预训练策略

**原理**：同时使用多种预训练目标

**实现**：
```python
class MixedPretraining(nn.Module):
    """混合预训练策略"""
    
    def __init__(self, encoders, decoder, generator, matching_head):
        super().__init__()
        self.encoders = encoders
        self.decoder = decoder
        self.generator = generator
        self.matching_head = matching_head
        
        # 损失权重
        self.contrastive_weight = nn.Parameter(torch.tensor(1.0))
        self.masked_weight = nn.Parameter(torch.tensor(1.0))
        self.generative_weight = nn.Parameter(torch.tensor(1.0))
        self.matching_weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, inputs):
        # 提取特征
        text_feat = self.encoders['text'](inputs['text'])
        image_feat = self.encoders['image'](inputs['image'])
        audio_feat = self.encoders['audio'](inputs['audio'])
        
        # 对比损失
        contrastive_loss_val = contrastive_loss([text_feat, image_feat, audio_feat])
        
        # 掩码建模损失
        masks = create_mask(inputs)
        masked_loss, _ = self.decoder(inputs, masks)
        
        # 生成损失
        gen_loss, _ = self.generator(inputs, 'text')
        
        # 匹配损失
        matching_loss, _ = self.matching_head(inputs['text'], inputs['image'], inputs['labels'])
        
        # 加权总损失
        total_loss = (
            self.contrastive_weight * contrastive_loss_val +
            self.masked_weight * masked_loss +
            self.generative_weight * gen_loss +
            self.matching_weight * matching_loss
        )
        
        return total_loss
```

---

## 5. 跨模态对齐技术

### 5.1 对比对齐

**原理**：通过对比学习对齐不同模态

**实现**：
```python
class ContrastiveAlignment(nn.Module):
    """对比对齐"""
    
    def __init__(self, text_encoder, image_encoder, hidden_dim=512, temperature=0.07):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.temperature = temperature
        
        # 投影层
        self.text_proj = nn.Linear(768, hidden_dim)
        self.image_proj = nn.Linear(2048, hidden_dim)
    
    def forward(self, text, image):
        # 编码
        text_feat = self.text_encoder(text)
        image_feat = self.image_encoder(image)
        
        # 投影到统一空间
        text_proj = F.normalize(self.text_proj(text_feat), dim=-1)
        image_proj = F.normalize(self.image_proj(image_feat), dim=-1)
        
        # 计算相似度
        sim = text_proj @ image_proj.t() / self.temperature
        batch_size = sim.shape[0]
        labels = torch.arange(batch_size)
        
        # 双向对比损失
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        
        return loss, text_proj, image_proj
```

### 5.2 生成对齐

**原理**：通过生成任务学习对齐

**实现**：
```python
class GenerativeAlignment(nn.Module):
    """生成对齐"""
    
    def __init__(self, text_encoder, image_encoder, decoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.decoder = decoder
    
    def forward(self, text, image):
        # 编码
        text_feat = self.text_encoder(text)
        image_feat = self.image_encoder(image)
        
        # 从文本生成图像特征
        generated_image = self.decoder(text_feat)
        
        # 计算重构损失
        loss = F.mse_loss(generated_image, image_feat)
        
        return loss, generated_image
```

### 5.3 结构对齐

**原理**：利用结构信息进行对齐

**实现**：
```python
class StructuralAlignment(nn.Module):
    """结构对齐"""
    
    def __init__(self, text_encoder, image_encoder, hidden_dim=512):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        
        # 图神经网络用于结构建模
        self.graph_conv = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, text, image, text_structure, image_structure):
        # 编码
        text_feat = self.text_encoder(text)
        image_feat = self.image_encoder(image)
        
        # 结构信息融合
        text_structured = text_feat + self.graph_conv(text_structure)
        image_structured = image_feat + self.graph_conv(image_structure)
        
        # 对齐损失
        loss = F.mse_loss(F.normalize(text_structured, dim=-1), 
                         F.normalize(image_structured, dim=-1))
        
        return loss
```

### 5.4 对齐评估指标

```python
def evaluate_alignment(image_features, text_features, labels):
    """评估跨模态对齐质量"""
    
    # 计算相似度矩阵
    sim = image_features @ text_features.t()
    
    # 计算检索准确率
    ranks = torch.argsort(sim, dim=1, descending=True)
    correct = (ranks[:, 0] == labels).float().mean()
    
    # 计算Recall@k
    recall_at_5 = (ranks[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean()
    recall_at_10 = (ranks[:, :10] == labels.unsqueeze(1)).any(dim=1).float().mean()
    
    # 计算Median Rank
    ranks = (ranks == labels.unsqueeze(1)).nonzero()[:, 1].float()
    median_rank = ranks.median().item()
    
    return {
        'recall@1': correct.item(),
        'recall@5': recall_at_5.item(),
        'recall@10': recall_at_10.item(),
        'median_rank': median_rank
    }
```

---

## 6. 进阶话题

### 6.1 模态缺失处理

**问题定义**：某些模态可能缺失或质量不佳

**解决方案**：
```python
class MissingModalityHandler(nn.Module):
    """模态缺失处理"""
    
    def __init__(self, text_dim=768, image_dim=2048, audio_dim=512, hidden_dim=512):
        super().__init__()
        
        # 模态编码器
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.image_encoder = nn.Linear(image_dim, hidden_dim)
        self.audio_encoder = nn.Linear(audio_dim, hidden_dim)
        
        # 缺失模态嵌入
        self.missing_text_emb = nn.Parameter(torch.randn(hidden_dim))
        self.missing_image_emb = nn.Parameter(torch.randn(hidden_dim))
        self.missing_audio_emb = nn.Parameter(torch.randn(hidden_dim))
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, inputs, masks):
        # 处理缺失模态
        text_feat = self.missing_text_emb.repeat(inputs['text_mask'].shape[0], 1)
        image_feat = self.missing_image_emb.repeat(inputs['image_mask'].shape[0], 1)
        audio_feat = self.missing_audio_emb.repeat(inputs['audio_mask'].shape[0], 1)
        
        if 'text' in inputs and inputs['text_mask'].sum() > 0:
            text_feat = torch.where(
                inputs['text_mask'].unsqueeze(-1),
                F.relu(self.text_encoder(inputs['text'])),
                text_feat
            )
        
        if 'image' in inputs and inputs['image_mask'].sum() > 0:
            image_feat = torch.where(
                inputs['image_mask'].unsqueeze(-1),
                F.relu(self.image_encoder(inputs['image'])),
                image_feat
            )
        
        if 'audio' in inputs and inputs['audio_mask'].sum() > 0:
            audio_feat = torch.where(
                inputs['audio_mask'].unsqueeze(-1),
                F.relu(self.audio_encoder(inputs['audio'])),
                audio_feat
            )
        
        # 门控融合
        gate_input = torch.cat([text_feat, image_feat, audio_feat], dim=-1)
        weights = self.gate(gate_input)
        
        fused = (weights[:, 0].unsqueeze(-1) * text_feat +
                 weights[:, 1].unsqueeze(-1) * image_feat +
                 weights[:, 2].unsqueeze(-1) * audio_feat)
        
        return fused, weights
```

### 6.2 自适应融合

**原理**：根据输入动态选择融合策略

**实现**：
```python
class AdaptiveFusion(nn.Module):
    """自适应融合"""
    
    def __init__(self, dim=512, num_modalities=3):
        super().__init__()
        
        # 融合策略选择器
        self.strategy_selector = nn.Sequential(
            nn.Linear(dim * num_modalities, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # 四种融合策略
        )
        
        # 四种融合模块
        self.concat_fusion = ConcatFusion([dim] * num_modalities, dim)
        self.attention_fusion = AttentionFusion(dim)
        self.gated_fusion = GatedFusion(dim, num_modalities)
        self.elementwise_fusion = ElementwiseMeanFusion()
    
    def forward(self, features):
        # features: [batch, num_modalities, dim]
        
        # 预测融合策略权重
        flattened = features.flatten(1)
        strategy_weights = F.softmax(self.strategy_selector(flattened), dim=-1)  # [batch, 4]
        
        # 执行四种融合
        concat_out = self.concat_fusion([features[:, i, :] for i in range(features.shape[1])])
        attn_out, _ = self.attention_fusion(features)
        gated_out, _ = self.gated_fusion(features)
        elem_out = self.elementwise_fusion([features[:, i, :] for i in range(features.shape[1])])
        
        # 加权融合结果
        fused = (strategy_weights[:, 0].unsqueeze(-1) * concat_out +
                 strategy_weights[:, 1].unsqueeze(-1) * attn_out +
                 strategy_weights[:, 2].unsqueeze(-1) * gated_out +
                 strategy_weights[:, 3].unsqueeze(-1) * elem_out)
        
        return fused, strategy_weights
```

### 6.3 多模态推理增强

**原理**：增强模型的推理能力

**实现**：
```python
class ReasoningEnhancer(nn.Module):
    """推理增强模块"""
    
    def __init__(self, hidden_dim=512, reasoning_steps=3):
        super().__init__()
        self.reasoning_steps = reasoning_steps
        self.reasoning_layers = nn.ModuleList([
            nn.GRUCell(hidden_dim, hidden_dim) for _ in range(reasoning_steps)
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, 8) for _ in range(reasoning_steps)
        ])
    
    def forward(self, features, context=None):
        # features: [batch, dim]
        # context: [batch, seq_len, dim] (可选)
        
        reasoning_state = features
        reasoning_history = [features]
        
        for i in range(self.reasoning_steps):
            # 如果有上下文，使用注意力
            if context is not None:
                query = reasoning_state.unsqueeze(0)  # [1, batch, dim]
                key = context.transpose(0, 1)         # [seq_len, batch, dim]
                value = context.transpose(0, 1)       # [seq_len, batch, dim]
                
                attn_out, _ = self.attention_layers[i](query, key, value)
                reasoning_state = attn_out.squeeze(0)
            
            # 更新推理状态
            reasoning_state = self.reasoning_layers[i](features, reasoning_state)
            reasoning_history.append(reasoning_state)
        
        # 最终推理结果
        final_reasoning = reasoning_history[-1]
        return final_reasoning, reasoning_history
```

---

## 7. 实战项目案例

### 7.1 多模态情感分析

**任务描述**：结合文本、图像、音频分析情感

**实现**：
```python
class MultimodalSentimentAnalysis(nn.Module):
    """多模态情感分析"""
    
    def __init__(self, text_dim=768, image_dim=2048, audio_dim=512, hidden_dim=512, num_classes=3):
        super().__init__()
        
        # 模态编码器
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.image_encoder = nn.Linear(image_dim, hidden_dim)
        self.audio_encoder = nn.Linear(audio_dim, hidden_dim)
        
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, 3),
            nn.Softmax(dim=-1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, text_feat, image_feat, audio_feat):
        # 编码
        text_proj = F.relu(self.text_encoder(text_feat))
        image_proj = F.relu(self.image_encoder(image_feat))
        audio_proj = F.relu(self.audio_encoder(audio_feat))
        
        # 门控融合
        gate_input = torch.cat([text_proj, image_proj, audio_proj], dim=-1)
        weights = self.gate(gate_input)
        
        fused = (weights[:, 0].unsqueeze(-1) * text_proj +
                 weights[:, 1].unsqueeze(-1) * image_proj +
                 weights[:, 2].unsqueeze(-1) * audio_proj)
        
        # 分类
        logits = self.classifier(fused)
        
        return logits, weights
```

### 7.2 视觉问答系统

**任务描述**：根据图像回答问题

**实现**：
```python
class VisualQuestionAnswering(nn.Module):
    """视觉问答系统"""
    
    def __init__(self, text_dim=768, image_dim=2048, hidden_dim=512, vocab_size=10000):
        super().__init__()
        
        # 编码器
        self.text_encoder = nn.Linear(text_dim, hidden_dim)
        self.image_encoder = nn.Linear(image_dim, hidden_dim)
        
        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 8)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, question_feat, image_feat, answer_input):
        # 编码
        question_proj = F.relu(self.text_encoder(question_feat))  # [batch, dim]
        image_proj = F.relu(self.image_encoder(image_feat))       # [batch, num_patches, dim]
        
        # 跨模态注意力
        query = question_proj.unsqueeze(1).transpose(0, 1)  # [1, batch, dim]
        key = image_proj.transpose(0, 1)                    # [num_patches, batch, dim]
        value = image_proj.transpose(0, 1)                  # [num_patches, batch, dim]
        
        attn_out, _ = self.cross_attn(query, key, value)
        fused = attn_out.squeeze(0)  # [batch, dim]
        
        # 解码生成答案
        decoder_input = torch.cat([fused.unsqueeze(1), answer_input], dim=1)
        decoder_out, _ = self.decoder[0](decoder_input)
        logits = self.decoder[1](decoder_out)
        
        return logits
```

### 7.3 多模态内容检索

**任务描述**：跨模态检索相关内容

**实现**：
```python
class MultimodalRetrieval(nn.Module):
    """多模态内容检索"""
    
    def __init__(self, text_dim=768, image_dim=2048, audio_dim=512, hidden_dim=512):
        super().__init__()
        
        # 投影层
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
    
    def encode(self, inputs):
        """编码输入"""
        features = {}
        
        if 'text' in inputs:
            features['text'] = F.normalize(self.text_proj(inputs['text']), dim=-1)
        
        if 'image' in inputs:
            features['image'] = F.normalize(self.image_proj(inputs['image']), dim=-1)
        
        if 'audio' in inputs:
            features['audio'] = F.normalize(self.audio_proj(inputs['audio']), dim=-1)
        
        return features
    
    def retrieve(self, query_feat, database_feats, top_k=5):
        """检索相关内容"""
        similarities = {}
        
        for modality, feats in database_feats.items():
            sim = query_feat @ feats.t()
            similarities[modality] = sim
        
        # 综合相似度
        combined_sim = sum(similarities.values()) / len(similarities)
        
        # 获取top-k
        top_k_indices = torch.topk(combined_sim, k=top_k, dim=1)[1]
        
        return top_k_indices, similarities
```

---

## 8. 模型优化与部署

### 8.1 模型压缩

**量化**：
```python
import torch.quantization

def quantize_model(model):
    """量化模型"""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized_model
```

**剪枝**：
```python
import torch.nn.utils.prune as prune

def prune_model(model, amount=0.5):
    """剪枝模型"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model
```

**知识蒸馏**：
```python
class Distiller(nn.Module):
    """知识蒸馏"""
    
    def __init__(self, teacher, student, temperature=3):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
    
    def forward(self, inputs):
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
        
        student_logits = self.student(inputs)
        
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        return distill_loss
```

### 8.2 ONNX导出

```python
import torch.onnx

def export_to_onnx(model, output_path):
    """导出模型到ONNX"""
    
    dummy_inputs = {
        'text': torch.randn(1, 10, 768),
        'image': torch.randn(1, 3, 224, 224),
        'audio': torch.randn(1, 1, 1000)
    }
    
    torch.onnx.export(
        model,
        (dummy_inputs,),
        output_path,
        opset_version=13,
        input_names=['text', 'image', 'audio'],
        output_names=['output'],
        dynamic_axes={
            'text': {0: 'batch_size', 1: 'seq_len'},
            'image': {0: 'batch_size'},
            'audio': {0: 'batch_size', 2: 'time'},
            'output': {0: 'batch_size'}
        }
    )
```

### 8.3 推理优化

**批量推理**：
```python
def batch_inference(model, dataloader):
    """批量推理"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            results.append(output)
    
    return torch.cat(results, dim=0)
```

**异步推理**：
```python
async def async_inference(model, inputs):
    """异步推理"""
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(
        None,
        lambda: model(inputs)
    )
    return output
```

---

## 9. 未来方向

### 9.1 研究趋势

| 方向 | 描述 | 关键技术 |
|------|------|---------|
| 原生多模态 | 设计时考虑多模态 | 统一架构、共享表示 |
| 动态融合 | 根据输入自适应融合 | 自适应门控、动态路由 |
| 持续学习 | 不断学习新模态 | 增量学习、迁移学习 |
| 可解释性 | 解释决策过程 | 注意力可视化、归因分析 |
| 高效计算 | 降低计算成本 | 稀疏计算、模型压缩 |

### 9.2 挑战与机遇

**挑战**：
- 模态异质性
- 数据稀缺
- 计算成本
- 可解释性

**机遇**：
- 通用人工智能
- 具身智能
- 边缘部署
- 个性化应用

### 9.3 推荐阅读

1. Baltrušaitis, T., et al. (2018). Multimodal machine learning: A survey and taxonomy.
2. Wang, P., et al. (2021). FLAVA: A Foundational Language and Vision Alignment Model.
3. Radford, A., et al. (2021). Learning transferable visual models from natural language supervision.

---

**下一节**：[音频-语言模型](02-audio-language.md)