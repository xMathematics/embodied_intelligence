# 2.2 视觉编码器

## 目录

- [1. 引言](#1-引言)
- [2. 视觉特征提取概述](#2-视觉特征提取概述)
- [3. 卷积神经网络（CNN）](#3-卷积神经网络cnn)
- [4. Vision Transformer（ViT）](#4-vision-transformer-vit)
- [5. Swin Transformer](#5-swin-transformer)
- [6. 其他视觉编码器](#6-其他视觉编码器)
- [7. 视觉编码器对比](#7-视觉编码器对比)
- [8. 视觉编码器在VLM中的应用](#8-视觉编码器在vlm中的应用)
- [9. 实践练习](#9-实践练习)

---

## 1. 引言

### 1.1 视觉编码器的作用

**视觉编码器**是将图像数据转换为机器可理解的特征表示的组件。在VLM中，视觉编码器负责将像素级的图像数据转换为高级语义特征，以便与语言特征进行融合。

### 1.2 发展历程

| 阶段 | 技术 | 代表模型 |
|------|------|---------|
| **早期** | 手工设计特征 | SIFT、HOG |
| **深度学习初期** | 卷积神经网络 | AlexNet、VGG |
| **深度学习中期** | 深度残差网络 | ResNet、DenseNet |
| **Transformer时代** | Vision Transformer | ViT、Swin |

---

## 2. 视觉特征提取概述

### 2.1 特征层次

视觉特征通常具有层次结构：

```
输入图像 (像素)
    ↓
低层特征 (边缘、纹理)
    ↓
中层特征 (形状、局部结构)
    ↓
高层特征 (对象、语义概念)
    ↓
输出特征向量
```

### 2.2 特征提取目标

| 目标 | 描述 |
|------|------|
| **局部特征** | 捕捉边缘、纹理等局部模式 |
| **全局特征** | 捕捉整体结构和语义 |
| **多尺度特征** | 捕捉不同分辨率的信息 |
| **不变性** | 对平移、缩放、旋转具有不变性 |

---

## 3. 卷积神经网络（CNN）

### 3.1 基本原理

**卷积层**：使用卷积核在图像上滑动，提取局部特征

```python
import torch
import torch.nn.functional as F

# 简单卷积示例
input = torch.randn(1, 3, 224, 224)  # batch, channels, height, width
kernel = torch.randn(16, 3, 3, 3)     # out_channels, in_channels, kernel_size
output = F.conv2d(input, kernel, padding=1)
```

### 3.2 经典CNN架构

| 模型 | 层数 | 关键创新 |
|------|------|---------|
| **AlexNet** | 8层 | 深度CNN、ReLU激活 |
| **VGG** | 16/19层 | 统一3x3卷积、深度堆叠 |
| **ResNet** | 50/101/152层 | 残差连接、解决梯度消失 |
| **Inception** | 多层 | 多尺度卷积、inception模块 |
| **DenseNet** | 多层 | 密集连接、特征复用 |

### 3.3 残差连接

**残差网络**（ResNet）通过残差连接解决深度网络训练困难的问题：

```python
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        
        # 残差连接的维度匹配
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, 1, stride)
        else:
            self.shortcut = torch.nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
```

---

## 4. Vision Transformer（ViT）

### 4.1 核心思想

将Transformer架构应用于图像处理，将图像分割为patch，然后使用Transformer进行处理。

### 4.2 架构设计

```
输入图像 (H x W x C)
    ↓
分割为patch (N x P² x C)，N = HW/P²
    ↓
patch嵌入 (N x D)
    ↓
添加位置编码 (N x D)
    ↓
Transformer编码器 (N x D)
    ↓
[CLS] token提取 (D)
    ↓
分类/特征输出
```

### 4.3 关键组件

| 组件 | 作用 | 实现细节 |
|------|------|---------|
| **Patch Embedding** | 将patch转换为向量 | 卷积或线性层 |
| **Position Embedding** | 添加位置信息 | 可学习或固定 |
| **Transformer Encoder** | 建模patch间关系 | 多头注意力 |
| **[CLS] Token** | 聚合全局信息 | 特殊token |

### 4.4 ViT变体

| 变体 | 参数 | 特点 |
|------|------|------|
| **ViT-B/16** | 86M | 基础模型，16x16 patch |
| **ViT-B/32** | 86M | 32x32 patch，更快 |
| **ViT-L/16** | 307M | 更大模型，更好性能 |
| **ViT-H/14** | 632M | 超大模型，最佳性能 |

### 4.5 代码示例

```python
import torch
from transformers import ViTModel

# 加载预训练ViT模型
model = ViTModel.from_pretrained("google/vit-base-patch16-224")
image = torch.randn(1, 3, 224, 224)

# 提取特征
outputs = model(image)
last_hidden_states = outputs.last_hidden_state  # [1, 197, 768]
cls_token = last_hidden_states[:, 0, :]        # [1, 768]
patch_tokens = last_hidden_states[:, 1:, :]    # [1, 196, 768]
```

---

## 5. Swin Transformer

### 5.1 核心思想

引入层次化设计和局部注意力，解决ViT的计算复杂度问题。

### 5.2 架构特点

| 特点 | 描述 |
|------|------|
| **层次化** | 类似CNN的层次结构 |
| **局部窗口注意力** | 限制注意力范围，降低复杂度 |
| **移位窗口** | 跨窗口交互，保持全局信息 |
| **Patch Merging** | 下采样，类似池化 |

### 5.3 移位窗口注意力

```
普通窗口注意力:
┌─────────┬─────────┐
│ 窗口1   │ 窗口2   │
│ ←→←→←→  │ ←→←→←→  │
├─────────┼─────────┤
│ 窗口3   │ 窗口4   │
│ ←→←→←→  │ ←→←→←→  │
└─────────┴─────────┘

移位窗口注意力:
┌───┬───┬───┬───┐
│ 1 │ 1 │ 2 │ 2 │
├───┼───┼───┼───┤
│ 1 │ 1 │ 2 │ 2 │
├───┼───┼───┼───┤
│ 3 │ 3 │ 4 │ 4 │
└───┴───┴───┴───┘
```

### 5.4 Swin变体

| 变体 | 参数 | 特点 |
|------|------|------|
| **Swin-T** | 28M | 小型模型 |
| **Swin-S** | 50M | 中型模型 |
| **Swin-B** | 88M | 大型模型 |
| **Swin-L** | 196M | 超大型模型 |

---

## 6. 其他视觉编码器

### 6.1 DETR

**论文**：End-to-End Object Detection with Transformers (Carion et al., 2020)

**特点**：
- 端到端目标检测
- 使用Transformer进行对象查询
- 无需手工设计的组件

### 6.2 EfficientNet

**论文**：EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (Tan & Le, 2019)

**特点**：
- 高效的模型缩放策略
- 平衡深度、宽度、分辨率
- 更高的参数效率

### 6.3 ConvNeXt

**论文**：ConvNeXt: A ConvNet for the 2020s (Liu et al., 2022)

**特点**：
- 将Transformer思想引入CNN
- 残差块重新设计
- 接近ViT的性能

---

## 7. 视觉编码器对比

### 7.1 性能对比

| 模型 | 参数 | ImageNet Top-1 | 计算量 |
|------|------|---------------|--------|
| **ResNet-50** | 25M | 76.1% | 4.1 GFLOPs |
| **ViT-B/16** | 86M | 77.9% | 17.5 GFLOPs |
| **Swin-B** | 88M | 83.2% | 15.4 GFLOPs |
| **ConvNeXt-B** | 89M | 82.6% | 15.3 GFLOPs |

### 7.2 选择建议

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **计算资源有限** | ResNet-50、EfficientNet | 高效 |
| **通用视觉任务** | ViT-B、Swin-B | 平衡性能与效率 |
| **高精度需求** | Swin-L、ViT-H | 最佳性能 |
| **目标检测** | DETR、Swin | 专门优化 |

---

## 8. 视觉编码器在VLM中的应用

### 8.1 特征提取策略

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| **整体特征** | 使用[CLS] token | 图像级别任务 |
| **局部特征** | 使用patch特征 | 细粒度任务 |
| **多尺度特征** | 融合不同层级 | 多尺度任务 |

### 8.2 与语言模型的融合

```
视觉特征 → 投影层 → 与文本特征融合
                ↓
           跨模态注意力
                ↓
           融合特征输出
```

### 8.3 示例：CLIP中的视觉编码器

```python
# CLIP使用ViT作为视觉编码器
from clip import clip

model, preprocess = clip.load("ViT-B/32")
image = preprocess(Image.open("example.jpg")).unsqueeze(0)
image_features = model.encode_image(image)  # [1, 512]
```

---

## 9. 实践练习

### 练习1：比较不同视觉编码器

```python
import torch
from transformers import ViTModel, SwinModel, ResNetModel

# 加载不同的视觉编码器
vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
swin = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
resnet = ResNetModel.from_pretrained("microsoft/resnet-50")

# 输入图像
image = torch.randn(1, 3, 224, 224)

# 提取特征
vit_features = vit(image).last_hidden_state[:, 0, :]    # [1, 768]
swin_features = swin(image).last_hidden_state[:, 0, :]  # [1, 1024]
resnet_features = resnet(image).pooler_output           # [1, 2048]

print(f"ViT特征维度: {vit_features.shape}")
print(f"Swin特征维度: {swin_features.shape}")
print(f"ResNet特征维度: {resnet_features.shape}")
```

### 练习2：可视化视觉注意力

```python
import torch
import matplotlib.pyplot as plt
from transformers import ViTModel, ViTImageProcessor
from PIL import Image

# 加载模型
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224", output_attentions=True)

# 加载图像
image = Image.open("example.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# 推理获取注意力权重
outputs = model(**inputs)
attentions = outputs.attentions  # [num_layers, batch, num_heads, seq_len, seq_len]

# 可视化第一层的注意力
first_layer_attn = attentions[0][0, 0, 1:, 0].detach().numpy()  # CLS token对其他token的注意力

# 调整形状为网格
grid_size = int(first_layer_attn.shape[0] ** 0.5)
attn_map = first_layer_attn.reshape(grid_size, grid_size)

plt.imshow(attn_map, cmap='viridis')
plt.title("ViT Attention Map (CLS token)")
plt.axis('off')
plt.show()
```

### 练习3：多尺度特征融合

```python
import torch
import torch.nn as nn

class MultiScaleFusion(nn.Module):
    def __init__(self, dims=[256, 512, 1024], output_dim=512):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(d, output_dim, 1), nn.ReLU())
            for d in dims
        ])
        self.fusion = nn.Conv2d(output_dim * len(dims), output_dim, 1)
    
    def forward(self, features):
        # features: 多尺度特征列表
        projected = [proj(f) for proj, f in zip(self.projections, features)]
        # 调整尺寸到相同大小
        target_size = projected[0].shape[-2:]
        projected = [nn.functional.interpolate(p, size=target_size) for p in projected]
        concatenated = torch.cat(projected, dim=1)
        fused = self.fusion(concatenated)
        return fused

# 测试
fusion = MultiScaleFusion()
features = [
    torch.randn(1, 256, 56, 56),   # 第一层特征
    torch.randn(1, 512, 28, 28),   # 第二层特征
    torch.randn(1, 1024, 14, 14)   # 第三层特征
]
output = fusion(features)
print(f"融合后特征形状: {output.shape}")
```

---

**下一节**：[跨模态对齐](03-cross-modal-alignment.md)

---

## 参考文献

1. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale.
2. Liu, Z., Lin, Y., Cao, Y., et al. (2021). Swin transformer: Hierarchical vision transformer using shifted windows.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
4. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-end object detection with transformers.
