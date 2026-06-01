# 1.3 语义分割

## 目录

- [1. 引言](#1-引言)
- [2. 语义分割概述](#2-语义分割概述)
- [3. 经典方法](#3-经典方法)
- [4. 现代方法](#4-现代方法)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 语义分割的重要性

**语义分割**是将图像中的每个像素分配到特定类别的任务。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **自动驾驶** | 道路分割、障碍物识别 | 车道线检测 |
| **医学影像** | 组织分割、病变识别 | 器官分割 |
| **遥感图像** | 土地利用分类 | 城市规划 |
| **视频监控** | 前景背景分离 | 行为分析 |

---

## 2. 语义分割概述

### 2.1 定义

**语义分割**：对图像中的每个像素进行分类，得到密集的像素级标签。

**输出格式**：与输入图像大小相同的标签图。

### 2.2 评价指标

| 指标 | 描述 | 计算公式 |
|------|------|---------|
| **IoU** | 交并比 | IoU = Area(∩) / Area(∪) |
| **mIoU** | 平均交并比 | 各类别IoU的平均值 |
| **Pixel Accuracy** | 像素准确率 | 正确分类像素数/总像素数 |
| **F1 Score** | F1分数 | 2 * precision * recall / (precision + recall) |

---

## 3. 经典方法

### 3.1 FCN (Fully Convolutional Network)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        
        # 编码器（基于VGG16）
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        # 分类层
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # 上采样
        self.upsample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16)
    
    def forward(self, x):
        """前向传播"""
        # 编码
        features = self.encoder(x)
        
        # 分类
        x = self.classifier(features)
        
        # 上采样到原始大小
        x = self.upsample(x)
        
        return x

# 测试
model = FCN(num_classes=21)
input = torch.randn(1, 3, 224, 224)
output = model(input)
print(f"FCN输出形状: {output.shape}")
```

### 3.2 U-Net

```python
class UNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # 编码器
        self.encoder = UNetEncoder()
        
        # 解码器
        self.decoder = UNetDecoder()
        
        # 最终分类
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        # 编码
        features = self.encoder(x)
        
        # 解码（需要传递编码器特征用于跳跃连接）
        x = self.decoder(features)
        
        # 最终分类
        x = self.final_conv(x)
        
        return x

class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.down1 = self._double_conv(3, 64)
        self.down2 = self._double_conv(64, 128)
        self.down3 = self._double_conv(128, 256)
        self.down4 = self._double_conv(256, 512)
        self.down5 = self._double_conv(512, 1024)
        
        self.pool = nn.MaxPool2d(2)
    
    def _double_conv(self, in_channels, out_channels):
        """双重卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """前向传播"""
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        x5 = self.down5(self.pool(x4))
        
        return [x1, x2, x3, x4, x5]

class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.up1 = self._up_conv(1024, 512)
        self.up2 = self._up_conv(512, 256)
        self.up3 = self._up_conv(256, 128)
        self.up4 = self._up_conv(128, 64)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def _up_conv(self, in_channels, out_channels):
        """上采样卷积"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, features):
        """前向传播"""
        x1, x2, x3, x4, x5 = features
        
        # 上采样和跳跃连接
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        return x

# 测试
model = UNet(num_classes=2)
input = torch.randn(1, 3, 256, 256)
output = model(input)
print(f"U-Net输出形状: {output.shape}")
```

---

## 4. 现代方法

### 4.1 DeepLab系列

```python
class DeepLabV3(nn.Module):
    def __init__(self, num_classes=21, backbone='resnet18'):
        super().__init__()
        
        # 骨干网络
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # ASPP模块
        self.aspp = ASPP(512)
        
        # 解码器
        self.decoder = DeepLabDecoder(num_classes)
    
    def forward(self, x):
        """前向传播"""
        # 骨干网络提取特征
        features = self.backbone(x)
        
        # ASPP
        x = self.aspp(features)
        
        # 解码器
        x = self.decoder(x)
        
        # 上采样到原始大小
        x = F.interpolate(x, size=x.shape[-2:], mode='bilinear', align_corners=True)
        
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.convs = nn.ModuleList([
            # 1x1卷积
            nn.Conv2d(in_channels, 256, kernel_size=1),
            
            # 3x3卷积，不同膨胀率
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=12, dilation=12),
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=18, dilation=18)
        ])
        
        # 全局平均池化
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, kernel_size=1)
        )
        
        # 融合层
        self.fusion = nn.Conv2d(5 * 256, 256, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        outputs = []
        
        for conv in self.convs:
            outputs.append(F.relu(conv(x)))
        
        # 全局特征
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[-2:], mode='bilinear', align_corners=True)
        outputs.append(F.relu(global_feat))
        
        # 融合
        x = torch.cat(outputs, dim=1)
        x = self.fusion(x)
        
        return x

class DeepLabDecoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        return self.conv(x)

# 需要导入torchvision
from torchvision import models

# 测试
model = DeepLabV3(num_classes=21)
input = torch.randn(1, 3, 224, 224)
output = model(input)
print(f"DeepLabV3输出形状: {output.shape}")
```

### 4.2 Transformer-based方法

```python
class SegFormer(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        
        # Patch Embedding
        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(3, 32, 7, 4),
            PatchEmbedding(32, 64, 3, 2),
            PatchEmbedding(64, 128, 3, 2),
            PatchEmbedding(128, 256, 3, 2)
        ])
        
        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(32, 8),
            TransformerBlock(64, 8),
            TransformerBlock(128, 8),
            TransformerBlock(256, 8)
        ])
        
        # 解码器
        self.decoder = SegFormerDecoder()
        
        # 最终分类
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        features = []
        
        # 编码阶段
        for i, (patch_embed, transformer) in enumerate(zip(self.patch_embeddings, self.transformer_blocks)):
            x = patch_embed(x)
            x = transformer(x)
            features.append(x)
        
        # 解码阶段
        x = self.decoder(features)
        
        # 最终分类
        x = self.final_conv(x)
        
        # 上采样
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.norm = nn.LayerNorm(out_channels)
    
    def forward(self, x):
        """前向传播"""
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        """前向传播"""
        # 多头注意力
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

class SegFormerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.decoder_layers = nn.ModuleList([
            nn.Conv2d(256, 64, kernel_size=1),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Conv2d(32, 64, kernel_size=1)
        ])
    
    def forward(self, features):
        """前向传播"""
        outputs = []
        
        for i, (feat, conv) in enumerate(zip(reversed(features), self.decoder_layers)):
            # 恢复形状
            b, n, c = feat.shape
            h = w = int(n ** 0.5)
            feat = feat.transpose(1, 2).view(b, c, h, w)
            
            # 1x1卷积
            x = conv(feat)
            
            # 上采样
            if i < 3:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            
            outputs.append(x)
        
        # 逐元素相加
        x = sum(outputs)
        
        return x

# 测试
model = SegFormer(num_classes=19)
input = torch.randn(1, 3, 224, 224)
output = model(input)
print(f"SegFormer输出形状: {output.shape}")
```

---

## 5. 实践练习

### 练习1：实现简单的分割网络

```python
class SimpleSegmentationNet(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 测试
model = SimpleSegmentationNet(num_classes=21)
input = torch.randn(1, 3, 128, 128)
output = model(input)
print(f"输出形状: {output.shape}")
```

### 练习2：实现Dice Loss

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        计算Dice Loss
        
        参数:
            logits: 模型输出 [batch, num_classes, H, W]
            targets: 标签 [batch, H, W]
        
        返回:
            Dice Loss
        """
        # 转换为概率
        probs = F.softmax(logits, dim=1)
        
        # 将标签转换为one-hot
        targets_onehot = F.one_hot(targets, num_classes=logits.size(1))
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()
        
        # 计算交集和并集
        intersection = torch.sum(probs * targets_onehot, dim=(2, 3))
        union = torch.sum(probs + targets_onehot, dim=(2, 3))
        
        # Dice系数
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Dice Loss
        loss = 1 - torch.mean(dice)
        
        return loss

# 测试
loss_fn = DiceLoss()

logits = torch.randn(2, 3, 32, 32)
targets = torch.randint(0, 3, (2, 32, 32))

loss = loss_fn(logits, targets)
print(f"Dice Loss: {loss.item()}")
```

### 练习3：实现分割评估指标

```python
class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = None
    
    def update(self, predictions, targets):
        """
        更新混淆矩阵
        
        参数:
            predictions: 预测标签 [batch, H, W]
            targets: 真实标签 [batch, H, W]
        """
        # 获取预测
        preds = predictions.flatten()
        tgts = targets.flatten()
        
        # 计算混淆矩阵
        mask = (tgts >= 0) & (tgts < self.num_classes)
        indices = self.num_classes * tgts[mask].long() + preds[mask]
        conf_mat = torch.bincount(indices, minlength=self.num_classes ** 2)
        conf_mat = conf_mat.view(self.num_classes, self.num_classes)
        
        if self.confusion_matrix is None:
            self.confusion_matrix = conf_mat
        else:
            self.confusion_matrix += conf_mat
    
    def compute_miou(self):
        """计算mIoU"""
        if self.confusion_matrix is None:
            return 0.0
        
        # 计算IoU
        intersection = torch.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) - intersection
        
        # 避免除以零
        iou = torch.where(union == 0, torch.tensor(0.0), intersection / union)
        
        # mIoU
        miou = torch.mean(iou).item()
        
        return miou
    
    def compute_pixel_accuracy(self):
        """计算像素准确率"""
        if self.confusion_matrix is None:
            return 0.0
        
        correct = torch.diag(self.confusion_matrix).sum().item()
        total = self.confusion_matrix.sum().item()
        
        return correct / total if total > 0 else 0.0
    
    def reset(self):
        """重置混淆矩阵"""
        self.confusion_matrix = None

# 测试
metrics = SegmentationMetrics(num_classes=3)

predictions = torch.randint(0, 3, (2, 32, 32))
targets = torch.randint(0, 3, (2, 32, 32))

metrics.update(predictions, targets)

miou = metrics.compute_miou()
pixel_acc = metrics.compute_pixel_accuracy()

print(f"mIoU: {miou:.4f}")
print(f"Pixel Accuracy: {pixel_acc:.4f}")
```

---

**下一节**：[实例分割](04-instance-segmentation.md)

---

## 参考文献

1. Long, J., et al. (2015). Fully Convolutional Networks for Semantic Segmentation.
2. Ronneberger, O., et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
3. Chen, L.-C., et al. (2018). DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.