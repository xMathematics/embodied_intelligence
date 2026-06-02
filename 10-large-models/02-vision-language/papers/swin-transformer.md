# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## 目录

- [1. 论文概述](#1-论文概述)
- [2. 核心思想](#2-核心思想)
- [3. 模型架构](#3-模型架构)
- [4. 训练方法](#4-训练方法)
- [5. 实验结果](#5-实验结果)
- [6. 创新点分析](#6-创新点分析)
- [7. 代码实现](#7-代码实现)
- [8. 总结](#8-总结)

---

## 1. 论文概述

### 1.1 基本信息

**论文标题**：Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

**作者**：Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo

**发表会议**：ICCV 2021

**引用格式**：
```
@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10012--10022},
  year={2021}
}
```

### 1.2 研究背景

**Transformer在视觉任务中的挑战**：
- 计算复杂度高（O(N^2)）
- 难以处理高分辨率图像
- 缺乏层次化表示

**研究目标**：
1. 降低Transformer的计算复杂度
2. 实现层次化视觉表示
3. 在多个视觉任务上取得SOTA性能

### 1.3 核心贡献

1. 提出Swin Transformer，使用移位窗口注意力
2. 实现层次化特征表示
3. 在图像分类、目标检测、分割等任务上取得SOTA

---

## 2. 核心思想

### 2.1 移位窗口注意力

**核心假设**：
- 图像中的局部区域相关性更强
- 限制注意力范围可以降低计算复杂度
- 移位窗口可以实现跨窗口通信

**窗口注意力**：
- 将图像划分为多个窗口
- 每个窗口内进行自注意力
- 计算复杂度从O(N^2)降低到O(N * W^2)

**移位窗口**：
- 下一层窗口位置偏移
- 实现跨窗口信息传递
- 保持局部性的同时实现全局交互

### 2.2 层次化结构

**设计思想**：
- 类似CNN的层次化结构
- 逐步降低特征图分辨率
- 增加通道数

**阶段划分**：
- Stage 1: 高分辨率，低通道数
- Stage 2: 中分辨率，中通道数
- Stage 3: 低分辨率，高通道数
- Stage 4: 最低分辨率，最高通道数

### 2.3 相对位置编码

**设计思想**：
- 使用相对位置编码而非绝对位置编码
- 减少位置编码参数
- 提高模型泛化能力

**公式表达**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + B}{\sqrt{d_k}}\right)V$$

其中 $B$ 是相对位置偏差。

---

## 3. 模型架构

### 3.1 整体架构

```
图像 → Patch Partition → Linear Embedding → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Classification
```

### 3.2 移位窗口注意力

```python
class WindowAttention(nn.Module):
    """窗口注意力层"""
    
    def __init__(self, dim, window_size=7, num_heads=8):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        
        # 相对位置编码
        self.relative_position_bias = nn.Parameter(
            torch.randn(2 * window_size - 1, 2 * window_size - 1, num_heads)
        )
        
        # 线性变换
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入特征 [B, H, W, D]
            mask: 窗口掩码 [num_windows, window_size^2, window_size^2]
        
        返回:
            out: 输出特征 [B, H, W, D]
        """
        B, H, W, D = x.shape
        
        # 划分窗口
        x = x.view(B, H // self.window_size, self.window_size, 
                   W // self.window_size, self.window_size, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, D)
        
        # QKV变换
        qkv = self.qkv(x).view(-1, self.window_size * self.window_size, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(2)  # [num_windows*B, N, H, D/H]
        
        # 计算注意力
        q = q.transpose(1, 2)  # [num_windows*B, H, N, D/H]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(D // self.num_heads).float())  # [num_windows*B, H, N, N]
        
        # 添加相对位置编码
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Ws, Ws]
        coords_flatten = coords.view(2, -1).transpose(0, 1)  # [N, 2]
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # [N, N, 2]
        relative_coords += self.window_size - 1  # 偏移到非负
        relative_coords = relative_coords[:, :, 0] * (2 * self.window_size - 1) + relative_coords[:, :, 1]
        relative_position_bias = self.relative_position_bias.view(-1, self.num_heads)[relative_coords.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, self.num_heads
        ).permute(2, 0, 1).contiguous()  # [H, N, N]
        
        scores = scores + relative_position_bias.unsqueeze(0)
        
        # 添加掩码
        if mask is not None:
            scores = scores + mask.unsqueeze(1)
        
        # Softmax和加权求和
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(-1, self.window_size * self.window_size, D)
        
        # 投影
        out = self.proj(out)
        
        # 恢复形状
        out = out.view(B, H // self.window_size, W // self.window_size, 
                       self.window_size, self.window_size, D)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, D)
        
        return out
```

### 3.3 Swin Transformer块

```python
class SwinTransformerBlock(nn.Module):
    """Swin Transformer块"""
    
    def __init__(self, dim, window_size=7, num_heads=8, shift_size=0):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        
        # 归一化和窗口注意力
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        
        # 归一化和MLP
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # 计算掩码（当shift_size > 0时）
        if self.shift_size > 0:
            H, W = self.window_size, self.window_size
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = self.create_mask_windows(img_mask)
            self.register_buffer("attn_mask", mask_windows)
        else:
            self.attn_mask = None
    
    def create_mask_windows(self, img_mask):
        """创建窗口掩码"""
        B, H, W, C = img_mask.shape
        mask_windows = img_mask.view(B, H // self.window_size, self.window_size, 
                                     W // self.window_size, self.window_size, C)
        mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size, C)
        
        # 计算掩码
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask
    
    def forward(self, x):
        """
        参数:
            x: 输入特征 [B, H, W, D]
        
        返回:
            out: 输出特征 [B, H, W, D]
        """
        B, H, W, D = x.shape
        
        # 残差连接1
        residual = x
        
        # 归一化和注意力
        x = self.norm1(x)
        
        # 移位（当shift_size > 0时）
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 窗口注意力
        attn_out = self.attn(shifted_x, mask=self.attn_mask)
        
        # 反向移位
        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        # 残差连接
        x = residual + attn_out
        
        # 残差连接2
        residual = x
        
        # 归一化和MLP
        x = self.norm2(x)
        x = self.mlp(x)
        
        # 残差连接
        out = residual + x
        
        return out
```

### 3.4 Swin Transformer模型

```python
class SwinTransformer(nn.Module):
    """Swin Transformer模型"""
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4.):
        super().__init__()
        
        self.num_classes = num_classes
        self.depths = depths
        
        # Patch分割和嵌入
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.LayerNorm(embed_dim)
        )
        
        # Swin Transformer层
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = nn.Sequential(*[
                SwinTransformerBlock(
                    dim=embed_dim * (2 ** i),
                    window_size=window_size,
                    num_heads=num_heads[i],
                    shift_size=0 if (j % 2 == 0) else window_size // 2
                ) for j in range(depths[i])
            ])
            self.layers.append(layer)
            
            # 下采样（除了最后一层）
            if i < len(depths) - 1:
                downsample = nn.Sequential(
                    nn.LayerNorm(embed_dim * (2 ** i)),
                    nn.Conv2d(embed_dim * (2 ** i), embed_dim * (2 ** (i + 1)),
                              kernel_size=2, stride=2)
                )
                self.layers.append(downsample)
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim * (2 ** (len(depths) - 1)))
        self.head = nn.Linear(embed_dim * (2 ** (len(depths) - 1)), num_classes)
    
    def forward(self, x):
        """
        参数:
            x: 输入图像 [B, C, H, W]
        
        返回:
            logits: 分类logits [B, num_classes]
        """
        # Patch嵌入
        x = self.patch_embed(x)  # [B, D, H/4, W/4]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H/4, W/4, D]
        
        # Swin Transformer层
        for layer in self.layers:
            x = layer(x)
        
        # 全局平均池化
        x = x.mean(dim=(1, 2))  # [B, D]
        
        # 分类
        x = self.norm(x)
        logits = self.head(x)
        
        return logits
```

---

## 4. 训练方法

### 4.1 数据集

**训练数据集**：
- ImageNet-1K：128万训练图像
- ImageNet-22K：1400万训练图像

**验证数据集**：
- ImageNet-1K验证集：5万图像

### 4.2 训练配置

**batch size**：1024

**学习率**：6e-3

**训练轮数**：300 epochs

**优化器**：AdamW

**权重衰减**：0.05

### 4.3 数据增强

**图像增强**：
- RandAugment
- Mixup
- Cutmix
- Random Erasing

### 4.4 训练流程

```
for epoch in range(num_epochs):
    for batch in dataloader:
        images, labels = batch
        
        # 前向传播
        logits = swin_transformer(images)
        
        # 计算损失
        loss = F.cross_entropy(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 5. 实验结果

### 5.1 图像分类结果

**ImageNet-1K分类结果**：

| 模型 | 参数量 | Top-1准确率 |
|------|--------|------------|
| Swin-T | 28M | 81.3% |
| Swin-S | 50M | 83.0% |
| Swin-B | 88M | 83.5% |
| Swin-L | 196M | 84.5% |

### 5.2 目标检测结果

**COCO目标检测结果**：

| 模型 | mAP |
|------|-----|
| Swin-T + DETR | 49.4% |
| Swin-S + DETR | 51.1% |
| Swin-B + DETR | 52.3% |

### 5.3 语义分割结果

**ADE20K语义分割结果**：

| 模型 | mIoU |
|------|------|
| Swin-T + UPerNet | 47.3% |
| Swin-S + UPerNet | 48.5% |
| Swin-B + UPerNet | 49.1% |

### 5.4 消融实验

**窗口大小影响**：

| 窗口大小 | Top-1准确率 | 计算复杂度 |
|----------|------------|-----------|
| 4 | 79.8% | 低 |
| 7 | 81.3% | 中 |
| 12 | 80.5% | 高 |

**层数影响**：

| 深度 | Top-1准确率 | 参数数量 |
|------|------------|---------|
| [2, 2, 6, 2] | 81.3% | 28M |
| [3, 4, 6, 3] | 82.1% | 50M |
| [4, 6, 12, 4] | 83.5% | 88M |

---

## 6. 创新点分析

### 6.1 移位窗口注意力

**创新之处**：
- 限制注意力范围，降低计算复杂度
- 移位窗口实现跨窗口通信
- 保持局部性的同时实现全局交互

**影响**：
- 使Transformer可以处理高分辨率图像
- 提高计算效率
- 为后续视觉Transformer提供参考

### 6.2 层次化结构

**创新之处**：
- 类似CNN的层次化特征表示
- 逐步降低分辨率，增加通道数
- 更好地捕捉多尺度特征

**影响**：
- 提高模型的表达能力
- 增强多尺度理解能力
- 改善下游任务性能

### 6.3 相对位置编码

**创新之处**：
- 使用相对位置编码
- 减少参数数量
- 提高模型泛化能力

**影响**：
- 降低模型复杂度
- 提高训练效率
- 改善模型的泛化能力

---

## 7. 代码实现

### 7.1 完整模型训练

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 初始化模型
swin_model = SwinTransformer(
    img_size=224,
    patch_size=4,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    num_classes=1000
)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swin_model.to(device)

# 优化器
optimizer = optim.AdamW(swin_model.parameters(), lr=6e-3, weight_decay=0.05)

# 数据加载器
dataloader = DataLoader(imagenet_dataset, batch_size=1024, shuffle=True)

# 训练循环
num_epochs = 300
for epoch in range(num_epochs):
    swin_model.train()
    total_loss = 0
    
    for batch in dataloader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        logits = swin_model(images)
        
        # 计算损失
        loss = F.cross_entropy(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 保存模型
torch.save(swin_model.state_dict(), "swin_transformer.pth")
```

### 7.2 推理示例

```python
class SwinClassifier:
    """Swin Transformer分类器"""
    
    def __init__(self, model, preprocessor, class_names):
        self.model = model
        self.preprocessor = preprocessor
        self.class_names = class_names
        self.device = next(model.parameters()).device
    
    def classify(self, image):
        """
        参数:
            image: 输入图像
        
        返回:
            prediction: 预测类别
            confidence: 置信度
        """
        # 预处理
        input_tensor = self.preprocessor(image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=-1)
        
        # 获取预测结果
        confidence, idx = torch.max(probabilities, dim=-1)
        prediction = self.class_names[idx.item()]
        
        return prediction, confidence.item()

# 使用示例
classifier = SwinClassifier(swin_model, preprocessor, imagenet_class_names)
image = Image.open("cat.jpg")
prediction, confidence = classifier.classify(image)
print(f"预测类别: {prediction}, 置信度: {confidence:.2f}")
```

---

## 8. 总结

### 8.1 核心贡献

1. **提出Swin Transformer**：使用移位窗口注意力
2. **实现层次化特征表示**：类似CNN的结构
3. **在多个任务上取得SOTA**：分类、检测、分割

### 8.2 影响与意义

**对视觉Transformer的影响**：
- 推动视觉Transformer的发展
- 提供了高效的注意力机制
- 为后续模型提供参考

**未来方向**：
- 更大规模的模型
- 更高效的注意力机制
- 多模态应用

---

## 9. 进阶话题

### 9.1 移位窗口注意力深度解析

**计算复杂度分析**：
标准Transformer的自注意力复杂度为O(N^2)，其中N是序列长度。对于图像，N = H * W，所以复杂度为O((H*W)^2)。

Swin Transformer使用窗口注意力后，复杂度降低为：
- 每个窗口内的注意力：O(W^2 * D)，其中W是窗口大小
- 总窗口数：O((H*W)/W^2) = O(H*W/W^2)
- 总复杂度：O(H*W * W^2 * D) = O(H*W * D * W^2) = O(N * D * W^2)

当W=7，H=W=224时：
- 标准Transformer：O((224*224)^2) ≈ O(2.5B)
- Swin Transformer：O(224*224 * 7^2) ≈ O(2.5M)

**窗口划分策略**：
```python
class WindowPartitioner:
    """窗口划分器"""
    
    def __init__(self, window_size=7, shift_size=0):
        self.window_size = window_size
        self.shift_size = shift_size
    
    def partition(self, x):
        """
        参数:
            x: 输入特征 [B, H, W, D]
        
        返回:
            windows: 划分后的窗口 [num_windows*B, window_size, window_size, D]
        """
        B, H, W, D = x.shape
        
        # 填充到窗口大小的倍数
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_w, 0, pad_h, 0))
        
        # 移位
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        
        # 划分窗口
        x = x.view(B, H // self.window_size, self.window_size, 
                   W // self.window_size, self.window_size, D)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, D)
        
        return windows
    
    def merge(self, windows, original_size):
        """
        参数:
            windows: 窗口特征 [num_windows*B, window_size, window_size, D]
            original_size: 原始尺寸 (H, W)
        
        返回:
            x: 合并后的特征 [B, H, W, D]
        """
        B, H, W = original_size
        num_windows_h = H // self.window_size
        num_windows_w = W // self.window_size
        
        # 恢复形状
        x = windows.view(B, num_windows_h, num_windows_w, 
                         self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        
        # 反向移位
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        return x
```

### 9.2 层次化特征融合

**多尺度特征融合**：
```python
class FeatureFusion(nn.Module):
    """多尺度特征融合"""
    
    def __init__(self, dims=[96, 192, 384, 768]):
        super().__init__()
        
        # 上采样层
        self.upsamplers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.upsamplers.append(nn.Sequential(
                nn.ConvTranspose2d(dims[i+1], dims[i], kernel_size=2, stride=2),
                nn.LayerNorm(dims[i])
            ))
        
        # 融合层
        self.fusion_layers = nn.ModuleList()
        for i in range(len(dims)):
            self.fusion_layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i]),
                nn.GELU(),
                nn.Linear(dims[i], dims[0])
            ))
    
    def forward(self, features):
        """
        参数:
            features: 各阶段特征列表
        
        返回:
            fused: 融合后的特征
        """
        # 从最深层开始上采样和融合
        fused = features[-1]
        
        for i in range(len(features)-2, -1, -1):
            # 上采样
            fused = self.upsamplers[i](fused.permute(0, 3, 1, 2))  # [B, D, H, W]
            fused = fused.permute(0, 2, 3, 1)  # [B, H, W, D]
            
            # 融合
            fused = fused + features[i]
        
        return fused
```

### 9.3 相对位置编码详解

**相对位置编码计算**：
```python
class RelativePositionEncoding(nn.Module):
    """相对位置编码"""
    
    def __init__(self, window_size=7, num_heads=8):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        
        # 相对位置偏差参数
        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # 计算相对位置索引
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Ws, Ws]
        coords_flatten = coords.view(2, -1).transpose(0, 1)  # [N, 2]
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # [N, N, 2]
        relative_coords += window_size - 1  # 偏移到非负
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.relative_position_index = relative_coords.sum(-1)  # [N, N]
    
    def forward(self):
        """
        返回:
            relative_position_bias: 相对位置偏差 [H, N, N]
        """
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        bias = bias.view(self.window_size * self.window_size, 
                         self.window_size * self.window_size, 
                         self.num_heads)
        bias = bias.permute(2, 0, 1).contiguous()  # [H, N, N]
        
        return bias
```

---

## 10. 高级应用技巧

### 10.1 目标检测

**Swin Transformer作为检测器骨干**：
```python
class SwinDetection(nn.Module):
    """基于Swin Transformer的目标检测器"""
    
    def __init__(self, swin_model, num_classes=80):
        super().__init__()
        self.backbone = swin_model
        
        # 检测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes + 4, kernel_size=1)
        )
    
    def forward(self, x):
        """
        参数:
            x: 输入图像 [B, C, H, W]
        
        返回:
            detections: 检测结果
        """
        # 提取特征
        features = self.backbone.extract_features(x)
        
        # 获取最后一层特征
        last_feature = features[-1]  # [B, D, H/32, W/32]
        
        # 检测头
        detections = self.detection_head(last_feature)
        
        return detections
```

### 10.2 语义分割

**Swin Transformer作为分割器骨干**：
```python
class SwinSegmentation(nn.Module):
    """基于Swin Transformer的语义分割器"""
    
    def __init__(self, swin_model, num_classes=150):
        super().__init__()
        self.backbone = swin_model
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(96, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        """
        参数:
            x: 输入图像 [B, C, H, W]
        
        返回:
            segmentation: 分割结果 [B, num_classes, H, W]
        """
        # 提取特征
        features = self.backbone.extract_features(x)
        
        # 获取最后一层特征
        last_feature = features[-1]  # [B, D, H/32, W/32]
        
        # 解码
        segmentation = self.decoder(last_feature)
        
        return segmentation
```

### 10.3 实例分割

**Mask R-CNN结合Swin**：
```python
class SwinMaskRCNN(nn.Module):
    """基于Swin的Mask R-CNN"""
    
    def __init__(self, swin_model, num_classes=80):
        super().__init__()
        self.backbone = swin_model
        
        # RPN
        self.rpn = RPN(in_channels=768)
        
        # ROI Head
        self.roi_head = ROIHead(
            in_channels=768,
            num_classes=num_classes
        )
    
    def forward(self, x, targets=None):
        """
        参数:
            x: 输入图像 [B, C, H, W]
            targets: 目标标签（训练时）
        
        返回:
            outputs: 检测和分割结果
        """
        # 提取特征
        features = self.backbone.extract_features(x)
        
        # RPN
        proposals = self.rpn(features, targets)
        
        # ROI Head
        outputs = self.roi_head(features, proposals, targets)
        
        return outputs
```

---

## 11. 模型优化与部署

### 11.1 模型压缩

**通道剪枝**：
```python
class ChannelPruner:
    """通道剪枝器"""
    
    def __init__(self, model):
        self.model = model
    
    def prune(self, prune_ratio=0.3):
        """
        参数:
            prune_ratio: 剪枝比例
        
        返回:
            pruned_model: 剪枝后的模型
        """
        # 对每个卷积层进行剪枝
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 计算通道重要性
                importance = self.compute_importance(module)
                
                # 选择要保留的通道
                num_channels = module.out_channels
                num_keep = int(num_channels * (1 - prune_ratio))
                keep_indices = torch.argsort(importance, descending=True)[:num_keep]
                
                # 剪枝
                module.weight.data = module.weight.data[keep_indices]
                if module.bias is not None:
                    module.bias.data = module.bias.data[keep_indices]
                
                # 更新后续层的输入通道数
                self.update_next_layer(name, keep_indices)
        
        return self.model
    
    def compute_importance(self, conv_layer):
        """计算通道重要性"""
        return torch.sum(torch.abs(conv_layer.weight.data), dim=(1, 2, 3))
    
    def update_next_layer(self, layer_name, keep_indices):
        """更新后续层的输入通道"""
        # 找到后续层并更新
        pass
```

### 11.2 量化

**INT8量化**：
```python
import torch.ao.quantization as quantization

# 配置量化
model.qconfig = quantization.get_default_qconfig("qnnpack")

# 准备量化
model_prepared = quantization.prepare(model)

# 校准
for batch in calibration_data:
    images = batch
    model_prepared(images)

# 量化
model_quantized = quantization.convert(model_prepared)
```

### 11.3 ONNX导出与优化

**导出模型**：
```python
# 准备输入
image = torch.randn(1, 3, 224, 224)

# 导出
torch.onnx.export(
    model,
    image,
    "swin_transformer.onnx",
    opset_version=14,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch_size", 2: "height", 3: "width"},
        "logits": {0: "batch_size"}
    }
)

# 使用ONNX Runtime优化
import onnxruntime as ort
session = ort.InferenceSession("swin_transformer.onnx", providers=["CPUExecutionProvider"])
```

---

## 12. 实战项目案例

### 12.1 图像分类服务

**FastAPI服务**：
```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI(title="Swin Transformer Classification API")

# 加载模型
model = load_swin_model()
preprocessor = load_preprocessor()

@app.post("/classify")
async def classify_image(image: UploadFile = File(...)):
    """图像分类"""
    # 读取图像
    image = Image.open(image.file).convert("RGB")
    
    # 预处理
    input_tensor = preprocessor(image).unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=-1)
    
    # 获取top-5预测
    top5_probs, top5_indices = torch.topk(probabilities, 5)
    
    results = []
    for prob, idx in zip(top5_probs[0], top5_indices[0]):
        results.append({
            "class": imagenet_class_names[idx],
            "confidence": prob.item()
        })
    
    return {"predictions": results}
```

### 12.2 实时目标检测系统

**实时检测系统**：
```python
class RealTimeDetector:
    """实时目标检测系统"""
    
    def __init__(self, model, processor, threshold=0.5):
        self.model = model
        self.processor = processor
        self.threshold = threshold
    
    def detect(self, frame):
        """
        参数:
            frame: 输入帧
        
        返回:
            detections: 检测结果
        """
        # 预处理
        inputs = self.processor(images=frame, return_tensors="pt")
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 解析结果
        detections = []
        for i in range(len(outputs["boxes"])):
            boxes = outputs["boxes"][i]
            scores = outputs["scores"][i]
            labels = outputs["labels"][i]
            
            for box, score, label in zip(boxes, scores, labels):
                if score > self.threshold:
                    detections.append({
                        "label": label,
                        "score": score.item(),
                        "box": box.tolist()
                    })
        
        return detections
    
    def run(self, video_source=0):
        """运行实时检测"""
        cap = cv2.VideoCapture(video_source)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测
            detections = self.detect(frame)
            
            # 绘制结果
            for det in detections:
                box = det["box"]
                cv2.rectangle(frame, (int(box[0]), int(box[1])), 
                              (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{det['label']}: {det['score']:.2f}", 
                            (int(box[0]), int(box[1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 显示
            cv2.imshow("Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
```

### 12.3 医学图像分割系统

**医学图像分割**：
```python
class MedicalSegmentation:
    """医学图像分割系统"""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def segment(self, image_path):
        """
        参数:
            image_path: 图像路径
        
        返回:
            mask: 分割掩码
        """
        # 加载图像
        image = Image.open(image_path)
        
        # 预处理
        input_tensor = self.processor(image).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        return mask
    
    def visualize(self, image_path):
        """可视化分割结果"""
        # 加载图像
        image = np.array(Image.open(image_path))
        
        # 分割
        mask = self.segment(image_path)
        
        # 可视化
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5, cmap="jet")
        plt.title("Segmentation Result")
        
        plt.show()
```

---

## 13. Swin Transformer变体

### 13.1 Swin Transformer V2

**改进点**：
- 更大的模型容量
- 改进的相对位置编码
- 更高的分辨率训练

```python
class SwinTransformerV2(nn.Module):
    """Swin Transformer V2"""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        # 改进的位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, 1024, kwargs["embed_dim"]))
        
        # 改进的注意力
        self.attention = nn.MultiheadAttention(
            kwargs["embed_dim"], 
            kwargs["num_heads"],
            dropout=0.1
        )
```

### 13.2 Swin Transformer Tiny

**轻量级版本**：
```python
class SwinTransformerTiny(SwinTransformer):
    """Swin Transformer Tiny"""
    
    def __init__(self):
        super().__init__(
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7
        )
```

---

## 14. 未来发展方向

### 14.1 研究挑战

**当前挑战**：
1. **计算效率**：虽然比标准Transformer高效，但仍需优化
2. **长距离依赖**：窗口注意力限制了长距离交互
3. **动态窗口**：固定窗口大小可能不是最优的
4. **多模态融合**：如何更好地与语言模型结合

### 14.2 潜在解决方案

**可能的方向**：
1. **动态窗口大小**：根据内容自适应调整窗口大小
2. **稀疏注意力**：只关注重要区域
3. **混合架构**：结合CNN和Transformer的优点
4. **高效训练**：使用更高效的训练策略

### 14.3 应用展望

**未来应用场景**：
1. **自动驾驶**：实时感知和决策
2. **医疗影像**：高精度医学图像分析
3. **机器人视觉**：实时场景理解
4. **视频理解**：长视频内容分析

---

## 附录：常见问题

### Q1：Swin Transformer与ViT有什么区别？

A：主要区别在于：
- ViT使用全局注意力，计算复杂度高
- Swin Transformer使用窗口注意力，更高效
- Swin Transformer具有层次化结构
- Swin Transformer使用相对位置编码

### Q2：如何选择窗口大小？

A：窗口大小的选择需要权衡：
- 小窗口：计算效率高，但局部性强
- 大窗口：建模能力强，但计算量大
- 通常选择7或12

### Q3：Swin Transformer适合哪些任务？

A：Swin Transformer适合多种视觉任务：
- 图像分类
- 目标检测
- 语义分割
- 实例分割
- 姿态估计

### Q4：如何训练更大的Swin模型？

A：训练更大模型需要：
- 更大的batch size
- 更长的训练时间
- 学习率调整
- 梯度累积

---

**返回**：[视觉编码器](../02-vision-encoder.md)