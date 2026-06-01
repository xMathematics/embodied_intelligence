# 1.5 关键点检测

## 目录

- [1. 引言](#1-引言)
- [2. 关键点检测概述](#2-关键点检测概述)
- [3. 人体关键点检测](#3-人体关键点检测)
- [4. 物体关键点检测](#4-物体关键点检测)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 关键点检测的重要性

**关键点检测**是识别图像中特定位置的任务，在人体姿态估计、物体识别和动作分析中至关重要。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **人体姿态估计** | 检测人体关节点 | 运动分析 |
| **面部关键点检测** | 检测面部特征点 | 表情识别 |
| **手势识别** | 检测手部关键点 | 交互控制 |
| **物体姿态估计** | 检测物体关键点 | 机器人抓取 |

---

## 2. 关键点检测概述

### 2.1 定义

**关键点检测**：定位图像中特定的兴趣点，如人体关节、面部特征等。

**输出格式**：
```
[
    {
        "keypoint_id": 0,
        "x": 120,
        "y": 85,
        "confidence": 0.98
    },
    ...
]
```

### 2.2 评价指标

| 指标 | 描述 | 计算公式 |
|------|------|---------|
| **PCK** | 正确关键点百分比 | 距离阈值内关键点比例 |
| **OKS** | 目标关键点相似度 | 考虑尺度的关键点匹配度 |
| **mAP** | 平均精度均值 | 不同阈值下的平均精度 |

---

## 3. 人体关键点检测

### 3.1 HRNet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HRNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()
        
        # 阶段1：高分辨率分支
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(2, stride=2)
        )
        
        # 阶段2：多分辨率融合
        self.stage2 = HRStage(64, [64, 128])
        
        # 阶段3：多分辨率融合
        self.stage3 = HRStage(128, [64, 128, 256])
        
        # 阶段4：多分辨率融合
        self.stage4 = HRStage(256, [64, 128, 256, 512])
        
        # 最终预测头
        self.final_layer = nn.Conv2d(64, num_keypoints, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        # 阶段1
        x = self.stage1(x)
        
        # 阶段2-4
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # 获取高分辨率特征
        x = x[0]
        
        # 最终预测
        x = self.final_layer(x)
        
        return x

class HRStage(nn.Module):
    def __init__(self, in_channels, out_channels_list):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        
        for out_channels in out_channels_list:
            block = HRBlock(in_channels, out_channels)
            self.blocks.append(block)
            in_channels = out_channels
    
    def forward(self, inputs):
        """前向传播"""
        outputs = []
        
        for i, block in enumerate(self.blocks):
            if i == 0:
                # 第一个分支使用输入
                x = block(inputs) if isinstance(inputs, torch.Tensor) else block(inputs[0])
            else:
                # 后续分支使用前一分支的下采样
                prev = outputs[i-1]
                x = F.interpolate(prev, scale_factor=0.5, mode='bilinear')
                x = block(x)
            
            outputs.append(x)
        
        # 跨分支融合
        outputs = self._fuse_branches(outputs)
        
        return outputs
    
    def _fuse_branches(self, outputs):
        """跨分支特征融合"""
        fused = []
        
        for i, out in enumerate(outputs):
            fused_out = out
            
            for j, other in enumerate(outputs):
                if i != j:
                    if j < i:
                        # 上采样
                        other_up = F.interpolate(other, scale_factor=2 ** (i - j), mode='bilinear')
                        fused_out += other_up
                    else:
                        # 下采样
                        other_down = F.interpolate(other, scale_factor=2 ** (i - j), mode='bilinear')
                        fused_out += other_down
            
            fused.append(fused_out)
        
        return fused

class HRBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        """前向传播"""
        residual = x
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        if residual.size(1) != x.size(1):
            residual = nn.Conv2d(residual.size(1), x.size(1), kernel_size=1)(residual)
        
        x += residual
        x = F.relu(x)
        
        return x

# 测试
model = HRNet(num_keypoints=17)
input = torch.randn(1, 3, 256, 256)
output = model(input)
print(f"HRNet输出形状: {output.shape}")
```

### 3.2 Simple Baseline

```python
class SimpleBaseline(nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()
        
        # 骨干网络
        self.backbone = nn.Sequential(
            # 阶段1
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # 阶段2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 阶段3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # 阶段4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 预测头
        self.final_layer = nn.Conv2d(64, num_keypoints, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        # 编码
        x = self.backbone(x)
        
        # 解码
        x = self.decoder(x)
        
        # 预测
        x = self.final_layer(x)
        
        return x

# 测试
model = SimpleBaseline(num_keypoints=17)
input = torch.randn(1, 3, 256, 256)
output = model(input)
print(f"SimpleBaseline输出形状: {output.shape}")
```

---

## 4. 物体关键点检测

### 4.1 CenterNet

```python
class CenterNet(nn.Module):
    def __init__(self, num_keypoints=17):
        super().__init__()
        
        # 骨干网络（Hourglass-104简化版）
        self.backbone = HourglassBackbone()
        
        # 检测头
        self.detection_head = CenterNetHead(num_keypoints)
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        features = self.backbone(x)
        
        # 检测
        outputs = self.detection_head(features)
        
        return outputs

class HourglassBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 残差模块
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256)
        )
        
        # Hourglass模块
        self.hourglass = HourglassModule(256)
    
    def forward(self, x):
        """前向传播"""
        x = self.init_conv(x)
        x = self.residual_blocks(x)
        x = self.hourglass(x)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 下采样残差
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
    
    def forward(self, x):
        """前向传播"""
        residual = self.downsample(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        x += residual
        x = F.relu(x)
        
        return x

class HourglassModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # 上分支
        self.up1 = ResidualBlock(channels, channels)
        
        # 下分支
        self.down1 = nn.MaxPool2d(2, stride=2)
        self.down2 = ResidualBlock(channels, channels)
        self.down3 = ResidualBlock(channels, channels)
        
        # 中间
        self.middle = ResidualBlock(channels, channels)
        
        # 上采样
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
    
    def forward(self, x):
        """前向传播"""
        # 上分支
        up1 = self.up1(x)
        
        # 下分支
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        
        # 中间
        middle = self.middle(down3)
        
        # 上采样融合
        up2 = self.up2(middle)
        out = up1 + up2
        
        return out

class CenterNetHead(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        
        # 热力图分支
        self.heatmap_conv = nn.Conv2d(256, num_keypoints, kernel_size=1)
        
        # 偏移量分支
        self.offset_conv = nn.Conv2d(256, 2, kernel_size=1)
        
        # 尺寸分支
        self.size_conv = nn.Conv2d(256, 2, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        heatmap = self.heatmap_conv(x)
        offset = self.offset_conv(x)
        size = self.size_conv(x)
        
        return {
            'heatmap': heatmap,
            'offset': offset,
            'size': size
        }

# 测试
model = CenterNet(num_keypoints=17)
input = torch.randn(1, 3, 512, 512)
outputs = model(input)
print(f"热力图形状: {outputs['heatmap'].shape}")
print(f"偏移量形状: {outputs['offset'].shape}")
print(f"尺寸形状: {outputs['size'].shape}")
```

---

## 5. 实践练习

### 练习1：实现简单的关键点检测网络

```python
class SimpleKeypointDetector(nn.Module):
    def __init__(self, num_keypoints=17):
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
            
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        # 预测头
        self.final_layer = nn.Conv2d(64, num_keypoints, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_layer(x)
        
        return x

# 测试
model = SimpleKeypointDetector(num_keypoints=17)
input = torch.randn(1, 3, 128, 128)
output = model(input)
print(f"关键点热力图形状: {output.shape}")
```

### 练习2：实现关键点检测损失函数

```python
class KeypointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, heatmap_pred, heatmap_target):
        """
        计算关键点检测损失
        
        参数:
            heatmap_pred: 预测热力图 [batch, num_keypoints, H, W]
            heatmap_target: 目标热力图 [batch, num_keypoints, H, W]
        
        返回:
            损失值
        """
        # 使用MSE损失
        loss = self.mse_loss(heatmap_pred, heatmap_target)
        
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, heatmap_pred, heatmap_target):
        """
        计算Focal Loss
        
        参数:
            heatmap_pred: 预测热力图 [batch, num_keypoints, H, W]
            heatmap_target: 目标热力图 [batch, num_keypoints, H, W]
        
        返回:
            损失值
        """
        # Sigmoid激活
        pred = torch.sigmoid(heatmap_pred)
        
        # 正样本损失
        pos_mask = heatmap_target == 1.0
        pos_loss = -(1 - pred[pos_mask]) ** self.alpha * torch.log(pred[pos_mask] + 1e-10)
        
        # 负样本损失
        neg_mask = heatmap_target == 0.0
        neg_loss = -(1 - heatmap_target[neg_mask]) ** self.beta * pred[neg_mask] ** self.alpha * torch.log(1 - pred[neg_mask] + 1e-10)
        
        # 总损失
        loss = (pos_loss.sum() + neg_loss.sum()) / (pos_mask.sum() + neg_mask.sum())
        
        return loss

# 测试
focal_loss = FocalLoss()

heatmap_pred = torch.randn(2, 17, 32, 32)
heatmap_target = torch.zeros(2, 17, 32, 32)

# 设置一些正样本点
heatmap_target[0, 0, 10, 10] = 1.0
heatmap_target[0, 1, 15, 15] = 1.0
heatmap_target[1, 0, 8, 8] = 1.0

loss = focal_loss(heatmap_pred, heatmap_target)
print(f"Focal Loss: {loss.item()}")
```

### 练习3：实现关键点后处理

```python
class KeypointPostProcessor:
    @staticmethod
    def extract_keypoints(heatmap, threshold=0.5):
        """
        从热力图中提取关键点
        
        参数:
            heatmap: 热力图 [num_keypoints, H, W]
            threshold: 置信度阈值
        
        返回:
            关键点坐标列表
        """
        num_keypoints, height, width = heatmap.shape
        
        keypoints = []
        
        for k in range(num_keypoints):
            # 获取当前关键点的热力图
            kp_heatmap = heatmap[k]
            
            # 找到最大值位置
            max_val, max_idx = torch.max(kp_heatmap.flatten(), dim=0)
            
            if max_val > threshold:
                # 转换为坐标
                y = max_idx // width
                x = max_idx % width
                
                keypoints.append({
                    'keypoint_id': k,
                    'x': int(x),
                    'y': int(y),
                    'confidence': float(max_val)
                })
            else:
                keypoints.append({
                    'keypoint_id': k,
                    'x': -1,
                    'y': -1,
                    'confidence': 0.0
                })
        
        return keypoints
    
    @staticmethod
    def nms_keypoints(heatmap, kernel_size=3, threshold=0.5):
        """
        使用NMS提取关键点
        
        参数:
            heatmap: 热力图 [num_keypoints, H, W]
            kernel_size: 最大池化核大小
            threshold: 置信度阈值
        
        返回:
            关键点坐标列表
        """
        num_keypoints, height, width = heatmap.shape
        
        keypoints = []
        
        # 最大池化
        max_pooled = F.max_pool2d(heatmap.unsqueeze(0), kernel_size=kernel_size, padding=kernel_size//2)
        max_pooled = max_pooled.squeeze(0)
        
        # 找到峰值点
        peak_mask = (heatmap == max_pooled) & (heatmap > threshold)
        
        for k in range(num_keypoints):
            peaks = torch.nonzero(peak_mask[k])
            
            if peaks.size(0) > 0:
                # 选择置信度最高的点
                max_idx = torch.argmax(heatmap[k][peak_mask[k]])
                y, x = peaks[max_idx]
                
                keypoints.append({
                    'keypoint_id': k,
                    'x': int(x),
                    'y': int(y),
                    'confidence': float(heatmap[k, y, x])
                })
            else:
                keypoints.append({
                    'keypoint_id': k,
                    'x': -1,
                    'y': -1,
                    'confidence': 0.0
                })
        
        return keypoints

# 测试
processor = KeypointPostProcessor()

heatmap = torch.zeros(5, 32, 32)
heatmap[0, 10, 10] = 0.9
heatmap[0, 11, 11] = 0.8
heatmap[1, 15, 20] = 0.95
heatmap[2, 8, 12] = 0.7
heatmap[3, 20, 25] = 0.85
heatmap[4, 5, 5] = 0.6

keypoints = processor.extract_keypoints(heatmap, threshold=0.7)
print("提取的关键点:")
for kp in keypoints:
    print(f"  关键点{kp['keypoint_id']}: ({kp['x']}, {kp['y']}) - 置信度: {kp['confidence']}")
```

---

## 参考文献

1. Sun, K., et al. (2019). Deep High-Resolution Representation Learning for Visual Recognition.
2. Xiao, B., et al. (2018). Simple Baselines for Human Pose Estimation and Tracking.
3. Zhou, X., et al. (2019). Objects as Points.