# 1.4 实例分割

## 目录

- [1. 引言](#1-引言)
- [2. 实例分割概述](#2-实例分割概述)
- [3. Mask R-CNN](#3-mask-r-cnn)
- [4. 其他方法](#4-其他方法)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 实例分割的重要性

**实例分割**不仅要分割图像，还要区分同一类别的不同实例。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **自动驾驶** | 识别不同车辆和行人 | 多目标跟踪 |
| **医学影像** | 分割多个病变区域 | 细胞计数 |
| **机器人抓取** | 识别不同物体实例 | 物体定位 |
| **视频分析** | 跟踪多个目标 | 行为识别 |

---

## 2. 实例分割概述

### 2.1 定义

**实例分割**：对图像中的每个物体实例进行像素级分割。

**区别于语义分割**：语义分割只关心类别，实例分割还要区分同一类别的不同个体。

### 2.2 输出格式

```
[
    {
        "class": "car",
        "instance_id": 1,
        "mask": [H, W]  # 二值掩码
    },
    ...
]
```

---

## 3. Mask R-CNN

### 3.1 整体架构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=91):
        super().__init__()
        
        # 骨干网络
        self.backbone = self._build_backbone()
        
        # RPN
        self.rpn = RegionProposalNetwork()
        
        # ROI Align
        self.roi_align = ROIAlign()
        
        # 头部
        self.head = MaskRCNNHead(num_classes)
    
    def _build_backbone(self):
        """构建骨干网络"""
        return nn.Sequential(
            # 简化的ResNet结构
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        features = self.backbone(x)
        
        # RPN生成候选框
        proposals, _ = self.rpn(features)
        
        # ROI Align
        roi_features = self.roi_align(features, proposals)
        
        # 头部预测
        outputs = self.head(roi_features)
        
        return outputs

class ROIAlign(nn.Module):
    def __init__(self, output_size=(7, 7)):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, features, proposals):
        """
        ROI Align
        
        参数:
            features: 特征图 [batch, channels, H, W]
            proposals: 候选框 [batch, num_proposals, 4]
        
        返回:
            ROI特征 [batch * num_proposals, channels, output_size, output_size]
        """
        batch_size = features.size(0)
        num_proposals = proposals.size(1)
        
        # 简化实现：采样固定大小
        roi_features = []
        
        for i in range(batch_size):
            for j in range(num_proposals):
                # 获取候选框
                x1, y1, x2, y2 = proposals[i, j]
                
                # 计算区域
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 裁剪并resize
                if x2 > x1 and y2 > y1:
                    roi = features[i, :, y1:y2, x1:x2]
                    roi = F.interpolate(roi.unsqueeze(0), size=self.output_size, mode='bilinear')
                else:
                    roi = torch.zeros(1, features.size(1), *self.output_size)
                
                roi_features.append(roi)
        
        return torch.cat(roi_features, dim=0)

class MaskRCNNHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # 共享全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU()
        )
        
        # 分类头
        self.cls_head = nn.Linear(1024, num_classes)
        
        # 边界框回归头
        self.bbox_head = nn.Linear(1024, num_classes * 4)
        
        # 掩码预测头
        self.mask_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, roi_features):
        """前向传播"""
        batch_size = roi_features.size(0)
        
        # 分类和边界框回归
        flattened = roi_features.view(batch_size, -1)
        fc_out = self.fc_layers(flattened)
        
        cls_logits = self.cls_head(fc_out)
        bbox_deltas = self.bbox_head(fc_out)
        
        # 掩码预测
        mask_logits = self.mask_head(roi_features)
        
        return {
            'cls_logits': cls_logits,
            'bbox_deltas': bbox_deltas,
            'mask_logits': mask_logits
        }

# 测试
model = MaskRCNN(num_classes=91)
input = torch.randn(1, 3, 600, 800)
outputs = model(input)
print(f"分类logits: {outputs['cls_logits'].shape}")
print(f"边界框回归: {outputs['bbox_deltas'].shape}")
print(f"掩码logits: {outputs['mask_logits'].shape}")
```

---

## 4. 其他方法

### 4.1 YOLACT

```python
class YOLACT(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        
        # 骨干网络
        self.backbone = self._build_backbone()
        
        # 特征金字塔
        self.fpn = FeaturePyramidNetwork()
        
        # 预测头
        self.prediction_head = YOLACTPredictionHead(num_classes)
        
        # 原型掩码
        self.proto_net = ProtoNet()
    
    def _build_backbone(self):
        """构建骨干网络"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
    
    def forward(self, x):
        """前向传播"""
        # 骨干网络
        features = self.backbone(x)
        
        # 特征金字塔
        pyramid_features = self.fpn(features)
        
        # 预测
        predictions = self.prediction_head(pyramid_features)
        
        # 原型掩码
        proto_masks = self.proto_net(pyramid_features[-1])
        
        return {
            'predictions': predictions,
            'proto_masks': proto_masks
        }

class FeaturePyramidNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(128, 256, kernel_size=1)
        ])
    
    def forward(self, features):
        """前向传播"""
        pyramid = []
        
        # 顶层
        p5 = self.lateral_convs[0](features)
        pyramid.append(p5)
        
        # 上采样融合
        p4 = F.interpolate(p5, scale_factor=2) + self.lateral_convs[1](features)
        pyramid.append(p4)
        
        return pyramid[::-1]

class YOLACTPredictionHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.num_classes = num_classes
        
        # 分类头
        self.cls_conv = nn.Conv2d(256, num_classes * 3, kernel_size=3, padding=1)
        
        # 边界框回归头
        self.bbox_conv = nn.Conv2d(256, 4 * 3, kernel_size=3, padding=1)
        
        # 掩码系数头
        self.mask_conv = nn.Conv2d(256, 32 * 3, kernel_size=3, padding=1)
    
    def forward(self, features):
        """前向传播"""
        cls_logits = []
        bbox_deltas = []
        mask_coeffs = []
        
        for feat in features:
            cls_logits.append(self.cls_conv(feat))
            bbox_deltas.append(self.bbox_conv(feat))
            mask_coeffs.append(self.mask_conv(feat))
        
        return {
            'cls_logits': cls_logits,
            'bbox_deltas': bbox_deltas,
            'mask_coeffs': mask_coeffs
        }

class ProtoNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 32, kernel_size=1)
        )
    
    def forward(self, x):
        """前向传播"""
        return self.convs(x)

# 测试
model = YOLACT(num_classes=80)
input = torch.randn(1, 3, 550, 550)
outputs = model(input)
print(f"原型掩码形状: {outputs['proto_masks'].shape}")
```

### 4.2 SOLO

```python
class SOLO(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        
        # 骨干网络
        self.backbone = self._build_backbone()
        
        # 特征金字塔
        self.fpn = FeaturePyramidNetwork()
        
        # 分类分支
        self.cate_head = SOLOCategoryHead(num_classes)
        
        # 掩码分支
        self.mask_head = SOLOSegmentationHead()
    
    def _build_backbone(self):
        """构建骨干网络"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        """前向传播"""
        # 骨干网络
        features = self.backbone(x)
        
        # 特征金字塔
        pyramid_features = self.fpn(features)
        
        # 分类预测
        cate_preds = self.cate_head(pyramid_features)
        
        # 掩码预测
        mask_preds = self.mask_head(pyramid_features)
        
        return {
            'cate_preds': cate_preds,
            'mask_preds': mask_preds
        }

class SOLOCategoryHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for _ in range(4):
            self.convs.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)
    
    def forward(self, features):
        """前向传播"""
        preds = []
        
        for feat in features:
            x = feat
            for conv in self.convs:
                x = F.relu(conv(x))
            pred = self.final_conv(x)
            preds.append(pred)
        
        return preds

class SOLOSegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convs = nn.ModuleList()
        for _ in range(4):
            self.convs.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        
        self.final_conv = nn.Conv2d(256, 256, kernel_size=1)
    
    def forward(self, features):
        """前向传播"""
        preds = []
        
        for feat in features:
            x = feat
            for conv in self.convs:
                x = F.relu(conv(x))
            pred = self.final_conv(x)
            preds.append(pred)
        
        return preds

# 测试
model = SOLO(num_classes=80)
input = torch.randn(1, 3, 512, 512)
outputs = model(input)
print(f"分类预测数量: {len(outputs['cate_preds'])}")
print(f"掩码预测形状: {outputs['mask_preds'][0].shape}")
```

---

## 5. 实践练习

### 练习1：实现简单的实例分割网络

```python
class SimpleInstanceSegmenter(nn.Module):
    def __init__(self, num_classes=20):
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
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        
        # 分类头
        self.cls_head = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # 掩码头
        self.mask_head = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        x = self.encoder(x)
        x = self.decoder(x)
        
        cls_logits = self.cls_head(x)
        mask_logits = self.mask_head(x)
        
        return {
            'cls_logits': cls_logits,
            'mask_logits': mask_logits
        }

# 测试
model = SimpleInstanceSegmenter(num_classes=20)
input = torch.randn(1, 3, 128, 128)
output = model(input)
print(f"分类输出: {output['cls_logits'].shape}")
print(f"掩码输出: {output['mask_logits'].shape}")
```

### 练习2：实现实例分割损失函数

```python
class InstanceSegmentationLoss(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        
        self.num_classes = num_classes
        self.cls_loss = nn.CrossEntropyLoss()
        self.mask_loss = nn.BCELoss()
    
    def forward(self, outputs, targets):
        """
        计算损失
        
        参数:
            outputs: 模型输出 {'cls_logits', 'mask_logits'}
            targets: 目标 {'labels', 'masks'}
        
        返回:
            总损失
        """
        # 分类损失
        cls_logits = outputs['cls_logits']
        labels = targets['labels']
        
        # 展平
        cls_logits_flat = cls_logits.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        labels_flat = labels.reshape(-1)
        
        cls_loss = self.cls_loss(cls_logits_flat, labels_flat)
        
        # 掩码损失
        mask_logits = outputs['mask_logits']
        masks = targets['masks']
        
        # Sigmoid激活
        mask_probs = torch.sigmoid(mask_logits)
        
        # 只计算正样本的掩码损失
        pos_mask = (labels > 0).unsqueeze(1).expand_as(mask_probs)
        mask_loss = self.mask_loss(mask_probs[pos_mask], masks[pos_mask].float())
        
        # 总损失
        total_loss = cls_loss + mask_loss
        
        return total_loss

# 测试
loss_fn = InstanceSegmentationLoss(num_classes=20)

outputs = {
    'cls_logits': torch.randn(2, 20, 32, 32),
    'mask_logits': torch.randn(2, 20, 32, 32)
}

targets = {
    'labels': torch.randint(0, 20, (2, 32, 32)),
    'masks': torch.randint(0, 2, (2, 20, 32, 32))
}

loss = loss_fn(outputs, targets)
print(f"实例分割损失: {loss.item()}")
```

### 练习3：实现实例掩码后处理

```python
class InstanceMaskProcessor:
    @staticmethod
    def generate_masks(cls_logits, mask_logits, threshold=0.5):
        """
        生成实例掩码
        
        参数:
            cls_logits: 分类logits [batch, num_classes, H, W]
            mask_logits: 掩码logits [batch, num_classes, H, W]
            threshold: 置信度阈值
        
        返回:
            实例掩码列表
        """
        batch_size, num_classes, height, width = cls_logits.shape
        
        results = []
        
        for i in range(batch_size):
            image_results = []
            
            # 获取类别预测
            cls_probs = F.softmax(cls_logits[i], dim=0)
            
            # 获取掩码概率
            mask_probs = torch.sigmoid(mask_logits[i])
            
            # 找到所有检测到的实例
            for c in range(1, num_classes):  # 跳过背景类
                # 检查是否有足够置信度的区域
                max_prob = cls_probs[c].max()
                
                if max_prob > threshold:
                    # 获取掩码
                    mask = (mask_probs[c] > threshold).float()
                    
                    # 寻找连通区域作为不同实例
                    instances = InstanceMaskProcessor._find_instances(mask)
                    
                    for j, instance_mask in enumerate(instances):
                        image_results.append({
                            'class_id': c,
                            'instance_id': j,
                            'mask': instance_mask,
                            'confidence': max_prob.item()
                        })
            
            results.append(image_results)
        
        return results
    
    @staticmethod
    def _find_instances(mask):
        """
        寻找连通区域
        
        参数:
            mask: 二值掩码
        
        返回:
            实例掩码列表
        """
        # 简化实现：返回原始掩码作为单个实例
        return [mask]

# 测试
processor = InstanceMaskProcessor()

cls_logits = torch.randn(1, 3, 32, 32)
mask_logits = torch.randn(1, 3, 32, 32)

results = processor.generate_masks(cls_logits, mask_logits)
print(f"检测到的实例数量: {len(results[0])}")
```

---

**下一节**：[关键点检测](05-keypoint-detection.md)

---

## 参考文献

1. He, K., et al. (2017). Mask R-CNN.
2. Bolya, D., et al. (2019). YOLACT: Real-Time Instance Segmentation.
3. Wang, X., et al. (2020). SOLO: Segmenting Objects by Locations.