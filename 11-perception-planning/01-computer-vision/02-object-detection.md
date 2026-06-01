# 1.2 目标检测

## 目录

- [1. 引言](#1-引言)
- [2. 目标检测概述](#2-目标检测概述)
- [3. 两阶段检测器](#3-两阶段检测器)
- [4. 一阶段检测器](#4-一阶段检测器)
- [5. Transformer检测器](#5-transformer检测器)
- [6. 实践练习](#6-实践练习)

---

## 1. 引言

### 1.1 目标检测的重要性

**目标检测**不仅要识别图像中的物体类别，还要定位物体的位置。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **自动驾驶** | 检测车辆、行人、交通标志 | 障碍物检测 |
| **安防监控** | 检测异常行为和可疑物体 | 入侵检测 |
| **工业质检** | 检测产品缺陷 | 零件检测 |
| **医学影像** | 检测病变区域 | 肿瘤检测 |

---

## 2. 目标检测概述

### 2.1 定义

**目标检测**：识别图像中多个物体的类别和位置。

**输出格式**：
```
[
    {
        "class": "car",
        "bbox": [x_min, y_min, x_max, y_max],
        "confidence": 0.95
    },
    ...
]
```

### 2.2 评价指标

| 指标 | 描述 | 计算公式 |
|------|------|---------|
| **IoU** | 交并比 | IoU = Area(∩) / Area(∪) |
| **mAP** | 平均精度均值 | 各类别AP的平均值 |
| **Recall** | 召回率 | TP / (TP + FN) |
| **FPS** | 每秒帧数 | 衡量实时性 |

---

## 3. 两阶段检测器

### 3.1 Faster R-CNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FasterRCNN(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        
        # 骨干网络
        self.backbone = self._build_backbone()
        
        # RPN
        self.rpn = RegionProposalNetwork()
        
        # 头部
        self.head = RCNNHead(num_classes)
    
    def _build_backbone(self):
        """构建骨干网络"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        features = self.backbone(x)
        
        # RPN生成候选框
        proposals, rpn_loss = self.rpn(features)
        
        # R-CNN头部
        detections, roi_loss = self.head(features, proposals)
        
        return detections, rpn_loss + roi_loss

class RegionProposalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 3x3卷积
        self.conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # 分类分支（前景/背景）
        self.cls_conv = nn.Conv2d(512, 18, kernel_size=1)
        
        # 回归分支（边界框偏移）
        self.reg_conv = nn.Conv2d(512, 36, kernel_size=1)
    
    def forward(self, features):
        """前向传播"""
        x = F.relu(self.conv(features))
        
        # 分类和回归
        cls_logits = self.cls_conv(x)  # [batch, 18, H, W]
        reg_deltas = self.reg_conv(x)  # [batch, 36, H, W]
        
        # 生成候选框
        proposals = self._generate_proposals(cls_logits, reg_deltas)
        
        # 计算损失（简化）
        loss = torch.tensor(0.0)
        
        return proposals, loss
    
    def _generate_proposals(self, cls_logits, reg_deltas):
        """生成候选框（简化实现）"""
        batch_size = cls_logits.size(0)
        num_proposals = 256
        
        # 返回模拟的候选框 [batch, num_proposals, 4]
        return torch.randn(batch_size, num_proposals, 4)

class RCNNHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout()
        )
        
        # 分类头
        self.cls_score = nn.Linear(4096, num_classes)
        
        # 回归头
        self.bbox_pred = nn.Linear(4096, num_classes * 4)
    
    def forward(self, features, proposals):
        """前向传播"""
        # RoI池化（简化）
        batch_size = features.size(0)
        num_proposals = proposals.size(1)
        
        # 模拟RoI特征
        roi_features = torch.randn(batch_size, num_proposals, 256, 7, 7)
        roi_features = roi_features.view(batch_size * num_proposals, 256 * 7 * 7)
        
        # 全连接层
        x = self.fc_layers(roi_features)
        
        # 分类和回归
        cls_logits = self.cls_score(x)  # [batch*num_proposals, num_classes]
        bbox_deltas = self.bbox_pred(x)  # [batch*num_proposals, num_classes*4]
        
        # 计算损失（简化）
        loss = torch.tensor(0.0)
        
        return {'cls_logits': cls_logits, 'bbox_deltas': bbox_deltas}, loss

# 测试
model = FasterRCNN(num_classes=20)
input = torch.randn(1, 3, 600, 800)
detections, loss = model(input)
print(f"检测结果: {detections.keys()}")
```

---

## 4. 一阶段检测器

### 4.1 YOLO系列

```python
class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        
        # Darknet-53骨干网络
        self.backbone = Darknet53()
        
        # 检测头
        self.detection_heads = nn.ModuleList([
            DetectionHead(1024, num_classes),
            DetectionHead(512, num_classes),
            DetectionHead(256, num_classes)
        ])
    
    def forward(self, x):
        """前向传播"""
        # 获取多尺度特征
        features = self.backbone(x)
        
        # 多尺度检测
        outputs = []
        for i, (feature, head) in enumerate(zip(features, self.detection_heads)):
            output = head(feature)
            outputs.append(output)
        
        return outputs

class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            # 输入层
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            
            # 残差块
            self._make_residual_block(32, 64),
            self._make_residual_block(64, 128),
            self._make_residual_block(128, 256),
            self._make_residual_block(256, 512),
            self._make_residual_block(512, 1024)
        )
    
    def _make_residual_block(self, in_channels, out_channels):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        """前向传播"""
        # 返回三个尺度的特征
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in [3, 4, 5]:  # 特定层输出
                features.append(x)
        
        return features[::-1]  # 从大到小

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels * 2, (num_classes + 5) * 3, kernel_size=1)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """前向传播"""
        output = self.conv_layers(x)
        
        # 调整形状: [batch, 3*(num_classes+5), H, W] -> [batch, 3, H, W, num_classes+5]
        batch_size = output.size(0)
        grid_size = output.size(2)
        
        output = output.view(batch_size, 3, self.num_classes + 5, grid_size, grid_size)
        output = output.permute(0, 1, 3, 4, 2)
        
        return output

# 测试
model = YOLOv3(num_classes=80)
input = torch.randn(1, 3, 416, 416)
outputs = model(input)
print(f"YOLOv3输出尺度: {[o.shape for o in outputs]}")
```

### 4.2 SSD

```python
class SSD(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        
        # VGG16骨干网络
        self.backbone = VGG16Backbone()
        
        # 额外特征层
        self.extra_layers = self._build_extra_layers()
        
        # 检测头
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        
        # 不同尺度的特征图
        feature_map_channels = [512, 1024, 512, 256, 256, 256]
        num_priors = [4, 6, 6, 6, 4, 4]
        
        for i, (channels, priors) in enumerate(zip(feature_map_channels, num_priors)):
            self.loc_layers.append(nn.Conv2d(channels, priors * 4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(channels, priors * num_classes, kernel_size=3, padding=1))
    
    def _build_extra_layers(self):
        """构建额外特征层"""
        layers = []
        in_channels = 1024
        
        for _ in range(4):
            layers.append(nn.Conv2d(in_channels, in_channels // 2, kernel_size=1))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_channels //= 2
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        features = []
        
        # 骨干网络特征
        x = self.backbone(x)
        features.append(x)
        
        # 额外特征层
        for layer in self.extra_layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        
        # 检测
        locs = []
        confs = []
        
        for i, (feature, loc_layer, conf_layer) in enumerate(zip(features, self.loc_layers, self.conf_layers)):
            loc = loc_layer(feature)
            conf = conf_layer(feature)
            
            # 展平
            batch_size = loc.size(0)
            loc = loc.permute(0, 2, 3, 1).contiguous()
            loc = loc.view(batch_size, -1, 4)
            
            conf = conf.permute(0, 2, 3, 1).contiguous()
            conf = conf.view(batch_size, -1, self.conf_layers[0].out_channels // 4)
            
            locs.append(loc)
            confs.append(conf)
        
        return torch.cat(locs, dim=1), torch.cat(confs, dim=1)

class VGG16Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
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
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.features(x)

# 测试
model = SSD(num_classes=21)
input = torch.randn(1, 3, 300, 300)
locs, confs = model(input)
print(f"位置预测: {locs.shape}")
print(f"置信度预测: {confs.shape}")
```

---

## 5. Transformer检测器

### 5.1 DETR

```python
class DETR(nn.Module):
    def __init__(self, num_classes=91, num_queries=100):
        super().__init__()
        
        # 骨干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 位置编码
        self.position_embedding = PositionEmbeddingSine()
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )
        
        # 查询向量
        self.query_embeddings = nn.Embedding(num_queries, 512)
        
        # 预测头
        self.class_head = nn.Linear(512, num_classes + 1)  # +1 for no object
        self.bbox_head = MLP(512, 512, 4, 3)
    
    def forward(self, x):
        """前向传播"""
        # 特征提取
        features = self.backbone(x)
        batch_size, channels, height, width = features.shape
        
        # 展平特征
        features = features.flatten(2).permute(2, 0, 1)  # [H*W, batch, channels]
        
        # 添加位置编码
        pos_embed = self.position_embedding(features, height, width)
        features = features + pos_embed
        
        # Transformer编码
        memory = self.transformer.encoder(features)
        
        # 查询向量
        queries = self.query_embeddings.weight.unsqueeze(1).repeat(1, batch_size, 1)
        
        # Transformer解码
        hs = self.transformer.decoder(queries, memory)
        
        # 预测
        logits = self.class_head(hs)
        bbox = self.bbox_head(hs).sigmoid()
        
        return {
            'pred_logits': logits.transpose(0, 1),
            'pred_boxes': bbox.transpose(0, 1)
        }

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
    
    def forward(self, features, height, width):
        """计算位置编码"""
        y_embed = torch.arange(height, dtype=torch.float32)
        x_embed = torch.arange(width, dtype=torch.float32)
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=1).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=1).flatten(1)
        
        pos = torch.cat((pos_y, pos_x), dim=1).unsqueeze(1)
        pos = pos.repeat(1, height, 1).flatten(0, 1).unsqueeze(1)
        
        return pos

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i != num_layers - 1:
                self.layers.append(nn.ReLU())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 测试
model = DETR(num_classes=91)
input = torch.randn(2, 3, 256, 256)
outputs = model(input)
print(f"预测logits: {outputs['pred_logits'].shape}")
print(f"预测框: {outputs['pred_boxes'].shape}")
```

---

## 6. 实践练习

### 练习1：实现简单的目标检测器

```python
class SimpleObjectDetector(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        
        # 骨干网络
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 检测头
        self.cls_head = nn.Conv2d(128, num_classes, kernel_size=1)
        self.bbox_head = nn.Conv2d(128, 4, kernel_size=1)
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        
        # 预测
        cls_logits = self.cls_head(features)
        bbox_pred = self.bbox_head(features)
        
        return {
            'cls_logits': cls_logits,
            'bbox_pred': bbox_pred
        }

# 测试
detector = SimpleObjectDetector(num_classes=20)
input = torch.randn(1, 3, 224, 224)
output = detector(input)
print(f"分类logits: {output['cls_logits'].shape}")
print(f"边界框预测: {output['bbox_pred'].shape}")
```

### 练习2：实现NMS算法

```python
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    非极大值抑制
    
    参数:
        boxes: 边界框列表 [N, 4]
        scores: 置信度列表 [N]
        iou_threshold: IoU阈值
    
    返回:
        保留的索引列表
    """
    if len(boxes) == 0:
        return []
    
    # 将boxes转为numpy数组
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 获取边界框坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # 计算面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # 按置信度排序
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        # 选择置信度最高的
        i = order[0]
        keep.append(i)
        
        # 计算与其他框的IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留IoU小于阈值的
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

# 测试
boxes = [
    [10, 10, 100, 100],
    [15, 15, 105, 105],
    [200, 200, 300, 300],
    [50, 50, 150, 150]
]

scores = [0.9, 0.85, 0.95, 0.7]

keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
print(f"保留的索引: {keep}")
```

### 练习3：实现IoU计算

```python
class IoUCalculator:
    @staticmethod
    def calculate_iou(box1, box2):
        """
        计算IoU
        
        参数:
            box1: 边界框1 [x1, y1, x2, y2]
            box2: 边界框2 [x1, y1, x2, y2]
        
        返回:
            IoU值
        """
        # 计算交集
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 交集面积
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 并集面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    @staticmethod
    def calculate_ciou(box1, box2):
        """计算CIoU"""
        iou = IoUCalculator.calculate_iou(box1, box2)
        
        # 中心距离
        c_x1 = (box1[0] + box1[2]) / 2
        c_y1 = (box1[1] + box1[3]) / 2
        c_x2 = (box2[0] + box2[2]) / 2
        c_y2 = (box2[1] + box2[3]) / 2
        
        dist = ((c_x1 - c_x2) ** 2 + (c_y1 - c_y2) ** 2) ** 0.5
        
        # 最小外接矩形对角线
        w = max(box1[2], box2[2]) - min(box1[0], box2[0])
        h = max(box1[3], box2[3]) - min(box1[1], box2[1])
        diag = (w ** 2 + h ** 2) ** 0.5
        
        # 宽高比一致性
        v = (4 / (np.pi ** 2)) * (np.arctan((box1[2]-box1[0])/(box1[3]-box1[1]+1e-10)) - 
                                   np.arctan((box2[2]-box2[0])/(box2[3]-box2[1]+1e-10))) ** 2
        
        alpha = v / (1 - iou + v)
        
        ciou = iou - (dist ** 2) / (diag ** 2) - alpha * v
        
        return ciou

# 测试
calculator = IoUCalculator()

box1 = [10, 10, 50, 50]
box2 = [20, 20, 60, 60]

iou = calculator.calculate_iou(box1, box2)
ciou = calculator.calculate_ciou(box1, box2)

print(f"IoU: {iou:.4f}")
print(f"CIoU: {ciou:.4f}")
```

---

**下一节**：[语义分割](03-semantic-segmentation.md)

---

## 参考文献

1. Girshick, R., et al. (2014). Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation.
2. Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection.
3. Carion, N., et al. (2020). End-to-End Object Detection with Transformers.