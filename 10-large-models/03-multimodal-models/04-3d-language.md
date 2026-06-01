# 3.4 3D-语言模型

## 目录

- [1. 引言](#1-引言)
- [2. 3D-语言学习概述](#2-3d-语言学习概述)
- [3. 3D数据表示](#3-3d数据表示)
- [4. 3D特征提取](#4-3d特征提取)
- [5. 3D-语言模型架构](#5-3d-语言模型架构)
- [6. 代表性模型](#6-代表性模型)
- [7. 3D-语言任务](#7-3d-语言任务)
- [8. 实践练习](#8-实践练习)

---

## 1. 引言

### 1.1 3D-语言模型的重要性

**3D-语言模型**是能够理解3D几何信息并与语言交互的AI模型，在机器人、增强现实、计算机图形学等领域有广泛应用。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **3D物体理解** | 理解3D物体的形状和结构 | 3D物体描述 |
| **3D场景理解** | 理解3D场景布局 | 场景描述 |
| **机器人操控** | 指导机器人操作 | 自然语言指令操控 |
| **增强现实** | 结合真实世界和虚拟内容 | AR交互 |

---

## 2. 3D-语言学习概述

### 2.1 3D数据特点

| 特点 | 描述 |
|------|------|
| **三维结构** | 包含x、y、z坐标 |
| **几何信息** | 形状、体积、表面 |
| **空间关系** | 物体间的空间位置关系 |
| **多种表示** | 点云、网格、体素等 |

### 2.2 3D-语言任务类型

| 任务类型 | 描述 | 示例 |
|---------|------|------|
| **3D物体描述** | 生成3D物体的文字描述 | 描述物体形状 |
| **3D场景描述** | 描述3D场景内容 | 场景布局描述 |
| **3D问答** | 根据3D内容回答问题 | "桌子上有什么？" |
| **3D生成** | 根据文本生成3D物体 | 文本到3D |
| **指令跟随** | 根据指令操作3D场景 | 机器人指令 |

---

## 3. 3D数据表示

### 3.1 点云表示

**定义**：三维空间中的点集合

**特点**：
- 稀疏表示
- 保留精确位置
- 无序性

**代码示例**：
```python
import numpy as np

# 生成简单点云
point_cloud = np.random.rand(1000, 3)  # 1000个点，每个点3个坐标
print(f"点云形状: {point_cloud.shape}")

# 可视化点云
import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
o3d.visualization.draw_geometries([pcd])
```

### 3.2 网格表示

**定义**：由顶点和多边形组成的表面表示

**特点**：
- 连续表面
- 适合渲染
- 拓扑结构复杂

### 3.3 体素表示

**定义**：三维网格中的体素集合

**特点**：
- 规则网格
- 适合卷积操作
- 内存消耗大

### 3.4 神经辐射场（NeRF）

**定义**：使用神经网络表示的连续场景

**特点**：
- 连续表示
- 高质量渲染
- 训练成本高

---

## 4. 3D特征提取

### 4.1 点云特征提取

**PointNet**：
```python
import torch
import torch.nn as nn

class PointNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, 1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, 1)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 1)
    
    def forward(self, x):
        # x: [batch, num_points, input_dim]
        x = x.transpose(1, 2)  # [batch, input_dim, num_points]
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        
        # 全局特征（最大池化）
        global_feat = torch.max(x, dim=2)[0]  # [batch, hidden_dim * 4]
        return global_feat

# 测试
model = PointNet()
point_cloud = torch.randn(8, 1024, 3)  # [batch, num_points, dim]
global_feat = model(point_cloud)
print(f"全局特征形状: {global_feat.shape}")  # [8, 256]
```

### 4.2 3D卷积

**3D CNN**：
```python
class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.fc = nn.Linear(64 * 8 * 8 * 8, 128)
    
    def forward(self, x):
        # x: [batch, 1, depth, height, width]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 测试
model = Simple3DCNN()
voxel = torch.randn(8, 1, 32, 32, 32)  # [batch, channels, depth, height, width]
feat = model(voxel)
print(f"3D CNN特征形状: {feat.shape}")  # [8, 128]
```

---

## 5. 3D-语言模型架构

### 5.1 融合策略

**早期融合**：
```
3D特征 + 文本特征 → 融合 → 任务输出
```

**代码示例**：
```python
class ThreeDLanguageModel(nn.Module):
    def __init__(self, pointnet_dim=256, text_dim=768, hidden_dim=512):
        super().__init__()
        self.pointnet = PointNet()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.point_proj = nn.Linear(pointnet_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, point_cloud, text_feat):
        # 提取3D特征
        point_feat = self.pointnet(point_cloud)
        
        # 投影
        point_proj = torch.tanh(self.point_proj(point_feat))
        text_proj = torch.tanh(self.text_proj(text_feat[:, 0, :]))
        
        # 融合
        fused = torch.cat([point_proj, text_proj], dim=-1)
        fused = torch.tanh(self.fusion(fused))
        
        return fused
```

### 5.2 跨模态注意力

**3D-文本注意力**：
```python
class ThreeDCrossAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads)
    
    def forward(self, point_features, text_features):
        # point_features: [num_points, batch, dim]
        # text_features: [seq_len, batch, dim]
        
        # 文本引导的点云注意力
        output, weights = self.multihead_attn(
            query=text_features,
            key=point_features,
            value=point_features
        )
        return output, weights
```

---

## 6. 代表性模型

### 6.1 PointNet++

**论文**：PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space (Qi et al., 2017)

**核心特点**：
- 层次化点云特征学习
- 局部特征聚合
- 处理非均匀点云

### 6.2 Point-BERT

**论文**：Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling (Li et al., 2022)

**核心特点**：
- Transformer架构
- 掩码点建模预训练
- 自监督学习

### 6.3 Scan2Cap

**论文**：Scan2Cap: Context-aware Dense Captioning in RGB-D Scans (Gupta et al., 2021)

**核心特点**：
- 3D场景描述
- 上下文感知
- RGB-D数据

---

## 7. 3D-语言任务

### 7.1 3D物体描述

**定义**：为3D物体生成文字描述

**代码示例**：
```python
from transformers import BertTokenizer, BertModel
import torch

# 加载BERT模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# 点云描述生成
class PointCloudCaptioner(nn.Module):
    def __init__(self, pointnet_dim=256, text_dim=768, vocab_size=30522):
        super().__init__()
        self.pointnet = PointNet()
        self.decoder = nn.Sequential(
            nn.Linear(pointnet_dim + text_dim, text_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(text_dim, vocab_size)
    
    def forward(self, point_cloud, text_input):
        point_feat = self.pointnet(point_cloud)  # [batch, 256]
        text_emb = bert_model(text_input).last_hidden_state  # [batch, seq_len, 768]
        
        # 将点云特征广播到所有位置
        point_feat_expanded = point_feat.unsqueeze(1).repeat(1, text_emb.shape[1], 1)
        fused = torch.cat([text_emb, point_feat_expanded], dim=-1)
        fused = self.decoder(fused)
        logits = self.classifier(fused)
        
        return logits
```

### 7.2 3D问答

**定义**：根据3D场景回答问题

**流程**：
```
3D场景 → 特征提取 → 问题理解 → 融合 → 答案生成
```

### 7.3 3D指令跟随

**定义**：根据自然语言指令操作3D场景

**示例**：
```
指令："把红色方块移动到桌子上"
动作：移动方块到指定位置
```

---

## 8. 实践练习

### 练习1：点云特征提取

```python
import torch
import torch.nn as nn

class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=512):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, output_dim, 1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # x: [batch, num_points, dim]
        x = x.transpose(1, 2)  # [batch, dim, num_points]
        
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        
        # 最大池化获取全局特征
        global_feat = torch.max(x, dim=2)[0]
        return global_feat

# 测试
encoder = PointNetEncoder()
point_cloud = torch.randn(8, 2048, 3)  # [batch, num_points, dim]
global_feat = encoder(point_cloud)
print(f"点云全局特征形状: {global_feat.shape}")  # [8, 512]
```

### 练习2：3D-文本匹配

```python
import torch
import torch.nn.functional as F

def three_d_text_matching(point_cloud, text_feat, encoder):
    """
    计算3D点云和文本的匹配度
    
    参数:
        point_cloud: [batch, num_points, dim]
        text_feat: [batch, dim]
        encoder: 点云编码器
    
    返回:
        匹配分数
    """
    # 提取点云特征
    point_feat = encoder(point_cloud)  # [batch, output_dim]
    
    # 归一化
    point_norm = F.normalize(point_feat, dim=-1)
    text_norm = F.normalize(text_feat, dim=-1)
    
    # 计算相似度
    similarity = (point_norm * text_norm).sum(dim=-1)
    return similarity

# 测试
encoder = PointNetEncoder()
point_cloud = torch.randn(8, 2048, 3)
text_feat = torch.randn(8, 512)

similarity = three_d_text_matching(point_cloud, text_feat, encoder)
print(f"3D-文本匹配分数: {similarity}")
```

### 练习3：简单3D问答系统

```python
class ThreeDQA(nn.Module):
    def __init__(self, pointnet_dim=512, text_dim=768, hidden_dim=512, num_answers=1000):
        super().__init__()
        self.pointnet = PointNetEncoder(output_dim=pointnet_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.point_proj = nn.Linear(pointnet_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(hidden_dim, num_answers)
    
    def forward(self, point_cloud, question_feat):
        # 提取点云特征
        point_feat = self.pointnet(point_cloud)
        point_proj = F.relu(self.point_proj(point_feat))
        
        # 问题特征（假设已经过BERT编码）
        question_proj = F.relu(self.text_proj(question_feat))
        
        # 融合
        fused = torch.cat([point_proj, question_proj], dim=-1)
        fused = self.fusion(fused)
        
        # 分类
        logits = self.classifier(fused)
        return logits

# 测试
model = ThreeDQA()
point_cloud = torch.randn(8, 2048, 3)   # [batch, num_points, dim]
question_feat = torch.randn(8, 768)     # [batch, text_dim]

logits = model(point_cloud, question_feat)
print(f"输出形状: {logits.shape}")  # [8, 1000]
```

---

**下一节**：[通用多模态模型](05-universal-multimodal.md)

---

## 参考文献

1. Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). PointNet++: Deep hierarchical feature learning on point sets in a metric space.
2. Li, B., Chen, L., Chen, Q., & Lin, Y. (2022). Point-BERT: Pre-training 3D point cloud transformers with masked point modeling.
3. Gupta, S., Malik, A., & Ramanan, D. (2021). Scan2Cap: Context-aware dense captioning in RGB-D scans.
