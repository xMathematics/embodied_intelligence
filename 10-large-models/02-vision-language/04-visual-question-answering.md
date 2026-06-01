# 2.4 视觉问答

## 目录

- [1. 引言](#1-引言)
- [2. 视觉问答概述](#2-视觉问答概述)
- [3. VQA任务类型](#3-vqa任务类型)
- [4. VQA数据集](#4-vqa数据集)
- [5. VQA模型架构](#5-vqa模型架构)
- [6. 视觉推理](#6-视觉推理)
- [7. VQA评估指标](#7-vqa评估指标)
- [8. 实践练习](#8-实践练习)

---

## 1. 引言

### 1.1 什么是视觉问答

**视觉问答**（Visual Question Answering, VQA）是一种结合计算机视觉和自然语言处理的任务，要求模型根据给定的图像回答问题。

**示例**：
```
图像：一张猫坐在沙发上的照片
问题：猫是什么颜色的？
答案：黑色
```

### 1.2 VQA的重要性

| 方面 | 说明 |
|------|------|
| **多模态理解** | 综合视觉和语言信息 |
| **智能交互** | 实现更自然的人机交互 |
| **实际应用** | 辅助视障人士、教育等 |
| **AI评测** | 衡量模型的综合理解能力 |

---

## 2. 视觉问答概述

### 2.1 任务定义

给定图像I和问题Q，预测答案A：
```
VQA(I, Q) → A
```

### 2.2 VQA的特点

| 特点 | 描述 |
|------|------|
| **多模态** | 涉及视觉和语言两种模态 |
| **开放性** | 答案可能是任意文本 |
| **推理能力** | 需要不同层次的推理 |
| **多样性** | 问题类型多样 |

### 2.3 VQA系统架构

```
┌─────────┐     ┌─────────┐
│  图像   │     │  问题   │
└────┬────┘     └────┬────┘
     │               │
     ▼               ▼
┌─────────┐     ┌─────────┐
│视觉编码器│     │语言编码器│
└────┬────┘     └────┬────┘
     │               │
     └───────┬───────┘
             ▼
    ┌─────────────┐
    │  特征融合   │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │   答案预测  │
    └─────────────┘
```

---

## 3. VQA任务类型

### 3.1 按问题类型分类

| 类型 | 描述 | 示例 |
|------|------|------|
| **事实型** | 关于图像内容的事实 | "图中有几只猫？" |
| **计数型** | 需要计数的问题 | "有多少个物体？" |
| **推理型** | 需要推理的问题 | "这个人在做什么？" |
| **比较型** | 需要比较的问题 | "哪个更大？" |
| **常识型** | 需要常识知识 | "这是什么季节？" |

### 3.2 按答案类型分类

| 类型 | 描述 | 示例 |
|------|------|------|
| **开放式** | 自由文本答案 | "猫在沙发上" |
| **多选式** | 从选项中选择 | A.猫 B.狗 C.鸟 |
| **布尔型** | 是/否回答 | "是"或"否" |

---

## 4. VQA数据集

### 4.1 主要数据集

| 数据集 | 规模 | 特点 |
|--------|------|------|
| **VQA v2** | ~1.1M问题，80K图像 | 平衡的答案分布 |
| **GQA** | ~1.4M问题，117K图像 | 复杂推理问题 |
| **COCO-QA** | ~123K问题，123K图像 | 基于COCO数据集 |
| **Visual7W** | ~47K问题，47K图像 | 基于Flickr30K |
| **CLEVR** | ~100K问题，100K图像 | 合成图像，可控推理 |

### 4.2 VQA v2数据集

**数据集结构**：
- 训练集：80K图像，~0.6M问题
- 验证集：40K图像，~0.3M问题
- 测试集：80K图像，~0.2M问题

**答案收集**：
- 每个问题收集10个人工答案
- 使用多数投票确定最终答案

---

## 5. VQA模型架构

### 5.1 经典模型

| 模型 | 架构特点 |
|------|---------|
| **VQA-Net** | 早期VQA模型，使用CNN+LSTM |
| **ViLBERT** | 双流Transformer，跨模态注意力 |
| **LXMERT** | 大规模预训练，多任务学习 |
| **BUTD** | Bottom-Up Attention，对象级特征 |

### 5.2 Bottom-Up Attention

**核心思想**：使用目标检测模型提取对象级特征

**流程**：
```
图像 → Faster R-CNN → 对象特征（36个对象）
问题 → LSTM/Transformer → 问题特征
对象特征 + 问题特征 → 融合 → 答案预测
```

### 5.3 代码示例：VILT模型

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# 加载预训练模型
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# 加载图像和问题
image = Image.open("example.jpg").convert("RGB")
question = "图中有几只猫？"

# 预处理
encoding = processor(image, question, return_tensors="pt")

# 推理
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
answer = model.config.id2label[idx]

print(f"问题: {question}")
print(f"答案: {answer}")
```

---

## 6. 视觉推理

### 6.1 推理层次

| 层次 | 描述 | 示例 |
|------|------|------|
| **感知推理** | 识别对象和属性 | "这是什么动物？" |
| **关系推理** | 理解对象间关系 | "猫在桌子上还是椅子上？" |
| **空间推理** | 理解空间位置 | "球在盒子里面还是外面？" |
| **时序推理** | 理解时间顺序 | "这个人刚刚做了什么？" |
| **常识推理** | 结合外部知识 | "为什么这个人戴着帽子？" |

### 6.2 推理增强技术

| 技术 | 描述 |
|------|------|
| **思维链** | 生成中间推理步骤 | "首先看到猫，然后看到沙发..." |
| **符号推理** | 使用符号逻辑进行推理 | 解析问题并应用规则 |
| **外部知识** | 结合知识库回答问题 | 查询百科知识 |

### 6.3 复杂推理示例

**问题**："如果把左边的杯子倒过来放在桌子上，会发生什么？"

**推理过程**：
1. 识别左边的杯子里有水
2. 理解"倒过来"的含义
3. 应用物理常识：水会流出
4. 预测结果：杯子倒过来后，水会洒在桌子上

---

## 7. VQA评估指标

### 7.1 标准指标

| 指标 | 计算方式 | 说明 |
|------|---------|------|
| **准确率** | 正确答案比例 | 简单直接 |
| **VQA Accuracy** | 考虑答案多样性 | 使用软匹配 |

### 7.2 VQA准确率计算

**公式**：
```
score = min(1, count / 3)
```
其中count是答案与ground truth的匹配数量（最多10个ground truth答案）

### 7.3 评估挑战

| 挑战 | 描述 |
|------|------|
| **答案多样性** | 同一问题可能有多个正确答案 |
| **歧义问题** | 问题可能有多种解释 |
| **数据集偏差** | 模型可能学习到数据集中的偏差 |
| **推理能力评估** | 评估复杂推理能力困难 |

---

## 8. 实践练习

### 练习1：使用VILT进行VQA

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

# 加载模型
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# 测试图像和问题
image_path = "cat.jpg"
questions = [
    "图中有几只猫？",
    "猫是什么颜色的？",
    "猫在做什么？",
    "背景是什么颜色？"
]

image = Image.open(image_path).convert("RGB")

# 对每个问题进行推理
for question in questions:
    encoding = processor(image, question, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    answer = model.config.id2label[idx]
    print(f"问题: {question}")
    print(f"答案: {answer}")
    print("---")
```

### 练习2：实现简单的VQA模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleVQA(nn.Module):
    def __init__(self, vision_dim=512, text_dim=768, hidden_dim=512, num_answers=3129):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(hidden_dim, num_answers)
    
    def forward(self, vision_feat, text_feat):
        # 投影
        vision_proj = F.relu(self.vision_proj(vision_feat))
        text_proj = F.relu(self.text_proj(text_feat))
        
        # 融合
        fused = torch.cat([vision_proj, text_proj], dim=-1)
        fused = self.fusion(fused)
        
        # 分类
        logits = self.classifier(fused)
        return logits

# 测试模型
model = SimpleVQA()
vision_feat = torch.randn(8, 512)
text_feat = torch.randn(8, 768)
logits = model(vision_feat, text_feat)
print(f"输出形状: {logits.shape}")  # [8, 3129]
```

### 练习3：VQA数据集处理

```python
import json
import os
from PIL import Image

def load_vqa_dataset(data_dir, split="train"):
    """
    加载VQA v2数据集
    
    参数:
        data_dir: 数据集目录
        split: 数据集划分（train/val/test）
    
    返回:
        数据列表，每个元素包含image_id, question, answers
    """
    # 加载问题
    questions_file = os.path.join(data_dir, f"v2_OpenEnded_mscoco_{split}2014_questions.json")
    with open(questions_file, 'r') as f:
        questions_data = json.load(f)
    
    # 加载答案
    answers_file = os.path.join(data_dir, f"v2_mscoco_{split}2014_annotations.json")
    with open(answers_file, 'r') as f:
        answers_data = json.load(f)
    
    # 组织数据
    data = []
    for q, a in zip(questions_data['questions'], answers_data['annotations']):
        assert q['question_id'] == a['question_id']
        
        # 获取答案（取最常见的答案）
        answers = [ans['answer'] for ans in a['answers']]
        answer_counts = {}
        for ans in answers:
            answer_counts[ans] = answer_counts.get(ans, 0) + 1
        most_common_answer = max(answer_counts, key=answer_counts.get)
        
        data.append({
            'image_id': q['image_id'],
            'question': q['question'],
            'answer': most_common_answer,
            'all_answers': answers
        })
    
    return data

# 测试
# dataset = load_vqa_dataset("path/to/vqa_v2")
# print(f"数据集大小: {len(dataset)}")
# print(f"示例: {dataset[0]}")
```

---

**下一节**：[图文生成](05-image-text-generation.md)

---

## 参考文献

1. Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C. L., & Parikh, D. (2015). Vqa: Visual question answering.
2. Lu, J., Batra, D., Parikh, D., & Lee, S. (2019). Vilbert: Pretraining task-agnostic visiolinguistic representations.
3. Tan, H., & Bansal, M. (2019). Lxmert: Learning cross-modality encoder representations.
4. Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., & Zhang, L. (2018). Bottom-up and top-down attention for image captioning and visual question answering.
