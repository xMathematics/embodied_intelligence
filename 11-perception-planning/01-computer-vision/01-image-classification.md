# 1.1 图像分类

## 目录

- [1. 引言](#1-引言)
- [2. 图像分类概述](#2-图像分类概述)
- [3. 传统方法](#3-传统方法)
- [4. 深度学习方法](#4-深度学习方法)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 图像分类的重要性

**图像分类**是计算机视觉的基础任务，其目标是将图像分配到预定义的类别中。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **安防监控** | 识别可疑行为和物体 | 人脸识别、异常检测 |
| **自动驾驶** | 识别道路、车辆、行人 | 交通标志识别 |
| **医疗诊断** | 医学影像分析 | 癌症筛查、病变检测 |
| **工业质检** | 产品质量检测 | 缺陷识别、分类 |

---

## 2. 图像分类概述

### 2.1 定义

**图像分类**：给定一幅图像，预测其所属的类别标签。

**形式化表达**：
```
y = f(x; θ)
```
其中 `x` 是输入图像，`θ` 是模型参数，`y` 是类别标签。

### 2.2 分类任务类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **二分类** | 两类区分 | 猫/狗、有病/无病 |
| **多分类** | 多个类别 | 10类动物、1000类ImageNet |
| **多标签分类** | 多个标签 | 图像中包含多种物体 |
| **层次分类** | 层次化类别 | 动物→哺乳动物→狗→金毛 |

---

## 3. 传统方法

### 3.1 特征提取

```python
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class TraditionalImageClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = SVC()
    
    def extract_hog_features(self, image):
        """
        提取HOG特征
        
        参数:
            image: 输入图像
        
        返回:
            HOG特征向量
        """
        # 转换为灰度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算HOG特征
        win_size = (64, 64)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(gray)
        
        return features.flatten()
    
    def extract_sift_features(self, image):
        """
        提取SIFT特征
        
        参数:
            image: 输入图像
        
        返回:
            SIFT特征描述符
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            return np.zeros(128)
        
        # 取前128维
        return descriptors.flatten()[:128]
    
    def train(self, images, labels):
        """
        训练分类器
        
        参数:
            images: 训练图像列表
            labels: 标签列表
        """
        features = []
        for image in images:
            hog = self.extract_hog_features(image)
            sift = self.extract_sift_features(image)
            combined = np.concatenate([hog, sift])
            features.append(combined)
        
        features = np.array(features)
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        
        self.classifier.fit(scaled_features, labels)
    
    def predict(self, image):
        """
        预测类别
        
        参数:
            image: 输入图像
        
        返回:
            预测标签
        """
        hog = self.extract_hog_features(image)
        sift = self.extract_sift_features(image)
        features = np.concatenate([hog, sift])
        
        scaled = self.scaler.transform([features])
        return self.classifier.predict(scaled)[0]

# 测试（需要实际图像数据）
# classifier = TraditionalImageClassifier()
# classifier.train(train_images, train_labels)
# prediction = classifier.predict(test_image)
```

### 3.2 机器学习分类器

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

class ClassifierEnsemble:
    def __init__(self):
        self.classifiers = {
            'svm': SVC(probability=True),
            'random_forest': RandomForestClassifier(),
            'naive_bayes': GaussianNB(),
            'knn': KNeighborsClassifier()
        }
    
    def train(self, features, labels):
        """训练所有分类器"""
        for name, clf in self.classifiers.items():
            clf.fit(features, labels)
            print(f"训练完成: {name}")
    
    def predict(self, features):
        """集成预测"""
        predictions = []
        for name, clf in self.classifiers.items():
            pred = clf.predict([features])[0]
            predictions.append(pred)
        
        # 投票
        from collections import Counter
        vote = Counter(predictions)
        return vote.most_common(1)[0][0]
    
    def predict_proba(self, features):
        """预测概率"""
        probas = []
        for name, clf in self.classifiers.items():
            prob = clf.predict_proba([features])[0]
            probas.append(prob)
        
        # 平均概率
        avg_proba = np.mean(probas, axis=0)
        return avg_proba
```

---

## 4. 深度学习方法

### 4.1 CNN分类器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 第三层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入图像 [batch, 3, 32, 32]
        
        返回:
            类别概率
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)

# 测试
model = SimpleCNN(num_classes=10)
input = torch.randn(32, 3, 32, 32)
output = model(input)
print(f"输出形状: {output.shape}")
print(f"类别数量: {output.size(1)}")
```

### 4.2 经典CNN架构

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# 测试
alexnet = AlexNet()
input = torch.randn(1, 3, 224, 224)
output = alexnet(input)
print(f"AlexNet输出形状: {output.shape}")
```

### 4.3 迁移学习

```python
from torchvision import models

class TransferLearningClassifier(nn.Module):
    def __init__(self, num_classes=10, backbone='resnet18'):
        super().__init__()
        
        # 加载预训练模型
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            num_features = self.backbone.fc.in_features
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            num_features = self.backbone.fc.in_features
        elif backbone == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 冻结骨干网络
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 替换分类头
        if backbone == 'efficientnet':
            self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
        else:
            self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_layers(self, num_layers=2):
        """解冻最后几层"""
        layers = list(self.backbone.children())[-num_layers:]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True

# 测试
model = TransferLearningClassifier(num_classes=10, backbone='resnet18')
input = torch.randn(32, 3, 224, 224)
output = model(input)
print(f"迁移学习模型输出: {output.shape}")
```

### 4.4 数据增强

```python
from torchvision import transforms

class DataAugmentation:
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def apply_train(self, image):
        """应用训练时的增强"""
        return self.train_transform(image)
    
    def apply_val(self, image):
        """应用验证时的变换"""
        return self.val_transform(image)

# 测试
# augment = DataAugmentation()
# image = Image.open("test.jpg")
# augmented = augment.apply_train(image)
# print(f"增强后形状: {augmented.shape}")
```

---

## 5. 实践练习

### 练习1：实现CNN图像分类器

```python
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv_blocks = nn.ModuleList([
            # Block 1
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            # Block 2
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            # Block 3
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ),
            # Block 4
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """前向传播"""
        for block in self.conv_blocks:
            x = block(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# 测试
model = CustomCNN(num_classes=10)
input = torch.randn(16, 3, 224, 224)
output = model(input)
print(f"输出形状: {output.shape}")
print(f"参数数量: {sum(p.numel() for p in model.parameters())}")
```

### 练习2：实现训练循环

```python
class ImageClassificationTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.001):
        """
        训练模型
        
        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += torch.sum(preds == labels.data)
            
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct.double() / len(train_loader.dataset)
            
            # 验证阶段
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n")
    
    def evaluate(self, loader, criterion):
        """评估模型"""
        self.model.eval()
        loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss += criterion(outputs, labels).item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
        
        loss = loss / len(loader.dataset)
        acc = correct.double() / len(loader.dataset)
        
        return loss, acc

# 测试（需要实际数据加载器）
# trainer = ImageClassificationTrainer(model)
# trainer.train(train_loader, val_loader, epochs=10)
```

### 练习3：实现混淆矩阵分析

```python
class ClassificationAnalyzer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def compute_confusion_matrix(self, dataloader, num_classes):
        """
        计算混淆矩阵
        
        参数:
            dataloader: 数据加载器
            num_classes: 类别数量
        
        返回:
            混淆矩阵
        """
        self.model.eval()
        conf_matrix = torch.zeros(num_classes, num_classes)
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    conf_matrix[t.long(), p.long()] += 1
        
        return conf_matrix
    
    def compute_metrics(self, conf_matrix):
        """
        计算评估指标
        
        参数:
            conf_matrix: 混淆矩阵
        
        返回:
            指标字典
        """
        num_classes = conf_matrix.size(0)
        
        # 计算准确率
        accuracy = torch.trace(conf_matrix) / conf_matrix.sum()
        
        # 计算每个类别的precision, recall, f1
        precision = []
        recall = []
        f1 = []
        
        for i in range(num_classes):
            tp = conf_matrix[i, i]
            fp = conf_matrix[:, i].sum() - tp
            fn = conf_matrix[i, :].sum() - tp
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            precision.append(prec)
            recall.append(rec)
            f1.append(f)
        
        return {
            'accuracy': accuracy.item(),
            'precision': [p.item() for p in precision],
            'recall': [r.item() for r in recall],
            'f1': [f.item() for f in f1],
            'macro_f1': sum(f1) / num_classes
        }

# 测试
# analyzer = ClassificationAnalyzer(model)
# conf_matrix = analyzer.compute_confusion_matrix(test_loader, 10)
# metrics = analyzer.compute_metrics(conf_matrix)
# print(f"准确率: {metrics['accuracy']:.4f}")
# print(f"Macro F1: {metrics['macro_f1']:.4f}")
```

---

**下一节**：[目标检测](02-object-detection.md)

---

## 参考文献

1. Krizhevsky, A., et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks.
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition.
3. Russakovsky, O., et al. (2015). ImageNet Large Scale Visual Recognition Challenge.