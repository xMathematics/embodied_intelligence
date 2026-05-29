# 3.2 卷积神经网络

## 目录

- [1. CNN概述](#1-cnn概述)
- [2. 卷积层](#2-卷积层)
- [3. 池化层](#3-池化层)
- [4. CNN架构](#4-cnn架构)
- [5. 重要论文详解](#5-重要论文详解)
- [6. 实践练习](#6-实践练习)

---

## 1. CNN概述

### 1.1 什么是CNN？

**卷积神经网络（CNN）**：专门用于处理网格状数据（如图像）的神经网络。

**核心思想**：
- 局部感受野
- 参数共享
- 空间不变性

### 1.2 CNN的优势

| 方面 | 传统MLP | CNN |
|------|---------|-----|
| 参数数量 | 多 | 少（参数共享） |
| 空间信息 | 丢失 | 保留 |
| 平移不变性 | 不具备 | 具备 |
| 图像处理 | 差 | 优秀 |

---

## 2. 卷积层

### 2.1 卷积运算

**二维卷积**：
$$(I * K)[i,j] = \sum_{m} \sum_{n} I[i+m, j+n] \cdot K[m,n]$$

**填充（Padding）**：
- 保持输出尺寸
- 常用：same padding、valid padding

**步幅（Stride）**：
- 滑动步长
- 控制输出尺寸

### 2.2 卷积核

**常见卷积核**：

| 类型 | 核 | 作用 |
|------|-----|------|
| 边缘检测 | `[-1,-1,-1],[-1,8,-1],[-1,-1,-1]` | 检测边缘 |
| 模糊 | `[1,1,1],[1,1,1],[1,1,1]`/9 | 平滑图像 |
| 锐化 | `[0,-1,0],[-1,5,-1],[0,-1,0]` | 增强细节 |

### 2.3 多通道卷积

**输入**：$C_{in} \times H \times W$
**卷积核**：$C_{out} \times C_{in} \times K \times K$
**输出**：$C_{out} \times H' \times W'$

---

## 3. 池化层

### 3.1 最大池化

**作用**：
- 降采样
- 保留关键特征
- 增加感受野

**公式**：
$$\text{MaxPool}(I)[i,j] = \max_{m,n} I[i \cdot s + m, j \cdot s + n]$$

### 3.2 平均池化

**公式**：
$$\text{AvgPool}(I)[i,j] = \frac{1}{k^2} \sum_{m,n} I[i \cdot s + m, j \cdot s + n]$$

### 3.3 全局平均池化

**作用**：
- 替代全连接层
- 减少参数
- 增加泛化能力

---

## 4. CNN架构

### 4.1 LeNet-5

**架构**：
- 输入：32x32灰度图像
- C1：6个5x5卷积
- S2：2x2最大池化
- C3：16个5x5卷积
- S4：2x2最大池化
- C5：120个5x5卷积
- F6：84维全连接
- 输出：10类

### 4.2 AlexNet

**架构**：
- 输入：224x224彩色图像
- 5个卷积层
- 3个全连接层
- ReLU激活
- Dropout正则化

### 4.3 VGGNet

**架构**：
- 16/19层
- 3x3小卷积核
- 相同通道数堆叠
- 规律的网络结构

### 4.4 ResNet

**架构**：
- 残差连接
- 解决梯度消失
- 深度可达152层

### 4.5 架构对比

| 模型 | 层数 | 参数 | 特点 |
|------|------|------|------|
| LeNet | 5 | 60k | 开创性 |
| AlexNet | 8 | 60M | 深度学习复兴 |
| VGG | 16/19 | 138M | 规整结构 |
| ResNet | 50/101/152 | 25M | 残差连接 |

---

## 5. 重要论文详解

### 5.1 AlexNet (2012)

**作者**：Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton

**发表会议**：NIPS

**详细分析**：[查看论文详解](../papers/alexnet.md)

**核心贡献**：
- 首次使用ReLU激活函数
- 使用Dropout防止过拟合
- 数据增强
- GPU加速训练

**为什么重要**：
- 赢得ImageNet 2012竞赛
- 开启深度学习时代

### 5.2 ResNet (2015)

**作者**：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

**发表会议**：CVPR

**详细分析**：[查看论文详解](../papers/resnet.md)

**核心贡献**：
- 残差连接
- 解决梯度消失问题
- 训练超深网络

**为什么重要**：
- 突破网络深度限制
- 成为后续架构的基础

---

## 6. 实践练习

### 练习1：使用PyTorch构建简单CNN

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=True)

# 训练
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

# 测试
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

print(f'Test Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')
```

### 练习2：使用预训练模型

```python
from torchvision import models

# 加载预训练模型
model = models.resnet18(pretrained=True)

# 修改最后一层进行迁移学习
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

# 冻结前面层
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

# 训练（仅训练最后一层）
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

---

**下一步**：[循环神经网络](03-rnn.md)