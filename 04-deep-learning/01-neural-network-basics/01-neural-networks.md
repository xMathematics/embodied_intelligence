# 3.1 神经网络基础

## 目录

- [1. 神经网络概述](#1-神经网络概述)
- [2. 前向传播](#2-前向传播)
- [3. 反向传播](#3-反向传播)
- [4. 激活函数](#4-激活函数)
- [5. 损失函数](#5-损失函数)
- [6. 优化器](#6-优化器)
- [7. 实践练习](#7-实践练习)

---

## 1. 神经网络概述

### 1.1 什么是神经网络？

**神经网络**：由大量神经元相互连接组成的计算模型，模拟人脑神经元的工作方式。

**结构**：
- 输入层：接收输入数据
- 隐藏层：进行特征变换
- 输出层：产生输出结果

### 1.2 感知机

**最简单的神经网络**：
$$y = \sigma(w^T x + b)$$

**多层感知机（MLP）**：
$$h_1 = \sigma(W_1 x + b_1)$$
$$h_2 = \sigma(W_2 h_1 + b_2)$$
$$y = \sigma(W_3 h_2 + b_3)$$

---

## 2. 前向传播

### 2.1 计算图

**张量运算**：
- 矩阵乘法
- 非线性变换
- 批量处理

### 2.2 示例计算

```python
import numpy as np

# 定义参数
W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
b1 = np.array([0.1, 0.2])
W2 = np.array([[0.5, 0.6]])
b2 = np.array([0.3])

# 输入
x = np.array([1.0, 2.0])

# 前向传播
h = np.tanh(np.dot(W1, x) + b1)
y = np.dot(W2, h) + b2

print(f"隐藏层输出: {h}")
print(f"最终输出: {y}")
```

---

## 3. 反向传播

### 3.1 链式法则

**梯度计算**：
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

### 3.2 反向传播算法

**步骤**：
1. 前向传播计算损失
2. 反向传播计算梯度
3. 更新参数

### 3.3 梯度下降

**参数更新**：
$$\theta := \theta - \alpha \frac{\partial L}{\partial \theta}$$

---

## 4. 激活函数

### 4.1 常用激活函数

| 函数 | 公式 | 特点 |
|------|------|------|
| Sigmoid | $\sigma(x) = \frac{1}{1+e^{-x}}$ | 输出在(0,1)，梯度消失问题 |
| Tanh | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | 输出在(-1,1)，零均值 |
| ReLU | $\text{ReLU}(x) = \max(0, x)$ | 解决梯度消失，计算快 |
| Leaky ReLU | $\text{LeakyReLU}(x) = \max(\alpha x, x)$ | 解决ReLU死亡问题 |
| Swish | $\text{Swish}(x) = x \sigma(\beta x)$ | 平滑非单调 |

### 4.2 激活函数对比

```python
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)

# Sigmoid
sigmoid = 1 / (1 + np.exp(-x))

# Tanh
tanh = np.tanh(x)

# ReLU
relu = np.maximum(0, x)

# Leaky ReLU
leaky_relu = np.maximum(0.1 * x, x)

# 绘制
plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid, label='Sigmoid')
plt.plot(x, tanh, label='Tanh')
plt.plot(x, relu, label='ReLU')
plt.plot(x, leaky_relu, label='Leaky ReLU')
plt.legend()
plt.title('Activation Functions')
plt.grid(True)
plt.show()
```

---

## 5. 损失函数

### 5.1 回归损失

**均方误差（MSE）**：
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**平均绝对误差（MAE）**：
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### 5.2 分类损失

**交叉熵损失**：
$$CE = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

**二元交叉熵**：
$$BCE = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

---

## 6. 优化器

### 6.1 梯度下降变体

| 优化器 | 特点 |
|--------|------|
| SGD | 基础梯度下降 |
| Momentum | 加速收敛，减少震荡 |
| AdaGrad | 自适应学习率 |
| RMSprop | 改进AdaGrad |
| Adam | 结合Momentum和RMSprop |

### 6.2 Adam优化器

**步骤**：
1. 计算一阶矩估计
2. 计算二阶矩估计
3. 偏差修正
4. 更新参数

---

## 7. 实践练习

### 练习1：手动实现神经网络

```python
import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def backward(self, X, y, output, learning_rate=0.01):
        # 反向传播
        m = X.shape[0]
        
        # 输出层梯度
        dz2 = output - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 隐藏层梯度
        dz1 = np.dot(dz2, self.W2.T) * (self.z1 > 0)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 更新参数
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

# 测试
nn = SimpleNN(2, 3, 1)
X = np.random.randn(10, 2)
y = np.random.randn(10, 1)

# 训练
for i in range(1000):
    output = nn.forward(X)
    loss = np.mean((output - y) ** 2)
    nn.backward(X, y, output)
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")
```

### 练习2：使用PyTorch构建神经网络

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型
model = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练数据
X = torch.randn(100, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + torch.randn(100) * 0.1
y = y.unsqueeze(1)

# 训练循环
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 测试
test_X = torch.tensor([[1.0, 2.0]])
print(f"预测结果: {model(test_X).item():.2f}")
print(f"真实结果: {2*1 + 3*2 = 8.0}")
```

---

**下一步**：[卷积神经网络](02-cnn.md)