# 3.3 循环神经网络

## 目录

- [1. RNN概述](#1-rnn概述)
- [2. RNN结构](#2-rnn结构)
- [3. LSTM](#3-lstm)
- [4. GRU](#4-gru)
- [5. 序列建模应用](#5-序列建模应用)
- [6. 实践练习](#6-实践练习)

---

## 1. RNN概述

### 1.1 什么是RNN？

**循环神经网络（RNN）**：专门处理序列数据的神经网络。

**核心特点**：
- 记忆性
- 参数共享
- 处理变长序列

### 1.2 序列数据类型

| 类型 | 示例 |
|------|------|
| 文本 | 句子、段落 |
| 时间序列 | 股票价格、传感器数据 |
| 语音 | 音频波形 |
| 视频 | 帧序列 |

---

## 2. RNN结构

### 2.1 基本RNN单元

**循环连接**：
$$h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

**展开图**：
```
x_1 -> [RNN] -> y_1
x_2 -> [RNN] -> y_2
x_3 -> [RNN] -> y_3
       ...
```

### 2.2 梯度消失问题

**问题**：
- 梯度随时间衰减
- 长距离依赖难以学习

**原因**：
- 梯度连乘导致指数衰减
- tanh/sigmoid导数小于1

---

## 3. LSTM

### 3.1 长短期记忆网络

**解决梯度消失问题**：
- 引入门控机制
- 选择性记忆和遗忘

### 3.2 LSTM单元结构

**遗忘门**：
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**输入门**：
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**细胞状态更新**：
$$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t$$

**输出门**：
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \cdot \tanh(C_t)$$

### 3.3 LSTM变体

| 变体 | 特点 |
|------|------|
| Peephole连接 | 细胞状态参与门计算 |
| GRU | 简化版LSTM |
| Bidirectional LSTM | 双向处理 |

---

## 4. GRU

### 4.1 门控循环单元

**简化版LSTM**：
- 合并遗忘门和输入门
- 去掉细胞状态

### 4.2 GRU单元结构

**更新门**：
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

**重置门**：
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

**候选隐藏状态**：
$$\tilde{h}_t = \tanh(W \cdot [r_t \cdot h_{t-1}, x_t])$$

**隐藏状态更新**：
$$h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \tilde{h}_t$$

### 4.3 LSTM vs GRU

| 比较 | LSTM | GRU |
|------|------|-----|
| 参数数量 | 多 | 少 |
| 表达能力 | 强 | 略弱 |
| 训练速度 | 慢 | 快 |
| 记忆能力 | 强 | 较强 |

---

## 5. 序列建模应用

### 5.1 序列分类

**示例**：情感分析

```python
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out
```

### 5.2 序列生成

**示例**：文本生成

```python
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
```

### 5.3 机器翻译

**Encoder-Decoder架构**：
```
Encoder: 输入序列 -> 上下文向量
Decoder: 上下文向量 -> 输出序列
```

---

## 6. 实践练习

### 练习1：使用LSTM进行时间序列预测

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
time = np.linspace(0, 10, 200)
data = np.sin(time) + np.random.randn(200) * 0.1

# 准备数据
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

seq_length = 10
X, y = create_sequences(data, seq_length)

# 转换为张量
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# 定义模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out

# 训练
model = LSTMModel(1, 32, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# 预测
model.eval()
with torch.no_grad():
    predictions = model(X).numpy()

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(time[seq_length:], y.numpy(), label='真实值')
plt.plot(time[seq_length:], predictions, label='预测值')
plt.legend()
plt.show()
```

### 练习2：使用GRU进行文本分类

```python
from torchtext.data import Field, TabularDataset, BucketIterator

# 定义字段
TEXT = Field(tokenize='spacy', lower=True)
LABEL = Field(sequential=False, use_vocab=False)

# 加载数据
train_data, test_data = TabularDataset.splits(
    path='./data',
    train='train.csv',
    test='test.csv',
    format='csv',
    fields=[('text', TEXT), ('label', LABEL)]
)

# 构建词汇表
TEXT.build_vocab(train_data, max_size=10000)

# 创建迭代器
train_iter, test_iter = BucketIterator.splits(
    (train_data, test_data),
    batch_size=64,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 定义模型
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        out = self.fc(h_n[-1])
        return out

# 训练
model = GRUClassifier(len(TEXT.vocab), 100, 128, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for batch in train_iter:
        optimizer.zero_grad()
        output = model(batch.text)
        loss = criterion(output, batch.label)
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_iter:
            output = model(batch.text)
            pred = output.argmax(dim=1)
            correct += (pred == batch.label).sum().item()
            total += len(batch.label)
    
    print(f'Epoch {epoch}, Accuracy: {100 * correct / total:.2f}%')
```

---

**下一步**：[Transformer架构](04-transformer.md)