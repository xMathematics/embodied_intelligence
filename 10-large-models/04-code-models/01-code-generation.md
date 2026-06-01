# 4.1 代码生成模型

## 目录

- [1. 引言](#1-引言)
- [2. 代码生成概述](#2-代码生成概述)
- [3. 代码生成模型架构](#3-代码生成模型架构)
- [4. 代表性代码生成模型](#4-代表性代码生成模型)
- [5. 代码生成任务](#5-代码生成任务)
- [6. 代码生成评估](#6-代码生成评估)
- [7. 实践练习](#7-实践练习)

---

## 1. 引言

### 1.1 代码生成的重要性

**代码生成**是指根据自然语言描述或其他输入自动生成代码的任务。这是AI辅助编程的核心功能，能够显著提高编程效率。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **辅助编程** | 帮助开发者快速生成代码 | 根据需求描述生成函数 |
| **代码补全** | 自动补全部分代码 | IDE中的智能补全 |
| **代码翻译** | 在不同语言间转换代码 | Python→Java |
| **代码修复** | 自动修复代码错误 | 检测并修复bug |

---

## 2. 代码生成概述

### 2.1 任务定义

**代码生成**：给定自然语言描述或其他输入，生成符合语法和语义的代码。

**形式化表达**：
```
CodeGeneration(description) → code
```

### 2.2 代码生成的特点

| 特点 | 描述 |
|------|------|
| **语法约束** | 生成的代码必须符合编程语言语法 |
| **语义正确性** | 代码必须实现预期功能 |
| **可读性** | 生成的代码应易于理解 |
| **多样性** | 同一需求可能有多种实现方式 |

---

## 3. 代码生成模型架构

### 3.1 编码器-解码器架构

```
输入描述 → 编码器 → 隐藏状态 → 解码器 → 代码输出
```

**代码示例**：
```python
import torch
import torch.nn as nn

class CodeGenerator(nn.Module):
    def __init__(self, vocab_size=50000, embedding_dim=512, hidden_dim=1024, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids, target_ids):
        # 编码输入
        embed = self.embedding(input_ids)
        encoder_out, _ = self.encoder(embed)
        
        # 解码输出
        decoder_embed = self.embedding(target_ids)
        decoder_out, _ = self.decoder(decoder_embed, (encoder_out[:, -1, :].unsqueeze(0).repeat(6, 1, 1),)*2)
        logits = self.classifier(decoder_out)
        
        return logits
```

### 3.2 Transformer架构

**代码生成Transformer**：
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练代码生成模型
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 生成代码
prompt = "def fibonacci(n):"
inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(code)
```

---

## 4. 代表性代码生成模型

### 4.1 CodeLlama

**论文**：CodeLlama: Open Foundation Models for Code (Meta, 2023)

**核心特点**：
- 基于LLaMA的代码专用模型
- 支持多种编程语言
- 代码填充能力
- 开源可商用

**模型系列**：
| 模型 | 参数 | 特点 |
|------|------|------|
| CodeLlama-7B | 7B | 基础模型 |
| CodeLlama-13B | 13B | 更好性能 |
| CodeLlama-70B | 70B | 最佳性能 |
| CodeLlama-7B-Instruct | 7B | 指令微调版本 |

### 4.2 CodeGen

**论文**：CodeGen: An Open Large Language Model for Code with Multi-Turn Program Synthesis (Nijkamp et al., 2022)

**核心特点**：
- 专门优化的代码生成模型
- 支持多轮程序合成
- 多种编程语言支持

### 4.3 StarCoder

**论文**：StarCoder: Open Source Code Models (Hugging Face, 2023)

**核心特点**：
- 开源代码模型
- 支持80+编程语言
- 基于The Stack数据集训练
- 支持填充功能

### 4.4 GitHub Copilot

**特点**：
- 基于GPT-4的代码补全工具
- 集成到IDE中
- 实时代码补全
- 支持多种语言

---

## 5. 代码生成任务

### 5.1 文本到代码生成

**定义**：根据自然语言描述生成代码

**示例**：
```
输入："写一个快速排序算法"
输出：
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```

### 5.2 代码补全

**定义**：根据部分代码补全剩余部分

**示例**：
```
输入："def fibonacci(n):\n    if n <= 1:\n        return n\n    else:"
输出："        return fibonacci(n-1) + fibonacci(n-2)"
```

### 5.3 代码修复

**定义**：检测并修复代码中的错误

**示例**：
```
输入（有bug的代码）：
def add(a, b)
    return a + b

输出（修复后）：
def add(a, b):
    return a + b
```

---

## 6. 代码生成评估

### 6.1 评估指标

| 指标 | 描述 | 适用场景 |
|------|------|---------|
| **BLEU** | 基于n-gram匹配 | 代码生成质量 |
| **CodeBLEU** | 专门针对代码的BLEU变体 | 代码生成质量 |
| **Pass@k** | 代码通过测试的概率 | 功能正确性 |
| **HumanEval** | 人工评估 | 综合质量 |

### 6.2 评估数据集

| 数据集 | 描述 | 规模 |
|--------|------|------|
| **HumanEval** | 手写编程问题 | 164个问题 |
| **MBPP** | Most Basic Python Programs | 974个问题 |
| **APPS** | 编程竞赛问题 | 2,500个问题 |

### 6.3 Pass@k计算

**公式**：
```
Pass@k = 1 - (1 - p)^k
```
其中p是单次生成正确代码的概率，k是生成次数。

---

## 7. 实践练习

### 练习1：使用CodeLlama生成代码

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载CodeLlama模型
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")

# 生成代码
prompt = """
# 写一个Python函数，计算两个矩阵的乘积
def matrix_multiply(A, B):
"""

inputs = tokenizer.encode(prompt, return_tensors="pt")
outputs = model.generate(
    inputs,
    max_length=150,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(code)
```

### 练习2：实现简单的代码生成模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCodeGenerator(nn.Module):
    def __init__(self, vocab_size=50000, embed_dim=512, hidden_dim=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=hidden_dim
        )
        self.fc = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, src, tgt):
        # src: [src_len, batch]
        # tgt: [tgt_len, batch]
        
        src_embed = self.embedding(src) * torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        tgt_embed = self.embedding(tgt) * torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        
        # 添加位置编码
        src_pos = self._generate_positions(src.size(0)).unsqueeze(1).repeat(1, src.size(1), 1)
        tgt_pos = self._generate_positions(tgt.size(0)).unsqueeze(1).repeat(1, tgt.size(1), 1)
        
        src_embed = src_embed + src_pos
        tgt_embed = tgt_embed + tgt_pos
        
        output = self.transformer(src_embed, tgt_embed)
        logits = self.fc(output)
        
        return logits
    
    def _generate_positions(self, max_len):
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 512, 2) * (-torch.log(torch.tensor(10000.0)) / 512))
        pos_enc = torch.zeros(max_len, 512)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        return pos_enc

# 测试
model = SimpleCodeGenerator()
src = torch.randint(0, 50000, (10, 8))
tgt = torch.randint(0, 50000, (20, 8))
output = model(src, tgt)
print(f"输出形状: {output.shape}")  # [20, 8, 50000]
```

### 练习3：代码生成评估

```python
def calculate_pass_at_k(results, k=1):
    """
    计算Pass@k指标
    
    参数:
        results: 测试结果列表，每个元素是布尔值表示是否通过测试
        k: 生成次数
    
    返回:
        Pass@k值
    """
    passed = sum(1 for r in results if any(r[:k]))
    return passed / len(results)

# 模拟测试结果
# 每个样本生成3次，True表示通过测试
test_results = [
    [True, False, False],   # 第1个样本
    [False, True, False],   # 第2个样本
    [False, False, False],  # 第3个样本
    [True, True, True],     # 第4个样本
    [False, False, True]    # 第5个样本
]

# 计算Pass@1, Pass@2, Pass@3
for k in [1, 2, 3]:
    pass_at_k = calculate_pass_at_k(test_results, k)
    print(f"Pass@{k}: {pass_at_k:.2f}")
```

---

**下一节**：[代码理解](02-code-understanding.md)

---

## 参考文献

1. CodeLlama: Open Foundation Models for Code. (2023). Meta.
2. Nijkamp, E., Pang, M., Pinto, A., et al. (2022). CodeGen: An open large language model for code with multi-turn program synthesis.
3. StarCoder: Open Source Code Models. (2023). Hugging Face.
