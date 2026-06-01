# 4.3 代码补全

## 目录

- [1. 引言](#1-引言)
- [2. 代码补全概述](#2-代码补全概述)
- [3. 代码补全技术](#3-代码补全技术)
- [4. 代表性代码补全工具](#4-代表性代码补全工具)
- [5. 代码补全评估](#5-代码补全评估)
- [6. 实践练习](#6-实践练习)

---

## 1. 引言

### 1.1 代码补全的重要性

**代码补全**是指在编写代码时，根据已输入的部分自动推断并补全剩余部分。这是提高编程效率的关键工具。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **IDE集成** | 集成到代码编辑器中 | VS Code的IntelliSense |
| **快速编码** | 减少键盘输入 | 自动补全函数名 |
| **学习辅助** | 帮助新手学习 | 提示API用法 |
| **代码规范** | 保持代码风格一致 | 自动格式化 |

---

## 2. 代码补全概述

### 2.1 任务定义

**代码补全**：根据部分输入代码，预测并补全剩余部分。

**形式化表达**：
```
CodeCompletion(prefix) → completion
```

### 2.2 代码补全类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **标识符补全** | 补全变量、函数、类名 | `prin` → `print()` |
| **语句补全** | 补全完整语句 | `for i in` → `for i in range(10):` |
| **代码块补全** | 补全代码块 | `if x > 0:` → 补全缩进块 |
| **参数补全** | 补全函数参数 | `print(` → 提示参数 |

---

## 3. 代码补全技术

### 3.1 基于规则的方法

**特点**：
- 使用语法规则和模板
- 简单快速
- 缺乏上下文理解

**示例**：
```python
# 基于语法规则的补全
def complete_statement(prefix):
    if prefix.endswith('if'):
        return prefix + ' condition:'
    elif prefix.endswith('for'):
        return prefix + ' item in iterable:'
    elif prefix.endswith('def'):
        return prefix + ' function_name():'
    else:
        return prefix
```

### 3.2 基于统计的方法

**特点**：
- 使用n-gram模型
- 基于代码语料库统计
- 考虑上下文

**示例**：
```python
# 简单的n-gram补全
from collections import defaultdict

class NGramModel:
    def __init__(self, n=3):
        self.n = n
        self.ngrams = defaultdict(list)
    
    def train(self, code_corpus):
        for code in code_corpus:
            tokens = code.split()
            for i in range(len(tokens) - self.n + 1):
                prefix = tuple(tokens[i:i+self.n-1])
                next_token = tokens[i+self.n-1]
                self.ngrams[prefix].append(next_token)
    
    def predict(self, prefix_tokens):
        key = tuple(prefix_tokens[-self.n+1:])
        if key in self.ngrams:
            return max(set(self.ngrams[key]), key=self.ngrams[key].count)
        return None
```

### 3.3 基于深度学习的方法

**特点**：
- 使用Transformer架构
- 强大的上下文建模
- 支持长距离依赖

**示例**：
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载代码补全模型
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")

# 代码补全
prefix = "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return"

inputs = tokenizer.encode(prefix, return_tensors="pt")
outputs = model.generate(
    inputs,
    max_length=100,
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id
)

completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(completion)
```

---

## 4. 代表性代码补全工具

### 4.1 GitHub Copilot

**特点**：
- 基于GPT-4
- 实时补全
- 支持多种语言
- 集成到主流IDE

**工作原理**：
```
用户输入 → 发送到服务器 → GPT-4生成补全 → 返回结果
```

### 4.2 CodeLlama

**特点**：
- 开源模型
- 支持填充（Fill-in-the-Middle）
- 多种规模
- 可部署在本地

### 4.3 Tabnine

**特点**：
- 基于深度学习
- 支持多种IDE
- 离线模式
- 团队协作功能

### 4.4 IntelliCode

**特点**：
- 微软开发
- 集成到VS Code
- 基于AI的补全
- 支持多种语言

---

## 5. 代码补全评估

### 5.1 评估指标

| 指标 | 描述 |
|------|------|
| **准确率** | 补全正确的比例 |
| **召回率** | 覆盖的补全数量 |
| **排名** | 正确补全在建议列表中的位置 |
| **用户满意度** | 用户体验评分 |

### 5.2 评估数据集

| 数据集 | 描述 | 用途 |
|--------|------|------|
| **CodeSearchNet** | 代码语料库 | 训练和评估 |
| **HumanEval** | 编程问题 | 功能正确性评估 |
| **MBPP** | 基础编程问题 | 补全质量评估 |

---

## 6. 实践练习

### 练习1：简单的代码补全器

```python
class SimpleCodeCompleter:
    def __init__(self):
        self.keywords = {'def', 'class', 'if', 'else', 'for', 'while', 'return', 'import', 'from'}
        self.functions = {'print', 'len', 'range', 'list', 'dict', 'set', 'str', 'int', 'float'}
    
    def complete(self, prefix):
        """
        根据前缀提供补全建议
        
        参数:
            prefix: 输入前缀
        
        返回:
            补全建议列表
        """
        suggestions = []
        
        # 关键词补全
        for keyword in self.keywords:
            if keyword.startswith(prefix.lower()):
                suggestions.append(keyword)
        
        # 函数补全
        for func in self.functions:
            if func.startswith(prefix.lower()):
                suggestions.append(func + '()')
        
        return suggestions[:5]  # 返回前5个建议

# 测试
completer = SimpleCodeCompleter()
print(completer.complete('pri'))  # ['print()']
print(completer.complete('de'))   # ['def']
print(completer.complete('fo'))   # ['for', 'from', 'format()']
```

### 练习2：基于Transformer的代码补全

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class TransformerCodeCompleter:
    def __init__(self, model_name="codellama/CodeLlama-7b-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
    
    def complete(self, prefix, max_length=50, num_suggestions=3):
        """
        使用Transformer模型进行代码补全
        
        参数:
            prefix: 输入前缀
            max_length: 最大生成长度
            num_suggestions: 建议数量
        
        返回:
            补全建议列表
        """
        inputs = self.tokenizer.encode(prefix, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        outputs = self.model.generate(
            inputs,
            max_length=len(inputs[0]) + max_length,
            num_return_sequences=num_suggestions,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        suggestions = []
        for output in outputs:
            completion = self.tokenizer.decode(output, skip_special_tokens=True)
            suggestions.append(completion[len(prefix):].strip())
        
        return suggestions

# 测试
# completer = TransformerCodeCompleter()
# suggestions = completer.complete("def quicksort(arr):")
# for i, suggestion in enumerate(suggestions, 1):
#     print(f"建议{i}:\n{suggestion}\n")
```

### 练习3：代码补全评估

```python
def evaluate_completion(actual, predictions):
    """
    评估代码补全质量
    
    参数:
        actual: 实际期望的补全
        predictions: 补全建议列表
    
    返回:
        评估结果
    """
    # 检查是否包含正确答案
    correct_found = actual in predictions
    
    # 计算排名
    rank = None
    if correct_found:
        rank = predictions.index(actual) + 1
    
    # 计算准确率（如果只有一个建议）
    accuracy = 1.0 if predictions and predictions[0] == actual else 0.0
    
    return {
        'correct_found': correct_found,
        'rank': rank,
        'accuracy': accuracy,
        'num_suggestions': len(predictions)
    }

# 测试
actual = "return fibonacci(n-1) + fibonacci(n-2)"
predictions = [
    "return fibonacci(n-1) + fibonacci(n-2)",
    "return n",
    "return fib(n-1) + fib(n-2)"
]

result = evaluate_completion(actual, predictions)
print(f"正确答案是否找到: {result['correct_found']}")
print(f"排名: {result['rank']}")
print(f"准确率: {result['accuracy']}")
print(f"建议数量: {result['num_suggestions']}")
```

---

**下一节**：[代码翻译](04-code-translation.md)

---

## 参考文献

1. CodeLlama: Open Foundation Models for Code. (2023). Meta.
2. GitHub Copilot. (2021). GitHub.
3. IntelliCode. (2018). Microsoft.
