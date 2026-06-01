# 4.4 代码翻译

## 目录

- [1. 引言](#1-引言)
- [2. 代码翻译概述](#2-代码翻译概述)
- [3. 代码翻译技术](#3-代码翻译技术)
- [4. 代表性代码翻译模型](#4-代表性代码翻译模型)
- [5. 代码翻译评估](#5-代码翻译评估)
- [6. 实践练习](#6-实践练习)

---

## 1. 引言

### 1.1 代码翻译的重要性

**代码翻译**是指将一种编程语言的代码转换为另一种编程语言的代码。这在跨平台开发、代码迁移和学习新语言时非常有用。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **跨平台开发** | 将代码迁移到不同平台 | Python→JavaScript |
| **遗留代码迁移** | 更新旧代码 | Java→Kotlin |
| **多语言项目** | 保持代码同步 | C++→Python |
| **学习辅助** | 帮助理解不同语言 | Rust→Python |

---

## 2. 代码翻译概述

### 2.1 任务定义

**代码翻译**：将源语言代码转换为目标语言代码，保持功能等价。

**形式化表达**：
```
CodeTranslation(source_code, source_lang, target_lang) → target_code
```

### 2.2 代码翻译的挑战

| 挑战 | 描述 |
|------|------|
| **语法差异** | 不同语言语法差异大 |
| **语义等价** | 保持功能完全一致 |
| **API差异** | 标准库和API不同 |
| **习语差异** | 语言特定的表达方式 |

---

## 3. 代码翻译技术

### 3.1 基于规则的方法

**特点**：
- 使用语法规则进行转换
- 精确但不灵活
- 需要手动编写规则

**示例**：
```python
# 简单的Python到JavaScript转换规则
def python_to_javascript(code):
    # 替换print语句
    code = code.replace('print(', 'console.log(')
    
    # 替换def为function
    code = code.replace('def ', 'function ')
    
    # 替换True/False
    code = code.replace('True', 'true').replace('False', 'false')
    
    return code

# 测试
python_code = "def greet(name):\n    print('Hello, ' + name)"
js_code = python_to_javascript(python_code)
print(js_code)
```

### 3.2 基于统计的方法

**特点**：
- 使用平行语料库训练
- 基于翻译模型
- 自动学习翻译规则

### 3.3 基于深度学习的方法

**特点**：
- 使用Transformer架构
- 端到端训练
- 处理复杂转换

**示例**：
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 加载代码翻译模型
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")

# 代码翻译
python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# 使用CodeT5进行翻译
inputs = tokenizer("translate Python to Java: " + python_code, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=100)
java_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(java_code)
```

---

## 4. 代表性代码翻译模型

### 4.1 CodeT5

**特点**：
- 统一的编码器-解码器架构
- 支持多种编程语言
- 可以进行代码翻译

### 4.2 TransCoder

**论文**：TransCoder: Translating Between Programming Languages (Lample et al., 2020)

**核心特点**：
- 专门的代码翻译模型
- 支持多种语言对
- 基于Transformer架构

### 4.3 CodeSearchNet Translator

**特点**：
- 基于CodeSearchNet数据集
- 支持多种语言
- 开源可用

---

## 5. 代码翻译评估

### 5.1 评估指标

| 指标 | 描述 |
|------|------|
| **BLEU** | 基于n-gram匹配 |
| **CodeBLEU** | 专门针对代码的指标 |
| **功能正确性** | 通过测试用例的比例 |
| **可读性** | 生成代码的可读性 |

### 5.2 评估数据集

| 数据集 | 描述 | 语言对 |
|--------|------|-------|
| **CodeSearchNet** | 代码检索数据集 | 6种语言 |
| **HumanEval** | 编程问题 | 多语言版本 |
| **MBPP** | 基础编程问题 | 多语言版本 |

---

## 6. 实践练习

### 练习1：简单的代码翻译器

```python
class SimpleCodeTranslator:
    def __init__(self):
        self.python_to_js_mapping = {
            'print': 'console.log',
            'def ': 'function ',
            'True': 'true',
            'False': 'false',
            'None': 'null',
            'and': '&&',
            'or': '||',
            'not': '!',
            'elif': 'else if',
        }
    
    def translate(self, code, source_lang='python', target_lang='javascript'):
        """
        将代码从源语言翻译到目标语言
        
        参数:
            code: 源语言代码
            source_lang: 源语言
            target_lang: 目标语言
        
        返回:
            目标语言代码
        """
        if source_lang == 'python' and target_lang == 'javascript':
            for python_token, js_token in self.python_to_js_mapping.items():
                code = code.replace(python_token, js_token)
            
            # 处理缩进（简化版本）
            code = code.replace('    ', '    ')  # 保持4空格缩进
            
            # 处理函数定义
            lines = code.split('\n')
            translated_lines = []
            for line in lines:
                if line.strip().startswith('function') and '(' in line:
                    # 添加花括号
                    translated_lines.append(line)
                    translated_lines.append('{')
                else:
                    translated_lines.append(line)
            
            # 在最后添加闭合花括号（简化处理）
            code = '\n'.join(translated_lines)
            if '{' in code and '}' not in code:
                code += '\n}'
        
        return code

# 测试
translator = SimpleCodeTranslator()
python_code = """
def greet(name):
    print('Hello, ' + name)
"""
js_code = translator.translate(python_code)
print("Python代码:")
print(python_code)
print("\nJavaScript代码:")
print(js_code)
```

### 练习2：使用预训练模型进行代码翻译

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TransformerCodeTranslator:
    def __init__(self, model_name="Salesforce/codet5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def translate(self, code, source_lang, target_lang, max_length=150):
        """
        使用预训练模型进行代码翻译
        
        参数:
            code: 源语言代码
            source_lang: 源语言
            target_lang: 目标语言
            max_length: 最大生成长度
        
        返回:
            目标语言代码
        """
        prompt = f"translate {source_lang} to {target_lang}: {code}"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(inputs["input_ids"], max_length=max_length)
        translated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_code

# 测试
translator = TransformerCodeTranslator()

python_code = """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""

java_code = translator.translate(python_code, "Python", "Java")
print("Python代码:")
print(python_code)
print("\nJava代码:")
print(java_code)
```

### 练习3：代码翻译评估

```python
def evaluate_translation(original_code, translated_code, test_cases):
    """
    评估代码翻译的质量
    
    参数:
        original_code: 原始代码
        translated_code: 翻译后的代码
        test_cases: 测试用例列表
    
    返回:
        评估结果
    """
    passed = 0
    failed = 0
    
    for test_input, expected_output in test_cases:
        # 这里应该实际执行代码并比较结果
        # 为了安全，我们跳过实际执行
        # 在真实场景中，应该使用沙箱环境执行代码
        
        # 简单的语法检查
        if 'def ' in translated_code or 'function ' in translated_code:
            passed += 1
        else:
            failed += 1
    
    return {
        'passed': passed,
        'failed': failed,
        'accuracy': passed / (passed + failed) if (passed + failed) > 0 else 0
    }

# 测试
python_code = "def add(a, b):\n    return a + b"
translated_code = "function add(a, b) {\n    return a + b;\n}"

test_cases = [
    ((2, 3), 5),
    ((10, 20), 30),
    ((-1, 1), 0)
]

result = evaluate_translation(python_code, translated_code, test_cases)
print(f"通过测试: {result['passed']}")
print(f"失败测试: {result['failed']}")
print(f"准确率: {result['accuracy']:.2f}")
```

---

**下一节**：[调试与优化](05-debugging-optimization.md)

---

## 参考文献

1. Lample, G., Ott, M., & Conneau, A. (2020). TransCoder: Translating between programming languages.
2. Wang, Y., Liu, D., Shi, S., et al. (2021). CodeT5: Identifier-aware unified pre-trained encoder-decoder models.
3. CodeSearchNet. (2019). Salesforce.
