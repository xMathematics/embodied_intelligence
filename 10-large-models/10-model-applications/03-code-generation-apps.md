# 10.3 代码生成应用

## 目录

- [1. 引言](#1-引言)
- [2. 代码生成概述](#2-代码生成概述)
- [3. 核心技术](#3-核心技术)
- [4. 应用场景](#4-应用场景)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 代码生成的重要性

**代码生成**是大型语言模型的重要应用领域，能够根据自然语言描述自动生成代码。

### 1.2 应用价值

| 维度 | 描述 |
|------|------|
| **提高效率** | 自动生成样板代码 |
| **降低门槛** | 让非专业开发者也能编程 |
| **代码补全** | 智能补全代码片段 |
| **代码翻译** | 不同语言间代码转换 |

---

## 2. 代码生成概述

### 2.1 定义

**代码生成**：根据自然语言描述或上下文，自动生成符合语法和语义的代码。

**形式化表达**：
```
code = f(description, context, language)
```

### 2.2 代码生成类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **文本到代码** | 自然语言描述生成代码 | "创建一个Python函数来计算斐波那契数列" |
| **代码补全** | 根据上下文补全代码 | 函数中间自动补全逻辑 |
| **代码翻译** | 一种语言转另一种语言 | Python转JavaScript |
| **代码优化** | 优化现有代码 | 改进代码性能 |

---

## 3. 核心技术

### 3.1 代码生成器

```python
class CodeGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_code(self, prompt, language='python', max_length=500):
        """
        生成代码
        
        参数:
            prompt: 自然语言描述
            language: 目标语言
            max_length: 最大长度
        
        返回:
            生成的代码
        """
        # 构建提示
        full_prompt = f"""
{prompt}

{language} code:
"""
        
        inputs = self.tokenizer(full_prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取代码部分
        code = code.replace(full_prompt, "").strip()
        
        return code
    
    def generate_function(self, description, language='python'):
        """
        生成函数
        
        参数:
            description: 函数描述
            language: 目标语言
        
        返回:
            函数代码
        """
        prompt = f"Write a {language} function that {description}. Include docstring and handle edge cases."
        return self.generate_code(prompt, language)
    
    def generate_class(self, description, language='python'):
        """
        生成类
        
        参数:
            description: 类描述
            language: 目标语言
        
        返回:
            类代码
        """
        prompt = f"Write a {language} class that {description}. Include constructor and methods with docstrings."
        return self.generate_code(prompt, language)

# 测试（需要加载代码模型）
# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
# model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf")
# 
# generator = CodeGenerator(model, tokenizer)
# code = generator.generate_function("calculates the factorial of a number")
# print(code)
```

### 3.2 代码优化

```python
class CodeOptimizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def optimize_code(self, code, optimization_type='performance'):
        """
        优化代码
        
        参数:
            code: 原始代码
            optimization_type: 优化类型
        
        返回:
            优化后的代码
        """
        prompt = f"""
Optimize the following {optimization_type}:

{code}

Optimized code:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=500,
            temperature=0.7,
            do_sample=True
        )
        
        optimized = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return optimized.replace(prompt, "").strip()
    
    def refactor_code(self, code):
        """
        重构代码
        
        参数:
            code: 原始代码
        
        返回:
            重构后的代码
        """
        prompt = f"""
Refactor the following code to improve readability and maintainability:

{code}

Refactored code:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=500,
            temperature=0.7,
            do_sample=True
        )
        
        refactored = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return refactored.replace(prompt, "").strip()
    
    def debug_code(self, code, error_message):
        """
        调试代码
        
        参数:
            code: 代码
            error_message: 错误信息
        
        返回:
            修复后的代码
        """
        prompt = f"""
Fix the following code that produces this error: {error_message}

{code}

Fixed code:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=500,
            temperature=0.7,
            do_sample=True
        )
        
        fixed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return fixed.replace(prompt, "").strip()

# 测试
# optimizer = CodeOptimizer(model, tokenizer)
# 
# code = """
# def fibonacci(n):
#     if n <= 1:
#         return n
#     return fibonacci(n-1) + fibonacci(n-2)
# """
# 
# optimized = optimizer.optimize_code(code, 'performance')
# print("优化后的代码:")
# print(optimized)
```

### 3.3 代码翻译

```python
class CodeTranslator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def translate_code(self, code, source_lang, target_lang):
        """
        翻译代码
        
        参数:
            code: 原始代码
            source_lang: 源语言
            target_lang: 目标语言
        
        返回:
            翻译后的代码
        """
        prompt = f"""
Translate the following {source_lang} code to {target_lang}:

{code}

{target_lang} code:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=500,
            temperature=0.7,
            do_sample=True
        )
        
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated.replace(prompt, "").strip()
    
    def summarize_code(self, code):
        """
        总结代码功能
        
        参数:
            code: 代码
        
        返回:
            代码功能描述
        """
        prompt = f"""
Explain what the following code does in plain English:

{code}

Explanation:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True
        )
        
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation.replace(prompt, "").strip()

# 测试
# translator = CodeTranslator(model, tokenizer)
# 
# python_code = """
# def greet(name):
#     print(f"Hello, {name}!")
# """
# 
# js_code = translator.translate_code(python_code, 'Python', 'JavaScript')
# print("JavaScript代码:")
# print(js_code)
```

---

## 4. 应用场景

### 4.1 代码补全

```python
class CodeCompletion:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def complete_code(self, prefix, max_length=100):
        """
        补全代码
        
        参数:
            prefix: 代码前缀
            max_length: 最大长度
        
        返回:
            补全后的代码
        """
        inputs = self.tokenizer(prefix, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=len(inputs['input_ids'][0]) + max_length,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        completed = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return completed
    
    def suggest_fix(self, code, line_number):
        """
        建议修复
        
        参数:
            code: 代码
            line_number: 行号
        
        返回:
            修复建议
        """
        lines = code.split('\n')
        context = '\n'.join(lines[max(0, line_number-3):line_number+3])
        
        prompt = f"""
Fix the issue in line {line_number} of this code:

{context}

Fixed code:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True
        )
        
        suggestion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return suggestion.replace(prompt, "").strip()

# 测试
# completion = CodeCompletion(model, tokenizer)
# 
# prefix = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]"
# completed = completion.complete_code(prefix)
# print("补全后的代码:")
# print(completed)
```

### 4.2 API生成

```python
class APIGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_api(self, description, framework='flask'):
        """
        生成API
        
        参数:
            description: API描述
            framework: 框架
        
        返回:
            API代码
        """
        prompt = f"""
Create a {framework} API that {description}. Include routes, request handling, and error handling.

{framework} code:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=800,
            temperature=0.7,
            do_sample=True
        )
        
        api_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return api_code.replace(prompt, "").strip()
    
    def generate_database_schema(self, description):
        """
        生成数据库模式
        
        参数:
            description: 需求描述
        
        返回:
            SQL代码
        """
        prompt = f"""
Create a SQL database schema for {description}. Include tables, columns, and relationships.

SQL:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=500,
            temperature=0.7,
            do_sample=True
        )
        
        schema = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return schema.replace(prompt, "").strip()

# 测试
# api_gen = APIGenerator(model, tokenizer)
# api = api_gen.generate_api("manages a todo list with CRUD operations")
# print("生成的API代码:")
# print(api)
```

### 4.3 测试生成

```python
class TestGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_tests(self, code, testing_framework='pytest'):
        """
        生成测试代码
        
        参数:
            code: 源代码
            testing_framework: 测试框架
        
        返回:
            测试代码
        """
        prompt = f"""
Generate {testing_framework} tests for the following code:

{code}

{testing_framework} tests:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=500,
            temperature=0.7,
            do_sample=True
        )
        
        tests = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return tests.replace(prompt, "").strip()
    
    def generate_mocks(self, function_name, dependencies):
        """
        生成mock对象
        
        参数:
            function_name: 函数名
            dependencies: 依赖列表
        
        返回:
            Mock代码
        """
        prompt = f"""
Generate mocks for the dependencies of {function_name}: {', '.join(dependencies)}

Mock code:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=300,
            temperature=0.7,
            do_sample=True
        )
        
        mocks = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return mocks.replace(prompt, "").strip()

# 测试
# test_gen = TestGenerator(model, tokenizer)
# 
# code = """
# def add(a, b):
#     return a + b
# 
# def divide(a, b):
#     if b == 0:
#         raise ValueError("Cannot divide by zero")
#     return a / b
# """
# 
# tests = test_gen.generate_tests(code)
# print("生成的测试代码:")
# print(tests)
```

---

## 5. 实践练习

### 练习1：实现代码生成器

```python
class AdvancedCodeGenerator:
    def __init__(self, model_name='codellama/CodeLlama-7b-hf'):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt, language='python', max_length=500, **kwargs):
        """
        生成代码
        
        参数:
            prompt: 描述
            language: 目标语言
            max_length: 最大长度
        
        返回:
            生成的代码
        """
        default_kwargs = {
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id
        }
        default_kwargs.update(kwargs)
        
        full_prompt = f"{prompt}\n\n{language} code:\n"
        
        inputs = self.tokenizer(full_prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            **default_kwargs
        )
        
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return code.replace(full_prompt, "").strip()
    
    def generate_project(self, requirements):
        """
        生成项目结构
        
        参数:
            requirements: 项目需求
        
        返回:
            项目文件内容字典
        """
        prompt = f"""
Create a complete project structure for: {requirements}

Include:
1. Project structure
2. Main files with implementation
3. Dependencies
4. README

Project:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=1000,
            temperature=0.7,
            do_sample=True
        )
        
        project = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return project.replace(prompt, "").strip()

# 测试
# generator = AdvancedCodeGenerator()
# code = generator.generate(
#     "Write a Python function to reverse a linked list",
#     language='python',
#     max_length=200
# )
# print("生成的代码:")
# print(code)
```

### 练习2：实现代码审查助手

```python
class CodeReviewAssistant:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def review_code(self, code, focus_areas=None):
        """
        审查代码
        
        参数:
            code: 代码
            focus_areas: 关注领域列表
        
        返回:
            审查结果
        """
        if focus_areas:
            focus_str = ", ".join(focus_areas)
            prompt = f"""
Review the following code focusing on: {focus_str}

{code}

Review (include issues and suggestions):
"""
        else:
            prompt = f"""
Review the following code for bugs, security issues, performance, and best practices.

{code}

Review:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=500,
            temperature=0.7,
            do_sample=True
        )
        
        review = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return review.replace(prompt, "").strip()
    
    def suggest_improvements(self, code):
        """
        建议改进
        
        参数:
            code: 代码
        
        返回:
            改进建议
        """
        prompt = f"""
Suggest improvements for the following code:

{code}

Improvements:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=300,
            temperature=0.7,
            do_sample=True
        )
        
        suggestions = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return suggestions.replace(prompt, "").strip()
    
    def check_security(self, code):
        """
        检查安全问题
        
        参数:
            code: 代码
        
        返回:
            安全问题列表
        """
        prompt = f"""
Check the following code for security vulnerabilities:

{code}

Security issues:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=300,
            temperature=0.7,
            do_sample=True
        )
        
        issues = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return issues.replace(prompt, "").strip()

# 测试
# assistant = CodeReviewAssistant(model, tokenizer)
# 
# code = """
# def login(username, password):
#     query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
#     execute_query(query)
# """
# 
# security_issues = assistant.check_security(code)
# print("安全问题:")
# print(security_issues)
```

### 练习3：实现代码学习助手

```python
class CodeLearningAssistant:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def explain_code(self, code, level='intermediate'):
        """
        解释代码
        
        参数:
            code: 代码
            level: 难度级别
        
        返回:
            解释内容
        """
        prompt = f"""
Explain the following code at a {level} level. Include:
1. What the code does
2. Key concepts
3. How it works step by step

{code}

Explanation:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=400,
            temperature=0.7,
            do_sample=True
        )
        
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation.replace(prompt, "").strip()
    
    def generate_exercise(self, topic, difficulty='medium'):
        """
        生成练习题
        
        参数:
            topic: 主题
            difficulty: 难度
        
        返回:
            练习题
        """
        prompt = f"""
Create a coding exercise about {topic} with {difficulty} difficulty.
Include:
1. Problem statement
2. Example input/output
3. Hints

Exercise:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=400,
            temperature=0.7,
            do_sample=True
        )
        
        exercise = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return exercise.replace(prompt, "").strip()
    
    def provide_feedback(self, user_code, expected_output, actual_output):
        """
        提供反馈
        
        参数:
            user_code: 用户代码
            expected_output: 期望输出
            actual_output: 实际输出
        
        返回:
            反馈内容
        """
        prompt = f"""
Provide feedback on this code:

User code:
{user_code}

Expected output: {expected_output}
Actual output: {actual_output}

Feedback:
"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=300,
            temperature=0.7,
            do_sample=True
        )
        
        feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return feedback.replace(prompt, "").strip()

# 测试
# assistant = CodeLearningAssistant(model, tokenizer)
# 
# code = """
# def binary_search(arr, target):
#     left, right = 0, len(arr) - 1
#     while left <= right:
#         mid = (left + right) // 2
#         if arr[mid] == target:
#             return mid
#         elif arr[mid] < target:
#             left = mid + 1
#         else:
#             right = mid - 1
#     return -1
# """
# 
# explanation = assistant.explain_code(code, 'beginner')
# print("代码解释:")
# print(explanation)
```

---

**下一节**：[多模态应用](04-multimodal-applications.md)

---

## 参考文献

1. Chen, M., et al. (2021). Evaluating Large Language Models Trained on Code.
2. Li, C., et al. (2023). CodeLlama: Open Foundation Models for Code.
3. Svyatkovskiy, A., et al. (2020). CodeSearchNet Challenge: Evaluating the State of Semantic Code Search.