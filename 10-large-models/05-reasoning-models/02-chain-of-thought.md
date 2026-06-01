# 5.2 思维链推理

## 目录

- [1. 引言](#1-引言)
- [2. 思维链推理概述](#2-思维链推理概述)
- [3. 思维链提示方法](#3-思维链提示方法)
- [4. 思维链推理变体](#4-思维链推理变体)
- [5. 思维链推理应用](#5-思维链推理应用)
- [6. 实践练习](#6-实践练习)

---

## 1. 引言

### 1.1 思维链推理的重要性

**思维链推理**（Chain of Thought, CoT）是一种提示工程技术，通过引导模型逐步思考来解决复杂问题。这是提升大语言模型推理能力的关键方法。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **数学问题** | 解决复杂数学题 | 多步骤算术题 |
| **逻辑推理** | 进行逻辑分析 | 推理谜题 |
| **代码生成** | 生成复杂代码 | 需要设计思路的代码 |
| **科学问题** | 解决科学问题 | 物理、化学问题 |

---

## 2. 思维链推理概述

### 2.1 定义

**思维链推理**：通过引导模型输出中间推理步骤来解决复杂问题。

**核心思想**：
```
问题 → 步骤1 → 步骤2 → ... → 步骤n → 答案
```

### 2.2 思维链推理的特点

| 特点 | 描述 |
|------|------|
| **透明性** | 展示推理过程 |
| **可验证性** | 可以检查每一步是否正确 |
| **可修正性** | 可以修正错误步骤 |
| **通用性** | 适用于多种推理任务 |

---

## 3. 思维链提示方法

### 3.1 基本提示格式

```python
def chain_of_thought_prompt(question):
    """
    构建思维链提示
    
    参数:
        question: 问题
    
    返回:
        完整的提示
    """
    prompt = f"""
    请解决以下问题，在给出最终答案前，请详细说明你的推理过程。
    
    问题：{question}
    
    思考：
    """
    return prompt
```

### 3.2 少样本思维链提示

```python
def few_shot_cot_prompt(question):
    """
    构建少样本思维链提示
    
    参数:
        question: 问题
    
    返回:
        完整的提示
    """
    examples = [
        {
            "question": "小明有5个苹果，小红有3个苹果，他们一共有多少个苹果？",
            "thinking": "小明有5个苹果，小红有3个苹果，所以总共是5+3=8个苹果。",
            "answer": "8"
        },
        {
            "question": "一个书架有3层，每层放10本书，一共有多少本书？",
            "thinking": "每层10本书，3层就是3×10=30本书。",
            "answer": "30"
        }
    ]
    
    prompt = "请解决以下问题，按照示例的格式回答：\n\n"
    for example in examples:
        prompt += f"问题：{example['question']}\n思考：{example['thinking']}\n答案：{example['answer']}\n\n"
    prompt += f"问题：{question}\n思考："
    
    return prompt
```

---

## 4. 思维链推理变体

### 4.1 自一致性（Self-Consistency）

**方法**：生成多个思维链，选择最一致的答案。

```python
def self_consistency_prompt(question, num_chains=3):
    """
    自一致性提示
    
    参数:
        question: 问题
        num_chains: 思维链数量
    
    返回:
        提示
    """
    prompt = f"""
    请解决以下问题，给出{num_chains}种不同的推理路径：
    
    问题：{question}
    
    推理路径1：
    
    推理路径2：
    
    推理路径3：
    
    最终答案：
    """
    return prompt
```

### 4.2 迭代优化（Iterative Refinement）

**方法**：逐步改进推理过程。

```python
def iterative_refinement_prompt(question):
    """
    迭代优化提示
    
    参数:
        question: 问题
    
    返回:
        提示
    """
    prompt = f"""
    请解决以下问题。首先给出初步答案，然后检查答案是否正确，如果不正确，请修正。
    
    问题：{question}
    
    初步答案：
    
    检查：
    
    修正后的答案：
    """
    return prompt
```

### 4.3 思维树（Tree of Thought）

**方法**：探索多种推理路径，形成树状结构。

```python
def tree_of_thought_prompt(question):
    """
    思维树提示
    
    参数:
        question: 问题
    
    返回:
        提示
    """
    prompt = f"""
    请解决以下问题，考虑多种可能的推理路径，形成思维树：
    
    问题：{question}
    
    思维树：
    - 分支1：
      - 步骤1：
      - 步骤2：
      - 结论：
    - 分支2：
      - 步骤1：
      - 步骤2：
      - 结论：
    - 分支3：
      - 步骤1：
      - 步骤2：
      - 结论：
    
    最佳答案：
    """
    return prompt
```

---

## 5. 思维链推理应用

### 5.1 数学问题解决

**示例**：
```
问题：一个水池有两个进水管和一个出水管。单独开甲管，6小时可以注满水池；单独开乙管，4小时可以注满水池；单独开丙管，3小时可以放完一池水。如果三管同时打开，几小时可以注满水池？

思考：
1. 甲管每小时注水量：1/6
2. 乙管每小时注水量：1/4
3. 丙管每小时放水量：1/3
4. 三管同时打开的净注水量：1/6 + 1/4 - 1/3 = (2 + 3 - 4)/12 = 1/12
5. 注满水池需要的时间：1 / (1/12) = 12小时

答案：12小时
```

### 5.2 逻辑推理

**示例**：
```
问题：甲、乙、丙三人中有一人是教师，一人是医生，一人是工程师。已知：
1. 甲比医生年龄大
2. 乙和工程师不同岁
3. 工程师比丙年龄小

请问三人各是什么职业？

思考：
1. 由条件2和3可知，工程师不是乙也不是丙，所以工程师是甲。
2. 由条件1可知，甲（工程师）比医生年龄大。
3. 由条件3可知，工程师（甲）比丙年龄小，所以丙不是医生。
4. 因此，医生是乙，剩下的丙是教师。

答案：甲是工程师，乙是医生，丙是教师。
```

---

## 6. 实践练习

### 练习1：实现思维链提示生成器

```python
class CoTPromptGenerator:
    def __init__(self):
        self.examples = []
    
    def add_example(self, question, thinking, answer):
        """
        添加示例
        
        参数:
            question: 问题
            thinking: 思考过程
            answer: 答案
        """
        self.examples.append({
            'question': question,
            'thinking': thinking,
            'answer': answer
        })
    
    def generate_prompt(self, question, include_thinking=True):
        """
        生成思维链提示
        
        参数:
            question: 问题
            include_thinking: 是否包含思考过程
        
        返回:
            提示字符串
        """
        prompt = "请解决以下问题。"
        
        if self.examples:
            prompt += "以下是一些示例：\n\n"
            for example in self.examples:
                prompt += f"问题：{example['question']}\n"
                if include_thinking:
                    prompt += f"思考：{example['thinking']}\n"
                prompt += f"答案：{example['answer']}\n\n"
        
        prompt += f"问题：{question}\n"
        if include_thinking:
            prompt += "思考："
        
        return prompt

# 测试
generator = CoTPromptGenerator()
generator.add_example(
    "小明有10颗糖，分给朋友5颗，还剩几颗？",
    "小明原来有10颗糖，分给朋友5颗，所以剩下10-5=5颗糖。",
    "5"
)
generator.add_example(
    "一个正方形的边长是4厘米，它的面积是多少？",
    "正方形面积=边长×边长=4×4=16平方厘米。",
    "16平方厘米"
)

question = "一本书有200页，小明每天看20页，需要几天看完？"
print(generator.generate_prompt(question))
```

### 练习2：思维链推理实现

```python
class ChainOfThoughtReasoner:
    def __init__(self, model):
        """
        初始化思维链推理器
        
        参数:
            model: 语言模型
        """
        self.model = model
    
    def reason(self, question, max_steps=10):
        """
        进行思维链推理
        
        参数:
            question: 问题
            max_steps: 最大推理步骤
        
        返回:
            推理结果
        """
        prompt = f"""
        请解决以下问题，一步一步地思考：
        
        问题：{question}
        
        思考过程：
        """
        
        # 逐步生成思考过程
        thinking = ""
        for step in range(max_steps):
            step_prompt = prompt + thinking
            response = self.model.generate(step_prompt, max_tokens=50)
            
            # 检查是否已经得出答案
            if "答案：" in response:
                break
            
            thinking += response
        
        # 提取答案
        if "答案：" in thinking:
            answer = thinking.split("答案：")[-1].strip()
        else:
            answer = "无法得出答案"
        
        return {
            'thinking': thinking,
            'answer': answer
        }

# 模拟模型
class MockModel:
    def generate(self, prompt, max_tokens=50):
        # 简单的模拟响应
        if "需要几天看完" in prompt:
            return "这本书有200页，每天看20页，所以需要200÷20=10天。\n答案：10天"
        return "思考中..."

# 测试
model = MockModel()
reasoner = ChainOfThoughtReasoner(model)
result = reasoner.reason("一本书有200页，小明每天看20页，需要几天看完？")
print(f"思考过程：{result['thinking']}")
print(f"答案：{result['answer']}")
```

### 练习3：自一致性推理

```python
import statistics
from collections import Counter

class SelfConsistencyReasoner:
    def __init__(self, model):
        self.model = model
    
    def reason(self, question, num_samples=5):
        """
        使用自一致性进行推理
        
        参数:
            question: 问题
            num_samples: 采样数量
        
        返回:
            推理结果
        """
        answers = []
        
        for _ in range(num_samples):
            prompt = f"""
            请解决以下问题，并给出最终答案：
            
            问题：{question}
            
            思考：
            """
            
            response = self.model.generate(prompt, max_tokens=100)
            
            # 提取答案
            if "答案：" in response:
                answer = response.split("答案：")[-1].strip()
                answers.append(answer)
        
        # 统计答案
        if answers:
            counter = Counter(answers)
            most_common = counter.most_common(1)[0][0]
            confidence = counter[most_common] / len(answers)
            
            return {
                'answers': answers,
                'final_answer': most_common,
                'confidence': confidence
            }
        else:
            return {
                'answers': [],
                'final_answer': None,
                'confidence': 0.0
            }

# 测试
model = MockModel()
reasoner = SelfConsistencyReasoner(model)
result = reasoner.reason("2+2等于多少？", num_samples=3)
print(f"所有答案：{result['answers']}")
print(f"最终答案：{result['final_answer']}")
print(f"置信度：{result['confidence']}")
```

---

**下一节**：[数学推理](03-mathematical-reasoning.md)

---

## 参考文献

1. Wei, J., et al. (2022). Chain of thought prompting elicits reasoning in large language models.
2. Wang, X., et al. (2023). Self-consistency improves chain of thought reasoning in language models.
3. Yao, S., et al. (2023). Tree of thoughts: Deliberate problem solving with large language models.
