# 5.4 符号推理

## 目录

- [1. 引言](#1-引言)
- [2. 符号推理概述](#2-符号推理概述)
- [3. 符号推理类型](#3-符号推理类型)
- [4. 符号推理方法](#4-符号推理方法)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 符号推理的重要性

**符号推理**是指使用符号逻辑来进行推理的能力。这是人工智能领域的传统方法，与神经网络的数值方法形成互补。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **定理证明** | 数学定理证明 | 几何定理证明 |
| **逻辑推理** | 形式逻辑推理 | 命题逻辑推理 |
| **知识表示** | 知识的符号化表示 | 本体论 |
| **专家系统** | 基于规则的推理 | 医疗诊断 |

---

## 2. 符号推理概述

### 2.1 定义

**符号推理**：使用符号和规则进行逻辑推理的过程。

**形式化表达**：
```
SymbolicReasoning(knowledge_base, query) → result
```

### 2.2 符号推理的特点

| 特点 | 描述 |
|------|------|
| **精确性** | 结果是确定的 |
| **可解释性** | 推理过程清晰 |
| **逻辑性** | 遵循形式逻辑 |
| **抽象性** | 处理抽象符号 |

---

## 3. 符号推理类型

### 3.1 命题逻辑

**定义**：处理命题和逻辑连接词。

**示例**：
```
命题P：今天下雨
命题Q：地面湿

推理：如果P则Q（P → Q）
已知P为真，所以Q为真
```

**代码示例**：
```python
from sympy import symbols, Implies, And, Or, Not, simplify

# 定义命题
P = symbols('P')
Q = symbols('Q')

# 创建逻辑表达式
implication = Implies(P, Q)

# 求值
print(f"P→Q 当P=True, Q=True时: {implication.subs({P: True, Q: True})}")
```

### 3.2 一阶逻辑

**定义**：处理谓词和量词。

**示例**：
```
∀x (Human(x) → Mortal(x))  # 所有人都会死
Human(Socrates)             # 苏格拉底是人
Mortal(Socrates)            # 苏格拉底会死
```

### 3.3 模态逻辑

**定义**：处理可能性和必然性。

**示例**：
```
可能：明天可能下雨
必然：2+2必然等于4
```

---

## 4. 符号推理方法

### 4.1 规则引擎

**特点**：
- 基于IF-THEN规则
- 前向/后向链推理

```python
class RuleEngine:
    def __init__(self):
        self.rules = []
    
    def add_rule(self, condition, action):
        """
        添加规则
        
        参数:
            condition: 条件
            action: 动作/结论
        """
        self.rules.append((condition, action))
    
    def forward_chaining(self, facts):
        """
        前向链推理
        
        参数:
            facts: 已知事实
        
        返回:
            推导出的结论
        """
        conclusions = set(facts)
        changed = True
        
        while changed:
            changed = False
            for condition, action in self.rules:
                if condition.issubset(conclusions) and action not in conclusions:
                    conclusions.add(action)
                    changed = True
        
        return conclusions

# 测试
engine = RuleEngine()
engine.add_rule({'下雨'}, {'地面湿'})
engine.add_rule({'地面湿'}, {'需要小心'})
engine.add_rule({'下雨'}, {'需要带伞'})

facts = {'下雨'}
print(engine.forward_chaining(facts))  # {'下雨', '地面湿', '需要小心', '需要带伞'}
```

### 4.2 定理证明器

**特点**：
- 自动化定理证明
- 使用形式逻辑

```python
from sympy import *

class TheoremProver:
    def __init__(self):
        pass
    
    def prove(self, premises, conclusion):
        """
        证明定理
        
        参数:
            premises: 前提列表
            conclusion: 结论
        
        返回:
            证明结果
        """
        # 创建逻辑表达式
        expr = And(*premises) >> conclusion
        
        # 简化并检查是否为重言式
        simplified = simplify(expr)
        return simplified == True

# 测试
prover = TheoremProver()
P, Q = symbols('P Q')

# 假言推理：P → Q, P ⊢ Q
premises = [Implies(P, Q), P]
conclusion = Q
print(f"假言推理成立: {prover.prove(premises, conclusion)}")  # True
```

### 4.3 知识图谱推理

**特点**：
- 基于图结构的知识
- 路径推理

```python
class KnowledgeGraph:
    def __init__(self):
        self.graph = {}
    
    def add_relation(self, subject, predicate, obj):
        """
        添加关系
        
        参数:
            subject: 主语
            predicate: 谓语
            obj: 宾语
        """
        if subject not in self.graph:
            self.graph[subject] = []
        self.graph[subject].append((predicate, obj))
    
    def query(self, subject, predicate=None):
        """
        查询知识
        
        参数:
            subject: 主语
            predicate: 谓语（可选）
        
        返回:
            结果列表
        """
        if subject not in self.graph:
            return []
        
        results = []
        for pred, obj in self.graph[subject]:
            if predicate is None or pred == predicate:
                results.append((subject, pred, obj))
        
        return results

# 测试
kg = KnowledgeGraph()
kg.add_relation('苏格拉底', '是', '人')
kg.add_relation('人', '都会', '死')

print(kg.query('苏格拉底'))  # [('苏格拉底', '是', '人')]
print(kg.query('人'))        # [('人', '都会', '死')]
```

---

## 5. 实践练习

### 练习1：实现命题逻辑推理器

```python
class PropositionalReasoner:
    def __init__(self):
        self.truth_table = {
            (True, True): {
                'AND': True,
                'OR': True,
                'IMPLIES': True,
                'XOR': False,
                'EQ': True
            },
            (True, False): {
                'AND': False,
                'OR': True,
                'IMPLIES': False,
                'XOR': True,
                'EQ': False
            },
            (False, True): {
                'AND': False,
                'OR': True,
                'IMPLIES': True,
                'XOR': True,
                'EQ': False
            },
            (False, False): {
                'AND': False,
                'OR': False,
                'IMPLIES': True,
                'XOR': False,
                'EQ': True
            }
        }
    
    def evaluate(self, expr, values):
        """
        评估命题逻辑表达式
        
        参数:
            expr: 表达式（字符串）
            values: 命题值字典
        
        返回:
            结果
        """
        # 简单实现：处理 AND, OR, NOT, IMPLIES
        expr = expr.replace('AND', 'and').replace('OR', 'or').replace('NOT', 'not')
        expr = expr.replace('IMPLIES', '<=')
        
        # 替换命题变量
        for var, val in values.items():
            expr = expr.replace(var, str(val))
        
        # 计算
        return eval(expr)
    
    def is_tautology(self, expr, variables):
        """
        检查是否为重言式
        
        参数:
            expr: 表达式
            variables: 变量列表
        
        返回:
            是否为重言式
        """
        from itertools import product
        
        # 生成所有可能的真值组合
        for values in product([True, False], repeat=len(variables)):
            value_dict = dict(zip(variables, values))
            if not self.evaluate(expr, value_dict):
                return False
        return True

# 测试
reasoner = PropositionalReasoner()

# 评估表达式
result = reasoner.evaluate('P AND Q', {'P': True, 'Q': False})
print(f"P AND Q = {result}")  # False

# 检查重言式
tautology = reasoner.is_tautology('P OR NOT P', ['P'])
print(f"P OR NOT P 是重言式: {tautology}")  # True
```

### 练习2：实现一阶逻辑推理器

```python
class FirstOrderReasoner:
    def __init__(self):
        self.facts = []
    
    def add_fact(self, fact):
        """
        添加事实
        
        参数:
            fact: 事实（元组）
        """
        self.facts.append(fact)
    
    def query(self, pattern):
        """
        查询匹配的事实
        
        参数:
            pattern: 查询模式
        
        返回:
            匹配结果
        """
        results = []
        for fact in self.facts:
            if self.match(fact, pattern):
                results.append(fact)
        return results
    
    def match(self, fact, pattern):
        """
        匹配事实和模式
        
        参数:
            fact: 事实
            pattern: 模式
        
        返回:
            是否匹配
        """
        if len(fact) != len(pattern):
            return False
        
        for f, p in zip(fact, pattern):
            if p != '_' and f != p:
                return False
        return True

# 测试
reasoner = FirstOrderReasoner()
reasoner.add_fact(('苏格拉底', '是', '人'))
reasoner.add_fact(('柏拉图', '是', '人'))
reasoner.add_fact(('人', '都会', '死'))

# 查询所有人
print(reasoner.query(('_', '是', '人')))  # [('苏格拉底', '是', '人'), ('柏拉图', '是', '人')]

# 查询苏格拉底是什么
print(reasoner.query(('苏格拉底', '_', '_')))  # [('苏格拉底', '是', '人')]
```

### 练习3：实现简单的专家系统

```python
class ExpertSystem:
    def __init__(self):
        self.rules = []
    
    def add_rule(self, conditions, conclusion, certainty=1.0):
        """
        添加规则
        
        参数:
            conditions: 条件列表
            conclusion: 结论
            certainty: 确定性因子
        """
        self.rules.append({
            'conditions': conditions,
            'conclusion': conclusion,
            'certainty': certainty
        })
    
    def infer(self, facts):
        """
        进行推理
        
        参数:
            facts: 已知事实
        
        返回:
            结论列表
        """
        conclusions = []
        
        for rule in self.rules:
            # 检查所有条件是否满足
            all_conditions_met = True
            for condition in rule['conditions']:
                if condition not in facts:
                    all_conditions_met = False
                    break
            
            if all_conditions_met:
                conclusions.append({
                    'conclusion': rule['conclusion'],
                    'certainty': rule['certainty']
                })
        
        return conclusions

# 测试医疗诊断专家系统
system = ExpertSystem()
system.add_rule(['发烧', '咳嗽'], '可能是感冒', 0.8)
system.add_rule(['头痛', '恶心', '颈部僵硬'], '可能是脑膜炎', 0.9)
system.add_rule(['发烧', '喉咙痛'], '可能是咽喉炎', 0.7)

# 患者症状
patient_symptoms = ['发烧', '咳嗽']

# 诊断
diagnosis = system.infer(patient_symptoms)
print("诊断结果:")
for result in diagnosis:
    print(f"- {result['conclusion']} (确定性: {result['certainty']})")
```

---

**下一节**：[工具使用](05-tool-use.md)

---

## 参考文献

1. Russell, S., & Norvig, P. (2021). Artificial Intelligence: A Modern Approach.
2. Boolos, G. S., & Jeffrey, R. C. (1989). Computability and Logic.
3. Robinson, J. A. (1965). A machine-oriented logic based on the resolution principle.
