# 7.5 大模型规划

## 目录

- [1. 引言](#1-引言)
- [2. 提示工程](#2-提示工程)
- [3. 思维链与思考](#3-思维链与思考)
- [4. 工具调用规划](#4-工具调用规划)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 大语言模型在规划中的优势

| 特性 | 传统规划 | LLM规划 |
|------|--------|
| 常识 | 无 | 有 |
| 知识 | 手工编码 | 预训练知识 |
| 可解释 | 低 | 高 |
| 泛化 | 差 | 好 |

```python
import random
from typing import List, Dict, Any
```

---

## 2. 提示工程

### 2.1 规划提示模板

```python
class LLMTemplate:
    """大模型规划模板"""
    
    @staticmethod
    def basic_prompt(goal, context=None):
        """基础规划提示"""
        template = f"""你是一个任务规划助手。

目标: {goal}

{'环境/上下文:
{context if context else '没有提供'}

请给出完成这个目标的步骤计划。
使用列表形式返回，格式：
1. 第一步...
2. 第二步...
"""
"""
        return template
    
    @staticmethod
    def step_by_step_prompt(goal, available_actions=None):
        """分步规划提示"""
        template = f"""你是一个任务规划专家。

目标: {goal}

{('可用动作: {available_actions if available_actions else '未指定'}

请逐步思考并给出完成这个目标的计划。

请用清晰的步骤列表。
"""
        return template
```

---

## 3. 思维链与思考

### 3.1 思维链规划

```python
class ChainOfThoughtPlanner:
    """思维链规划器"""
    
    def __init__(self):
        self.thoughts = []
        self.plan = []
    
    def reason_and_plan(self, goal, context=None):
        """推理并规划"""
        # 阶段1: 理解目标
        thought1 = f"首先，我需要理解目标: {goal}"
        self.thoughts.append(thought1)
        
        # 阶段2: 分解子目标
        thought2 = "将目标分解为子任务"
        self.thoughts.append(thought2)
        
        # 阶段3: 排序
        thought3 = "确定子任务的执行顺序"
        self.thoughts.append(thought3)
        
        # 阶段4: 生成计划
        plan = [
            "第一步: 分析当前状态",
            "第二步: 执行核心动作",
            "第三步: 验证"
        ]
        self.plan = plan
        
        return self.thoughts, self.plan
```

---

## 4. 工具调用规划

### 4.1 ReAct模式

```python
class ReActPlanner:
    """ReAct (Reason + Act) 规划器"""
    
    def __init__(self):
        self.tools = {}
        self.max_steps = 10
    
    def add_tool(self, name, func, description):
        """添加工具"""
        self.tools[name] = {
            'func': func,
            'description': description
        }
    
    def plan_and_execute(self, query):
        """规划与执行"""
        trajectory = []
        state = {'query': query}
        
        for step in range(self.max_steps):
            # 思考
            thought = self._think(state)
            
            # 行动
            action, tool_input = self._choose_action(state)
            
            # 观察
            observation = self._execute_tool(action, tool_input)
            
            trajectory.append({
                'step': step+1,
                'thought': thought,
                'action': action,
                'input': tool_input,
                'observation': observation
            })
            
            state['observation'] = observation
            
            if 'done' in observation and observation['done']:
                break
        
        return trajectory
    
    def _think(self, state):
        """思考"""
        return "思考中..."
    
    def _choose_action(self, state):
        """选择动作"""
        return "answer", "最终答案"
    
    def _execute_tool(self, action, tool_input):
        """执行工具"""
        return {'result': '完成', 'done': True}
```

### 4.2 简单工具

```python
class SimpleTools:
    """简单工具集"""
    
    @staticmethod
    def search(query):
        """搜索"""
        return f"搜索结果: {query}"
    
    @staticmethod
    def calculate(expression):
        """计算"""
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except:
            return "计算错误"
    
    @staticmethod
    def check_condition(condition):
        """检查条件"""
        return "条件检查完成"
```

---

## 5. 实践练习

### 练习1：简单LLM规划模拟

```python
def exercise_llm_planning():
    """大模型规划练习"""
    print("=== 大模型规划 (模拟)\n")
    
    # 1. 提示工程
    print("=== 1. 提示工程示例\n")
    
    goal = "准备一份给朋友的生日惊喜派对"
    prompt = LLMTemplate.basic_prompt(goal)
    print("提示词：")
    print(prompt)
    
    # 2. 思维链
    print("\n\n=== 2. 思维链规划 (模拟) ===")
    cot_planner = ChainOfThoughtPlanner()
    thoughts, plan = cot_planner.reason_and_plan(goal)
    print("思考过程:")
    for i, t in enumerate(thoughts):
        print(f"  思考 {i+1}: {t}")
    
    print("\n计划:")
    for i, step in enumerate(plan):
        print(f"  {i+1}. {step}")
    
    # 3. ReAct
    print("\n\n=== 3. ReAct规划 (模拟) ===")
    print("ReAct模式: 思考 -> 行动 -> 观察 -> 重复")
    
    react_planner = ReActPlanner()
    react_planner.add_tool('search', SimpleTools.search, "搜索信息")
    react_planner.add_tool('calc', SimpleTools.calculate, "计算")
    
    print("\n可用工具:")
    for name, info in react_planner.tools.items():
        print(f"  - {name}: {info['description']}")
    
    print("\n典型步骤示例:")
    print("  1. 思考: 我需要规划生日派对...")
    print("  2. 行动: search('朋友的生日日期")
    print("  3. 观察: 朋友生日是...")
    print("  4. 思考: 现在我知道...")
    print("  5. 行动: search('附近的蛋糕店')")
    print("  6. 思考: 好的，我有计划了")
    print("  7. 行动: answer('最终计划...')")

# exercise_llm_planning()
```

---

恭喜！你已经完成了第七部分：任务规划（2周）的全部内容！

---

## 参考文献

1. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
2. Yao, S., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models.
3. Huang, W., et al. (2022). Inner Monologue: Embodied Reasoning through Planning with Language Models.
4. Ahn, M., et al. (2022). Do As I Can, Not As I Say: Grounding Language in Robotic Affordances.
5. Brohan, A., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.
6. Du, Y., et al. (2023). LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models.
