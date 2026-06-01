# 7.3 具身推理

## 目录

- [1. 引言](#1-引言)
- [2. 具身推理概述](#2-具身推理概述)
- [3. 推理类型](#3-推理类型)
- [4. 推理方法](#4-推理方法)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 具身推理的重要性

**具身推理**是指智能体在物理环境中进行推理的能力，包括空间推理、因果推理、工具使用等。这是实现高级具身智能的关键。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **空间推理** | 理解空间关系 | "杯子在桌子上" |
| **因果推理** | 理解因果关系 | "推杯子会掉落" |
| **工具使用** | 理解工具用途 | "用锤子钉钉子" |
| **任务规划** | 规划复杂任务 | "做早餐的步骤" |

---

## 2. 具身推理概述

### 2.1 定义

**具身推理**：智能体在物理环境中，基于感知和经验进行推理的能力。

**形式化表达**：
```
Reasoning: (Perception, Knowledge, Goal) → Plan
```

### 2.2 具身推理的特点

| 特点 | 描述 |
|------|------|
| **情境性** | 依赖当前环境情境 |
| **多模态** | 结合视觉、语言、触觉等 |
| **实用性** | 推理结果可执行 |
| **适应性** | 适应环境变化 |

---

## 3. 推理类型

### 3.1 空间推理

**定义**：理解空间关系和空间变换。

```python
class SpatialReasoner:
    def __init__(self):
        self.spatial_relations = {
            'on': lambda obj1, obj2: self._check_on(obj1, obj2),
            'in': lambda obj1, obj2: self._check_in(obj1, obj2),
            'near': lambda obj1, obj2: self._check_near(obj1, obj2),
            'above': lambda obj1, obj2: self._check_above(obj1, obj2),
            'below': lambda obj1, obj2: self._check_below(obj1, obj2)
        }
    
    def check_relation(self, obj1, obj2, relation):
        """
        检查空间关系
        
        参数:
            obj1: 物体1
            obj2: 物体2
            relation: 关系类型
        
        返回:
            是否满足关系
        """
        if relation in self.spatial_relations:
            return self.spatial_relations[relation](obj1, obj2)
        return False
    
    def _check_on(self, obj1, obj2):
        """检查obj1在obj2上"""
        return (obj1['position'][2] > obj2['position'][2] and
                obj1['position'][0] > obj2['position'][0] and
                obj1['position'][0] < obj2['position'][0] + obj2['size'][0] and
                obj1['position'][1] > obj2['position'][1] and
                obj1['position'][1] < obj2['position'][1] + obj2['size'][1])
    
    def _check_near(self, obj1, obj2, threshold=0.5):
        """检查obj1靠近obj2"""
        distance = np.linalg.norm(
            np.array(obj1['position']) - np.array(obj2['position'])
        )
        return distance < threshold

# 测试
reasoner = SpatialReasoner()
table = {'position': [0, 0, 0], 'size': [1, 1, 0.5]}
cup = {'position': [0.5, 0.5, 0.6], 'size': [0.1, 0.1, 0.2]}

print(f"杯子在桌子上: {reasoner.check_relation(cup, table, 'on')}")  # True
```

### 3.2 因果推理

**定义**：理解因果关系和因果链。

```python
class CausalReasoner:
    def __init__(self):
        self.causal_rules = {
            'push': {
                'effect': 'move',
                'conditions': ['contact', 'force_applied']
            },
            'drop': {
                'effect': 'fall',
                'conditions': ['no_support']
            },
            'pour': {
                'effect': 'liquid_flows',
                'conditions': ['liquid', 'tilted']
            }
        }
    
    def predict_effect(self, action, object_state):
        """
        预测动作效果
        
        参数:
            action: 动作
            object_state: 物体状态
        
        返回:
            预测效果
        """
        if action in self.causal_rules:
            rule = self.causal_rules[action]
            if all(cond in object_state for cond in rule['conditions']):
                return rule['effect']
        return None
    
    def infer_cause(self, effect, object_state):
        """
        推断原因
        
        参数:
            effect: 效果
            object_state: 物体状态
        
        返回:
            可能的原因
        """
        possible_causes = []
        for action, rule in self.causal_rules.items():
            if rule['effect'] == effect:
                if all(cond in object_state for cond in rule['conditions']):
                    possible_causes.append(action)
        return possible_causes

# 测试
reasoner = CausalReasoner()
object_state = {'contact': True, 'force_applied': True}
effect = reasoner.predict_effect('push', object_state)
print(f"预测效果: {effect}")  # move

causes = reasoner.infer_cause('fall', {'no_support': True})
print(f"可能原因: {causes}")  # ['drop']
```

### 3.3 工具使用推理

**定义**：理解工具的用途和使用方法。

```python
class ToolUseReasoner:
    def __init__(self):
        self.tool_knowledge = {
            'hammer': {
                'purpose': 'driving nails',
                'requires': ['nail', 'surface'],
                'action': 'strike'
            },
            'screwdriver': {
                'purpose': 'driving screws',
                'requires': ['screw'],
                'action': 'rotate'
            },
            'cup': {
                'purpose': 'holding liquids',
                'requires': ['liquid'],
                'action': 'pour'
            }
        }
    
    def select_tool(self, task, available_tools):
        """
        选择合适的工具
        
        参数:
            task: 任务描述
            available_tools: 可用工具列表
        
        返回:
            最合适的工具
        """
        best_tool = None
        best_score = 0
        
        for tool in available_tools:
            if tool in self.tool_knowledge:
                knowledge = self.tool_knowledge[tool]
                # 简单的关键词匹配
                score = 0
                if knowledge['purpose'] in task:
                    score += 2
                for req in knowledge['requires']:
                    if req in task:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_tool = tool
        
        return best_tool
    
    def plan_tool_use(self, tool, task):
        """
        规划工具使用步骤
        
        参数:
            tool: 工具
            task: 任务
        
        返回:
            使用步骤
        """
        if tool in self.tool_knowledge:
            knowledge = self.tool_knowledge[tool]
            steps = [
                f"1. 获取{tool}",
                f"2. 准备{', '.join(knowledge['requires'])}",
                f"3. 使用{knowledge['action']}动作完成任务"
            ]
            return steps
        return []

# 测试
reasoner = ToolUseReasoner()
task = "drive a nail into the wall"
available_tools = ['hammer', 'screwdriver', 'cup']

best_tool = reasoner.select_tool(task, available_tools)
print(f"最佳工具: {best_tool}")  # hammer

steps = reasoner.plan_tool_use(best_tool, task)
print("使用步骤:")
for step in steps:
    print(step)
```

---

## 4. 推理方法

### 4.1 基于规则的推理

```python
class RuleBasedReasoner:
    def __init__(self):
        self.rules = []
    
    def add_rule(self, conditions, conclusion):
        """
        添加规则
        
        参数:
            conditions: 条件列表
            conclusion: 结论
        """
        self.rules.append({
            'conditions': conditions,
            'conclusion': conclusion
        })
    
    def reason(self, facts):
        """
        基于规则推理
        
        参数:
            facts: 事实集合
        
        返回:
            推导出的结论
        """
        conclusions = set(facts)
        changed = True
        
        while changed:
            changed = False
            for rule in self.rules:
                if all(cond in conclusions for cond in rule['conditions']):
                    if rule['conclusion'] not in conclusions:
                        conclusions.add(rule['conclusion'])
                        changed = True
        
        return conclusions

# 测试
reasoner = RuleBasedReasoner()
reasoner.add_rule(['cup_on_table', 'push_cup'], 'cup_falls')
reasoner.add_rule(['cup_falls', 'cup_contains_water'], 'water_spills')

facts = {'cup_on_table', 'push_cup', 'cup_contains_water'}
conclusions = reasoner.reason(facts)
print(f"推导的结论: {conclusions}")  # {'cup_falls', 'water_spills', ...}
```

### 4.2 基于神经网络的推理

```python
import torch
import torch.nn as nn

class NeuralReasoner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.reasoning_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, facts):
        """
        前向传播
        
        参数:
            facts: 事实向量 [batch, input_dim]
        
        返回:
            结论概率 [batch, output_dim]
        """
        # 编码
        encoded = self.encoder(facts)
        
        # 推理
        reasoning = self.reasoning_net(encoded)
        
        # 解码
        conclusions = self.decoder(reasoning)
        
        return conclusions
    
    def multi_step_reasoning(self, facts, num_steps=3):
        """
        多步推理
        
        参数:
            facts: 事实向量
            num_steps: 推理步数
        
        返回:
            每一步的结论
        """
        encoded = self.encoder(facts)
        conclusions = []
        
        for _ in range(num_steps):
            reasoning = self.reasoning_net(encoded)
            conclusion = self.decoder(reasoning)
            conclusions.append(conclusion)
            encoded = reasoning
        
        return conclusions

# 测试
model = NeuralReasoner(input_dim=10, hidden_dim=64, output_dim=5)
facts = torch.randn(1, 10)

conclusions = model(facts)
print(f"结论形状: {conclusions.shape}")  # [1, 5]

multi_step = model.multi_step_reasoning(facts, num_steps=3)
print(f"多步推理结论数量: {len(multi_step)}")  # 3
```

### 4.3 混合推理方法

```python
class HybridReasoner:
    def __init__(self, rule_reasoner, neural_reasoner):
        self.rule_reasoner = rule_reasoner
        self.neural_reasoner = neural_reasoner
    
    def reason(self, facts, rule_based=True):
        """
        混合推理
        
        参数:
            facts: 事实
            rule_based: 是否优先使用规则
        
        返回:
            推理结果
        """
        results = {}
        
        if rule_based:
            # 先尝试规则推理
            rule_results = self.rule_reasoner.reason(facts)
            results['rule_based'] = rule_results
        
        # 神经网络推理
        neural_results = self.neural_reasoner(facts)
        results['neural'] = neural_results
        
        # 融合结果
        if 'rule_based' in results:
            # 规则结果优先
            final_results = results['rule_based']
        else:
            final_results = results['neural']
        
        return final_results, results

# 测试
rule_reasoner = RuleBasedReasoner()
rule_reasoner.add_rule(['A'], 'B')
rule_reasoner.add_rule(['B'], 'C')

neural_reasoner = NeuralReasoner(input_dim=3, hidden_dim=16, output_dim=3)

hybrid_reasoner = HybridReasoner(rule_reasoner, neural_reasoner)
facts = {'A'}
final, all_results = hybrid_reasoner.reason(facts)
print(f"最终结果: {final}")
```

---

## 5. 实践练习

### 练习1：实现空间推理系统

```python
import numpy as np

class SpatialReasoningSystem:
    def __init__(self):
        self.objects = {}
    
    def add_object(self, name, position, size):
        """
        添加物体
        
        参数:
            name: 物体名称
            position: 位置 [x, y, z]
            size: 尺寸 [width, height, depth]
        """
        self.objects[name] = {
            'position': np.array(position),
            'size': np.array(size)
        }
    
    def query(self, query_type, obj1, obj2=None):
        """
        查询空间关系
        
        参数:
            query_type: 查询类型
            obj1: 物体1
            obj2: 物体2（可选）
        
        返回:
            查询结果
        """
        if query_type == 'position':
            return self.objects[obj1]['position']
        
        elif query_type == 'distance':
            if obj2 is None:
                return None
            pos1 = self.objects[obj1]['position']
            pos2 = self.objects[obj2]['position']
            return np.linalg.norm(pos1 - pos2)
        
        elif query_type == 'on':
            return self._check_on(obj1, obj2)
        
        elif query_type == 'inside':
            return self._check_inside(obj1, obj2)
        
        return None
    
    def _check_on(self, obj1, obj2):
        """检查obj1在obj2上"""
        o1 = self.objects[obj1]
        o2 = self.objects[obj2]
        
        # z方向上
        z_condition = o1['position'][2] >= o2['position'][2] + o2['size'][2]
        
        # xy方向上在范围内
        x_condition = (o1['position'][0] >= o2['position'][0] and
                      o1['position'][0] + o1['size'][0] <= o2['position'][0] + o2['size'][0])
        y_condition = (o1['position'][1] >= o2['position'][1] and
                      o1['position'][1] + o1['size'][1] <= o2['position'][1] + o2['size'][1])
        
        return z_condition and x_condition and y_condition
    
    def _check_inside(self, obj1, obj2):
        """检查obj1在obj2内部"""
        o1 = self.objects[obj1]
        o2 = self.objects[obj2]
        
        for i in range(3):
            if not (o1['position'][i] >= o2['position'][i] and
                    o1['position'][i] + o1['size'][i] <= o2['position'][i] + o2['size'][i]):
                return False
        return True

# 测试
system = SpatialReasoningSystem()
system.add_object('table', [0, 0, 0], [1, 1, 0.5])
system.add_object('cup', [0.5, 0.5, 0.5], [0.1, 0.1, 0.1])

print(f"杯子在桌子上: {system.query('on', 'cup', 'table')}")  # True
print(f"杯子位置: {system.query('position', 'cup')}")
```

### 练习2：实现任务规划器

```python
class TaskPlanner:
    def __init__(self):
        self.primitives = {
            'pick': {'preconditions': ['reachable', 'graspable'], 'effects': ['holding']},
            'place': {'preconditions': ['holding', 'place_reachable'], 'effects': ['placed']},
            'move': {'preconditions': ['path_clear'], 'effects': ['moved']},
            'open': {'preconditions': ['reachable', 'closed'], 'effects': ['opened']},
            'close': {'preconditions': ['reachable', 'opened'], 'effects': ['closed']}
        }
    
    def plan(self, goal, initial_state):
        """
        规划任务
        
        参数:
            goal: 目标状态
            initial_state: 初始状态
        
        返回:
            动作序列
        """
        plan = []
        current_state = set(initial_state)
        target_state = set(goal)
        
        # 简单的向前规划
        while not target_state.issubset(current_state):
            action_found = False
            
            for action, info in self.primitives.items():
                # 检查前置条件
                if all(cond in current_state for cond in info['preconditions']):
                    # 检查是否能达到目标
                    if any(effect in target_state for effect in info['effects']):
                        plan.append(action)
                        current_state.update(info['effects'])
                        action_found = True
                        break
            
            if not action_found:
                # 无法继续规划
                break
        
        return plan
    
    def hierarchical_plan(self, goal, initial_state):
        """
        层次化规划
        
        参数:
            goal: 目标
            initial_state: 初始状态
        
        返回:
            层次化计划
        """
        high_level_plan = []
        low_level_plan = []
        
        # 高层规划
        high_level_actions = self.plan(goal, initial_state)
        
        # 将高层动作分解为低层动作
        for action in high_level_actions:
            high_level_plan.append(action)
            if action == 'pick':
                low_level_plan.extend(['move_to_object', 'grasp', 'lift'])
            elif action == 'place':
                low_level_plan.extend(['move_to_location', 'release'])
            elif action == 'move':
                low_level_plan.extend(['plan_path', 'execute_path'])
        
        return high_level_plan, low_level_plan

# 测试
planner = TaskPlanner()
goal = ['placed']
initial_state = ['reachable', 'graspable', 'place_reachable']

plan = planner.plan(goal, initial_state)
print(f"动作序列: {plan}")  # ['pick', 'place']

high_level, low_level = planner.hierarchical_plan(goal, initial_state)
print(f"高层计划: {high_level}")
print(f"低层计划: {low_level}")
```

### 练习3：实现因果推理系统

```python
class CausalInferenceSystem:
    def __init__(self):
        self.causal_graph = {}
    
    def add_causal_relation(self, cause, effect, strength=1.0):
        """
        添加因果关系
        
        参数:
            cause: 原因
            effect: 效果
            strength: 因果强度
        """
        if cause not in self.causal_graph:
            self.causal_graph[cause] = {}
        self.causal_graph[cause][effect] = strength
    
    def predict_effects(self, cause, max_depth=2):
        """
        预测效果
        
        参数:
            cause: 原因
            max_depth: 最大深度
        
        返回:
            效果列表
        """
        effects = []
        visited = set()
        
        def dfs(current_cause, depth):
            if depth > max_depth or current_cause in visited:
                return
            
            visited.add(current_cause)
            
            if current_cause in self.causal_graph:
                for effect, strength in self.causal_graph[current_cause].items():
                    effects.append((effect, strength, depth))
                    dfs(effect, depth + 1)
        
        dfs(cause, 0)
        return effects
    
    def infer_causes(self, effect, threshold=0.5):
        """
        推断原因
        
        参数:
            effect: 效果
            threshold: 阈值
        
        返回:
            可能的原因
        """
        possible_causes = []
        
        for cause, effects in self.causal_graph.items():
            if effect in effects and effects[effect] >= threshold:
                possible_causes.append((cause, effects[effect]))
        
        # 按强度排序
        possible_causes.sort(key=lambda x: x[1], reverse=True)
        return possible_causes
    
    def compute_causal_path(self, start, end):
        """
        计算因果路径
        
        参数:
            start: 起点
            end: 终点
        
        返回:
            因果路径
        """
        from collections import deque
        
        queue = deque([(start, [])])
        visited = set()
        
        while queue:
            current, path = queue.popleft()
            
            if current == end:
                return path + [current]
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current in self.causal_graph:
                for effect in self.causal_graph[current]:
                    queue.append((effect, path + [current]))
        
        return None

# 测试
system = CausalInferenceSystem()
system.add_causal_relation('push', 'move', 0.9)
system.add_causal_relation('move', 'fall', 0.7)
system.add_causal_relation('fall', 'break', 0.8)
system.add_causal_relation('drop', 'fall', 0.95)

effects = system.predict_effects('push')
print("推的效果:")
for effect, strength, depth in effects:
    print(f"  {effect} (强度: {strength}, 深度: {depth})")

causes = system.infer_causes('fall')
print("掉落的原因:")
for cause, strength in causes:
    print(f"  {cause} (强度: {strength})")

path = system.compute_causal_path('push', 'break')
print(f"因果路径: {path}")
```

---

**下一节**：[机器人操控](04-robot-manipulation.md)

---

## 参考文献

1. Tenenbaum, J. B., et al. (2011). How to Grow a Mind: Statistics, Structure, and Abstraction.
2. Lake, B. M., et al. (2017). Building Machines That Learn and Think Like People.
3. Battaglia, P., et al. (2018). Relational inductive biases, deep learning, and graph networks.