# 7.2 分层任务网络 (HTN)

## 目录

- [1. 引言](#1-引言)
- [2. 任务和方法](#2-任务和方法)
- [3. HTN规划器](#3-htn规划器)
- [4. 示例](#4-示例)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 HTN vs STRIPS

| | STRIPS | HTN |
|--|------|-----|
| 操作 | 原子动作 | 任务分解 |
| 控制 | 无 | 有（领域知识） |
| 效率 | 低 | 高（利用知识） |

```python
import numpy as np
from collections import defaultdict
```

---

## 2. 任务和方法

### 2.1 任务表示

```python
class Task:
    """任务"""
    
    def __init__(self, name, args):
        self.name = name
        self.args = tuple(args)
        self.is_primitive = False
    
    def __eq__(self, other):
        return self.name == other.name and self.args == other.args
    
    def __hash__(self):
        return hash((self.name, self.args))
    
    def __repr__(self):
        return f"{self.name}({', '.join(self.args)})"


class PrimitiveTask(Task):
    """原始任务（不可分解）"""
    
    def __init__(self, name, args, preconditions, effects):
        super().__init__(name, args)
        self.is_primitive = True
        self.preconditions = preconditions
        self.effects = effects


class CompoundTask(Task):
    """复合任务（可分解）"""
    
    def __init__(self, name, args):
        super().__init__(name, args)
        self.is_primitive = False


class Method:
    """分解方法"""
    
    def __init__(self, name, task, preconditions, subtasks):
        self.name = name
        self.task = task  # 要分解的复合任务
        self.preconditions = preconditions
        self.subtasks = subtasks  # 子任务列表
    
    def is_applicable(self, state):
        """方法是否适用"""
        for pre in self.preconditions:
            if not state.holds(pre):
                return False
        return True
```

---

## 3. HTN规划器

### 3.1 全序HTN规划器

```python
class TotalOrderHTNPlanner:
    """全序HTN规划器"""
    
    def __init__(self, methods, actions):
        self.methods = methods  # 字典：任务 -> [方法列表]
        self.actions = actions  # 动作字典
    
    def plan(self, state, tasks):
        """规划"""
        final_plan = []
        if self._decompose(state, tasks, [], final_plan):
            return final_plan
        return None
    
    def _decompose(self, state, tasks, current_plan, final_plan):
        """递归分解"""
        if not tasks:
            final_plan.extend(current_plan)
            return True
        
        task = tasks[0]
        remaining_tasks = tasks[1:]
        
        if task.is_primitive:
            # 原始任务：查找动作
            action = self._find_action(task)
            if action and action.is_applicable(state):
                new_state = action.apply(state)
                if self._decompose(new_state, remaining_tasks, 
                                    current_plan + [action], final_plan):
                    return True
        else:
            # 复合任务：尝试方法
            if task in self.methods:
                for method in self.methods[task]:
                    if method.is_applicable(state):
                        new_tasks = method.subtasks + remaining_tasks
                        if self._decompose(state, new_tasks, current_plan, final_plan):
                            return True
        
        return False
    
    def _find_action(self, task):
        """查找匹配的动作（简化）"""
        for a in self.actions:
            if a.name == task.name:
                return a
        return None
```

---

## 4. 示例

### 4.1 旅行HTN

```python
from .01-symbolic-planning import Predicate, State, Action


def create_travel_htn():
    """旅行HTN示例"""
    
    # 原始任务/动作
    go_by_taxi = PrimitiveTask('Go', ['?from', '?to'], 
                              [], [])  # 简化
    
    walk = PrimitiveTask('Walk', ['?from', '?to'], 
                        [Predicate('LessThan', ['?distance', '1km'])], 
                        [])
    
    # 复合任务
    travel = CompoundTask('Travel', ['?from', '?to'])
    
    # 方法
    method_taxi = Method(
        name='TravelByTaxi',
        task=travel,
        preconditions=[Predicate('LongDistance', ['?from', '?to'])],
        subtasks=[
            PrimitiveTask('CallTaxi', ['?from'], [], []),
            go_by_taxi
        ]
    )
    
    method_walk = Method(
        name='TravelByWalk',
        task=travel,
        preconditions=[Predicate('ShortDistance', ['?from', '?to'])],
        subtasks=[walk]
    )
    
    methods = {
        travel: [method_taxi, method_walk]
    }
    
    # 状态
    initial = State([
        Predicate('At', ['me', 'home']),
        Predicate('ShortDistance', ['home', 'park']),
        Predicate('LongDistance', ['park', 'airport'])
    ])
    
    # 目标任务
    goal_tasks = [CompoundTask('Travel', ['home', 'airport'])]
    
    return methods, [walk, go_by_taxi], initial, goal_tasks
```

---

## 5. 实践练习

### 练习1：简单HTN示例

```python
def exercise_htn():
    """HTN练习"""
    print("=== 分层任务网络 (HTN) ===")
    
    # 创建积木世界HTN
    from .01-symbolic-planning import create_blocks_world
    
    initial, goal, primitive_actions = create_blocks_world()
    
    # 创建复合任务
    stack_ab = CompoundTask('StackAB', [])
    
    # 方法：先移动A到桌上，再移动A到B上
    # 简化演示，实际需要更完整的HTN定义
    print("HTN规划演示:")
    print("- 复合任务: StackAB")
    print("- 分解: MoveA -> StackAB")
    
    # 运行符号规划来演示
    from .01-symbolic-planning import BFSPlanner
    
    planner = BFSPlanner(primitive_actions)
    plan = planner.plan(initial, goal)
    
    if plan:
        print("\n符号规划得到的计划可以作为HTN的原语:")
        for i, action in enumerate(plan):
            print(f"  {i+1}. {action}")

# exercise_htn()
```

---

**下一节**：[规划域定义](03-pddl.md)

---

## 参考文献

1. Erol, K., et al. (1994). HTN Planning: Complexity and Expressivity.
2. Nau, D. S., et al. (2003). SHOP2: An HTN Planning System.
3. Ghallab, M., et al. (2004). Automated Planning: Theory & Practice.
4. Sacerdoti, E. D. (1975). The Nonlinear Nature of Plans.
5. Wilkins, D. E. (1988). Practical Planning: Extending the Classical AI Planning Paradigm.
