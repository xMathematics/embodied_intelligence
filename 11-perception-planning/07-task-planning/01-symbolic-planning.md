# 7.1 符号规划

## 目录

- [1. 引言](#1-引言)
- [2. STRIPS](#2-strips)
- [3. 状态空间搜索](#3-状态空间搜索)
- [4. 规划图](#4-规划图)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 任务规划 vs 运动规划

|  | 任务规划 | 运动规划 |
|--|--------|--------|
| 抽象层级 | 符号级 | 几何级 |
| 时间 | 数秒/分钟 | 数毫秒 |
| 示例 | "去厨房拿水" | "从A移动到B" |

```python
import numpy as np
from collections import defaultdict, deque
```

---

## 2. STRIPS

### 2.1 经典表示

```python
class Predicate:
    """谓词"""
    
    def __init__(self, name, args):
        self.name = name
        self.args = tuple(args)
    
    def __hash__(self):
        return hash((self.name, self.args))
    
    def __eq__(self, other):
        return self.name == other.name and self.args == other.args
    
    def __repr__(self):
        return f"{self.name}({', '.join(self.args)})"


class State:
    """状态（谓词集合）"""
    
    def __init__(self, predicates):
        self.preds = set(predicates)
    
    def holds(self, predicate):
        """检查是否成立"""
        return predicate in self.preds
    
    def __eq__(self, other):
        return self.preds == other.preds
    
    def __hash__(self):
        return hash(frozenset(self.preds))
    
    def __repr__(self):
        return str(self.preds)


class Action:
    """动作（STRIPS表示）"""
    
    def __init__(self, name, parameters, preconditions, add_list, delete_list):
        self.name = name
        self.parameters = parameters
        self.preconditions = preconditions
        self.add_list = add_list
        self.delete_list = delete_list
    
    def is_applicable(self, state):
        """检查是否可应用"""
        for pre in self.preconditions:
            if not state.holds(pre):
                return False
        return True
    
    def apply(self, state):
        """应用动作"""
        if not self.is_applicable(state):
            return None
        
        new_preds = set(state.preds)
        for p in self.delete_list:
            if p in new_preds:
                new_preds.remove(p)
        for p in self.add_list:
            new_preds.add(p)
        
        return State(new_preds)
    
    def __repr__(self):
        return f"{self.name}({', '.join(self.parameters)})"
```

### 2.2 示例：积木世界

```python
def create_blocks_world():
    """创建积木世界示例"""
    
    # 动作
    def move(x, from_y, to_z):
        pre = [
            Predicate('on', [x, from_y]),
            Predicate('clear', [x]),
            Predicate('clear', [to_z])
        ]
        add = [
            Predicate('on', [x, to_z]),
            Predicate('clear', [from_y])
        ]
        delete = [
            Predicate('on', [x, from_y]),
            Predicate('clear', [to_z])
        ]
        return Action('move', [x, from_y, to_z], pre, add, delete)
    
    def move_to_table(x, from_y):
        pre = [
            Predicate('on', [x, from_y]),
            Predicate('clear', [x])
        ]
        add = [
            Predicate('on', [x, 'table']),
            Predicate('clear', [from_y])
        ]
        delete = [
            Predicate('on', [x, from_y])
        ]
        return Action('move_to_table', [x, from_y], pre, add, delete)
    
    # 初始状态
    initial = State([
        Predicate('on', ['A', 'table']),
        Predicate('on', ['B', 'table']),
        Predicate('on', ['C', 'A']),
        Predicate('clear', ['B']),
        Predicate('clear', ['C'])
    ])
    
    # 目标状态
    goal = State([
        Predicate('on', ['A', 'B']),
        Predicate('on', ['B', 'C']),
        Predicate('on', ['C', 'table']),
        Predicate('clear', ['A'])
    ])
    
    # 动作模板
    blocks = ['A', 'B', 'C']
    actions = []
    for x in blocks:
        for y in blocks + ['table']:
            for z in blocks:
                if x != y and x != z and y != z:
                    actions.append(move(x, y, z))
            if y in blocks:
                actions.append(move_to_table(x, y))
    
    return initial, goal, actions
```

---

## 3. 状态空间搜索

### 3.1 BFS规划

```python
class BFSPlanner:
    """广度优先搜索规划器"""
    
    def __init__(self, actions):
        self.actions = actions
    
    def plan(self, initial, goal):
        """规划"""
        queue = deque([(initial, [])])
        visited = set([initial])
        
        while queue:
            state, path = queue.popleft()
            
            if self.is_goal(state, goal):
                return path
            
            for action in self.actions:
                if action.is_applicable(state):
                    new_state = action.apply(state)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state, path + [action]))
        
        return None
    
    def is_goal(self, state, goal):
        """检查是否到达目标"""
        for p in goal.preds:
            if not state.holds(p):
                return False
        return True
```

### 3.2 A*规划

```python
class AStarPlanner:
    """A*搜索规划器"""
    
    def __init__(self, actions, heuristic=None):
        self.actions = actions
        self.heuristic = heuristic if heuristic else self._default_heuristic
    
    def _default_heuristic(self, state, goal):
        """默认启发式：未满足的目标数"""
        count = 0
        for p in goal.preds:
            if not state.holds(p):
                count += 1
        return count
    
    def plan(self, initial, goal):
        """规划"""
        import heapq
        
        open_list = []
        heapq.heappush(open_list, (0, 0, initial, []))
        
        g_score = {initial: 0}
        
        while open_list:
            f, g, state, path = heapq.heappop(open_list)
            
            if self._is_goal(state, goal):
                return path
            
            for action in self.actions:
                if action.is_applicable(state):
                    new_state = action.apply(state)
                    new_g = g + 1
                    
                    if new_state not in g_score or new_g < g_score.get(new_state, float('inf')):
                        g_score[new_state] = new_g
                        h = self.heuristic(new_state, goal)
                        f = new_g + h
                        heapq.heappush(open_list, (f, new_g, new_state, path + [action]))
        
        return None
    
    def _is_goal(self, state, goal):
        """目标检查"""
        for p in goal.preds:
            if not state.holds(p):
                return False
        return True
```

---

## 4. 规划图

### 4.1 Graphplan

```python
class PlanningGraph:
    """规划图"""
    
    def __init__(self, actions):
        self.actions = actions
        self.layer_states = []
        self.layer_actions = []
        self.mutexes = []
    
    def build(self, initial, max_layers=10):
        """构建规划图"""
        self.layer_states = [initial.preds.copy()]
        self.layer_actions = []
        self.mutexes = []
        
        for i in range(max_layers):
            # 动作层
            applicable = []
            for a in self.actions:
                if self._is_action_applicable(a, self.layer_states[i]):
                    applicable.append(a)
            
            self.layer_actions.append(applicable)
            
            # 互斥
            mutex_pairs = self._compute_mutexes(applicable, self.layer_states[i])
            self.mutexes.append(mutex_pairs)
            
            # 下一层状态
            next_preds = set(self.layer_states[i])
            for a in applicable:
                for p in a.add_list:
                    next_preds.add(p)
            
            # 检查是否饱和
            if next_preds == self.layer_states[i]:
                break
            
            self.layer_states.append(next_preds)
    
    def _is_action_applicable(self, action, preds):
        """动作是否可应用"""
        for pre in action.preconditions:
            if pre not in preds:
                return False
        return True
    
    def _compute_mutexes(self, actions, state):
        """计算互斥"""
        mutexes = []
        # 简化：假设没有冲突
        return mutexes
    
    def extract_solution(self, goal):
        """解提取（简化）"""
        # 完整的Graphplan需要回溯，这里简化
        return None
```

---

## 5. 实践练习

### 练习1：积木世界规划

```python
def exercise_blocks_planning():
    """积木世界练习"""
    print("=== 符号规划 (积木世界) ===")
    
    initial, goal, actions = create_blocks_world()
    
    print("\n初始状态:")
    print(initial)
    print("\n目标状态:")
    print(goal)
    
    # BFS
    print("\n=== BFS规划 ===")
    planner_bfs = BFSPlanner(actions)
    plan_bfs = planner_bfs.plan(initial, goal)
    
    if plan_bfs:
        print(f"找到计划，{len(plan_bfs)}步:")
        for i, action in enumerate(plan_bfs):
            print(f"  {i+1}. {action}")
    else:
        print("未找到计划")
    
    # A*
    print("\n=== A*规划 ===")
    planner_astar = AStarPlanner(actions)
    plan_astar = planner_astar.plan(initial, goal)
    
    if plan_astar:
        print(f"找到计划，{len(plan_astar)}步:")
        for i, action in enumerate(plan_astar):
            print(f"  {i+1}. {action}")
    else:
        print("未找到计划")

# exercise_blocks_planning()
```

---

**下一节**：[分层任务网络](02-htn-planning.md)

---

## 参考文献

1. Fikes, R. E., & Nilsson, N. J. (1971). STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving.
2. Ghallab, M., et al. (2004). Automated Planning: Theory & Practice.
3. Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.).
4. Blum, A. L., & Furst, M. L. (1997). Fast Planning Through Planning Graph Analysis.
5. Weld, D. S. (1999). Recent Advances in AI Planning.
