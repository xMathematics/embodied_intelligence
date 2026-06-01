# 7.5 具身规划

## 目录

- [1. 引言](#1-引言)
- [2. 具身规划概述](#2-具身规划概述)
- [3. 规划方法](#3-规划方法)
- [4. 规划架构](#4-规划架构)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 具身规划的重要性

**具身规划**是指智能体在物理环境中规划行动序列以实现目标的能力。这是实现复杂具身任务的关键，包括任务规划、运动规划、路径规划等。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **任务规划** | 规划任务步骤 | "做早餐"的步骤 |
| **路径规划** | 规划运动路径 | 从A点到B点的路径 |
| **运动规划** | 规划关节运动 | 机械臂的运动轨迹 |
| **多智能体规划** | 多个智能体协作 | 多机器人协作 |

---

## 2. 具身规划概述

### 2.1 定义

**具身规划**：在给定初始状态和目标的情况下，找到从初始状态到目标状态的动作序列。

**形式化表达**：
```
Planning: (InitialState, Goal) → ActionSequence
```

### 2.2 规划层次

| 层次 | 时间尺度 | 内容 |
|------|---------|------|
| **任务规划** | 分钟到小时 | 任务分解和排序 |
| **行为规划** | 秒到分钟 | 行为选择 |
| **运动规划** | 毫秒到秒 | 关节轨迹生成 |

---

## 3. 规划方法

### 3.1 基于搜索的规划

**A*算法**：

```python
import heapq

class AStarPlanner:
    def __init__(self, heuristic):
        self.heuristic = heuristic
    
    def plan(self, start, goal, get_neighbors, get_cost):
        """
        A*规划
        
        参数:
            start: 起始状态
            goal: 目标状态
            get_neighbors: 获取邻居状态的函数
            get_cost: 计算代价的函数
        
        返回:
            路径
        """
        # 优先队列: (f, g, state, path)
        open_set = []
        heapq.heappush(open_set, (0, 0, start, []))
        
        # 已访问状态
        closed_set = set()
        
        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            
            # 检查是否到达目标
            if current == goal:
                return path + [current]
            
            # 跳过已访问
            if current in closed_set:
                continue
            closed_set.add(current)
            
            # 探索邻居
            for neighbor in get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # 计算代价
                new_g = g + get_cost(current, neighbor)
                h = self.heuristic(neighbor, goal)
                f = new_g + h
                
                heapq.heappush(open_set, (f, new_g, neighbor, path + [current]))
        
        return None  # 无解

# 测试
def euclidean_heuristic(state1, state2):
    """欧几里得距离启发式"""
    return ((state1[0] - state2[0])**2 + (state1[1] - state2[1])**2)**0.5

def get_neighbors(state):
    """获取邻居（4连通）"""
    x, y = state
    neighbors = [
        (x+1, y), (x-1, y),
        (x, y+1), (x, y-1)
    ]
    return neighbors

def get_cost(state1, state2):
    """计算代价"""
    return 1  # 均匀代价

planner = AStarPlanner(euclidean_heuristic)
path = planner.plan((0, 0), (5, 5), get_neighbors, get_cost)
print(f"路径长度: {len(path) if path else 0}")
```

### 3.2 基于采样的规划

**RRT算法**：

```python
import random
import numpy as np

class RRTPlanner:
    def __init__(self, step_size=0.5, max_iterations=1000):
        self.step_size = step_size
        self.max_iterations = max_iterations
    
    def plan(self, start, goal, sample_free, is_collision_free):
        """
        RRT规划
        
        参数:
            start: 起始状态
            goal: 目标状态
            sample_free: 采样自由空间的函数
            is_collision_free: 检查碰撞的函数
        
        返回:
            路径
        """
        # 树结构
        tree = {start: None}
        
        for _ in range(self.max_iterations):
            # 随机采样
            if random.random() < 0.1:  # 10%概率采样目标
                sample = goal
            else:
                sample = sample_free()
            
            # 找到最近节点
            nearest = self._find_nearest(tree, sample)
            
            # 扩展
            new_node = self._steer(nearest, sample)
            
            # 检查碰撞
            if is_collision_free(nearest, new_node):
                tree[new_node] = nearest
                
                # 检查是否到达目标
                if self._distance(new_node, goal) < self.step_size:
                    tree[goal] = new_node
                    return self._extract_path(tree, goal)
        
        return None  # 无解
    
    def _find_nearest(self, tree, sample):
        """找到最近节点"""
        nearest = None
        min_dist = float('inf')
        
        for node in tree:
            dist = self._distance(node, sample)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_node, to_node):
        """向目标方向扩展"""
        direction = np.array(to_node) - np.array(from_node)
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return to_node
        else:
            direction = direction / distance * self.step_size
            return tuple(np.array(from_node) + direction)
    
    def _distance(self, node1, node2):
        """计算距离"""
        return np.linalg.norm(np.array(node1) - np.array(node2))
    
    def _extract_path(self, tree, goal):
        """提取路径"""
        path = [goal]
        current = goal
        
        while current is not None:
            path.append(current)
            current = tree[current]
        
        path.reverse()
        return path

# 测试
def sample_free():
    """采样自由空间"""
    return (random.uniform(0, 10), random.uniform(0, 10))

def is_collision_free(node1, node2):
    """检查碰撞（简化版）"""
    return True  # 假设无碰撞

planner = RRTPlanner(step_size=0.5)
path = planner.plan((0, 0), (9, 9), sample_free, is_collision_free)
print(f"路径长度: {len(path) if path else 0}")
```

### 3.3 基于学习的规划

**使用神经网络进行规划**：

```python
import torch
import torch.nn as nn

class LearnedPlanner(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 目标编码器
        self.goal_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 规划网络
        self.planning_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state, goal):
        """
        前向传播
        
        参数:
            state: 当前状态 [batch, state_dim]
            goal: 目标状态 [batch, state_dim]
        
        返回:
            动作 [batch, action_dim]
        """
        state_feat = self.state_encoder(state)
        goal_feat = self.goal_encoder(goal)
        
        combined = torch.cat([state_feat, goal_feat], dim=-1)
        action = self.planning_net(combined)
        
        return action
    
    def plan(self, start, goal, horizon=10):
        """
        规划动作序列
        
        参数:
            start: 起始状态
            goal: 目标状态
            horizon: 规划步数
        
        返回:
            动作序列
        """
        actions = []
        current_state = torch.FloatTensor(start).unsqueeze(0)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0)
        
        for _ in range(horizon):
            with torch.no_grad():
                action = self.forward(current_state, goal_tensor)
            actions.append(action.squeeze().numpy())
            
            # 简化：假设动作直接改变状态
            current_state = current_state + action
            
            # 检查是否到达目标
            distance = torch.norm(current_state - goal_tensor)
            if distance < 0.1:
                break
        
        return np.array(actions)

# 测试
planner = LearnedPlanner(state_dim=4, action_dim=2)
start = np.array([0, 0, 0, 0])
goal = np.array([5, 5, 0, 0])

actions = planner.plan(start, goal, horizon=20)
print(f"动作序列形状: {actions.shape}")  # [horizon, 2]
```

---

## 4. 规划架构

### 4.1 层次化规划

```python
class HierarchicalPlanner:
    def __init__(self):
        self.task_planner = None
        self.motion_planner = None
    
    def set_task_planner(self, planner):
        """设置任务规划器"""
        self.task_planner = planner
    
    def set_motion_planner(self, planner):
        """设置运动规划器"""
        self.motion_planner = planner
    
    def plan(self, start, goal):
        """
        层次化规划
        
        参数:
            start: 起始状态
            goal: 目标状态
        
        返回:
            完整规划
        """
        # 任务规划
        subgoals = self.task_planner.plan(start, goal)
        
        # 运动规划
        trajectory = []
        current = start
        
        for subgoal in subgoals:
            path = self.motion_planner.plan(current, subgoal)
            if path:
                trajectory.extend(path)
                current = subgoal
            else:
                print(f"无法规划到子目标: {subgoal}")
                return None
        
        return trajectory
```

### 4.2 反应式规划

```python
class ReactivePlanner:
    def __init__(self, policy):
        self.policy = policy
    
    def plan(self, start, goal, environment, max_steps=100):
        """
        反应式规划
        
        参数:
            start: 起始状态
            goal: 目标状态
            environment: 环境
            max_steps: 最大步数
        
        返回:
            轨迹
        """
        trajectory = [start]
        current_state = start
        
        for step in range(max_steps):
            # 获取观测
            observation = environment.get_observation(current_state)
            
            # 策略选择动作
            action = self.policy.select_action(observation, goal)
            
            # 执行动作
            next_state = environment.step(current_state, action)
            
            trajectory.append(next_state)
            current_state = next_state
            
            # 检查是否到达目标
            if self._is_goal_reached(current_state, goal):
                break
        
        return trajectory
    
    def _is_goal_reached(self, current, goal, threshold=0.1):
        """检查是否到达目标"""
        distance = np.linalg.norm(np.array(current) - np.array(goal))
        return distance < threshold
```

### 4.3 混合规划

```python
class HybridPlanner:
    def __init__(self, global_planner, local_planner):
        self.global_planner = global_planner
        self.local_planner = local_planner
    
    def plan(self, start, goal, environment):
        """
        混合规划
        
        参数:
            start: 起始状态
            goal: 目标状态
            environment: 环境
        
        返回:
            轨迹
        """
        # 全局规划
        global_path = self.global_planner.plan(start, goal)
        
        if not global_path:
            return None
        
        # 局部规划
        trajectory = []
        current_state = start
        
        for waypoint in global_path[1:]:
            local_path = self.local_planner.plan(
                current_state,
                waypoint,
                environment
            )
            
            if local_path:
                trajectory.extend(local_path)
                current_state = waypoint
            else:
                print(f"局部规划失败: {current_state} -> {waypoint}")
                return None
        
        return trajectory
```

---

## 5. 实践练习

### 练习1：实现任务规划器

```python
class TaskPlanner:
    def __init__(self):
        self.actions = {}
        self.preconditions = {}
        self.effects = {}
    
    def add_action(self, name, preconditions, effects):
        """
        添加动作
        
        参数:
            name: 动作名称
            preconditions: 前置条件
            effects: 效果
        """
        self.actions[name] = {
            'preconditions': set(preconditions),
            'effects': set(effects)
        }
    
    def plan(self, initial_state, goal_state):
        """
        任务规划（简化版的前向搜索）
        
        参数:
            initial_state: 初始状态
            goal_state: 目标状态
        
        返回:
            动作序列
        """
        current_state = set(initial_state)
        goal = set(goal_state)
        plan = []
        
        max_iterations = 100
        iteration = 0
        
        while not goal.issubset(current_state) and iteration < max_iterations:
            action_found = False
            
            for action_name, action_info in self.actions.items():
                # 检查前置条件
                if action_info['preconditions'].issubset(current_state):
                    # 检查是否能达到目标
                    if any(effect in goal for effect in action_info['effects']):
                        # 执行动作
                        current_state.update(action_info['effects'])
                        plan.append(action_name)
                        action_found = True
                        break
            
            if not action_found:
                print("无法继续规划")
                return None
            
            iteration += 1
        
        return plan if goal.issubset(current_state) else None

# 测试
planner = TaskPlanner()

# 定义动作
planner.add_action('pick', ['hand_empty', 'graspable'], ['holding'])
planner.add_action('place', ['holding', 'place_reachable'], ['placed'])
planner.add_action('move_to_object', ['hand_empty'], ['graspable'])
planner.add_action('move_to_place', ['holding'], ['place_reachable'])

# 规划
initial_state = ['hand_empty']
goal_state = ['placed']

plan = planner.plan(initial_state, goal_state)
print(f"任务计划: {plan}")
```

### 练习2：实现运动规划器

```python
import numpy as np

class MotionPlanner:
    def __init__(self, step_size=0.1, max_iterations=1000):
        self.step_size = step_size
        self.max_iterations = max_iterations
    
    def plan(self, start, goal, obstacles=None):
        """
        运动规划（简化的RRT）
        
        参数:
            start: 起始位置
            goal: 目标位置
            obstacles: 障碍物列表
        
        返回:
            路径
        """
        if obstacles is None:
            obstacles = []
        
        # 树结构
        tree = {tuple(start): None}
        
        for _ in range(self.max_iterations):
            # 随机采样
            if np.random.random() < 0.1:
                sample = goal
            else:
                sample = self._sample_free_space(obstacles)
            
            # 找到最近节点
            nearest = self._find_nearest(tree, sample)
            
            # 扩展
            new_node = self._steer(nearest, sample)
            
            # 检查碰撞
            if self._is_collision_free(nearest, new_node, obstacles):
                tree[new_node] = nearest
                
                # 检查是否到达目标
                if np.linalg.norm(np.array(new_node) - np.array(goal)) < self.step_size:
                    tree[tuple(goal)] = new_node
                    return self._extract_path(tree, tuple(goal))
        
        return None
    
    def _sample_free_space(self, obstacles):
        """采样自由空间"""
        return np.random.uniform(-5, 5, 2)
    
    def _find_nearest(self, tree, sample):
        """找到最近节点"""
        nearest = None
        min_dist = float('inf')
        
        for node in tree:
            dist = np.linalg.norm(np.array(node) - np.array(sample))
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_node, to_node):
        """向目标方向扩展"""
        direction = np.array(to_node) - np.array(from_node)
        distance = np.linalg.norm(direction)
        
        if distance <= self.step_size:
            return tuple(to_node)
        else:
            direction = direction / distance * self.step_size
            return tuple(np.array(from_node) + direction)
    
    def _is_collision_free(self, node1, node2, obstacles):
        """检查碰撞"""
        # 简化：假设无碰撞
        return True
    
    def _extract_path(self, tree, goal):
        """提取路径"""
        path = [goal]
        current = goal
        
        while current is not None:
            path.append(current)
            current = tree[current]
        
        path.reverse()
        return path

# 测试
planner = MotionPlanner(step_size=0.5)
start = [0, 0]
goal = [4, 4]

path = planner.plan(start, goal)
print(f"路径长度: {len(path) if path else 0}")
if path:
    print(f"路径: {path[:5]}...")  # 打印前5个点
```

### 练习3：实现完整的具身规划系统

```python
class EmbodiedPlanningSystem:
    def __init__(self, task_planner, motion_planner):
        self.task_planner = task_planner
        self.motion_planner = motion_planner
    
    def plan(self, initial_state, goal, environment):
        """
        完整规划
        
        参数:
            initial_state: 初始状态
            goal: 目标
            environment: 环境
        
        返回:
            完整的执行计划
        """
        # 任务规划
        task_plan = self.task_planner.plan(
            initial_state['task_state'],
            goal['task_state']
        )
        
        if not task_plan:
            print("任务规划失败")
            return None
        
        # 为每个任务步骤规划运动
        execution_plan = []
        current_position = initial_state['position']
        
        for task_step in task_plan:
            # 确定目标位置
            target_position = self._get_target_position(
                task_step,
                goal,
                environment
            )
            
            # 运动规划
            motion_path = self.motion_planner.plan(
                current_position,
                target_position,
                environment.get_obstacles()
            )
            
            if motion_path:
                execution_plan.append({
                    'task': task_step,
                    'path': motion_path
                })
                current_position = target_position
            else:
                print(f"运动规划失败: {task_step}")
                return None
        
        return execution_plan
    
    def _get_target_position(self, task_step, goal, environment):
        """获取任务步骤的目标位置"""
        # 简化：返回目标位置
        return goal.get('position', [0, 0])
    
    def execute(self, plan, environment):
        """
        执行计划
        
        参数:
            plan: 执行计划
            environment: 环境
        
        返回:
            执行结果
        """
        for step in plan:
            print(f"执行任务: {step['task']}")
            
            # 执行路径
            for waypoint in step['path']:
                environment.move_to(waypoint)
            
            # 执行任务动作
            environment.execute_action(step['task'])
        
        return True

# 测试
task_planner = TaskPlanner()
motion_planner = MotionPlanner()

task_planner.add_action('pick', ['hand_empty'], ['holding'])
task_planner.add_action('place', ['holding'], ['placed'])

system = EmbodiedPlanningSystem(task_planner, motion_planner)

initial_state = {
    'task_state': ['hand_empty'],
    'position': [0, 0]
}

goal = {
    'task_state': ['placed'],
    'position': [3, 3]
}

class MockEnvironment:
    def get_obstacles(self):
        return []
    
    def move_to(self, position):
        pass
    
    def execute_action(self, action):
        print(f"  执行动作: {action}")

environment = MockEnvironment()

plan = system.plan(initial_state, goal, environment)
if plan:
    print(f"执行计划包含 {len(plan)} 个步骤")
    result = system.execute(plan, environment)
    print(f"执行结果: {result}")
```

---

**返回**：[具身智能概述](01-embodied-intelligence-overview.md)

---

## 参考文献

1. LaValle, S. M. (2006). Planning Algorithms.
2. Kavraki, L. E., et al. (1996). Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces.
3. Lynch, K. M., & Park, F. C. (2017). Modern Robotics: Mechanics, Planning, and Control.