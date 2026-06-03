# 7.5 具身规划

## 目录

- [1. 引言](#1-引言)
- [2. 具身规划概述](#2-具身规划概述)
- [3. 经典规划算法](#3-经典规划算法)
- [4. 采样-based规划算法](#4-采样-based规划算法)
- [5. 基于学习的规划方法](#5-基于学习的规划方法)
- [6. 规划架构](#6-规划架构)
- [7. 复杂任务规划案例](#7-复杂任务规划案例)
- [8. 规划评估指标](#8-规划评估指标)
- [9. 实践练习](#9-实践练习)
- [10. 总结与展望](#10-总结与展望)

---

## 1. 引言

### 1.1 具身规划的重要性

**具身规划**是指智能体在物理环境中规划行动序列以实现目标的能力。这是实现复杂具身任务的关键，包括任务规划、运动规划、路径规划等。

在具身智能系统中，规划扮演着核心角色：
- **任务分解**：将复杂任务分解为可执行的子任务
- **动作序列生成**：确定完成任务所需的动作顺序
- **资源分配**：合理分配时间、能量等资源
- **风险评估**：评估执行过程中的潜在风险

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **任务规划** | 规划任务步骤 | "做早餐"的步骤 |
| **路径规划** | 规划运动路径 | 从A点到B点的路径 |
| **运动规划** | 规划关节运动 | 机械臂的运动轨迹 |
| **多智能体规划** | 多个智能体协作 | 多机器人协作装配 |
| **人机协作规划** | 人与机器人协作 | 协作搬运重物 |
| **应急规划** | 应对突发情况 | 避障、故障恢复 |

### 1.3 规划系统的核心要求

一个优秀的具身规划系统需要满足以下要求：
1. **完备性**：保证找到可行解（如果存在）
2. **最优性**：找到最优或近似最优解
3. **效率**：在合理时间内完成规划
4. **鲁棒性**：应对环境不确定性
5. **可扩展性**：处理大规模问题

---

## 2. 具身规划概述

### 2.1 定义

**具身规划**：在给定初始状态和目标的情况下，找到从初始状态到目标状态的动作序列。

**形式化表达**：
```
Planning: (InitialState, Goal, Environment) → ActionSequence
```

其中：
- **InitialState**：智能体的初始状态（位置、姿态、内部状态等）
- **Goal**：目标状态或目标描述
- **Environment**：环境模型（障碍物、约束、动态等）
- **ActionSequence**：动作序列（时间上有序的动作集合）

### 2.2 规划层次

| 层次 | 时间尺度 | 内容 | 典型算法 |
|------|---------|------|----------|
| **任务规划** | 分钟到小时 | 任务分解和排序 | STRIPS、PDDL、HTN |
| **行为规划** | 秒到分钟 | 行为选择和协调 | 有限状态机、行为树 |
| **运动规划** | 毫秒到秒 | 关节轨迹生成 | RRT、A*、PRM |
| **控制规划** | 微秒到毫秒 | 低级控制信号 | PID、MPC |

### 2.3 规划问题分类

根据问题特性，规划可以分为：

**按状态空间类型**：
- **离散规划**：状态空间有限且离散（如网格世界）
- **连续规划**：状态空间连续（如机器人关节角度）
- **混合规划**：同时包含离散和连续状态

**按环境信息**：
- **确定性规划**：环境完全可预测
- **概率规划**：环境存在不确定性
- **在线规划**：边执行边规划

**按时间特性**：
- **瞬时规划**：不考虑时间因素
- **时序规划**：考虑时间约束
- **长期规划**：长期目标的规划

---

## 3. 经典规划算法

### 3.1 广度优先搜索（BFS）

**BFS**是一种盲目搜索算法，按层遍历状态空间。

```python
from collections import deque

class BFSPlanner:
    def __init__(self):
        pass
    
    def plan(self, start, goal, get_neighbors):
        """
        BFS规划
        
        参数:
            start: 起始状态
            goal: 目标状态
            get_neighbors: 获取邻居状态的函数
        
        返回:
            路径
        """
        # 队列: (state, path)
        queue = deque([(start, [start])])
        
        # 已访问状态
        visited = set([start])
        
        while queue:
            current, path = queue.popleft()
            
            # 检查是否到达目标
            if current == goal:
                return path
            
            # 探索邻居
            for neighbor in get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # 无解

# 测试
def get_grid_neighbors(state):
    """获取网格邻居（4连通）"""
    x, y = state
    neighbors = [
        (x+1, y), (x-1, y),
        (x, y+1), (x, y-1)
    ]
    # 边界检查
    return [(nx, ny) for nx, ny in neighbors if 0 <= nx < 10 and 0 <= ny < 10]

planner = BFSPlanner()
path = planner.plan((0, 0), (9, 9), get_grid_neighbors)
print(f"BFS路径长度: {len(path) if path else 0}")
```

**BFS特性**：
- **完备性**：是（有限状态空间）
- **最优性**：是（单位代价）
- **时间复杂度**：O(b^d)，b是分支因子，d是解深度
- **空间复杂度**：O(b^d)

### 3.2 深度优先搜索（DFS）

**DFS**优先探索深度方向。

```python
class DFSPlanner:
    def __init__(self, max_depth=100):
        self.max_depth = max_depth
    
    def plan(self, start, goal, get_neighbors):
        """
        DFS规划
        
        参数:
            start: 起始状态
            goal: 目标状态
            get_neighbors: 获取邻居状态的函数
        
        返回:
            路径
        """
        return self._dfs(start, goal, get_neighbors, [start], 0)
    
    def _dfs(self, current, goal, get_neighbors, path, depth):
        """递归DFS"""
        if depth > self.max_depth:
            return None
        
        if current == goal:
            return path
        
        for neighbor in get_neighbors(current):
            if neighbor not in path:
                result = self._dfs(neighbor, goal, get_neighbors, path + [neighbor], depth + 1)
                if result:
                    return result
        
        return None

# 测试
planner = DFSPlanner(max_depth=50)
path = planner.plan((0, 0), (9, 9), get_grid_neighbors)
print(f"DFS路径长度: {len(path) if path else 0}")
```

**DFS特性**：
- **完备性**：否（可能陷入无限循环）
- **最优性**：否
- **时间复杂度**：O(b^m)，m是最大深度
- **空间复杂度**：O(b*m)

### 3.3 Dijkstra算法

**Dijkstra算法**是一种加权图的最短路径算法。

```python
import heapq

class DijkstraPlanner:
    def __init__(self):
        pass
    
    def plan(self, start, goal, get_neighbors, get_cost):
        """
        Dijkstra规划
        
        参数:
            start: 起始状态
            goal: 目标状态
            get_neighbors: 获取邻居状态的函数
            get_cost: 计算代价的函数
        
        返回:
            路径和总代价
        """
        # 优先队列: (cost, state, path)
        open_set = []
        heapq.heappush(open_set, (0, start, [start]))
        
        # 已访问状态及其代价
        closed_set = {}
        
        while open_set:
            cost, current, path = heapq.heappop(open_set)
            
            # 检查是否到达目标
            if current == goal:
                return path, cost
            
            # 跳过已访问且代价更高的路径
            if current in closed_set and cost >= closed_set[current]:
                continue
            closed_set[current] = cost
            
            # 探索邻居
            for neighbor in get_neighbors(current):
                new_cost = cost + get_cost(current, neighbor)
                
                if neighbor not in closed_set or new_cost < closed_set[neighbor]:
                    heapq.heappush(open_set, (new_cost, neighbor, path + [neighbor]))
        
        return None, None  # 无解

# 测试
def get_weighted_cost(state1, state2):
    """计算加权代价"""
    # 对角线移动代价更高
    dx = abs(state1[0] - state2[0])
    dy = abs(state1[1] - state2[1])
    if dx + dy == 2:  # 对角线
        return 1.414
    return 1.0

planner = DijkstraPlanner()
path, cost = planner.plan((0, 0), (9, 9), get_grid_neighbors, get_weighted_cost)
print(f"Dijkstra路径长度: {len(path) if path else 0}, 代价: {cost:.2f}")
```

**Dijkstra特性**：
- **完备性**：是（非负代价）
- **最优性**：是
- **时间复杂度**：O((V+E)logV)
- **空间复杂度**：O(V)

### 3.4 A*算法

**A*算法**结合了Dijkstra算法和贪心策略，使用启发式函数引导搜索。

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

def manhattan_heuristic(state1, state2):
    """曼哈顿距离启发式"""
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

planner_euclidean = AStarPlanner(euclidean_heuristic)
path_euclidean = planner_euclidean.plan((0, 0), (5, 5), get_grid_neighbors, get_weighted_cost)
print(f"A* (欧几里得)路径长度: {len(path_euclidean) if path_euclidean else 0}")

planner_manhattan = AStarPlanner(manhattan_heuristic)
path_manhattan = planner_manhattan.plan((0, 0), (5, 5), get_grid_neighbors, get_weighted_cost)
print(f"A* (曼哈顿)路径长度: {len(path_manhattan) if path_manhattan else 0}")
```

**A*特性**：
- **完备性**：是（启发式可采纳）
- **最优性**：是（启发式可采纳）
- **时间复杂度**：O(b^d)，实际中远小于此
- **启发式要求**：可采纳（h(n) ≤ h*(n)）、一致（h(n) ≤ c(n,n') + h(n')）

### 3.5 双向搜索

**双向搜索**同时从起点和终点进行搜索，在中间相遇。

```python
class BidirectionalSearchPlanner:
    def __init__(self):
        pass
    
    def plan(self, start, goal, get_neighbors):
        """
        双向搜索规划
        
        参数:
            start: 起始状态
            goal: 目标状态
            get_neighbors: 获取邻居状态的函数
        
        返回:
            路径
        """
        if start == goal:
            return [start]
        
        # 正向搜索队列和访问记录
        forward_queue = deque([(start, [start])])
        forward_visited = {start: [start]}
        
        # 反向搜索队列和访问记录
        backward_queue = deque([(goal, [goal])])
        backward_visited = {goal: [goal]}
        
        while forward_queue and backward_queue:
            # 正向搜索一步
            if forward_queue:
                current_forward, path_forward = forward_queue.popleft()
                for neighbor in get_neighbors(current_forward):
                    if neighbor in backward_visited:
                        # 找到连接点
                        path_backward = backward_visited[neighbor][::-1]
                        return path_forward + path_backward[1:]
                    if neighbor not in forward_visited:
                        forward_visited[neighbor] = path_forward + [neighbor]
                        forward_queue.append((neighbor, path_forward + [neighbor]))
            
            # 反向搜索一步
            if backward_queue:
                current_backward, path_backward = backward_queue.popleft()
                for neighbor in get_neighbors(current_backward):
                    if neighbor in forward_visited:
                        # 找到连接点
                        path_forward = forward_visited[neighbor]
                        return path_forward + path_backward[::-1][1:]
                    if neighbor not in backward_visited:
                        backward_visited[neighbor] = path_backward + [neighbor]
                        backward_queue.append((neighbor, path_backward + [neighbor]))
        
        return None

# 测试
planner = BidirectionalSearchPlanner()
path = planner.plan((0, 0), (9, 9), get_grid_neighbors)
print(f"双向搜索路径长度: {len(path) if path else 0}")
```

**双向搜索特性**：
- **完备性**：是
- **最优性**：是（单位代价）
- **时间复杂度**：O(b^(d/2))
- **空间复杂度**：O(b^(d/2))

---

## 4. 采样-based规划算法

### 4.1 RRT算法

**RRT（Rapidly-exploring Random Trees）**是一种基于采样的运动规划算法。

```python
import random
import numpy as np

class RRTPlanner:
    def __init__(self, step_size=0.5, max_iterations=1000, goal_bias=0.1):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias
    
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
        # 树结构: {node: parent}
        tree = {tuple(start): None}
        
        for _ in range(self.max_iterations):
            # 随机采样
            if random.random() < self.goal_bias:
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
                    tree[tuple(goal)] = new_node
                    return self._extract_path(tree, tuple(goal))
        
        return None
    
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
            return tuple(to_node)
        else:
            direction = direction / distance * self.step_size
            return tuple(np.array(from_node) + direction)
    
    def _distance(self, node1, node2):
        """计算距离"""
        return np.linalg.norm(np.array(node1) - np.array(node2))
    
    def _extract_path(self, tree, goal):
        """提取路径"""
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = tree[current]
        
        path.reverse()
        return path

# 测试
def sample_free_2d():
    """采样2D自由空间"""
    return (random.uniform(-5, 5), random.uniform(-5, 5))

def is_collision_free_2d(node1, node2):
    """检查2D碰撞（简化版）"""
    # 模拟一些障碍物
    obstacles = [(-2, 0), (0, 2), (2, -1)]
    line = np.linspace(node1, node2, 10)
    
    for point in line:
        for obs in obstacles:
            if np.linalg.norm(np.array(point) - np.array(obs)) < 0.5:
                return False
    return True

planner = RRTPlanner(step_size=0.5, max_iterations=2000)
path = planner.plan((-4, -4), (4, 4), sample_free_2d, is_collision_free_2d)
print(f"RRT路径长度: {len(path) if path else 0}")
```

**RRT特性**：
- **完备性**：概率完备（无限迭代时概率趋近于1）
- **最优性**：否（但有RRT*变体）
- **时间复杂度**：O(n log n)，n是节点数
- **适用于**：高维空间、连续状态

### 4.2 RRT*算法

**RRT*（RRT-star）**是RRT的优化版本，保证渐近最优。

```python
class RRTStarPlanner(RRTPlanner):
    def __init__(self, step_size=0.5, max_iterations=1000, goal_bias=0.1, rewire_radius=1.0):
        super().__init__(step_size, max_iterations, goal_bias)
        self.rewire_radius = rewire_radius
    
    def plan(self, start, goal, sample_free, is_collision_free, get_cost):
        """
        RRT*规划
        
        参数:
            start: 起始状态
            goal: 目标状态
            sample_free: 采样自由空间的函数
            is_collision_free: 检查碰撞的函数
            get_cost: 计算代价到起点的函数
        
        返回:
            路径和总代价
        """
        # 树结构: {node: (parent, cost)}
        tree = {tuple(start): (None, 0.0)}
        
        for _ in range(self.max_iterations):
            # 随机采样
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = sample_free()
            
            # 找到最近节点
            nearest = self._find_nearest(tree, sample)
            
            # 扩展
            new_node = self._steer(nearest, sample)
            
            # 检查碰撞
            if is_collision_free(nearest, new_node):
                # 找到最佳父节点
                best_parent, best_cost = self._find_best_parent(tree, new_node, is_collision_free, get_cost)
                
                if best_parent is not None:
                    tree[new_node] = (best_parent, best_cost)
                    
                    # 重布线
                    self._rewire(tree, new_node, is_collision_free, get_cost)
                    
                    # 检查是否到达目标
                    if self._distance(new_node, goal) < self.step_size:
                        # 尝试直接连接到目标
                        if is_collision_free(new_node, tuple(goal)):
                            goal_cost = best_cost + get_cost(new_node, tuple(goal))
                            tree[tuple(goal)] = (new_node, goal_cost)
                            return self._extract_path(tree, tuple(goal)), goal_cost
        
        # 返回最佳路径（如果存在）
        if tuple(goal) in tree:
            return self._extract_path(tree, tuple(goal)), tree[tuple(goal)][1]
        
        return None, None
    
    def _find_best_parent(self, tree, new_node, is_collision_free, get_cost):
        """找到最佳父节点"""
        best_parent = None
        best_cost = float('inf')
        
        # 在重布线半径内寻找候选父节点
        for node in tree:
            if self._distance(node, new_node) <= self.rewire_radius:
                if is_collision_free(node, new_node):
                    cost = tree[node][1] + get_cost(node, new_node)
                    if cost < best_cost:
                        best_cost = cost
                        best_parent = node
        
        return best_parent, best_cost
    
    def _rewire(self, tree, new_node, is_collision_free, get_cost):
        """重布线"""
        new_node_cost = tree[new_node][1]
        
        for node in list(tree.keys()):
            if node == new_node:
                continue
            
            if self._distance(node, new_node) <= self.rewire_radius:
                if is_collision_free(new_node, node):
                    potential_cost = new_node_cost + get_cost(new_node, node)
                    if potential_cost < tree[node][1]:
                        tree[node] = (new_node, potential_cost)

# 测试
def get_euclidean_cost(node1, node2):
    """计算欧几里得距离代价"""
    return np.linalg.norm(np.array(node1) - np.array(node2))

planner = RRTStarPlanner(step_size=0.5, max_iterations=3000, rewire_radius=2.0)
path, cost = planner.plan((-4, -4), (4, 4), sample_free_2d, is_collision_free_2d, get_euclidean_cost)
print(f"RRT*路径长度: {len(path) if path else 0}, 代价: {cost:.2f}")
```

**RRT*特性**：
- **完备性**：概率完备
- **最优性**：渐近最优（迭代次数趋近于无穷时）
- **时间复杂度**：O(n log n)
- **代价**：比RRT慢，但找到更优路径

### 4.3 PRM算法

**PRM（Probabilistic Roadmap）**是一种多查询规划算法。

```python
class PRMPlanner:
    def __init__(self, num_samples=500, connection_radius=1.0):
        self.num_samples = num_samples
        self.connection_radius = connection_radius
    
    def build_roadmap(self, sample_free, is_collision_free):
        """
        构建概率路图
        
        参数:
            sample_free: 采样自由空间的函数
            is_collision_free: 检查碰撞的函数
        
        返回:
            路图（邻接表形式）
        """
        # 采样节点
        nodes = []
        for _ in range(self.num_samples):
            node = sample_free()
            nodes.append(tuple(node))
        
        # 添加起点和终点占位
        nodes.append('start')
        nodes.append('goal')
        
        # 构建邻接表
        roadmap = {node: [] for node in nodes}
        
        # 连接节点
        for i, node1 in enumerate(nodes[:-2]):
            for j, node2 in enumerate(nodes[:-2]):
                if i != j:
                    distance = np.linalg.norm(np.array(node1) - np.array(node2))
                    if distance <= self.connection_radius:
                        if is_collision_free(node1, node2):
                            roadmap[node1].append(node2)
                            roadmap[node2].append(node1)
        
        return roadmap, nodes[:-2]
    
    def query(self, roadmap, start, goal, is_collision_free, get_cost):
        """
        查询路径
        
        参数:
            roadmap: 路图
            start: 起始状态
            goal: 目标状态
            is_collision_free: 检查碰撞的函数
            get_cost: 计算代价的函数
        
        返回:
            路径
        """
        # 连接起点和终点到路图
        temp_roadmap = {k: list(v) for k, v in roadmap.items()}
        
        # 找到起点附近的节点
        for node in roadmap:
            if node not in ['start', 'goal']:
                if is_collision_free(tuple(start), node):
                    temp_roadmap['start'].append(node)
                    temp_roadmap[node].append('start')
        
        # 找到终点附近的节点
        for node in roadmap:
            if node not in ['start', 'goal']:
                if is_collision_free(tuple(goal), node):
                    temp_roadmap['goal'].append(node)
                    temp_roadmap[node].append('goal')
        
        # 使用A*查询
        open_set = []
        heapq.heappush(open_set, (0, 'start', ['start']))
        closed_set = {}
        
        while open_set:
            f, current, path = heapq.heappop(open_set)
            
            if current == 'goal':
                # 提取实际路径
                actual_path = [start]
                for i in range(1, len(path)-1):
                    actual_path.append(path[i])
                actual_path.append(goal)
                return actual_path
            
            if current in closed_set and f >= closed_set[current]:
                continue
            closed_set[current] = f
            
            for neighbor in temp_roadmap[current]:
                if neighbor in ['start', 'goal']:
                    g = path_cost(path, get_cost) + 0
                else:
                    g = path_cost(path, get_cost) + get_cost(current, neighbor)
                
                if current == 'start':
                    h = np.linalg.norm(np.array(goal) - np.array(neighbor))
                elif current == 'goal':
                    h = 0
                else:
                    h = np.linalg.norm(np.array(goal) - np.array(current))
                
                f = g + h
                
                if neighbor not in closed_set or g < closed_set.get(neighbor, float('inf')):
                    heapq.heappush(open_set, (f, neighbor, path + [neighbor]))
        
        return None

def path_cost(path, get_cost):
    """计算路径代价"""
    cost = 0
    for i in range(len(path)-1):
        if path[i] != 'start' and path[i+1] != 'goal':
            cost += get_cost(path[i], path[i+1])
    return cost

# 测试
planner = PRMPlanner(num_samples=200, connection_radius=2.0)
roadmap, nodes = planner.build_roadmap(sample_free_2d, is_collision_free_2d)
print(f"PRM节点数: {len(nodes)}")

path = planner.query(roadmap, (-4, -4), (4, 4), is_collision_free_2d, get_euclidean_cost)
print(f"PRM路径长度: {len(path) if path else 0}")
```

**PRM特性**：
- **完备性**：概率完备
- **最优性**：否（但有PRM*变体）
- **时间复杂度**：构建O(n^2)，查询O(n log n)
- **适用于**：多查询场景、静态环境

---

## 5. 基于学习的规划方法

### 5.1 神经规划器

**神经规划器**使用神经网络直接学习规划策略。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralPlanner(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, horizon=10):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        
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
        
        # 规划网络（Transformer架构）
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=hidden_dim * 4
        )
        
        # 动作解码器
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # 位置编码
        self.positional_encoding = nn.Parameter(
            torch.randn(horizon, 1, hidden_dim)
        )
    
    def forward(self, state, goal):
        """
        前向传播
        
        参数:
            state: 当前状态 [batch, state_dim]
            goal: 目标状态 [batch, state_dim]
        
        返回:
            动作序列 [batch, horizon, action_dim]
        """
        batch_size = state.size(0)
        
        # 编码状态和目标
        state_feat = self.state_encoder(state).unsqueeze(0)  # [1, batch, hidden]
        goal_feat = self.goal_encoder(goal).unsqueeze(0)    # [1, batch, hidden]
        
        # 编码器输入: [state, goal]
        encoder_input = torch.cat([state_feat, goal_feat], dim=0)  # [2, batch, hidden]
        
        # 解码器输入: 位置编码
        decoder_input = self.positional_encoding.repeat(1, batch_size, 1)  # [horizon, batch, hidden]
        
        # Transformer前向
        transformer_output = self.transformer(encoder_input, decoder_input)  # [horizon, batch, hidden]
        
        # 解码动作
        actions = self.action_decoder(transformer_output)  # [horizon, batch, action_dim]
        
        return actions.permute(1, 0, 2)  # [batch, horizon, action_dim]
    
    def plan(self, start, goal):
        """
        规划动作序列
        
        参数:
            start: 起始状态
            goal: 目标状态
        
        返回:
            动作序列
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(start).unsqueeze(0)
            goal_tensor = torch.FloatTensor(goal).unsqueeze(0)
            actions = self.forward(state_tensor, goal_tensor)
        return actions.squeeze(0).numpy()

# 训练示例
def train_neural_planner(planner, dataloader, epochs=100):
    """训练神经规划器"""
    optimizer = optim.Adam(planner.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for states, goals, target_actions in dataloader:
            optimizer.zero_grad()
            pred_actions = planner(states, goals)
            loss = criterion(pred_actions, target_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

# 测试
planner = NeuralPlanner(state_dim=4, action_dim=2, horizon=15)
start = np.array([0, 0, 0, 0])
goal = np.array([5, 5, 0, 0])
actions = planner.plan(start, goal)
print(f"神经规划器输出形状: {actions.shape}")  # [horizon, action_dim]
```

### 5.2 强化学习规划

**强化学习规划**通过与环境交互学习最优策略。

```python
class RLPlanner:
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # state + goal
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # 价值网络
        self.value_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + 
                                    list(self.value_net.parameters()), lr=3e-4)
    
    def select_action(self, state, goal, epsilon=0.1):
        """
        选择动作（带探索）
        
        参数:
            state: 当前状态
            goal: 目标状态
            epsilon: 探索概率
        
        返回:
            动作
        """
        if np.random.random() < epsilon:
            return np.random.uniform(-1, 1, self.action_dim)
        
        self.policy_net.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(np.concatenate([state, goal])).unsqueeze(0)
            action = self.policy_net(input_tensor).squeeze(0).numpy()
        return action
    
    def compute_returns(self, rewards, gamma=0.99):
        """计算折扣回报"""
        returns = []
        running_sum = 0
        for r in reversed(rewards):
            running_sum = r + gamma * running_sum
            returns.insert(0, running_sum)
        return np.array(returns)
    
    def train(self, trajectories, gamma=0.99):
        """
        训练策略
        
        参数:
            trajectories: 轨迹数据
            gamma: 折扣因子
        """
        self.policy_net.train()
        self.value_net.train()
        
        states = []
        goals = []
        actions = []
        returns = []
        
        for traj in trajectories:
            for step in traj:
                states.append(step['state'])
                goals.append(step['goal'])
                actions.append(step['action'])
            
            # 计算回报
            traj_returns = self.compute_returns([step['reward'] for step in traj], gamma)
            returns.extend(traj_returns)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states)
        goals_tensor = torch.FloatTensor(goals)
        actions_tensor = torch.FloatTensor(actions)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1)
        
        # 计算优势
        inputs = torch.cat([states_tensor, goals_tensor], dim=1)
        values = self.value_net(inputs)
        advantages = returns_tensor - values.detach()
        
        # 计算策略损失（PPO风格）
        old_actions = actions_tensor.detach()
        new_actions = self.policy_net(inputs)
        
        # 计算概率比
        action_dist = torch.distributions.Normal(new_actions, torch.ones_like(new_actions) * 0.1)
        old_dist = torch.distributions.Normal(old_actions, torch.ones_like(old_actions) * 0.1)
        
        log_probs = action_dist.log_prob(actions_tensor).sum(dim=1, keepdim=True)
        old_log_probs = old_dist.log_prob(actions_tensor).sum(dim=1, keepdim=True)
        
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
        
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # 计算价值损失
        value_loss = nn.MSELoss()(values, returns_tensor)
        
        # 总损失
        total_loss = policy_loss + 0.5 * value_loss
        
        # 更新
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

# 测试
planner = RLPlanner(state_dim=4, action_dim=2)
state = np.array([0, 0, 0, 0])
goal = np.array([5, 5, 0, 0])
action = planner.select_action(state, goal, epsilon=0.0)
print(f"RL规划器动作: {action}")
```

### 5.3 模仿学习规划

**模仿学习**从专家演示中学习规划策略。

```python
class ImitationPlanner(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
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
        inputs = torch.cat([state, goal], dim=1)
        return self.model(inputs)
    
    def train(self, demonstrations, epochs=500, batch_size=32):
        """
        训练模仿学习模型
        
        参数:
            demonstrations: 演示数据
            epochs: 训练轮数
            batch_size: 批次大小
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        states = torch.FloatTensor([d['state'] for d in demonstrations])
        goals = torch.FloatTensor([d['goal'] for d in demonstrations])
        actions = torch.FloatTensor([d['action'] for d in demonstrations])
        
        dataset = torch.utils.data.TensorDataset(states, goals, actions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_states, batch_goals, batch_actions in dataloader:
                optimizer.zero_grad()
                pred_actions = self(batch_states, batch_goals)
                loss = criterion(pred_actions, batch_actions)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 50 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
    
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
        self.eval()
        actions = []
        current_state = np.array(start)
        
        for _ in range(horizon):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
                goal_tensor = torch.FloatTensor(goal).unsqueeze(0)
                action = self(state_tensor, goal_tensor).squeeze(0).numpy()
            
            actions.append(action)
            # 简化：假设动作直接改变状态
            current_state = current_state + action * 0.5
            
            # 检查是否到达目标
            if np.linalg.norm(current_state - goal) < 0.1:
                break
        
        return np.array(actions)

# 生成演示数据
def generate_demonstrations(num_demos=1000):
    """生成演示数据"""
    demonstrations = []
    for _ in range(num_demos):
        start = np.random.uniform(-2, 2, 4)
        goal = np.random.uniform(-2, 2, 4)
        # 专家策略：直接向目标移动
        action = (goal - start) * 0.1
        demonstrations.append({
            'state': start,
            'goal': goal,
            'action': action
        })
    return demonstrations

# 测试
demonstrations = generate_demonstrations()
planner = ImitationPlanner(state_dim=4, action_dim=4)
planner.train(demonstrations, epochs=200)

start = np.array([-1, -1, 0, 0])
goal = np.array([1, 1, 0, 0])
actions = planner.plan(start, goal, horizon=20)
print(f"模仿学习规划器动作序列长度: {len(actions)}")
```

---

## 6. 规划架构

### 6.1 层次化规划

**层次化规划**将规划任务分解为多个层次。

```python
class HierarchicalPlanner:
    def __init__(self):
        self.task_planner = None
        self.behavior_planner = None
        self.motion_planner = None
    
    def set_task_planner(self, planner):
        """设置任务规划器"""
        self.task_planner = planner
    
    def set_behavior_planner(self, planner):
        """设置行为规划器"""
        self.behavior_planner = planner
    
    def set_motion_planner(self, planner):
        """设置运动规划器"""
        self.motion_planner = planner
    
    def plan(self, initial_state, goal):
        """
        层次化规划
        
        参数:
            initial_state: 初始状态
            goal: 目标状态
        
        返回:
            完整规划
        """
        # 任务规划：分解为子任务
        subtasks = self.task_planner.decompose(initial_state['task_state'], goal['task_state'])
        print(f"子任务: {subtasks}")
        
        # 完整执行计划
        execution_plan = []
        current_state = initial_state
        
        for subtask in subtasks:
            # 行为规划：确定行为策略
            behavior = self.behavior_planner.select_behavior(subtask, current_state)
            print(f"行为选择: {behavior}")
            
            # 运动规划：生成轨迹
            motion_goal = self._get_motion_goal(subtask, current_state, goal)
            trajectory = self.motion_planner.plan(
                current_state['position'],
                motion_goal,
                current_state['obstacles']
            )
            
            if trajectory:
                execution_plan.append({
                    'subtask': subtask,
                    'behavior': behavior,
                    'trajectory': trajectory
                })
                current_state['position'] = motion_goal
            else:
                print(f"运动规划失败: {subtask}")
                return None
        
        return execution_plan
    
    def _get_motion_goal(self, subtask, current_state, goal):
        """获取运动目标"""
        # 简化实现
        return goal.get('position', current_state['position'])

# 测试层次化规划
class SimpleTaskPlanner:
    def decompose(self, initial, goal):
        """分解任务"""
        if goal == ['placed']:
            return ['move_to_object', 'grasp', 'move_to_place', 'release']
        return []

class SimpleBehaviorPlanner:
    def select_behavior(self, subtask, state):
        """选择行为"""
        behaviors = {
            'move_to_object': 'navigate',
            'grasp': 'precision_grasp',
            'move_to_place': 'navigate',
            'release': 'release_object'
        }
        return behaviors.get(subtask, 'default')

class SimpleMotionPlanner:
    def plan(self, start, goal, obstacles=None):
        """简化运动规划"""
        if obstacles is None:
            obstacles = []
        # 直线路径
        steps = 10
        path = []
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + (goal[0] - start[0]) * t
            y = start[1] + (goal[1] - start[1]) * t
            path.append((x, y))
        return path

# 创建层次化规划器
hierarchical_planner = HierarchicalPlanner()
hierarchical_planner.set_task_planner(SimpleTaskPlanner())
hierarchical_planner.set_behavior_planner(SimpleBehaviorPlanner())
hierarchical_planner.set_motion_planner(SimpleMotionPlanner())

initial_state = {
    'task_state': ['hand_empty'],
    'position': (0, 0),
    'obstacles': []
}

goal = {
    'task_state': ['placed'],
    'position': (3, 3)
}

plan = hierarchical_planner.plan(initial_state, goal)
print(f"层次化规划步骤数: {len(plan) if plan else 0}")
```

### 6.2 反应式规划

**反应式规划**直接根据当前感知选择动作。

```python
class ReactivePlanner:
    def __init__(self, policy):
        self.policy = policy
    
    def plan(self, initial_state, goal, environment, max_steps=100):
        """
        反应式规划
        
        参数:
            initial_state: 初始状态
            goal: 目标状态
            environment: 环境
            max_steps: 最大步数
        
        返回:
            轨迹和动作序列
        """
        trajectory = [initial_state]
        actions = []
        current_state = initial_state
        
        for step in range(max_steps):
            # 获取观测
            observation = environment.get_observation(current_state)
            
            # 策略选择动作
            action = self.policy.select_action(observation, goal)
            actions.append(action)
            
            # 执行动作
            next_state = environment.step(current_state, action)
            
            trajectory.append(next_state)
            current_state = next_state
            
            # 检查是否到达目标
            if self._is_goal_reached(current_state, goal):
                print(f"在第 {step+1} 步到达目标")
                break
        
        return trajectory, actions
    
    def _is_goal_reached(self, current, goal, threshold=0.1):
        """检查是否到达目标"""
        distance = np.linalg.norm(np.array(current['position']) - np.array(goal['position']))
        return distance < threshold

# 简单反应式策略
class SimpleReactivePolicy:
    def select_action(self, observation, goal):
        """选择动作"""
        current_pos = observation['position']
        goal_pos = goal['position']
        
        # 计算方向
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        
        # 简单避障
        for obstacle in observation.get('obstacles', []):
            obs_dist = np.linalg.norm(np.array(current_pos) - np.array(obstacle))
            if obs_dist < 0.5:
                # 避开障碍物
                dx += (current_pos[0] - obstacle[0]) * 2
                dy += (current_pos[1] - obstacle[1]) * 2
        
        # 归一化
        distance = (dx**2 + dy**2)**0.5
        if distance > 0.01:
            dx /= distance
            dy /= distance
        
        return {'type': 'move', 'direction': (dx, dy), 'speed': 0.2}

# 模拟环境
class SimpleEnvironment:
    def get_observation(self, state):
        """获取观测"""
        return {
            'position': state['position'],
            'obstacles': state.get('obstacles', [])
        }
    
    def step(self, state, action):
        """执行动作"""
        pos = np.array(state['position'])
        dir_vec = np.array(action['direction'])
        new_pos = pos + dir_vec * action['speed']
        
        # 边界检查
        new_pos = np.clip(new_pos, -5, 5)
        
        return {
            'position': tuple(new_pos),
            'obstacles': state.get('obstacles', [])
        }

# 测试反应式规划
policy = SimpleReactivePolicy()
planner = ReactivePlanner(policy)
environment = SimpleEnvironment()

initial_state = {'position': (0, 0), 'obstacles': [(1.5, 1.5)]}
goal = {'position': (4, 4)}

trajectory, actions = planner.plan(initial_state, goal, environment, max_steps=50)
print(f"反应式规划步数: {len(actions)}")
print(f"最终位置: {trajectory[-1]['position']}")
```

### 6.3 混合规划

**混合规划**结合全局规划和局部规划。

```python
class HybridPlanner:
    def __init__(self, global_planner, local_planner):
        self.global_planner = global_planner
        self.local_planner = local_planner
    
    def plan(self, start, goal, environment, replan_threshold=0.5):
        """
        混合规划
        
        参数:
            start: 起始状态
            goal: 目标状态
            environment: 环境
            replan_threshold: 重规划阈值
        
        返回:
            轨迹
        """
        # 全局规划
        global_path = self.global_planner.plan(start, goal, environment.get_static_obstacles())
        
        if not global_path:
            print("全局规划失败")
            return None
        
        print(f"全局路径点数: {len(global_path)}")
        
        # 局部规划和执行
        trajectory = [start]
        current_state = start
        waypoint_index = 1
        
        while waypoint_index < len(global_path):
            # 获取当前子目标
            waypoint = global_path[waypoint_index]
            
            # 局部规划
            local_path = self.local_planner.plan(
                current_state,
                waypoint,
                environment.get_dynamic_obstacles()
            )
            
            if local_path:
                # 执行局部路径
                for next_state in local_path[1:]:
                    trajectory.append(next_state)
                    current_state = next_state
                    
                    # 检查是否偏离全局路径过多
                    if waypoint_index + 1 < len(global_path):
                        expected_pos = global_path[waypoint_index + 1]
                        deviation = np.linalg.norm(np.array(current_state) - np.array(expected_pos))
                        if deviation > replan_threshold:
                            print(f"偏离全局路径，重新规划...")
                            global_path = self.global_planner.plan(current_state, goal, 
                                                                  environment.get_static_obstacles())
                            waypoint_index = 1
                            break
                
                waypoint_index += 1
            else:
                print(f"局部规划失败，尝试下一个航点")
                waypoint_index += 1
        
        return trajectory

# 测试混合规划
class GlobalAStarPlanner:
    def plan(self, start, goal, obstacles):
        """全局A*规划"""
        # 简化网格规划
        grid_start = (int(start[0]), int(start[1]))
        grid_goal = (int(goal[0]), int(goal[1]))
        
        def get_neighbors(state):
            x, y = state
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            return [n for n in neighbors if 0 <= n[0] <= 10 and 0 <= n[1] <= 10]
        
        def get_cost(a, b):
            return 1.0
        
        planner = AStarPlanner(lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1]))
        return planner.plan(grid_start, grid_goal, get_neighbors, get_cost)

class LocalRRTPlanner:
    def plan(self, start, goal, dynamic_obstacles):
        """局部RRT规划"""
        def sample_free():
            return (random.uniform(start[0]-2, start[0]+2), 
                    random.uniform(start[1]-2, start[1]+2))
        
        def is_collision_free(a, b):
            for obs in dynamic_obstacles:
                # 简化碰撞检测
                if np.linalg.norm(np.array(a) - np.array(obs)) < 0.3:
                    return False
            return True
        
        planner = RRTPlanner(step_size=0.3, max_iterations=500)
        return planner.plan(start, goal, sample_free, is_collision_free)

class HybridEnvironment:
    def get_static_obstacles(self):
        """获取静态障碍物"""
        return [(3, 3), (5, 5)]
    
    def get_dynamic_obstacles(self):
        """获取动态障碍物"""
        return [(4, 4)]  # 模拟动态障碍物

# 创建混合规划器
global_planner = GlobalAStarPlanner()
local_planner = LocalRRTPlanner()
hybrid_planner = HybridPlanner(global_planner, local_planner)
environment = HybridEnvironment()

start = (0, 0)
goal = (10, 10)

trajectory = hybrid_planner.plan(start, goal, environment)
print(f"混合规划轨迹长度: {len(trajectory) if trajectory else 0}")
```

---

## 7. 复杂任务规划案例

### 7.1 机器人装配任务规划

```python
class AssemblyTaskPlanner:
    def __init__(self):
        self.actions = {}
        self.predicates = set()
    
    def define_action(self, name, preconditions, effects, duration=1.0):
        """
        定义动作
        
        参数:
            name: 动作名称
            preconditions: 前置条件（谓词集合）
            effects: 效果（谓词集合）
            duration: 动作持续时间
        """
        self.actions[name] = {
            'preconditions': set(preconditions),
            'effects': set(effects),
            'duration': duration
        }
        # 添加谓词
        self.predicates.update(preconditions)
        self.predicates.update(effects)
    
    def plan(self, initial_state, goal_state, max_depth=20):
        """
        任务规划（前向搜索）
        
        参数:
            initial_state: 初始状态（谓词集合）
            goal_state: 目标状态（谓词集合）
            max_depth: 最大搜索深度
        
        返回:
            动作序列和总时间
        """
        return self._forward_search(initial_state, goal_state, [], 0, max_depth)
    
    def _forward_search(self, current_state, goal_state, plan, depth, max_depth):
        """递归前向搜索"""
        if depth >= max_depth:
            return None, None
        
        # 检查是否达到目标
        if goal_state.issubset(current_state):
            return plan, sum(self.actions[a]['duration'] for a in plan)
        
        # 尝试每个可用动作
        for action_name, action_info in self.actions.items():
            # 检查前置条件
            if action_info['preconditions'].issubset(current_state):
                # 应用效果
                new_state = current_state.copy()
                # 删除负效果（以not_开头）
                for effect in action_info['effects']:
                    if effect.startswith('not_'):
                        new_state.discard(effect[4:])
                    else:
                        new_state.add(effect)
                
                # 递归搜索
                result, time = self._forward_search(new_state, goal_state, 
                                                   plan + [action_name], 
                                                   depth + 1, max_depth)
                if result:
                    return result, time
        
        return None, None

# 定义装配任务
planner = AssemblyTaskPlanner()

# 定义动作
planner.define_action(
    'fetch_part_a',
    ['robot_idle', 'part_a_available'],
    ['holding_a', 'not_robot_idle', 'not_part_a_available']
)

planner.define_action(
    'fetch_part_b',
    ['robot_idle', 'part_b_available'],
    ['holding_b', 'not_robot_idle', 'not_part_b_available']
)

planner.define_action(
    'assemble_ab',
    ['holding_a', 'holding_b'],
    ['assembled_ab', 'not_holding_a', 'not_holding_b', 'robot_idle']
)

planner.define_action(
    'place_assembly',
    ['assembled_ab', 'robot_idle'],
    ['assembly_placed', 'not_assembled_ab']
)

planner.define_action(
    'release_robot',
    ['robot_idle'],
    ['robot_ready']
)

# 规划
initial_state = {'robot_idle', 'part_a_available', 'part_b_available'}
goal_state = {'assembly_placed', 'robot_ready'}

plan, total_time = planner.plan(initial_state, goal_state)
print(f"装配任务规划: {plan}")
print(f"总时间: {total_time} 秒")
```

### 7.2 多机器人协作规划

```python
class MultiRobotPlanner:
    def __init__(self, num_robots):
        self.num_robots = num_robots
        self.robot_states = [{'id': i, 'position': (0, 0), 'status': 'idle'} 
                           for i in range(num_robots)]
    
    def assign_task(self, tasks):
        """
        任务分配
        
        参数:
            tasks: 任务列表
        
        返回:
            分配方案
        """
        assignments = []
        
        for task in tasks:
            # 找到最合适的机器人
            best_robot = None
            best_score = float('inf')
            
            for robot in self.robot_states:
                if robot['status'] == 'idle':
                    # 计算代价（距离 + 能力匹配）
                    distance = np.linalg.norm(np.array(robot['position']) - 
                                            np.array(task['location']))
                    capability_score = 0 if task.get('requires_heavy_lift') else 0
                    score = distance + capability_score
                    
                    if score < best_score:
                        best_score = score
                        best_robot = robot
            
            if best_robot:
                assignments.append({
                    'robot_id': best_robot['id'],
                    'task': task['name'],
                    'location': task['location']
                })
                best_robot['status'] = 'busy'
        
        return assignments
    
    def coordinate_movement(self, assignments):
        """
        协调机器人移动
        
        参数:
            assignments: 分配方案
        
        返回:
            协调后的路径
        """
        paths = []
        collision_graph = self._build_collision_graph(assignments)
        
        # 简单协调：按顺序执行
        for i, assignment in enumerate(assignments):
            robot = next(r for r in self.robot_states if r['id'] == assignment['robot_id'])
            
            # 规划路径
            path = self._plan_path(robot['position'], assignment['location'])
            paths.append({
                'robot_id': assignment['robot_id'],
                'task': assignment['task'],
                'path': path,
                'start_time': i * 2.0  # 错开启动时间
            })
            
            # 更新机器人状态
            robot['position'] = assignment['location']
            robot['status'] = 'idle'
        
        return paths
    
    def _build_collision_graph(self, assignments):
        """构建碰撞图"""
        graph = {}
        for i, a1 in enumerate(assignments):
            graph[i] = []
            for j, a2 in enumerate(assignments):
                if i != j:
                    # 检查路径是否可能冲突
                    if self._paths_conflict(a1['location'], a2['location']):
                        graph[i].append(j)
        return graph
    
    def _paths_conflict(self, loc1, loc2):
        """检查路径是否冲突"""
        distance = np.linalg.norm(np.array(loc1) - np.array(loc2))
        return distance < 2.0  # 距离太近可能冲突
    
    def _plan_path(self, start, goal):
        """规划路径"""
        # 简化：直线路径
        steps = 5
        path = []
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + (goal[0] - start[0]) * t
            y = start[1] + (goal[1] - start[1]) * t
            path.append((x, y))
        return path

# 测试多机器人规划
tasks = [
    {'name': 'fetch_box', 'location': (5, 0), 'requires_heavy_lift': False},
    {'name': 'move_table', 'location': (0, 5), 'requires_heavy_lift': True},
    {'name': 'deliver_package', 'location': (5, 5), 'requires_heavy_lift': False}
]

planner = MultiRobotPlanner(num_robots=2)
assignments = planner.assign_task(tasks)
print(f"任务分配: {assignments}")

paths = planner.coordinate_movement(assignments)
print(f"协调路径:")
for path_info in paths:
    print(f"  机器人 {path_info['robot_id']}: {path_info['task']}")
```

### 7.3 动态环境规划

```python
class DynamicEnvironmentPlanner:
    def __init__(self, replan_interval=5):
        self.replan_interval = replan_interval
        self.current_plan = None
        self.step_count = 0
    
    def plan(self, start, goal, environment):
        """
        动态环境规划
        
        参数:
            start: 起始状态
            goal: 目标状态
            environment: 动态环境
        
        返回:
            执行轨迹
        """
        trajectory = [start]
        current_state = start
        
        while not self._is_goal(current_state, goal):
            # 定期重规划
            if self.step_count % self.replan_interval == 0:
                print(f"第 {self.step_count} 步: 重新规划")
                self.current_plan = self._generate_plan(current_state, goal, environment)
            
            if not self.current_plan or len(self.current_plan) == 0:
                print("无法规划路径")
                break
            
            # 执行下一步
            next_waypoint = self.current_plan.pop(0)
            current_state = self._move_to(current_state, next_waypoint, environment)
            trajectory.append(current_state)
            
            # 更新环境
            environment.update()
            
            # 检查是否需要紧急避障
            if self._detect_collision(current_state, environment):
                print("检测到碰撞风险，紧急停止")
                # 后退并重新规划
                current_state = trajectory[-2] if len(trajectory) > 1 else current_state
                self.current_plan = None
            
            self.step_count += 1
        
        return trajectory
    
    def _generate_plan(self, start, goal, environment):
        """生成规划"""
        obstacles = environment.get_obstacles()
        
        def get_neighbors(state):
            x, y = state
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            return [n for n in neighbors if 0 <= n[0] <= 10 and 0 <= n[1] <= 10]
        
        def get_cost(a, b):
            return 1.0
        
        # 考虑动态障碍物的启发式
        def dynamic_heuristic(state, goal):
            base_h = abs(state[0]-goal[0]) + abs(state[1]-goal[1])
            # 避开动态障碍物
            for obs in obstacles:
                dist = abs(state[0]-obs[0]) + abs(state[1]-obs[1])
                if dist < 3:
                    base_h += (3 - dist) * 10
            return base_h
        
        planner = AStarPlanner(dynamic_heuristic)
        return planner.plan(start, goal, get_neighbors, get_cost)
    
    def _move_to(self, current, target, environment):
        """移动到目标"""
        # 简化移动
        return target
    
    def _is_goal(self, current, goal, threshold=1):
        """检查是否到达目标"""
        distance = abs(current[0]-goal[0]) + abs(current[1]-goal[1])
        return distance <= threshold
    
    def _detect_collision(self, state, environment):
        """检测碰撞"""
        obstacles = environment.get_obstacles()
        for obs in obstacles:
            distance = abs(state[0]-obs[0]) + abs(state[1]-obs[1])
            if distance < 2:
                return True
        return False

# 动态环境
class DynamicEnvironment:
    def __init__(self):
        self.obstacles = [(5, 5)]
        self.time = 0
    
    def update(self):
        """更新环境（移动障碍物）"""
        self.time += 1
        # 障碍物周期性移动
        if self.time % 10 < 5:
            self.obstacles = [(5, 5 + (self.time % 5))]
        else:
            self.obstacles = [(5, 5 - ((self.time - 5) % 5))]
    
    def get_obstacles(self):
        """获取当前障碍物"""
        return self.obstacles

# 测试动态环境规划
environment = DynamicEnvironment()
planner = DynamicEnvironmentPlanner(replan_interval=3)
trajectory = planner.plan((0, 0), (10, 10), environment)
print(f"动态环境规划轨迹长度: {len(trajectory) if trajectory else 0}")
```

---

## 8. 规划评估指标

### 8.1 规划质量指标

| 指标 | 描述 | 计算公式 |
|------|------|----------|
| **路径长度** | 路径的总长度 | $\sum_{i=1}^{n-1} distance(p_i, p_{i+1})$ |
| **路径平滑度** | 路径的平滑程度 | $\sum_{i=1}^{n-2} angle(p_i, p_{i+1}, p_{i+2})$ |
| **时间最优性** | 执行时间是否最短 | 实际时间 / 最优时间 |
| **能量消耗** | 执行路径所需能量 | $\sum action^T \cdot M \cdot action$ |
| **安全性** | 与障碍物的最小距离 | $\min distance(path, obstacle)$ |

### 8.2 规划效率指标

```python
class PlanningMetrics:
    def __init__(self):
        self.metrics = {}
    
    def compute_path_length(self, path):
        """计算路径长度"""
        length = 0
        for i in range(len(path)-1):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))
        return length
    
    def compute_path_smoothness(self, path):
        """计算路径平滑度"""
        if len(path) < 3:
            return 0
        
        total_angle = 0
        for i in range(len(path)-2):
            # 计算连续三点的夹角
            v1 = np.array(path[i+1]) - np.array(path[i])
            v2 = np.array(path[i+2]) - np.array(path[i+1])
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                total_angle += angle
        
        return total_angle / (len(path) - 2)
    
    def compute_safety_margin(self, path, obstacles):
        """计算安全裕度"""
        min_distance = float('inf')
        
        for point in path:
            for obstacle in obstacles:
                distance = np.linalg.norm(np.array(point) - np.array(obstacle))
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def compute_execution_time(self, path, velocity=1.0):
        """计算执行时间"""
        length = self.compute_path_length(path)
        return length / velocity
    
    def evaluate(self, path, obstacles, start_time, end_time):
        """
        综合评估规划结果
        
        参数:
            path: 规划路径
            obstacles: 障碍物列表
            start_time: 规划开始时间
            end_time: 规划结束时间
        
        返回:
            评估结果字典
        """
        results = {
            'path_length': self.compute_path_length(path),
            'smoothness': self.compute_path_smoothness(path),
            'safety_margin': self.compute_safety_margin(path, obstacles),
            'execution_time': self.compute_execution_time(path),
            'planning_time': end_time - start_time,
            'path_quality': self._compute_quality_score(path, obstacles),
            'efficiency': self._compute_efficiency_score(path, start_time, end_time)
        }
        
        self.metrics = results
        return results
    
    def _compute_quality_score(self, path, obstacles):
        """计算路径质量分数"""
        length_score = max(0, 1 - self.compute_path_length(path) / 100)
        smooth_score = max(0, 1 - self.compute_path_smoothness(path) / np.pi)
        safety_score = min(1, self.compute_safety_margin(path, obstacles))
        
        return (length_score + smooth_score + safety_score) / 3
    
    def _compute_efficiency_score(self, path, start_time, end_time):
        """计算效率分数"""
        planning_time = end_time - start_time
        execution_time = self.compute_execution_time(path)
        
        if planning_time > 10:
            planning_score = 0
        else:
            planning_score = 1 - planning_time / 10
        
        return (planning_score + min(1, 10 / execution_time)) / 2

# 测试评估指标
metrics = PlanningMetrics()
path = [(0, 0), (2, 2), (4, 3), (6, 4), (8, 5), (10, 10)]
obstacles = [(3, 3), (5, 5)]

results = metrics.evaluate(path, obstacles, 0, 0.5)
print("规划评估结果:")
for key, value in results.items():
    print(f"  {key}: {value:.4f}")
```

### 8.3 规划鲁棒性指标

```python
class RobustnessMetrics:
    def __init__(self):
        pass
    
    def test_perturbation_robustness(self, planner, start, goal, environment, num_trials=10):
        """
        测试扰动鲁棒性
        
        参数:
            planner: 规划器
            start: 起始状态
            goal: 目标状态
            environment: 环境
            num_trials: 试验次数
        
        返回:
            成功率和平均偏差
        """
        successes = 0
        total_deviation = 0
        
        for _ in range(num_trials):
            # 添加扰动
            perturbed_start = (
                start[0] + np.random.uniform(-0.5, 0.5),
                start[1] + np.random.uniform(-0.5, 0.5)
            )
            
            path = planner.plan(perturbed_start, goal, environment.get_static_obstacles())
            
            if path:
                successes += 1
                # 计算与原始路径的偏差
                original_path = planner.plan(start, goal, environment.get_static_obstacles())
                if original_path:
                    deviation = self._compute_path_deviation(path, original_path)
                    total_deviation += deviation
        
        success_rate = successes / num_trials
        avg_deviation = total_deviation / successes if successes > 0 else float('inf')
        
        return {
            'success_rate': success_rate,
            'avg_deviation': avg_deviation
        }
    
    def _compute_path_deviation(self, path1, path2):
        """计算两条路径之间的偏差"""
        # 采样相同数量的点
        num_samples = max(len(path1), len(path2))
        
        points1 = self._resample_path(path1, num_samples)
        points2 = self._resample_path(path2, num_samples)
        
        total_distance = 0
        for p1, p2 in zip(points1, points2):
            total_distance += np.linalg.norm(np.array(p1) - np.array(p2))
        
        return total_distance / num_samples
    
    def _resample_path(self, path, num_samples):
        """重采样路径"""
        if len(path) <= 1:
            return path
        
        resampled = []
        for i in range(num_samples):
            t = i / (num_samples - 1)
            index = int(t * (len(path) - 1))
            
            if index < len(path) - 1:
                # 线性插值
                alpha = t * (len(path) - 1) - index
                point = (
                    path[index][0] * (1 - alpha) + path[index+1][0] * alpha,
                    path[index][1] * (1 - alpha) + path[index+1][1] * alpha
                )
            else:
                point = path[-1]
            
            resampled.append(point)
        
        return resampled
    
    def test_environment_changes(self, planner, start, goal, base_environment, changes):
        """
        测试环境变化鲁棒性
        
        参数:
            planner: 规划器
            start: 起始状态
            goal: 目标状态
            base_environment: 基础环境
            changes: 环境变化列表
        
        返回:
            适应成功率
        """
        successes = 0
        
        for change in changes:
            # 应用环境变化
            environment = copy.deepcopy(base_environment)
            environment.apply_change(change)
            
            path = planner.plan(start, goal, environment.get_static_obstacles())
            
            if path:
                # 检查路径是否有效
                if self._validate_path(path, environment):
                    successes += 1
        
        return successes / len(changes)
    
    def _validate_path(self, path, environment):
        """验证路径是否有效"""
        obstacles = environment.get_static_obstacles()
        
        for point in path:
            for obstacle in obstacles:
                if np.linalg.norm(np.array(point) - np.array(obstacle)) < 0.3:
                    return False
        
        return True

# 测试鲁棒性指标
import copy

class TestEnvironment:
    def __init__(self):
        self.obstacles = [(5, 5)]
    
    def get_static_obstacles(self):
        return self.obstacles
    
    def apply_change(self, change):
        """应用环境变化"""
        if change['type'] == 'add_obstacle':
            self.obstacles.append(change['position'])
        elif change['type'] == 'remove_obstacle':
            self.obstacles = [obs for obs in self.obstacles if obs != change['position']]

robustness = RobustnessMetrics()
environment = TestEnvironment()
planner = GlobalAStarPlanner()

# 测试扰动鲁棒性
robustness_results = robustness.test_perturbation_robustness(planner, (0, 0), (10, 10), environment)
print(f"扰动鲁棒性 - 成功率: {robustness_results['success_rate']:.2f}, 平均偏差: {robustness_results['avg_deviation']:.4f}")

# 测试环境变化鲁棒性
changes = [
    {'type': 'add_obstacle', 'position': (3, 3)},
    {'type': 'add_obstacle', 'position': (7, 7)},
    {'type': 'remove_obstacle', 'position': (5, 5)},
    {'type': 'add_obstacle', 'position': (8, 8)}
]

adaptation_rate = robustness.test_environment_changes(planner, (0, 0), (10, 10), environment, changes)
print(f"环境变化适应率: {adaptation_rate:.2f}")
```

---

## 9. 实践练习

### 练习1：实现完整的A*规划器

```python
class CompleteAStarPlanner:
    """完整的A*规划器实现"""
    
    def __init__(self, heuristic):
        self.heuristic = heuristic
    
    def plan(self, start, goal, get_neighbors, get_cost, is_valid=None):
        """
        A*规划主函数
        
        参数:
            start: 起始状态
            goal: 目标状态
            get_neighbors: 获取邻居状态的函数
            get_cost: 计算代价的函数
            is_valid: 状态有效性检查函数
        
        返回:
            路径和统计信息
        """
        import heapq
        
        # 优先队列: (f, g, state, path)
        open_set = []
        heapq.heappush(open_set, (0, 0, start, []))
        
        # 已访问状态及其最优代价
        closed_set = {}
        
        statistics = {
            'nodes_expanded': 0,
            'nodes_generated': 1,
            'planning_time': 0
        }
        
        start_time = time.time()
        
        while open_set:
            f, g, current, path = heapq.heappop(open_set)
            
            # 检查是否到达目标
            if current == goal:
                statistics['planning_time'] = time.time() - start_time
                statistics['path_length'] = g
                statistics['path_nodes'] = len(path) + 1
                return path + [current], statistics
            
            # 跳过已访问且代价更高的路径
            if current in closed_set and g >= closed_set[current]:
                continue
            
            closed_set[current] = g
            statistics['nodes_expanded'] += 1
            
            # 探索邻居
            for neighbor in get_neighbors(current):
                # 检查状态有效性
                if is_valid and not is_valid(neighbor):
                    continue
                
                new_g = g + get_cost(current, neighbor)
                
                # 如果邻居未访问或找到更优路径
                if neighbor not in closed_set or new_g < closed_set.get(neighbor, float('inf')):
                    h = self.heuristic(neighbor, goal)
                    f = new_g + h
                    
                    heapq.heappush(open_set, (f, new_g, neighbor, path + [current]))
                    statistics['nodes_generated'] += 1
        
        statistics['planning_time'] = time.time() - start_time
        return None, statistics

# 测试完整A*规划器
import time

def grid_get_neighbors(state):
    """获取网格邻居（8连通）"""
    x, y = state
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx <= 20 and 0 <= ny <= 20:
                neighbors.append((nx, ny))
    return neighbors

def grid_get_cost(state1, state2):
    """计算网格移动代价"""
    dx = abs(state1[0] - state2[0])
    dy = abs(state1[1] - state2[1])
    if dx + dy == 2:  # 对角线
        return 1.414
    return 1.0

def grid_is_valid(state):
    """检查状态有效性（避开障碍物）"""
    obstacles = [(10, 10), (10, 11), (11, 10), (11, 11)]
    return state not in obstacles

planner = CompleteAStarPlanner(euclidean_heuristic)
path, stats = planner.plan((0, 0), (20, 20), grid_get_neighbors, grid_get_cost, grid_is_valid)

print("A*规划器统计信息:")
print(f"  扩展节点数: {stats['nodes_expanded']}")
print(f"  生成节点数: {stats['nodes_generated']}")
print(f"  规划时间: {stats['planning_time']:.4f} 秒")
print(f"  路径代价: {stats['path_length']:.2f}")
print(f"  路径节点数: {stats['path_nodes']}")
```

### 练习2：实现RRT*规划器

```python
class CompleteRRTStarPlanner:
    """完整的RRT*规划器实现"""
    
    def __init__(self, step_size=0.5, max_iterations=5000, goal_bias=0.1, rewire_radius=1.0):
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_bias = goal_bias
        self.rewire_radius = rewire_radius
    
    def plan(self, start, goal, sample_free, is_collision_free, get_cost):
        """
        RRT*规划主函数
        
        参数:
            start: 起始状态
            goal: 目标状态
            sample_free: 采样自由空间的函数
            is_collision_free: 检查碰撞的函数
            get_cost: 计算代价的函数
        
        返回:
            路径、代价和统计信息
        """
        import random
        import numpy as np
        
        # 树结构: {node: (parent, cost)}
        tree = {tuple(start): (None, 0.0)}
        
        statistics = {
            'nodes_generated': 1,
            'rewire_count': 0,
            'planning_time': 0
        }
        
        start_time = time.time()
        
        for iteration in range(self.max_iterations):
            # 随机采样
            if random.random() < self.goal_bias:
                sample = goal
            else:
                sample = sample_free()
            
            # 找到最近节点
            nearest = self._find_nearest(tree, sample)
            
            # 扩展
            new_node = self._steer(nearest, sample)
            
            # 检查碰撞
            if is_collision_free(nearest, new_node):
                # 找到最佳父节点
                best_parent, best_cost = self._find_best_parent(tree, new_node, is_collision_free, get_cost)
                
                if best_parent is not None:
                    tree[new_node] = (best_parent, best_cost)
                    statistics['nodes_generated'] += 1
                    
                    # 重布线
                    rewire_count = self._rewire(tree, new_node, is_collision_free, get_cost)
                    statistics['rewire_count'] += rewire_count
                    
                    # 检查是否到达目标
                    if self._distance(new_node, goal) < self.step_size:
                        # 尝试直接连接到目标
                        if is_collision_free(new_node, tuple(goal)):
                            goal_cost = best_cost + get_cost(new_node, tuple(goal))
                            tree[tuple(goal)] = (new_node, goal_cost)
                            
                            statistics['planning_time'] = time.time() - start_time
                            statistics['iterations'] = iteration + 1
                            
                            path = self._extract_path(tree, tuple(goal))
                            return path, goal_cost, statistics
        
        # 返回最佳路径（如果存在）
        if tuple(goal) in tree:
            statistics['planning_time'] = time.time() - start_time
            statistics['iterations'] = self.max_iterations
            path = self._extract_path(tree, tuple(goal))
            return path, tree[tuple(goal)][1], statistics
        
        statistics['planning_time'] = time.time() - start_time
        statistics['iterations'] = self.max_iterations
        return None, None, statistics
    
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
            return tuple(to_node)
        else:
            direction = direction / distance * self.step_size
            return tuple(np.array(from_node) + direction)
    
    def _distance(self, node1, node2):
        """计算距离"""
        return np.linalg.norm(np.array(node1) - np.array(node2))
    
    def _find_best_parent(self, tree, new_node, is_collision_free, get_cost):
        """找到最佳父节点"""
        best_parent = None
        best_cost = float('inf')
        
        for node in tree:
            if self._distance(node, new_node) <= self.rewire_radius:
                if is_collision_free(node, new_node):
                    cost = tree[node][1] + get_cost(node, new_node)
                    if cost < best_cost:
                        best_cost = cost
                        best_parent = node
        
        return best_parent, best_cost
    
    def _rewire(self, tree, new_node, is_collision_free, get_cost):
        """重布线"""
        new_node_cost = tree[new_node][1]
        rewire_count = 0
        
        for node in list(tree.keys()):
            if node == new_node:
                continue
            
            if self._distance(node, new_node) <= self.rewire_radius:
                if is_collision_free(new_node, node):
                    potential_cost = new_node_cost + get_cost(new_node, node)
                    if potential_cost < tree[node][1]:
                        tree[node] = (new_node, potential_cost)
                        rewire_count += 1
        
        return rewire_count
    
    def _extract_path(self, tree, goal):
        """提取路径"""
        path = []
        current = goal
        
        while current is not None:
            path.append(current)
            current = tree[current][0]
        
        path.reverse()
        return path

# 测试完整RRT*规划器
def rrt_sample_free():
    """采样自由空间"""
    return (random.uniform(-10, 10), random.uniform(-10, 10))

def rrt_is_collision_free(node1, node2):
    """检查碰撞"""
    obstacles = [(0, 0), (2, 3), (-3, 2)]
    line = np.linspace(node1, node2, 20)
    
    for point in line:
        for obs in obstacles:
            if np.linalg.norm(np.array(point) - np.array(obs)) < 1.0:
                return False
    return True

def rrt_get_cost(node1, node2):
    """计算代价"""
    return np.linalg.norm(np.array(node1) - np.array(node2))

planner = CompleteRRTStarPlanner(step_size=0.5, max_iterations=5000, rewire_radius=2.0)
path, cost, stats = planner.plan((-8, -8), (8, 8), rrt_sample_free, rrt_is_collision_free, rrt_get_cost)

print("\nRRT*规划器统计信息:")
print(f"  生成节点数: {stats['nodes_generated']}")
print(f"  重布线次数: {stats['rewire_count']}")
print(f"  迭代次数: {stats['iterations']}")
print(f"  规划时间: {stats['planning_time']:.4f} 秒")
print(f"  路径代价: {cost:.2f}")
if path:
    print(f"  路径节点数: {len(path)}")
```

### 练习3：实现层次化规划系统

```python
class CompleteHierarchicalPlanner:
    """完整的层次化规划系统"""
    
    def __init__(self):
        self.task_planner = None
        self.behavior_planner = None
        self.motion_planner = None
        self.executor = None
    
    def set_planners(self, task_planner, behavior_planner, motion_planner, executor):
        """设置各层规划器"""
        self.task_planner = task_planner
        self.behavior_planner = behavior_planner
        self.motion_planner = motion_planner
        self.executor = executor
    
    def plan_and_execute(self, initial_state, goal):
        """
        完整的规划和执行流程
        
        参数:
            initial_state: 初始状态
            goal: 目标状态
        
        返回:
            执行结果
        """
        print("=== 层次化规划系统 ===")
        
        # 任务规划
        print("\n1. 任务规划阶段")
        subtasks = self.task_planner.decompose(initial_state, goal)
        print(f"   子任务序列: {subtasks}")
        
        if not subtasks:
            print("   任务规划失败")
            return False
        
        # 执行每个子任务
        current_state = initial_state
        execution_log = []
        
        for i, subtask in enumerate(subtasks):
            print(f"\n2. 执行子任务 {i+1}/{len(subtasks)}: {subtask}")
            
            # 行为规划
            behavior = self.behavior_planner.select_behavior(subtask, current_state)
            print(f"   选择行为: {behavior}")
            
            # 运动规划
            motion_goal = self.behavior_planner.get_motion_goal(subtask, current_state, goal)
            print(f"   运动目标: {motion_goal}")
            
            trajectory = self.motion_planner.plan(
                current_state['position'],
                motion_goal,
                current_state.get('obstacles', [])
            )
            
            if not trajectory:
                print(f"   运动规划失败")
                return False
            
            print(f"   轨迹点数: {len(trajectory)}")
            
            # 执行轨迹
            execution_result = self.executor.execute(trajectory, subtask, behavior)
            
            if execution_result['success']:
                execution_log.append({
                    'subtask': subtask,
                    'behavior': behavior,
                    'success': True,
                    'duration': execution_result['duration']
                })
                current_state['position'] = motion_goal
                current_state['task_state'] = execution_result['new_task_state']
                print(f"   执行成功，耗时: {execution_result['duration']:.2f} 秒")
            else:
                print(f"   执行失败: {execution_result['error']}")
                return False
        
        # 检查是否达到最终目标
        if self._check_goal(current_state, goal):
            print("\n=== 任务完成 ===")
            return {
                'success': True,
                'execution_log': execution_log,
                'final_state': current_state
            }
        else:
            print("\n=== 任务未完成 ===")
            return {'success': False, 'error': '未达到目标状态'}
    
    def _check_goal(self, current_state, goal):
        """检查是否达到目标"""
        # 检查任务状态
        if 'task_state' in goal:
            for condition in goal['task_state']:
                if condition not in current_state.get('task_state', []):
                    return False
        
        # 检查位置
        if 'position' in goal:
            distance = np.linalg.norm(
                np.array(current_state['position']) - np.array(goal['position'])
            )
            if distance > 0.5:
                return False
        
        return True

# 定义各层组件
class HTNTaskPlanner:
    """HTN任务规划器"""
    def decompose(self, initial_state, goal):
        """分解任务"""
        if goal.get('task_state') == ['object_placed']:
            return [
                'navigate_to_object',
                'grasp_object',
                'navigate_to_destination',
                'place_object',
                'return_home'
            ]
        return []

class BehaviorSelector:
    """行为选择器"""
    def select_behavior(self, subtask, state):
        """选择行为"""
        behaviors = {
            'navigate_to_object': 'safe_navigation',
            'grasp_object': 'precision_grasp',
            'navigate_to_destination': 'safe_navigation',
            'place_object': 'gentle_placement',
            'return_home': 'fast_navigation'
        }
        return behaviors.get(subtask, 'default')
    
    def get_motion_goal(self, subtask, current_state, goal):
        """获取运动目标"""
        locations = {
            'navigate_to_object': goal.get('object_position', (5, 0)),
            'grasp_object': goal.get('object_position', (5, 0)),
            'navigate_to_destination': goal.get('destination', (0, 5)),
            'place_object': goal.get('destination', (0, 5)),
            'return_home': (0, 0)
        }
        return locations.get(subtask, current_state['position'])

class MotionExecutor:
    """运动执行器"""
    def execute(self, trajectory, subtask, behavior):
        """执行轨迹"""
        import time
        
        duration = len(trajectory) * 0.5  # 模拟执行时间
        
        # 模拟状态更新
        new_task_state = []
        if subtask == 'grasp_object':
            new_task_state = ['holding_object']
        elif subtask == 'place_object':
            new_task_state = ['object_placed']
        elif subtask == 'return_home':
            new_task_state = ['at_home']
        
        time.sleep(0.1)  # 模拟执行
        
        return {
            'success': True,
            'duration': duration,
            'new_task_state': new_task_state
        }

# 测试层次化规划系统
hierarchical_system = CompleteHierarchicalPlanner()
hierarchical_system.set_planners(
    task_planner=HTNTaskPlanner(),
    behavior_planner=BehaviorSelector(),
    motion_planner=SimpleMotionPlanner(),
    executor=MotionExecutor()
)

initial_state = {
    'task_state': ['at_home', 'hand_empty'],
    'position': (0, 0),
    'obstacles': []
}

goal = {
    'task_state': ['object_placed', 'at_home'],
    'object_position': (5, 0),
    'destination': (0, 5)
}

result = hierarchical_system.plan_and_execute(initial_state, goal)
print(f"\n最终结果: {result}")
```

---

## 10. 总结与展望

### 10.1 总结

具身规划是具身智能系统的核心组成部分，涵盖多个层次和多种方法：

**经典规划算法**：
- BFS/DFS：基础搜索算法
- Dijkstra：加权图最短路径
- A*：结合启发式的最优搜索
- 双向搜索：从两端同时搜索

**采样-based规划算法**：
- RRT：快速探索随机树
- RRT*：渐近最优版本
- PRM：概率路图（多查询）

**基于学习的规划方法**：
- 神经规划器：Transformer架构直接预测动作序列
- 强化学习规划：通过交互学习最优策略
- 模仿学习：从专家演示中学习

**规划架构**：
- 层次化规划：任务-行为-运动三层
- 反应式规划：直接感知-动作映射
- 混合规划：全局+局部结合

### 10.2 未来展望

具身规划领域的未来发展方向包括：

1. **大语言模型与规划结合**：利用LLM的推理能力进行高层任务规划
2. **终身规划**：在动态环境中持续学习和适应
3. **多智能体协同规划**：多个智能体共同完成复杂任务
4. **安全关键规划**：保证规划的安全性和可靠性
5. **实时规划**：在毫秒级时间内完成规划决策
6. **可解释规划**：提供规划决策的可解释性

### 10.3 关键挑战

当前具身规划面临的主要挑战：

| 挑战 | 描述 | 研究方向 |
|------|------|----------|
| **环境不确定性** | 动态环境中的障碍物移动 | 概率规划、在线重规划 |
| **高维状态空间** | 机器人关节空间维度高 | 采样-based方法、维度缩减 |
| **实时性要求** | 快速响应的规划需求 | 增量规划、并行计算 |
| **安全性保障** | 保证操作安全 | 安全约束规划、验证 |
| **泛化能力** | 跨环境的迁移能力 | 元学习、领域自适应 |

---

## 参考文献

1. LaValle, S. M. (2006). *Planning Algorithms*. Cambridge University Press.
2. Kavraki, L. E., Svestka, P., Latombe, J. C., & Overmars, M. H. (1996). Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces. *IEEE Transactions on Robotics and Automation*.
3. Karaman, S., & Frazzoli, E. (2011). Sampling-based Algorithms for Optimal Motion Planning. *International Journal of Robotics Research*.
4. Nilsson, N. J. (1980). *Principles of Artificial Intelligence*. Tioga Publishing.
5. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*. Pearson.

---

**返回**：[具身智能概述](01-embodied-intelligence-overview.md)
