# 5.5 动态规划

## 目录

- [1. 引言](#1-引言)
- [2. 值迭代](#2-值迭代)
- [3. 策略迭代](#3-策略迭代)
- [4. D* Lite](#4-d-lite)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 动态规划在规划中的应用

动态规划是求解序列决策问题的有效方法，适用于可完全观察的马尔可夫决策过程。

### 1.2 基本概念

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class GridMDP:
    """网格马尔可夫决策过程"""
    
    def __init__(self, gridmap):
        self.map = gridmap
        self.states = []
        
        for gy in range(gridmap.height):
            for gx in range(gridmap.width):
                if not gridmap.is_occupied(gx, gy):
                    self.states.append((gx, gy))
        
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上下右左
    
    def transition(self, state, action):
        """状态转移"""
        gx, gy = state
        dx, dy = action
        
        ngx = gx + dx
        ngy = gy + dy
        
        if 0 <= ngx < self.map.width and 0 <= ngy < self.map.height:
            if not self.map.is_occupied(ngx, ngy):
                return (ngx, ngy)
        
        return state
    
    def reward(self, state, action, goal_state):
        """奖励"""
        if state == goal_state:
            return 100.0
        
        return -1.0
```

---

## 2. 值迭代

### 2.1 值迭代实现

```python
class ValueIteration:
    """值迭代"""
    
    def __init__(self, mdp, gamma=0.99, theta=1e-3):
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        
        self.value = defaultdict(float)
        self.policy = {}
    
    def bellman_optimality_update(self, state, goal_state):
        """贝尔曼最优更新"""
        if state == goal_state:
            return 0.0
        
        max_q = -float('inf')
        
        for action in self.mdp.actions:
            next_state = self.mdp.transition(state, action)
            q = self.mdp.reward(state, action, goal_state) + \
                self.gamma * self.value[next_state]
            max_q = max(max_q, q)
        
        return max_q
    
    def train(self, goal_state, max_iterations=1000):
        """训练"""
        for iteration in range(max_iterations):
            delta = 0
            new_value = self.value.copy()
            
            for state in self.mdp.states:
                v = self.value[state]
                new_value[state] = self.bellman_optimality_update(state, goal_state)
                delta = max(delta, abs(v - new_value[state]))
            
            self.value = new_value
            
            if delta < self.theta:
                print(f"值迭代在 {iteration + 1} 轮后收敛")
                break
        
        # 提取策略
        for state in self.mdp.states:
            if state == goal_state:
                continue
            
            best_action = None
            best_q = -float('inf')
            
            for action in self.mdp.actions:
                next_state = self.mdp.transition(state, action)
                q = self.mdp.reward(state, action, goal_state) + \
                    self.gamma * self.value[next_state]
                
                if q > best_q:
                    best_q = q
                    best_action = action
            
            self.policy[state] = best_action
    
    def get_policy(self):
        """获取策略"""
        return self.policy
```

---

## 3. 策略迭代

### 3.1 策略迭代实现

```python
class PolicyIteration:
    """策略迭代"""
    
    def __init__(self, mdp, gamma=0.99, theta=1e-3):
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        
        self.value = defaultdict(float)
        self.policy = {}
        
        # 初始化随机策略
        for state in self.mdp.states:
            self.policy[state] = self.mdp.actions[np.random.randint(4)]
    
    def policy_evaluation(self, goal_state):
        """策略评估"""
        for _ in range(1000):
            delta = 0
            new_value = self.value.copy()
            
            for state in self.mdp.states:
                if state == goal_state:
                    continue
                
                v = self.value[state]
                action = self.policy[state]
                next_state = self.mdp.transition(state, action)
                
                new_value[state] = self.mdp.reward(state, action, goal_state) + \
                    self.gamma * self.value[next_state]
                
                delta = max(delta, abs(v - new_value[state]))
            
            self.value = new_value
            
            if delta < self.theta:
                break
    
    def policy_improvement(self, goal_state):
        """策略改进"""
        policy_stable = True
        
        for state in self.mdp.states:
            if state == goal_state:
                continue
            
            old_action = self.policy[state]
            
            best_action = None
            best_q = -float('inf')
            
            for action in self.mdp.actions:
                next_state = self.mdp.transition(state, action)
                q = self.mdp.reward(state, action, goal_state) + \
                    self.gamma * self.value[next_state]
                
                if q > best_q:
                    best_q = q
                    best_action = action
            
            self.policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def train(self, goal_state):
        """训练"""
        iteration = 0
        while True:
            print(f"策略迭代第 {iteration + 1} 轮")
            self.policy_evaluation(goal_state)
            stable = self.policy_improvement(goal_state)
            
            if stable:
                print("策略稳定")
                break
            
            iteration += 1
```

---

## 4. D* Lite

### 4.1 D* Lite实现

```python
class DLite:
    """D* Lite算法"""
    
    def __init__(self, gridmap):
        self.map = gridmap
        self.g = defaultdict(lambda: float('inf'))
        self.rhs = defaultdict(lambda: float('inf'))
        self.queue = []
        self.key_dict = {}
        
        self.goal = None
        self.start = None
    
    def key(self, state):
        """计算键值"""
        min_g_rhs = min(self.g[state], self.rhs[state])
        k1 = min_g_rhs + self.h(state, self.goal)
        k2 = min_g_rhs
        return (k1, k2)
    
    def h(self, a, b):
        """启发式 (L1距离)"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def initialize(self, start, goal):
        """初始化"""
        self.start = start
        self.goal = goal
        
        self.g = defaultdict(lambda: float('inf'))
        self.rhs = defaultdict(lambda: float('inf'))
        self.queue = []
        self.key_dict = {}
        
        self.rhs[goal] = 0
        k = self.key(goal)
        self.queue.append((k, goal))
        self.key_dict[goal] = k
    
    def update_vertex(self, state):
        """更新顶点"""
        if state != self.goal:
            # 计算最小rhs
            min_rhs = float('inf')
            for action in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                gx, gy = state
                dx, dy = action
                ngx, ngy = gx + dx, gy + dy
                
                if 0 <= ngx < self.map.width and 0 <= ngy < self.map.height:
                    if not self.map.is_occupied(ngx, ngy):
                        cost = 1.0
                        min_rhs = min(min_rhs, cost + self.g[(ngx, ngy)])
            
            self.rhs[state] = min_rhs
        
        # 从队列移除
        if state in self.key_dict:
            # 简单的移除（实际需要更高效的结构）
            pass
        
        if self.g[state] != self.rhs[state]:
            k = self.key(state)
            self.queue.append((k, state))
            self.key_dict[state] = k
    
    def compute_shortest_path(self):
        """计算最短路径"""
        self.queue.sort()
        
        while self.queue and (
            self.queue[0][0] < self.key(self.start) or
            self.rhs[self.start] != self.g[self.start]
        ):
            k_old, u = self.queue.pop(0)
            k_new = self.key(u)
            
            if k_old < k_new:
                self.queue.append((k_new, u))
                self.queue.sort()
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                gx, gy = u
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ngx, ngy = gx + dx, gy + dy
                    if 0 <= ngx < self.map.width and 0 <= ngy < self.map.height:
                        if not self.map.is_occupied(ngx, ngy):
                            self.update_vertex((ngx, ngy))
            else:
                g_old = self.g[u]
                self.g[u] = float('inf')
                self.update_vertex(u)
                
                gx, gy = u
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ngx, ngy = gx + dx, gy + dy
                    if 0 <= ngx < self.map.width and 0 <= ngy < self.map.height:
                        if not self.map.is_occupied(ngx, ngy):
                            self.update_vertex((ngx, ngy))
            
            self.queue.sort()
    
    def extract_path(self, start, goal):
        """提取路径"""
        path = [start]
        current = start
        
        while current != goal:
            # 选择最佳后继
            gx, gy = current
            best_next = None
            best_g = float('inf')
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ngx, ngy = gx + dx, gy + dy
                if 0 <= ngx < self.map.width and 0 <= ngy < self.map.height:
                    if not self.map.is_occupied(ngx, ngy):
                        if self.g[(ngx, ngy)] < best_g:
                            best_g = self.g[(ngx, ngy)]
                            best_next = (ngx, ngy)
            
            if best_next is None:
                return None
            
            path.append(best_next)
            current = best_next
        
        return path
```

---

## 5. 实践练习

### 练习1：值迭代演示

```python
def exercise_value_iteration():
    """值迭代练习"""
    print("=== 值迭代 ===")
    
    # 创建简单地图
    from .01-environment-representation import OccupancyGrid
    
    grid = OccupancyGrid(10, 10, 1.0)  # 10x10 grid, 1m per cell
    
    # 添加障碍物
    for i in range(3, 7):
        grid.set_occupancy(i, 3, True)
        grid.set_occupancy(i, 6, True)
    
    for j in range(3, 7):
        grid.set_occupancy(3, j, True)
        grid.set_occupancy(6, j, True)
    
    # MDP
    mdp = GridMDP(grid)
    goal_state = (8, 8)
    
    # 值迭代
    vi = ValueIteration(mdp)
    vi.train(goal_state)
    
    # 可视化值函数
    value_grid = np.zeros((grid.height, grid.width))
    for gy in range(grid.height):
        for gx in range(grid.width):
            if (gx, gy) in vi.value:
                value_grid[gy, gx] = vi.value[(gx, gy)]
            else:
                value_grid[gy, gx] = -float('inf')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    im = plt.imshow(grid.grid, origin='lower', cmap='gray_r')
    plt.plot(1, 1, 'go', markersize=10, label='Start')
    plt.plot(8, 8, 'r*', markersize=15, label='Goal')
    plt.title('Grid World')
    plt.legend()
    
    plt.subplot(122)
    im = plt.imshow(value_grid, origin='lower')
    plt.colorbar(im, label='Value V(s)')
    plt.plot(1, 1, 'go', markersize=10)
    plt.plot(8, 8, 'r*', markersize=15)
    plt.title('Value Function')
    
    plt.savefig('value_iteration.png')
    print("值迭代演示已保存")

# exercise_value_iteration()
```

### 练习2：策略提取

```python
def exercise_policy_extraction():
    """策略提取"""
    print("=== 策略提取 ===")
    
    from .01-environment-representation import OccupancyGrid
    
    grid = OccupancyGrid(10, 10, 1.0)
    
    for i in range(3, 7):
        grid.set_occupancy(i, 3, True)
        grid.set_occupancy(i, 6, True)
    for j in range(3, 7):
        grid.set_occupancy(3, j, True)
        grid.set_occupancy(6, j, True)
    
    mdp = GridMDP(grid)
    goal_state = (8, 8)
    
    # 训练
    pi = PolicyIteration(mdp)
    pi.train(goal_state)
    
    # 可视化策略
    policy_grid = np.zeros((grid.height, grid.width), dtype=int)
    arrows = ['v', '^', '>', '<']
    
    for state in pi.policy:
        gx, gy = state
        action = pi.policy[state]
        if action == (0, 1):
            policy_grid[gy, gx] = 0
        elif action == (0, -1):
            policy_grid[gy, gx] = 1
        elif action == (1, 0):
            policy_grid[gy, gx] = 2
        elif action == (-1, 0):
            policy_grid[gy, gx] = 3
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.grid, origin='lower', cmap='gray_r', alpha=0.5)
    
    for gy in range(grid.height):
        for gx in range(grid.width):
            if (gx, gy) in pi.policy:
                a_idx = policy_grid[gy, gx]
                plt.text(gx, gy, arrows[a_idx], ha='center', va='center', fontsize=12)
    
    plt.plot(1, 1, 'go', markersize=10, label='Start')
    plt.plot(8, 8, 'r*', markersize=15, label='Goal')
    plt.title('Policy')
    plt.legend()
    plt.savefig('policy_extraction.png')
    print("策略图已保存")

# exercise_policy_extraction()
```

---

恭喜！你已经完成了路径规划的所有内容！

## 参考文献

1. Bellman, R. E. (1957). A Markovian Decision Process.
2. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.).
3. Koenig, S., & Likhachev, M. (2002). D* Lite.
4. Stentz, A. (1994). Optimal and Efficient Path Planning for Partially-Known Environments.
5. Bertsekas, D. P. (2012). Dynamic Programming and Optimal Control (4th ed.).
