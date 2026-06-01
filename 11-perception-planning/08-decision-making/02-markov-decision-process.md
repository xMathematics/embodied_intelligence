# 8.2 马尔可夫决策过程

## 目录

- [1. 引言](#1-引言)
- [2. MDP定义](#2-mdp定义)
- [3. 值迭代与策略迭代](#3-值迭代与策略迭代)
- [4. 策略评估](#4-策略评估)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 MDP与规划

MDP是序列决策的标准框架：
- **状态** s ∈ S
- **动作** a ∈ A
- **转移** P(s' | s, a)
- **奖励** R(s, a, s')

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## 2. MDP定义

### 2.1 MDP类

```python
class MDP:
    """马尔可夫决策过程"""
    
    def __init__(self, states, actions, transition, reward, gamma=0.99):
        self.states = states
        self.actions = actions
        self.transition = transition
        self.reward = reward
        self.gamma = gamma


class GridWorldMDP(MDP):
    """网格世界 MDP"""
    
    def __init__(self, grid_size=4, gamma=0.99):
        self.grid_size = grid_size
        states = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        
        actions = [0, 1, 2, 3]  # up, down, left, right
        
        self.gamma = gamma
        
        # 初始化为super的简化版
        self.states = states
        self.actions = actions
        self.gamma = gamma
    
    def transition(self, state, action):
        """转移函数"""
        x, y = state
        dx, dy = [(0,-1), (0,1), (-1,0), (1,0)][action]
        
        new_x = max(0, min(self.grid_size - 1, x + dx))
        new_y = max(0, min(self.grid_size - 1, y + dy))
        
        return {(new_x, new_y): 1.0}
    
    def reward(self, state, action, next_state):
        """奖励函数"""
        goal = (self.grid_size - 1, self.grid_size - 1)
        
        if next_state == goal:
            return 10
        
        # 障碍惩罚
        if next_state[0] == 2 and next_state[1] == 2:
            return -10
        
        return -0.1
```

---

## 3. 值迭代与策略迭代

### 3.1 值迭代

```python
class ValueIteration:
    """值迭代"""
    
    def __init__(self, mdp):
        self.mdp = mdp
        self.V = {s: 0 for s in mdp.states}
        self.policy = None
    
    def iterate(self, tol=1e-6, max_iter=1000):
        """迭代"""
        for i in range(max_iter):
            delta = 0
            new_V = {}
            
            for s in self.mdp.states:
                max_q = -float('inf')
                
                for a in self.mdp.actions:
                    q = self._compute_q(s, a)
                    if q > max_q:
                        max_q = q
                
                new_V[s] = max_q
                delta = max(delta, abs(new_V[s] - self.V[s]))
            
            self.V = new_V
            
            if delta < tol:
                print(f"值迭代在 {i+1} 轮收敛")
                break
        
        self.extract_policy()
    
    def _compute_q(self, state, action):
        """计算Q值"""
        q = 0
        for next_state, prob in self.mdp.transition(state, action).items():
            r = self.mdp.reward(state, action, next_state)
            q += prob * (r + self.mdp.gamma * self.V[next_state])
        return q
    
    def extract_policy(self):
        """提取策略"""
        self.policy = {}
        
        for s in self.mdp.states:
            best_a = None
            max_q = -float('inf')
            
            for a in self.mdp.actions:
                q = self._compute_q(s, a)
                if q > max_q:
                    max_q = q
                    best_a = a
            
            self.policy[s] = best_a
```

### 3.2 策略迭代

```python
class PolicyIteration:
    """策略迭代"""
    
    def __init__(self, mdp):
        self.mdp = mdp
        self.V = {s: 0 for s in mdp.states}
        # 随机初始化策略
        self.policy = {s: np.random.choice(mdp.actions) for s in mdp.states}
    
    def policy_evaluation(self, tol=1e-6):
        """策略评估"""
        while True:
            delta = 0
            new_V = {}
            
            for s in self.mdp.states:
                a = self.policy[s]
                q = 0
                
                for next_state, prob in self.mdp.transition(s, a).items():
                    r = self.mdp.reward(s, a, next_state)
                    q += prob * (r + self.mdp.gamma * self.V[next_state])
                
                new_V[s] = q
                delta = max(delta, abs(new_V[s] - self.V[s]))
            
            self.V = new_V
            
            if delta < tol:
                break
    
    def policy_improvement(self):
        """策略改进"""
        policy_stable = True
        
        for s in self.mdp.states:
            old_a = self.policy[s]
            
            best_a = None
            max_q = -float('inf')
            
            for a in self.mdp.actions:
                q = 0
                for next_state, prob in self.mdp.transition(s, a).items():
                    r = self.mdp.reward(s, a, next_state)
                    q += prob * (r + self.mdp.gamma * self.V[next_state])
                
                if q > max_q:
                    max_q = q
                    best_a = a
            
            self.policy[s] = best_a
            
            if best_a != old_a:
                policy_stable = False
        
        return policy_stable
    
    def iterate(self, max_iter=100):
        """策略迭代"""
        for i in range(max_iter):
            self.policy_evaluation()
            stable = self.policy_improvement()
            
            print(f"策略迭代第 {i+1} 轮")
            
            if stable:
                print("策略稳定！")
                break
```

---

## 4. 策略评估

### 4.1 可视化工具

```python
class MDPVisualizer:
    """MDP可视化"""
    
    @staticmethod
    def plot_value_function(V, grid_size):
        """绘制值函数"""
        grid = np.zeros((grid_size, grid_size))
        
        for (x, y), v in V.items():
            grid[y, x] = v
        
        plt.figure(figsize=(8, 8))
        plt.imshow(grid, origin='lower', cmap='viridis')
        plt.colorbar(label='Value')
        plt.title('Value Function')
        plt.grid(True, alpha=0.3)
        plt.savefig('value_function.png')
        print("值函数图已保存")
    
    @staticmethod
    def plot_policy(policy, grid_size):
        """绘制策略"""
        directions = ['↑', '↓', '←', '→']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for (x, y), a in policy.items():
            ax.text(x, y, directions[a], ha='center', va='center', 
                    fontsize=16, fontweight='bold')
        
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_title('Policy')
        ax.grid(True)
        plt.savefig('policy.png')
        print("策略图已保存")
```

---

## 5. 实践练习

### 练习1：网格世界值迭代

```python
def exercise_value_iteration():
    """值迭代练习"""
    print("=== 值迭代 (MDP) ===")
    
    mdp = GridWorldMDP(grid_size=4)
    vi = ValueIteration(mdp)
    vi.iterate()
    
    print("值函数:")
    for s in sorted(vi.V.items()):
        print(f"  {s[0]}: {s[1]:.2f}")
    
    MDPVisualizer.plot_value_function(vi.V, mdp.grid_size)
    MDPVisualizer.plot_policy(vi.policy, mdp.grid_size)

# exercise_value_iteration()
```

### 练习2：策略迭代

```python
def exercise_policy_iteration():
    """策略迭代练习"""
    print("=== 策略迭代 (MDP) ===")
    
    mdp = GridWorldMDP(grid_size=4)
    pi = PolicyIteration(mdp)
    pi.iterate()
    
    print("策略:")
    sorted_states = sorted(pi.policy.items())
    for s, a in sorted_states:
        print(f"  {s} -> {['↑', '↓', '←', '→'][a]}")

# exercise_policy_iteration()
```

---

**下一节**：[部分可观测MDP](03-partially-observable.md)

---

## 参考文献

1. Bellman, R. (1957). A Markovian Decision Process.
2. Howard, R. A. (1960). Dynamic Programming and Markov Processes.
3. Puterman, M. L. (1994). Markov Decision Processes: Discrete Stochastic Dynamic Programming.
4. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.).
