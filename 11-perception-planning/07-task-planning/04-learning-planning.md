# 7.4 学习型规划

## 目录

- [1. 引言](#1-引言)
- [2. 强化学习规划](#2-强化学习规划)
- [3. 模仿学习](#3-模仿学习)
- [4. 神经符号规划](#4-神经符号规划)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 为什么学习型规划？

传统规划的限制：
- 需要手工设计领域知识
- 难以处理不确定性
- 难以迁移到新领域

```python
import numpy as np
import random
```

---

## 2. 强化学习规划

### 2.1 基于Q-learning的规划

```python
class QLearningPlanner:
    """Q-learning规划器"""
    
    def __init__(self, state_dim, action_dim, lr=0.1, gamma=0.99, epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = defaultdict(lambda: np.zeros(action_dim))
    
    def select_action(self, state):
        """epsilon-greedy选择"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return int(np.argmax(self.Q[state]))
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning更新"""
        q_old = self.Q[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        self.Q[state][action] = q_old + self.lr * (target - q_old)
    
    def plan(self, state):
        """规划（推理）"""
        return int(np.argmax(self.Q[state]))
```

### 2.2 Value Iteration (规划用)

```python
class MDPPlanner:
    """基于MDP的规划"""
    
    def __init__(self, states, actions, transition, reward, gamma=0.99):
        self.states = states
        self.actions = actions
        self.transition = transition
        self.reward = reward
        self.gamma = gamma
        
        self.V = {s: 0 for s in states}
        self.policy = {s: 0 for s in states}
    
    def value_iteration(self, tol=1e-6, max_iter=1000):
        """值迭代"""
        for i in range(max_iter):
            delta = 0
            new_V = {}
            
            for s in self.states:
                max_q = -float('inf')
                for a in self.actions:
                    q = self._compute_q(s, a)
                    if q > max_q:
                        max_q = q
                
                new_V[s] = max_q
                delta = max(delta, abs(new_V[s] - self.V[s]))
            
            self.V = new_V
            
            if delta < tol:
                print(f"值迭代在 {i+1} 轮收敛")
                break
        
        # 提取策略
        for s in self.states:
            best_action = None
            best_q = -float('inf')
            for a in self.actions:
                q = self._compute_q(s, a)
                if q > best_q:
                    best_q = q
                    best_action = a
            self.policy[s] = best_action
    
    def _compute_q(self, state, action):
        """计算Q值"""
        q = 0
        for next_state, prob in self.transition(state, action).items():
            r = self.reward(state, action, next_state)
            q += prob * (r + self.gamma * self.V[next_state])
        return q
```

---

## 3. 模仿学习

### 3.1 行为克隆

```python
class BehaviorCloning:
    """行为克隆"""
    
    def __init__(self, model=None):
        self.model = model
    
    def train(self, demonstrations):
        """从演示学习"""
        # demonstrations: [(state, action)] list
        states = []
        actions = []
        for s, a in demonstrations:
            states.append(s)
            actions.append(a)
        
        # 简单实现：最近邻
        self.states = states
        self.actions = actions
    
    def predict(self, state):
        """预测动作"""
        # 最近邻
        best_idx = 0
        best_dist = float('inf')
        
        for i, s in enumerate(self.states):
            dist = np.linalg.norm(np.array(state) - np.array(s))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        return self.actions[best_idx]
```

---

## 4. 神经符号规划

### 4.1 简单神经符号搜索

```python
class NeuralSymbolicPlanner:
    """神经符号规划器"""
    
    def __init__(self, heuristic_model, base_planner):
        self.heuristic_model = heuristic_model
        self.base_planner = base_planner
    
    def plan(self, state, goal):
        """规划（神经启发式+符号搜索）"""
        # 用神经网络预测启发式
        heuristic = self.heuristic_model.predict((state, goal))
        
        # 传递给A*
        plan = self.base_planner.plan(state, goal, heuristic)
        return plan
```

---

## 5. 实践练习

### 练习1：QLearning简单示例

```python
def exercise_rl_planning():
    """强化学习规划练习"""
    print("=== 学习型规划 (Q-learning) ===")
    
    # 简单的网格世界
    # 状态: (x,y)，动作: 上下左右
    grid_size = 5
    states = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    actions = [0, 1, 2, 3]  # up, down, left, right
    
    def transition(state, action):
        """简单转移函数（确定）"""
        x, y = state
        dx, dy = [(0,-1), (0,1), (-1,0), (1,0)][action]
        
        x_new = max(0, min(grid_size-1, x + dx))
        y_new = max(0, min(grid_size-1, y + dy))
        
        return {(x_new, y_new): 1.0}
    
    def reward(state, action, next_state):
        """奖励"""
        goal = (grid_size-1, grid_size-1)
        if next_state == goal:
            return 10
        return -0.1
    
    # Q learning
    planner = QLearningPlanner(state_dim=2, action_dim=4, epsilon=0.2)
    
    # 训练
    goal = (grid_size-1, grid_size-1)
    start = (0, 0)
    
    for episode in range(500):
        state = start
        total_reward = 0
        
        for t in range(50):
            # 简化状态表示
            state_key = state
            
            action = planner.select_action(state_key)
            
            next_state = list(transition(state, action).keys())[0]
            r = reward(state, action, next_state)
            done = (next_state == goal)
            
            planner.update(state_key, action, r, next_state, done)
            
            total_reward += r
            state = next_state
            
            if done:
                break
        
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.1f}")
    
    print("\n训练完成!")
    
    # 测试
    state = start
    path = [state]
    for t in range(20):
        state_key = state
        action = np.argmax(planner.Q[state_key])
        next_state = list(transition(state, action).keys())[0]
        path.append(next_state)
        
        if next_state == goal:
            break
        state = next_state
    
    print(f"路径: {path}")

# exercise_rl_planning()
```

---

**下一节**：[大模型规划](05-llm-planning.md)

---

## 参考文献

1. Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search.
2. Levine, S., et al. (2016). End-to-end training of deep visuomotor policies.
3. Ho, J., & Ermon, S. (2016). Generative Adversarial Imitation Learning.
4. Garnelo, M., et al. (2016). Towards Deep Symbolic Reinforcement Learning.
5. Kaelbling, L. P., et al. (1996). Reinforcement Learning: A Survey.
