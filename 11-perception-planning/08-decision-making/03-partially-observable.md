# 8.3 部分可观测MDP

## 目录

- [1. 引言](#1-引言)
- [2. POMDP定义](#2-pomdp定义)
- [3. 信念状态](#3-信念状态)
- [4. 求解方法](#4-求解方法)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 从MDP到POMDP

POMDP考虑了**部分可观测性**：
- 不能直接观测状态 s
- 只能通过观测 z 来推断
- 需要维护**信念状态** b

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
```

---

## 2. POMDP定义

### 2.1 POMDP类

```python
class POMDP:
    """部分可观测马尔可夫决策过程"""
    
    def __init__(self, states, actions, observations, 
                 transition, reward, observation, 
                 initial_belief, gamma=0.99):
        self.states = states
        self.actions = actions
        self.observations = observations
        self.transition = transition
        self.reward = reward
        self.observation = observation
        self.initial_belief = initial_belief
        self.gamma = gamma


class SimplePOMDP(POMDP):
    """简单示例POMDP"""
    
    def __init__(self):
        states = ['good', 'bad']
        actions = ['wait', 'act']
        observations = ['ok', 'warning']
        
        # 转移
        def transition(s, a):
            if a == 'wait':
                if s == 'good':
                    return {'good': 0.9, 'bad': 0.1}
                else:
                    return {'good': 0.1, 'bad': 0.9}
            else:
                return {'good': 0.7, 'bad': 0.3}
        
        # 奖励
        def reward(s, a, s_next):
            if a == 'act' and s == 'bad':
                return 10
            elif a == 'act':
                return -1
            return 0
        
        # 观测
        def observation(s_next, a):
            if s_next == 'good':
                return {'ok': 0.9, 'warning': 0.1}
            else:
                return {'ok': 0.2, 'warning': 0.8}
        
        initial_belief = {'good': 0.5, 'bad': 0.5}
        
        super().__init__(states, actions, observations, 
                         transition, reward, observation, 
                         initial_belief, gamma=0.9)
```

---

## 3. 信念状态

### 3.1 信念更新

```python
class BeliefState:
    """信念状态"""
    
    def __init__(self, pomdp):
        self.pomdp = pomdp
        self.belief = pomdp.initial_belief.copy()
    
    def update(self, action, observation):
        """信念更新 (Bayesian filtering)"""
        new_belief = {}
        
        # 预测
        predict = {}
        for s in self.pomdp.states:
            predict[s] = 0
            for s_prev in self.pomdp.states:
                t = self.pomdp.transition(s_prev, action)
                predict[s] += self.belief[s_prev] * t.get(s, 0)
        
        # 更新
        total_prob = 0
        for s in self.pomdp.states:
            o_prob = self.pomdp.observation(s, action)[observation]
            new_belief[s] = predict[s] * o_prob
            total_prob += new_belief[s]
        
        # 归一化
        if total_prob > 0:
            for s in self.pomdp.states:
                new_belief[s] /= total_prob
        
        self.belief = new_belief
        return self.belief
    
    def entropy(self):
        """计算信念熵"""
        ent = 0
        for s in self.pomdp.states:
            if self.belief[s] > 0:
                ent -= self.belief[s] * np.log2(self.belief[s])
        return ent
```

---

## 4. 求解方法

### 4.1 QMDP (确定性近似)

```python
class QMDP:
    """QMDP: 忽略未来部分可观测性"""
    
    def __init__(self, pomdp):
        self.pomdp = pomdp
        self.V = {}
        self.Q = {}
        
        # 先求解MDP
        self._solve_mdp()
    
    def _solve_mdp(self):
        """求解底层MDP"""
        gamma = self.pomdp.gamma
        
        # 值迭代
        V = {s: 0 for s in self.pomdp.states}
        
        for _ in range(100):
            new_V = V.copy()
            for s in self.pomdp.states:
                max_q = -float('inf')
                for a in self.pomdp.actions:
                    q = 0
                    for s_next, p in self.pomdp.transition(s, a).items():
                        r = self.pomdp.reward(s, a, s_next)
                        q += p * (r + gamma * V[s_next])
                    max_q = max(max_q, q)
                new_V[s] = max_q
            V = new_V
        
        self.V = V
        
        # Q值
        Q = {}
        for s in self.pomdp.states:
            Q[s] = {}
            for a in self.pomdp.actions:
                q = 0
                for s_next, p in self.pomdp.transition(s, a).items():
                    r = self.pomdp.reward(s, a, s_next)
                    q += p * (r + gamma * V[s_next])
                Q[s][a] = q
        self.Q = Q
    
    def select_action(self, belief):
        """根据信念选择动作"""
        best_action = None
        best_v = -float('inf')
        
        for a in self.pomdp.actions:
            v = 0
            for s in self.pomdp.states:
                v += belief[s] * self.Q[s][a]
            
            if v > best_v:
                best_v = v
                best_action = a
        
        return best_action
```

---

## 5. 实践练习

### 练习1：信念更新示例

```python
def exercise_belief_update():
    """信念更新练习"""
    print("=== 部分可观测MDP (POMDP) ===")
    
    pomdp = SimplePOMDP()
    belief = BeliefState(pomdp)
    
    print(f"初始信念: {belief.belief}")
    print(f"初始熵: {belief.entropy():.2f}")
    print()
    
    # 模拟几个步骤
    trajectory = [
        ('wait', 'ok'),
        ('wait', 'ok'),
        ('wait', 'warning'),
        ('act', 'ok')
    ]
    
    for a, o in trajectory:
        belief.update(a, o)
        print(f"行动: {a}, 观测: {o}")
        print(f"信念: {belief.belief}")
        print(f"熵: {belief.entropy():.2f}")
        print()

# exercise_belief_update()
```

### 练习2：QMDP决策

```python
def exercise_qmdp():
    """QMDP练习"""
    print("=== QMDP求解 ===")
    
    pomdp = SimplePOMDP()
    qmdp = QMDP(pomdp)
    
    print("底层MDP的Q值:")
    for s in pomdp.states:
        for a in pomdp.actions:
            print(f"  Q({s},{a}) = {qmdp.Q[s][a]:.2f}")
    
    print()
    belief = BeliefState(pomdp)
    a = qmdp.select_action(belief.belief)
    print(f"初始信念下选择: {a}")
    
    belief.update('wait', 'warning')
    a = qmdp.select_action(belief.belief)
    print(f"看到警告后选择: {a}")

# exercise_qmdp()
```

---

**下一节**：[多目标决策](04-multi-objective.md)

---

## 参考文献

1. Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and Acting in Partially Observable Stochastic Domains.
2. Lovejoy, W. S. (1991). A Survey of Algorithmic Methods for Partially Observable Markov Decision Processes.
3. Pineau, J., Gordon, G., & Thrun, S. (2003). Point-based Value Iteration: An Anytime Algorithm for POMDPs.
4. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics.
