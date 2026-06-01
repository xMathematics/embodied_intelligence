# 8.5 在线决策

## 目录

- [1. 引言](#1-引言)
- [2. 多臂老虎机](#2-多臂老虎机)
- [3. 强化学习在线决策](#3-强化学习在线决策)
- [4. 在线学习后悔分析](#4-在线学习后悔分析)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 在线决策

在线决策需要在不确定下**实时**决策：
- 数据随时间到达
- 决策会影响未来

```python
import numpy as np
import matplotlib.pyplot as plt
import random
```

---

## 2. 多臂老虎机

### 2.1 经典算法

```python
class Bandit:
    """多臂老虎机"""
    
    def __init__(self, n_arms, true_rewards):
        self.n_arms = n_arms
        self.true_rewards = true_rewards
    
    def pull(self, arm):
        """拉臂，返回随机奖励"""
        return np.random.normal(self.true_rewards[arm], 1.0)


class EpsilonGreedy:
    """Epsilon-Greedy策略"""
    
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms)
    
    def select_arm(self):
        """选择臂"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)
        else:
            return np.argmax(self.values)
    
    def update(self, arm, reward):
        """更新估计"""
        self.counts[arm] += 1
        n = self.counts[arm]
        val = self.values[arm]
        self.values[arm] = val + (reward - val) / n


class UCB:
    """UCB (Upper Confidence Bound)"""
    
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms)
        self.total_steps = 0
    
    def select_arm(self):
        """选择臂"""
        # 先确保每个臂都拉过一次
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a
        
        # UCB公式
        ucb_values = []
        for a in range(self.n_arms):
            n_a = self.counts[a]
            ucb = self.values[a] + self.c * np.sqrt(np.log(self.total_steps) / n_a)
            ucb_values.append(ucb)
        
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        """更新"""
        self.total_steps += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        val = self.values[arm]
        self.values[arm] = val + (reward - val) / n


class ThompsonSampling:
    """Thompson Sampling"""
    
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)
    
    def select_arm(self):
        """采样Beta分布选择"""
        theta_samples = [np.random.beta(self.successes[a], self.failures[a]) 
                         for a in range(self.n_arms)]
        return np.argmax(theta_samples)
    
    def update(self, arm, reward):
        """更新（假设是0-1伯努利奖励）"""
        if reward > 0.5:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
```

---

## 3. 强化学习在线决策

### 3.1 在线TD学习

```python
class OnlineTD:
    """在线时序差分学习 (TD(0))"""
    
    def __init__(self, states, actions, alpha=0.1, gamma=0.99):
        self.Q = {}
        for s in states:
            for a in actions:
                self.Q[(s, a)] = 0
        
        self.alpha = alpha
        self.gamma = gamma
    
    def update(self, s, a, r, s_next):
        """在线更新"""
        # 贪婪地选下一个动作（Q学习）
        max_q = max(self.Q.get((s_next, a_prime), 0) for a_prime in actions)
        
        self.Q[(s, a)] += self.alpha * (r + self.gamma * max_q - self.Q[(s, a)])
    
    def select_action(self, s, epsilon=0.1):
        """Epsilon-Greedy选择"""
        if random.random() < epsilon:
            return random.randint(0, 3)
        
        q_values = [self.Q.get((s, a), 0) for a in range(4)]
        return np.argmax(q_values)
```

---

## 4. 在线学习后悔分析

### 4.1 后悔计算

```python
class OnlineLearner:
    """在线学习器基类"""
    
    def __init__(self):
        self.cumulative_regret = 0
    
    def compute_regret(self, chosen_reward, best_reward):
        """计算单次后悔"""
        regret = best_reward - chosen_reward
        self.cumulative_regret += regret
        return regret
```

---

## 5. 实践练习

### 练习1：多臂老虎机对比

```python
def exercise_bandits():
    """多臂老虎机练习"""
    print("=== 多臂老虎机 ===")
    
    n_arms = 5
    true_rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bandit = Bandit(n_arms, true_rewards)
    
    n_steps = 1000
    
    # 算法对比
    algorithms = [
        ("Epsilon-Greedy", EpsilonGreedy(n_arms, epsilon=0.1)),
        ("UCB", UCB(n_arms, c=2.0)),
        ("Thompson", ThompsonSampling(n_arms))
    ]
    
    results = {}
    for name, algo in algorithms:
        regrets = []
        regrets_learner = OnlineLearner()
        rewards = []
        
        for _ in range(n_steps):
            arm = algo.select_arm()
            reward = bandit.pull(arm)
            algo.update(arm, reward)
            
            # 计算后悔
            best_r = true_rewards.max()
            regret = regrets_learner.compute_regret(reward, best_r)
            rewards.append(reward)
            regrets.append(regret)
        
        results[name] = {
            'rewards': rewards,
            'regrets': regrets
        }
        print(f"{name}: 平均奖励 = {np.mean(rewards):.2f}")
        print(f"        最终累计后悔 = {regrets_learner.cumulative_regret:.1f}")
    
    # 画图
    plt.figure(figsize=(12, 6))
    
    for name, data in results.items():
        regrets = np.array(data['regrets'])
        plt.plot(np.cumsum(regrets), label=name)
    
    plt.xlabel('Step')
    plt.ylabel('Cumulative Regret')
    plt.title('Bandit Algorithm Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('bandit_comparison.png')
    print()
    print("结果图已保存")

# exercise_bandits()
```

---

恭喜！你已经完成了第八部分：决策方法（1周）！

---

## 参考文献

1. Robbins, H. (1952). Some Aspects of the Sequential Design of Experiments.
2. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem.
3. Thompson, W. R. (1933). On the Likelihood that One Unknown Probability Exceeds Another in View of Two Samples.
4. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction.
5. Cesa-Bianchi, N., & Lugosi, G. (2006). Prediction, Learning, and Games.
