# 8.1 决策理论

## 目录

- [1. 引言](#1-引言)
- [2. 效用理论](#2-效用理论)
- [3. 贝叶斯决策](#3-贝叶斯决策)
- [4. 决策树](#4-决策树)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 决策理论基础

决策是在不确定性下选择行动的过程。

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## 2. 效用理论

### 2.1 期望效用

```python
class ExpectedUtility:
    """期望效用理论"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def expected_value(outcomes, probabilities):
        """计算期望值"""
        return sum(o * p for o, p in zip(outcomes, probabilities))
    
    @staticmethod
    def expected_utility(outcomes, probabilities, utility_func):
        """计算期望效用"""
        return sum(utility_func(o) * p for o, p in zip(outcomes, probabilities))


class UtilityFunctions:
    """效用函数"""
    
    @staticmethod
    def linear_utility(x):
        """线性效用（风险中性）"""
        return x
    
    @staticmethod
    def logarithmic_utility(x):
        """对数效用（风险厌恶）"""
        return np.log(x) if x > 0 else -float('inf')
    
    @staticmethod
    def exponential_utility(x, eta=1.0):
        """指数效用（风险厌恶）"""
        return -np.exp(-eta * x)
```

---

## 3. 贝叶斯决策

### 3.1 贝叶斯决策理论

```python
class BayesianDecision:
    """贝叶斯决策"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def maximum_a_posteriori(prior, likelihood):
        """最大后验 (MAP) 决策"""
        # 计算后验
        posterior = prior * likelihood
        posterior = posterior / posterior.sum()
        
        return np.argmax(posterior)
    
    @staticmethod
    def minimum_expected_loss(prior, likelihood, loss_matrix):
        """最小期望损失决策"""
        n_states, n_actions = loss_matrix.shape
        
        expected_loss = np.zeros(n_actions)
        
        for a in range(n_actions):
            for s in range(n_states):
                posterior = prior[s] * likelihood[s]
                expected_loss[a] += posterior * loss_matrix[s, a]
        
        return np.argmin(expected_loss)
```

---

## 4. 决策树

### 4.1 决策树结构

```python
class DecisionNode:
    """决策节点"""
    
    def __init__(self, name):
        self.name = name
        self.children = []
    
    def add_child(self, branch_name, node):
        """添加子节点"""
        self.children.append((branch_name, node))
    
    def __repr__(self):
        return f"Decision({self.name})"


class ChanceNode:
    """机会节点"""
    
    def __init__(self, name, probabilities=None):
        self.name = name
        self.children = []
        self.probabilities = probabilities if probabilities else []
    
    def add_child(self, outcome_name, probability, node):
        """添加子节点"""
        self.children.append((outcome_name, probability, node))
        self.probabilities.append(probability)
    
    def __repr__(self):
        return f"Chance({self.name})"


class TerminalNode:
    """终端节点"""
    
    def __init__(self, name, utility):
        self.name = name
        self.utility = utility
    
    def __repr__(self):
        return f"Terminal({self.name}, U={self.utility})"


class DecisionTree:
    """决策树"""
    
    def __init__(self, root):
        self.root = root
    
    def evaluate(self, node=None):
        """评估树（计算期望效用）"""
        if node is None:
            node = self.root
        
        if isinstance(node, TerminalNode):
            return node.utility
        
        if isinstance(node, ChanceNode):
            # 加权平均
            expected_u = 0
            for _, p, child in node.children:
                expected_u += p * self.evaluate(child)
            return expected_u
        
        if isinstance(node, DecisionNode):
            # 最大效用
            max_u = -float('inf')
            best_action = None
            
            for name, child in node.children:
                u = self.evaluate(child)
                if u > max_u:
                    max_u = u
                    best_action = name
            
            return max_u
```

---

## 5. 实践练习

### 练习1：期望效用示例

```python
def exercise_utility():
    """效用理论练习"""
    print("=== 效用理论 ===")
    
    # 选项A: 确定获得 100
    outcomes_a = [100]
    probs_a = [1.0]
    
    # 选项B: 50%获得 200, 50%获得 0
    outcomes_b = [200, 0]
    probs_b = [0.5, 0.5]
    
    print("选项A: 确定拿到 $100")
    print("选项B: 50%拿 $200, 50%拿 $0")
    print()
    
    ev_a = ExpectedUtility.expected_value(outcomes_a, probs_a)
    ev_b = ExpectedUtility.expected_value(outcomes_b, probs_b)
    print(f"期望值 A: {ev_a}, B: {ev_b}")
    
    # 线性效用（风险中性）
    eu_a_linear = ExpectedUtility.expected_utility(outcomes_a, probs_a, 
                                                     UtilityFunctions.linear_utility)
    eu_b_linear = ExpectedUtility.expected_utility(outcomes_b, probs_b, 
                                                     UtilityFunctions.linear_utility)
    print(f"线性效用 A: {eu_a_linear:.1f}, B: {eu_b_linear:.1f}")
    
    # 对数效用（风险厌恶）
    eu_a_log = ExpectedUtility.expected_utility(outcomes_a, probs_a, 
                                                  lambda x: UtilityFunctions.logarithmic_utility(x))
    eu_b_log = ExpectedUtility.expected_utility(outcomes_b, probs_b, 
                                                  lambda x: UtilityFunctions.logarithmic_utility(x))
    print(f"对数效用 A: {eu_a_log:.1f}, B: {eu_b_log:.1f}")
    print("风险厌恶者会选选项A！")

# exercise_utility()
```

### 练习2：简单决策树

```python
def exercise_decision_tree():
    """决策树练习"""
    print("=== 决策树示例 ===")
    
    # 构建决策树
    # 决策: 带伞还是不带伞
    decision_node = DecisionNode("带伞?")
    
    # 分支: 带伞
    chance_rain_with = ChanceNode("下雨?")
    chance_rain_with.add_child("下雨", 0.3, TerminalNode("有点麻烦但不湿", 5))
    chance_rain_with.add_child("不下雨", 0.7, TerminalNode("白带了但没事", 7))
    decision_node.add_child("带伞", chance_rain_with)
    
    # 分支: 不带伞
    chance_rain_without = ChanceNode("下雨?")
    chance_rain_without.add_child("下雨", 0.3, TerminalNode("淋湿", 0))
    chance_rain_without.add_child("不下雨", 0.7, TerminalNode("轻松", 10))
    decision_node.add_child("不带伞", chance_rain_without)
    
    # 评估
    tree = DecisionTree(decision_node)
    print("决策树已构建")
    print(f"最优期望效用: {tree.evaluate():.1f}")

# exercise_decision_tree()
```

---

**下一节**：[马尔可夫决策过程](02-markov-decision-process.md)

---

## 参考文献

1. von Neumann, J., & Morgenstern, O. (1944). Theory of Games and Economic Behavior.
2. Savage, L. J. (1954). The Foundations of Statistics.
3. Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach (4th ed.).
4. Berger, J. O. (1985). Statistical Decision Theory and Bayesian Analysis.
