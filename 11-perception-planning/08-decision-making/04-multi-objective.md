# 8.4 多目标决策

## 目录

- [1. 引言](#1-引言)
- [2. 帕累托最优](#2-帕累托最优)
- [3. 效用函数](#3-效用函数)
- [4. 多目标优化](#4-多目标优化)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 多目标问题

现实中常有多个相互冲突的目标：
- 速度 vs 安全
- 成本 vs 质量
- 探索 vs 利用

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## 2. 帕累托最优

### 2.1 帕累托支配

```python
class ParetoAnalysis:
    """帕累托分析"""
    
    @staticmethod
    def dominates(a, b, minimize=True):
        """判断a是否支配b"""
        n = len(a)
        if minimize:
            at_least_as_good = all(ai <= bi for ai, bi in zip(a, b))
            strictly_better = any(ai < bi for ai, bi in zip(a, b))
        else:
            at_least_as_good = all(ai >= bi for ai, bi in zip(a, b))
            strictly_better = any(ai > bi for ai, bi in zip(a, b))
        
        return at_least_as_good and strictly_better
    
    @staticmethod
    def find_pareto_front(objectives, minimize=True):
        """寻找帕累托前沿"""
        front = []
        
        for i, a in enumerate(objectives):
            is_dominated = False
            
            for j, b in enumerate(objectives):
                if i != j and ParetoAnalysis.dominates(b, a, minimize):
                    is_dominated = True
                    break
            
            if not is_dominated:
                front.append(a)
        
        return front
```

---

## 3. 效用函数

### 3.1 线性加权

```python
class LinearWeighted:
    """线性加权求和"""
    
    def __init__(self, weights):
        self.weights = np.array(weights)
    
    def utility(self, objectives):
        """计算效用"""
        return np.dot(self.weights, objectives)
    
    def select_best(self, candidates):
        """选择最佳"""
        best_u = -float('inf')
        best = None
        
        for cand in candidates:
            u = self.utility(cand)
            if u > best_u:
                best_u = u
                best = cand
        
        return best


class LexicographicOrder:
    """字典序排序"""
    
    def __init__(self, priorities):
        self.priorities = priorities
    
    def select_best(self, candidates):
        """按优先级选择"""
        current = candidates
        
        for idx in self.priorities:
            if len(current) <= 1:
                break
            
            best_val = max(c[idx] for c in current)
            current = [c for c in current if c[idx] == best_val]
        
        return current[0] if current else None
```

---

## 4. 多目标优化

### 4.1 多目标优化方法

```python
class ConstrainedMethod:
    """约束法：优化一个，约束其他"""
    
    def __init__(self, primary_idx, constraints):
        self.primary_idx = primary_idx
        self.constraints = constraints
    
    def select_feasible(self, candidates):
        """筛选可行解"""
        feasible = []
        
        for cand in candidates:
            ok = True
            
            for idx, min_val in self.constraints.items():
                if cand[idx] < min_val:
                    ok = False
                    break
            
            if ok:
                feasible.append(cand)
        
        return feasible
    
    def select_best(self, candidates):
        """选择最佳"""
        feasible = self.select_feasible(candidates)
        if not feasible:
            return None
        
        return max(feasible, key=lambda x: x[self.primary_idx])
```

---

## 5. 实践练习

### 练习1：帕累托前沿

```python
def exercise_pareto():
    """帕累托分析练习"""
    print("=== 帕累托最优 ===")
    
    # 示例候选解（双目标：越小越好）
    candidates = [
        (1, 9), (2, 7), (3, 5), (4, 4), (5, 3),
        (6, 2), (7, 1), (2, 5), (3, 4), (5, 2)
    ]
    
    print("候选解:")
    for i, c in enumerate(candidates):
        print(f"  {i}: {c}")
    
    # 找帕累托前沿
    front = ParetoAnalysis.find_pareto_front(candidates, minimize=True)
    print()
    print("帕累托前沿解:")
    for c in front:
        print(f"  {c}")
    
    # 可视化
    cand_array = np.array(candidates)
    front_array = np.array(front)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(cand_array[:, 0], cand_array[:, 1], 
                c='gray', s=50, alpha=0.5, label='Dominated')
    plt.scatter(front_array[:, 0], front_array[:, 1], 
                c='red', s=100, marker='*', label='Pareto Front')
    plt.xlabel('Objective 1 (min)')
    plt.ylabel('Objective 2 (min)')
    plt.title('Pareto Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig('pareto_front.png')
    print()
    print("帕累托图已保存")

# exercise_pareto()
```

### 练习2：多目标决策

```python
def exercise_multiobjective():
    """多目标决策练习"""
    print("=== 多目标决策 ===")
    
    # 候选：(速度, 安全性, 成本)，越大越好
    candidates = [
        (8, 5, 4),   # 快，中等，便宜
        (5, 8, 6),   # 中等，安全，较贵
        (6, 6, 5),   # 平衡
        (7, 7, 3),   # 好，但贵
        (4, 9, 7)    # 慢，很安全，很贵
    ]
    
    # 线性加权
    print("线性加权选择:")
    print("方案1: 看重速度")
    w1 = [0.6, 0.3, 0.1]
    best1 = LinearWeighted(w1).select_best(candidates)
    print(f"  权重: {w1}")
    print(f"  选择: {best1}")
    
    print()
    print("方案2: 看重安全")
    w2 = [0.1, 0.7, 0.2]
    best2 = LinearWeighted(w2).select_best(candidates)
    print(f"  权重: {w2}")
    print(f"  选择: {best2}")
    
    print()
    print("字典序选择 (安全>速度>成本):")
    lex = LexicographicOrder([1, 0, 2])
    best_lex = lex.select_best(candidates)
    print(f"  选择: {best_lex}")

# exercise_multiobjective()
```

---

**下一节**：[在线决策](05-online-decision.md)

---

## 参考文献

1. Pareto, V. (1906). Manual of Political Economy.
2. Zeleny, M. (1982). Multiple Criteria Decision Making.
3. Branke, J., et al. (2008). Multiobjective Optimization: Interactive and Evolutionary Approaches.
4. Deb, K. (2001). Multi-Objective Optimization Using Evolutionary Algorithms.
