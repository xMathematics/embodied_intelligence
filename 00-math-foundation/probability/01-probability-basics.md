# 3.1 概率基础

## 目录

- [1. 概率的定义](#1-概率的定义)
- [2. 条件概率与独立性](#2-条件概率与独立性)
- [3. 概率公理](#3-概率公理)
- [4. 常见概率模型](#4-常见概率模型)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 概率的定义

### 1.1 古典概率

**定义**：在有限等可能样本空间中，事件 $A$ 的概率为：

$$P(A) = \frac{\text{事件} A \text{包含的样本点数}}{\text{样本空间的总样本点数}}$$

**适用条件**：
- 样本空间有限
- 每个样本点等可能

**示例**：掷骰子，出现偶数的概率

$$P(\text{偶数}) = \frac{3}{6} = \frac{1}{2}$$

### 1.2 频率概率（统计概率）

**定义**：通过大量重复试验，事件 $A$ 发生的频率：

$$P(A) = \lim_{n \to \infty} \frac{n_A}{n}$$

其中 $n_A$ 是 $n$ 次试验中 $A$ 发生的次数。

**大数定律**：当 $n \to \infty$ 时，频率收敛到概率。

### 1.3 主观概率

**定义**：基于个人信念和信息的概率度量。

**特点**：
- 依赖于可用信息
- 可以随新信息更新
- 适用于一次性事件

---

## 2. 条件概率与独立性

### 2.1 条件概率

**定义**：在事件 $B$ 发生的条件下，事件 $A$ 发生的概率：

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**直观理解**：缩小样本空间到 $B$，计算 $A$ 在 $B$ 中的比例。

### 2.2 乘法法则

**两个事件**：

$$P(A \cap B) = P(A|B)P(B) = P(B|A)P(A)$$

**多个事件**：

$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1)P(A_2|A_1)P(A_3|A_1 \cap A_2) \cdots P(A_n|A_1 \cap \cdots \cap A_{n-1})$$

### 2.3 全概率公式

**定义**：设 $B_1, B_2, \ldots, B_n$ 是样本空间的划分，则：

$$P(A) = \sum_{i=1}^{n} P(A|B_i)P(B_i)$$

**应用**：通过条件概率计算无条件概率。

### 2.4 贝叶斯定理

**基本形式**：

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

**扩展形式**：

$$P(A_i|B) = \frac{P(B|A_i)P(A_i)}{\sum_{j=1}^{n} P(B|A_j)P(A_j)}$$

**术语**：
- $P(A_i)$：先验概率
- $P(B|A_i)$：似然
- $P(A_i|B)$：后验概率
- $P(B)$：证据

### 2.5 独立性

**定义**：事件 $A$ 和 $B$ 独立，如果：

$$P(A \cap B) = P(A)P(B)$$

**等价条件**：
- $P(A|B) = P(A)$
- $P(B|A) = P(B)$

**多个事件的独立性**：事件 $A_1, A_2, \ldots, A_n$ 相互独立，如果对任意子集：

$$P(A_{i_1} \cap A_{i_2} \cap \cdots \cap A_{i_k}) = P(A_{i_1})P(A_{i_2}) \cdots P(A_{i_k})$$

**注意**：两两独立不一定意味着相互独立。

---

## 3. 概率公理

### 3.1 柯尔莫哥洛夫公理

**公理1（非负性）**：对任意事件 $A$，

$$P(A) \geq 0$$

**公理2（归一性）**：

$$P(\Omega) = 1$$

其中 $\Omega$ 是样本空间。

**公理3（可列可加性）**：对于互不相容的事件序列 $A_1, A_2, \ldots$，

$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

### 3.2 基本性质

**补集**：

$$P(A^c) = 1 - P(A)$$

**容斥原理**：

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**推广**：

$$P\left(\bigcup_{i=1}^{n} A_i\right) = \sum_{i=1}^{n} P(A_i) - \sum_{i<j} P(A_i \cap A_j) + \sum_{i<j<k} P(A_i \cap A_j \cap A_k) - \cdots$$

**单调性**：若 $A \subseteq B$，则 $P(A) \leq P(B)$

**界**：$0 \leq P(A) \leq 1$

---

## 4. 常见概率模型

### 4.1 伯努利试验

**定义**：只有两种可能结果的试验（成功/失败）

**参数**：$p = P(\text{成功})$

**示例**：抛硬币、二分类

### 4.2 二项分布

**定义**：$n$ 次独立伯努利试验中成功 $k$ 次的概率：

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

**性质**：
- 期望：$E[X] = np$
- 方差：$\text{Var}(X) = np(1-p)$

**应用**：检测成功率、投票结果

### 4.3 几何分布

**定义**：首次成功所需的试验次数：

$$P(X = k) = (1-p)^{k-1}p$$

**性质**：
- 期望：$E[X] = \frac{1}{p}$
- 方差：$\text{Var}(X) = \frac{1-p}{p^2}$

**无记忆性**：$P(X > m+n | X > m) = P(X > n)$

### 4.4 泊松分布

**定义**：单位时间内事件发生 $k$ 次的概率：

$$P(X = k) = \frac{e^{-\lambda}\lambda^k}{k!}$$

**性质**：
- 期望：$E[X] = \lambda$
- 方差：$\text{Var}(X) = \lambda$

**应用**：稀有事件计数（如传感器故障、网络请求）

### 4.5 均匀分布

**离散均匀**：$P(X = k) = \frac{1}{n}$，$k = 1, 2, \ldots, n$

**连续均匀**：$f(x) = \frac{1}{b-a}$，$a \leq x \leq b$

**性质**：
- 期望：$E[X] = \frac{a+b}{2}$
- 方差：$\text{Var}(X) = \frac{(b-a)^2}{12}$

---

## 5. 在具身智能中的应用

### 5.1 传感器不确定性建模

**传感器噪声模型**：使用概率分布建模传感器读数的不确定性。

**示例**：激光雷达测距误差建模为高斯分布。

### 5.2 贝叶斯滤波

**状态估计**：递归地更新状态的后验概率。

$$P(x_t | z_{1:t}) = \frac{P(z_t | x_t) P(x_t | z_{1:t-1})}{P(z_t | z_{1:t-1})}$$

**应用**：卡尔曼滤波、粒子滤波。

### 5.3 决策理论

**最大后验概率（MAP）决策**：

$$\hat{y} = \arg\max_{y} P(y | x) = \arg\max_{y} P(x | y) P(y)$$

**应用**：分类、目标识别。

### 5.4 不确定性传播

**蒙特卡洛方法**：通过采样传播不确定性。

**应用**：SLAM、运动规划。

---

## 6. 相关论文

### 概率论基础

1. **"Foundations of the Theory of Probability"** - Kolmogorov (1933)
   - 概率论公理化的奠基之作

2. **"Probability Theory: The Logic of Science"** - Jaynes (2003)
   - 贝叶斯概率的经典教材

### 贝叶斯方法

3. **"Pattern Recognition and Machine Learning"** - Bishop (2006)
   - 贝叶斯方法在模式识别中的应用

4. **"Bayesian Data Analysis"** - Gelman et al. (2013)
   - 贝叶斯数据分析权威教材

### 应用相关

5. **"Probabilistic Robotics"** - Thrun et al. (2005)
   - 概率机器人学经典教材

6. **"Kalman Filtering: Theory and Practice"** - Grewal & Andrews (2014)
   - 卡尔曼滤波理论与实践

---

## 7. 实践练习

### 练习1：基本概率计算

```python
import numpy as np

# 1. 古典概率：掷骰子
def dice_probability(event):
    """
    计算掷骰子事件的概率
    event: 事件描述，如 'even', 'greater_than_3'
    """
    outcomes = np.arange(1, 7)  # [1, 2, 3, 4, 5, 6]
    
    if event == 'even':
        favorable = outcomes[outcomes % 2 == 0]
    elif event == 'greater_than_3':
        favorable = outcomes[outcomes > 3]
    elif event == 'prime':
        favorable = outcomes[np.isin(outcomes, [2, 3, 5])]
    else:
        return None
    
    return len(favorable) / len(outcomes)

print(f"偶数概率: {dice_probability('even')}")
print(f"大于3的概率: {dice_probability('greater_than_3')}")
print(f"质数概率: {dice_probability('prime')}")

# 2. 频率概率：模拟抛硬币
def simulate_coin_flips(n_flips, n_trials=10000):
    """
    模拟抛硬币，计算正面朝上的频率
    """
    results = []
    for _ in range(n_trials):
        flips = np.random.choice(['H', 'T'], size=n_flips)
        n_heads = np.sum(flips == 'H')
        results.append(n_heads / n_flips)
    
    return np.mean(results), np.std(results)

mean_freq, std_freq = simulate_coin_flips(100)
print(f"\n100次抛硬币，正面朝上的平均频率: {mean_freq:.4f} ± {std_freq:.4f}")
```

### 练习2：条件概率与贝叶斯定理

```python
import numpy as np

# 贝叶斯定理示例：疾病检测
def bayes_theorem_example():
    """
    疾病检测示例
    - P(D) = 0.01 (患病概率)
    - P(+|D) = 0.99 (真阳性率)
    - P(+|~D) = 0.05 (假阳性率)
    """
    P_D = 0.01  # 患病概率
    P_not_D = 1 - P_D
    P_pos_given_D = 0.99  # 真阳性率
    P_pos_given_not_D = 0.05  # 假阳性率
    
    # 计算检测为阳性的概率
    P_pos = P_pos_given_D * P_D + P_pos_given_not_D * P_not_D
    
    # 应用贝叶斯定理
    P_D_given_pos = (P_pos_given_D * P_D) / P_pos
    
    print("疾病检测示例:")
    print(f"  患病概率 P(D): {P_D}")
    print(f"  真阳性率 P(+|D): {P_pos_given_D}")
    print(f"  假阳性率 P(+|~D): {P_pos_given_not_D}")
    print(f"  检测为阳性的概率 P(+): {P_pos:.4f}")
    print(f"  检测为阳性时确实患病的概率 P(D|+): {P_D_given_pos:.4f}")
    print(f"\n  结论: 即使检测为阳性，患病的概率也只有 {P_D_given_pos*100:.2f}%")

bayes_theorem_example()

# 全概率公式示例
def total_probability_example():
    """
    全概率公式示例：从不同工厂采购零件
    """
    # 三个工厂的供货比例
    P_F1 = 0.5
    P_F2 = 0.3
    P_F3 = 0.2
    
    # 各工厂的次品率
    P_defect_given_F1 = 0.02
    P_defect_given_F2 = 0.03
    P_defect_given_F3 = 0.05
    
    # 应用全概率公式
    P_defect = (P_defect_given_F1 * P_F1 + 
                P_defect_given_F2 * P_F2 + 
                P_defect_given_F3 * P_F3)
    
    print("\n全概率公式示例:")
    print(f"  工厂1供货比例: {P_F1}, 次品率: {P_defect_given_F1}")
    print(f"  工厂2供货比例: {P_F2}, 次品率: {P_defect_given_F2}")
    print(f"  工厂3供货比例: {P_F3}, 次品率: {P_defect_given_F3}")
    print(f"  总次品率: {P_defect:.4f}")

total_probability_example()
```

### 练习3：独立性验证

```python
import numpy as np
from scipy.stats import chi2_contingency

def test_independence(observed):
    """
    使用卡方检验验证独立性
    observed: 观测频数表
    """
    chi2, p_value, dof, expected = chi2_contingency(observed)
    
    print(f"卡方统计量: {chi2:.4f}")
    print(f"P值: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"拒绝原假设，变量不独立 (p < {alpha})")
    else:
        print(f"不能拒绝原假设，变量可能独立 (p >= {alpha})")
    
    return p_value

# 示例1：性别与偏好（可能独立）
print("示例1: 性别与偏好")
observed1 = np.array([
    [20, 30, 50],  # 男性
    [25, 25, 50]   # 女性
])
print("观测频数:")
print(observed1)
test_independence(observed1)

# 示例2：年龄与疾病（可能不独立）
print("\n示例2: 年龄与疾病")
observed2 = np.array([
    [10, 40],  # 年轻人
    [30, 20]   # 老年人
])
print("观测频数:")
print(observed2)
test_independence(observed2)
```

### 练习4：常见分布模拟

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, geom, poisson

# 1. 二项分布
def plot_binomial_distribution(n, p):
    """
    绘制二项分布
    """
    k = np.arange(0, n+1)
    probabilities = binom.pmf(k, n, p)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.bar(k, probabilities, alpha=0.7)
    plt.xlabel('k')
    plt.ylabel('P(X=k)')
    plt.title(f'Binomial(n={n}, p={p})')
    plt.grid(True, alpha=0.3)
    
    return probabilities

# 2. 几何分布
def plot_geometric_distribution(p, max_k=20):
    """
    绘制几何分布
    """
    k = np.arange(1, max_k+1)
    probabilities = geom.pmf(k, p)
    
    plt.subplot(1, 3, 2)
    plt.bar(k, probabilities, alpha=0.7)
    plt.xlabel('k')
    plt.ylabel('P(X=k)')
    plt.title(f'Geometric(p={p})')
    plt.grid(True, alpha=0.3)
    
    return probabilities

# 3. 泊松分布
def plot_poisson_distribution(lam, max_k=20):
    """
    绘制泊松分布
    """
    k = np.arange(0, max_k+1)
    probabilities = poisson.pmf(k, lam)
    
    plt.subplot(1, 3, 3)
    plt.bar(k, probabilities, alpha=0.7)
    plt.xlabel('k')
    plt.ylabel('P(X=k)')
    plt.title(f'Poisson(λ={lam})')
    plt.grid(True, alpha=0.3)
    
    return probabilities

# 绘制三种分布
plt.figure(figsize=(15, 4))
plot_binomial_distribution(n=20, p=0.3)
plot_geometric_distribution(p=0.2)
plot_poisson_distribution(lam=5)
plt.tight_layout()
plt.show()

# 计算期望和方差
print("\n分布参数:")
print(f"二项分布 B(20, 0.3):")
print(f"  期望: {binom.mean(20, 0.3):.2f}")
print(f"  方差: {binom.var(20, 0.3):.2f}")

print(f"\n几何分布 G(0.2):")
print(f"  期望: {geom.mean(0.2):.2f}")
print(f"  方差: {geom.var(0.2):.2f}")

print(f"\n泊松分布 P(5):")
print(f"  期望: {poisson.mean(5):.2f}")
print(f"  方差: {poisson.var(5):.2f}")
```

### 练习5：蒙特卡洛模拟

```python
import numpy as np

def monte_carlo_probability(n_simulations=100000):
    """
    蒙特卡洛方法估计概率
    问题：随机选择3个不同的数字（1-10），求和大于15的概率
    """
    # 解析解
    from itertools import combinations
    total = 0
    favorable = 0
    for combo in combinations(range(1, 11), 3):
        total += 1
        if sum(combo) > 15:
            favorable += 1
    exact_prob = favorable / total
    
    # 蒙特卡洛估计
    sums = []
    for _ in range(n_simulations):
        numbers = np.random.choice(range(1, 11), size=3, replace=False)
        sums.append(np.sum(numbers))
    
    estimated_prob = np.mean(np.array(sums) > 15)
    
    print(f"解析解: {exact_prob:.6f}")
    print(f"蒙特卡洛估计: {estimated_prob:.6f}")
    print(f"误差: {abs(exact_prob - estimated_prob):.6f}")
    
    return exact_prob, estimated_prob

# 运行模拟
exact, estimated = monte_carlo_probability()

# 不同模拟次数的比较
print("\n不同模拟次数的比较:")
n_sim_list = [100, 1000, 10000, 100000]
for n in n_sim_list:
    _, est = monte_carlo_probability(n)
```

---

**下一步**：[随机变量与分布](02-random-variables.md)