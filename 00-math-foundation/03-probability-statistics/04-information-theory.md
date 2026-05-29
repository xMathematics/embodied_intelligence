# 3.4 信息论

## 目录

- [1. 信息量](#1-信息量)
- [2. 熵](#2-熵)
- [3. 互信息](#3-互信息)
- [4. 相对熵（KL散度）](#4-相对熵kl散度)
- [5. 交叉熵](#5-交叉熵)
- [6. 在具身智能中的应用](#6-在具身智能中的应用)
- [7. 相关论文](#7-相关论文)
- [8. 实践练习](#8-实践练习)

---

## 1. 信息量

### 1.1 自信息

**定义**：事件 $x$ 的自信息：

$$I(x) = -\log_2 P(x)$$

**性质**：
- 不确定性越高，信息量越大
- $P(x) = 1 \Rightarrow I(x) = 0$
- $P(x) = 0 \Rightarrow I(x) = \infty$

**单位**：比特（bit，以2为底）、奈特（nat，以e为底）

### 1.2 联合自信息

$$I(x, y) = -\log_2 P(x, y)$$

### 1.3 条件自信息

$$I(x|y) = -\log_2 P(x|y)$$

---

## 2. 熵

### 2.1 香农熵

**定义**：随机变量 $X$ 的熵：

$$H(X) = -\sum_{x} P(x) \log_2 P(x)$$

**连续型**（微分熵）：

$$H(X) = -\int f(x) \log f(x) dx$$

**性质**：
- $H(X) \geq 0$
- 最大熵：均匀分布
- 确定性变量的熵为0

### 2.2 联合熵

$$H(X, Y) = -\sum_{x, y} P(x, y) \log_2 P(x, y)$$

**性质**：

$$H(X, Y) \leq H(X) + H(Y)$$

等号成立当且仅当 $X$ 和 $Y$ 独立。

### 2.3 条件熵

$$H(Y|X) = -\sum_{x, y} P(x, y) \log_2 P(y|x)$$

**性质**：

$$H(Y|X) = H(X, Y) - H(X)$$

$$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$

### 2.4 熵的链式法则

$$H(X_1, X_2, \ldots, X_n) = \sum_{i=1}^{n} H(X_i | X_1, \ldots, X_{i-1})$$

### 2.5 最大熵分布

**约束**：给定期望 $E[X] = \mu$

**最大熵分布**：指数分布

**约束**：给定期望和方差

**最大熵分布**：正态分布

---

## 3. 互信息

### 3.1 定义

$$I(X; Y) = \sum_{x, y} P(x, y) \log_2 \frac{P(x, y)}{P(x)P(y)}$$

**等价形式**：

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$$I(X; Y) = H(X) + H(Y) - H(X, Y)$$

### 3.2 性质

- $I(X; Y) \geq 0$
- $I(X; Y) = 0 \Leftrightarrow X$ 和 $Y$ 独立
- 对称性：$I(X; Y) = I(Y; X)$

### 3.3 条件互信息

$$I(X; Y | Z) = \sum_{x, y, z} P(x, y, z) \log_2 \frac{P(x, y | z)}{P(x | z) P(y | z)}$$

### 3.4 互信息的链式法则

$$I(X_1, X_2, \ldots, X_n; Y) = \sum_{i=1}^{n} I(X_i; Y | X_1, \ldots, X_{i-1})$$

---

## 4. 相对熵（KL散度）

### 4.1 定义

$$D_{KL}(P || Q) = \sum_{x} P(x) \log_2 \frac{P(x)}{Q(x)}$$

**连续型**：

$$D_{KL}(P || Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx$$

### 4.2 性质

- $D_{KL}(P || Q) \geq 0$
- $D_{KL}(P || Q) = 0 \Leftrightarrow P = Q$
- 不对称性：$D_{KL}(P || Q) \neq D_{KL}(Q || P)$

### 4.3 与互信息的关系

$$I(X; Y) = D_{KL}(P(X, Y) || P(X)P(Y))$$

### 4.4 JS散度

**定义**：

$$D_{JS}(P || Q) = \frac{1}{2} D_{KL}(P || M) + \frac{1}{2} D_{KL}(Q || M)$$

其中 $M = \frac{1}{2}(P + Q)$

**性质**：
- 对称性
- 有界：$0 \leq D_{JS}(P || Q) \leq 1$

---

## 5. 交叉熵

### 5.1 定义

$$H(P, Q) = -\sum_{x} P(x) \log_2 Q(x)$$

### 5.2 与KL散度的关系

$$H(P, Q) = H(P) + D_{KL}(P || Q)$$

### 5.3 在机器学习中的应用

**损失函数**：分类问题的交叉熵损失

$$L = -\sum_{i} y_i \log \hat{y}_i$$

其中 $y_i$ 是真实标签，$\hat{y}_i$ 是预测概率。

---

## 6. 在具身智能中的应用

### 6.1 传感器融合

**互信息最大化**：选择信息量最大的传感器。

$$I(Z; X) = H(Z) - H(Z|X)$$

### 6.2 主动感知

**信息增益**：

$$IG(X; Z) = H(X) - H(X|Z)$$

**应用**：选择最佳观测位置。

### 6.3 不确定性量化

**熵**：衡量状态估计的不确定性。

$$H(X) = -\int p(x) \log p(x) dx$$

### 6.4 模型选择

**最小描述长度（MDL）**：

$$\text{MDL} = L(D|M) + L(M)$$

其中 $L(D|M)$ 是数据编码长度，$L(M)$ 是模型编码长度。

### 6.5 强化学习

**熵正则化**：鼓励探索。

$$\pi^*(a|s) = \arg\max_{\pi} \mathbb{E}[R(s, a) + \alpha H(\pi(\cdot|s))]$$

---

## 7. 相关论文

### 信息论基础

1. **"A Mathematical Theory of Communication"** - Shannon (1948)
   - 信息论奠基之作

2. **"Elements of Information Theory"** - Cover & Thomas (2006)
   - 信息论经典教材

### 应用相关

3. **"Information-Theoretic Active Perception"** - Chen et al. (2015)
   - 信息论在主动感知中的应用

4. **"Maximum Entropy Inverse Reinforcement Learning"** - Ziebart (2008)
   - 最大熵逆强化学习

---

## 8. 实践练习

### 练习1：计算熵

```python
import numpy as np

def entropy(p):
    """
    计算离散分布的熵
    """
    p = np.array(p)
    p = p[p > 0]  # 移除零概率
    return -np.sum(p * np.log2(p))

# 示例1：均匀分布
p_uniform = [0.25, 0.25, 0.25, 0.25]
print(f"均匀分布 {p_uniform} 的熵: {entropy(p_uniform):.4f} bits")

# 示例2：确定性分布
p_deterministic = [1, 0, 0, 0]
print(f"确定性分布 {p_deterministic} 的熵: {entropy(p_deterministic):.4f} bits")

# 示例3：偏斜分布
p_skewed = [0.9, 0.05, 0.03, 0.02]
print(f"偏斜分布 {p_skewed} 的熵: {entropy(p_skewed):.4f} bits")

# 示例4：硬币抛掷
p_fair = [0.5, 0.5]
p_biased = [0.7, 0.3]
print(f"\n公平硬币 {p_fair} 的熵: {entropy(p_fair):.4f} bits")
print(f"有偏硬币 {p_biased} 的熵: {entropy(p_biased):.4f} bits")

# 连续分布的微分熵
from scipy.stats import norm

def differential_entropy(dist):
    """
    计算连续分布的微分熵
    """
    return dist.entropy()

# 正态分布
dist1 = norm(0, 1)
dist2 = norm(0, 2)

print(f"\nN(0, 1) 的微分熵: {differential_entropy(dist1):.4f} nats")
print(f"N(0, 2) 的微分熵: {differential_entropy(dist2):.4f} nats")
```

### 练习2：条件熵和互信息

```python
import numpy as np

def joint_entropy(p_xy):
    """
    计算联合熵
    """
    p_xy = np.array(p_xy)
    p_xy = p_xy[p_xy > 0]
    return -np.sum(p_xy * np.log2(p_xy))

def conditional_entropy(p_xy, p_x):
    """
    计算条件熵 H(Y|X)
    """
    p_xy = np.array(p_xy)
    p_x = np.array(p_x)
    
    h_cond = 0
    for i, px in enumerate(p_x):
        if px > 0:
            py_given_x = p_xy[i] / px
            if py_given_x > 0:
                h_cond -= px * py_given_x * np.log2(py_given_x)
    
    return h_cond

def mutual_information(p_xy, p_x, p_y):
    """
    计算互信息
    """
    p_xy = np.array(p_xy)
    p_x = np.array(p_x)
    p_y = np.array(p_y)
    
    mi = 0
    for i in range(len(p_x)):
        for j in range(len(p_y)):
            if p_xy[i, j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    
    return mi

# 示例：两个二元变量
p_xy = np.array([
    [0.3, 0.2],  # X=0
    [0.1, 0.4]   # X=1
])

p_x = np.sum(p_xy, axis=1)  # [0.5, 0.5]
p_y = np.sum(p_xy, axis=0)  # [0.4, 0.6]

h_x = entropy(p_x)
h_y = entropy(p_y)
h_xy = joint_entropy(p_xy)
h_y_given_x = conditional_entropy(p_xy, p_x)
mi = mutual_information(p_xy, p_x, p_y)

print("联合分布:")
print(p_xy)
print(f"\n边缘分布 P(X): {p_x}")
print(f"边缘分布 P(Y): {p_y}")

print(f"\nH(X) = {h_x:.4f} bits")
print(f"H(Y) = {h_y:.4f} bits")
print(f"H(X, Y) = {h_xy:.4f} bits")
print(f"H(Y|X) = {h_y_given_x:.4f} bits")
print(f"I(X; Y) = {mi:.4f} bits")

# 验证关系
print(f"\n验证:")
print(f"H(X) + H(Y|X) = {h_x + h_y_given_x:.4f} bits")
print(f"H(X, Y) = {h_xy:.4f} bits")
print(f"H(X) + H(Y) - I(X; Y) = {h_x + h_y - mi:.4f} bits")
```

### 练习3：KL散度和JS散度

```python
import numpy as np

def kl_divergence(p, q):
    """
    计算KL散度 D_KL(P || Q)
    """
    p = np.array(p)
    q = np.array(q)
    
    # 只计算p > 0的部分
    mask = p > 0
    p = p[mask]
    q = q[mask]
    
    return np.sum(p * np.log2(p / q))

def js_divergence(p, q):
    """
    计算JS散度
    """
    p = np.array(p)
    q = np.array(q)
    
    m = 0.5 * (p + q)
    
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# 示例1：相似分布
p1 = [0.5, 0.3, 0.2]
q1 = [0.5, 0.25, 0.25]

print("示例1: 相似分布")
print(f"P = {p1}")
print(f"Q = {q1}")
print(f"D_KL(P || Q) = {kl_divergence(p1, q1):.4f} bits")
print(f"D_KL(Q || P) = {kl_divergence(q1, p1):.4f} bits")
print(f"D_JS(P, Q) = {js_divergence(p1, q1):.4f} bits")

# 示例2：不同分布
p2 = [0.9, 0.05, 0.05]
q2 = [0.1, 0.45, 0.45]

print("\n示例2: 不同分布")
print(f"P = {p2}")
print(f"Q = {q2}")
print(f"D_KL(P || Q) = {kl_divergence(p2, q2):.4f} bits")
print(f"D_KL(Q || P) = {kl_divergence(q2, p2):.4f} bits")
print(f"D_JS(P, Q) = {js_divergence(p2, q2):.4f} bits")

# 示例3：相同分布
p3 = [0.4, 0.3, 0.3]
q3 = [0.4, 0.3, 0.3]

print("\n示例3: 相同分布")
print(f"P = {p3}")
print(f"Q = {q3}")
print(f"D_KL(P || Q) = {kl_divergence(p3, q3):.4f} bits")
print(f"D_JS(P, Q) = {js_divergence(p3, q3):.4f} bits")

# 连续分布的KL散度
from scipy.stats import norm

def kl_divergence_continuous(dist1, dist2, x_range, n_points=1000):
    """
    计算连续分布的KL散度（数值积分）
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    dx = x[1] - x[0]
    
    p1 = dist1.pdf(x)
    p2 = dist2.pdf(x)
    
    # 避免数值问题
    p1 = np.maximum(p1, 1e-10)
    p2 = np.maximum(p2, 1e-10)
    
    kl = np.sum(p1 * np.log(p1 / p2)) * dx
    
    return kl

# 两个正态分布
dist_a = norm(0, 1)
dist_b = norm(1, 1)
dist_c = norm(0, 2)

print("\n连续分布的KL散度:")
kl_ab = kl_divergence_continuous(dist_a, dist_b, (-5, 5))
kl_ac = kl_divergence_continuous(dist_a, dist_c, (-5, 5))
print(f"D_KL(N(0,1) || N(1,1)) = {kl_ab:.4f} nats")
print(f"D_KL(N(0,1) || N(0,2)) = {kl_ac:.4f} nats")
```

### 练习4：交叉熵

```python
import numpy as np

def cross_entropy(p, q):
    """
    计算交叉熵 H(P, Q)
    """
    p = np.array(p)
    q = np.array(q)
    
    # 只计算p > 0的部分
    mask = p > 0
    p = p[mask]
    q = q[mask]
    
    return -np.sum(p * np.log2(q))

# 示例：分类问题
# 真实标签（one-hot编码）
y_true = [0, 1, 0]

# 预测概率（模型输出）
y_pred_good = [0.1, 0.8, 0.1]  # 预测准确
y_pred_bad = [0.4, 0.3, 0.3]   # 预测不准确
y_pred_random = [1/3, 1/3, 1/3]  # 随机预测

print("分类问题的交叉熵损失:")
print(f"真实标签: {y_true}")
print(f"预测 (好): {y_pred_good}, 交叉熵: {cross_entropy(y_true, y_pred_good):.4f} bits")
print(f"预测 (差): {y_pred_bad}, 交叉熵: {cross_entropy(y_true, y_pred_bad):.4f} bits")
print(f"预测 (随机): {y_pred_random}, 交叉熵: {cross_entropy(y_true, y_pred_random):.4f} bits")

# 验证关系
print(f"\n验证关系: H(P, Q) = H(P) + D_KL(P || Q)")
h_p = entropy(y_true)
kl_pq = kl_divergence(y_true, y_pred_good)
h_pq = cross_entropy(y_true, y_pred_good)
print(f"H(P) = {h_p:.4f} bits")
print(f"D_KL(P || Q) = {kl_pq:.4f} bits")
print(f"H(P) + D_KL(P || Q) = {h_p + kl_pq:.4f} bits")
print(f"H(P, Q) = {h_pq:.4f} bits")
```

### 练习5：信息增益

```python
import numpy as np

def information_gain(p_xy, p_x):
    """
    计算信息增益 IG(X; Y) = H(Y) - H(Y|X)
    """
    p_y = np.sum(p_xy, axis=0)
    h_y = entropy(p_y)
    h_y_given_x = conditional_entropy(p_xy, p_x)
    
    return h_y - h_y_given_x

# 示例：特征选择
# 目标变量Y（是否购买）
# 特征X（年龄段）

# 联合分布
p_xy = np.array([
    [0.15, 0.05],  # 青年 (购买, 不购买)
    [0.20, 0.10],  # 中年
    [0.10, 0.40]   # 老年
])

p_x = np.sum(p_xy, axis=1)  # [0.20, 0.30, 0.50]
p_y = np.sum(p_xy, axis=0)  # [0.45, 0.55]

print("特征选择示例:")
print("联合分布 (年龄段 × 购买):")
print(p_xy)
print(f"\n边缘分布 P(年龄段): {p_x}")
print(f"边缘分布 P(购买): {p_y}")

h_y = entropy(p_y)
h_y_given_x = conditional_entropy(p_xy, p_x)
ig = information_gain(p_xy, p_x)

print(f"\nH(购买) = {h_y:.4f} bits")
print(f"H(购买 | 年龄段) = {h_y_given_x:.4f} bits")
print(f"信息增益 IG(年龄段; 购买) = {ig:.4f} bits")

# 比较不同特征
print("\n比较不同特征的信息增益:")

# 特征1：年龄段
p_xy1 = np.array([
    [0.15, 0.05],
    [0.20, 0.10],
    [0.10, 0.40]
])
p_x1 = np.sum(p_xy1, axis=1)
ig1 = information_gain(p_xy1, p_x1)

# 特征2：性别
p_xy2 = np.array([
    [0.25, 0.25],  # 男性
    [0.20, 0.30]   # 女性
])
p_x2 = np.sum(p_xy2, axis=1)
ig2 = information_gain(p_xy2, p_x2)

# 特征3：收入
p_xy3 = np.array([
    [0.30, 0.05],  # 高收入
    [0.10, 0.15],  # 中收入
    [0.05, 0.35]   # 低收入
])
p_x3 = np.sum(p_xy3, axis=1)
ig3 = information_gain(p_xy3, p_x3)

print(f"年龄段的信息增益: {ig1:.4f} bits")
print(f"性别的信息增益: {ig2:.4f} bits")
print(f"收入的信息增益: {ig3:.4f} bits")
print(f"\n最佳特征: {['年龄段', '性别', '收入'][np.argmax([ig1, ig2, ig3])]}")
```

---

**下一步**：[蒙特卡洛方法](05-monte-carlo.md)