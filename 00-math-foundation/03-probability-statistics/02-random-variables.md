# 3.2 随机变量与分布

## 目录

- [1. 随机变量](#1-随机变量)
- [2. 离散随机变量](#2-离散随机变量)
- [3. 连续随机变量](#3-连续随机变量)
- [4. 多维随机变量](#4-多维随机变量)
- [5. 期望与方差](#5-期望与方差)
- [6. 在具身智能中的应用](#6-在具身智能中的应用)
- [7. 相关论文](#7-相关论文)
- [8. 实践练习](#8-实践练习)

---

## 1. 随机变量

### 1.1 定义

**随机变量**是从样本空间到实数的函数：

$$X: \Omega \to \mathbb{R}$$

**分类**：
- **离散随机变量**：取值为可数集
- **连续随机变量**：取值为连续区间

### 1.2 分布函数

**累积分布函数（CDF）**：

$$F_X(x) = P(X \leq x)$$

**性质**：
- 单调不减：$x_1 \leq x_2 \Rightarrow F(x_1) \leq F(x_2)$
- 右连续
- $\lim_{x \to -\infty} F(x) = 0$
- $\lim_{x \to +\infty} F(x) = 1$

---

## 2. 离散随机变量

### 2.1 概率质量函数（PMF）

**定义**：

$$p_X(x) = P(X = x)$$

**性质**：
- $p_X(x) \geq 0$
- $\sum_{x} p_X(x) = 1$

### 2.2 常见离散分布

#### 伯努利分布

$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

**期望**：$E[X] = p$

**方差**：$\text{Var}(X) = p(1-p)$

#### 二项分布

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

**期望**：$E[X] = np$

**方差**：$\text{Var}(X) = np(1-p)$

#### 泊松分布

$$P(X = k) = \frac{e^{-\lambda}\lambda^k}{k!}, \quad k = 0, 1, 2, \ldots$$

**期望**：$E[X] = \lambda$

**方差**：$\text{Var}(X) = \lambda$

#### 几何分布

$$P(X = k) = (1-p)^{k-1}p, \quad k = 1, 2, \ldots$$

**期望**：$E[X] = \frac{1}{p}$

**方差**：$\text{Var}(X) = \frac{1-p}{p^2}$

---

## 3. 连续随机变量

### 3.1 概率密度函数（PDF）

**定义**：

$$f_X(x) = \frac{d}{dx}F_X(x)$$

**性质**：
- $f_X(x) \geq 0$
- $\int_{-\infty}^{\infty} f_X(x) dx = 1$
- $P(a \leq X \leq b) = \int_a^b f_X(x) dx$

### 3.2 常见连续分布

#### 均匀分布

$$f_X(x) = \begin{cases} \frac{1}{b-a}, & a \leq x \leq b \\ 0, & \text{其他} \end{cases}$$

**期望**：$E[X] = \frac{a+b}{2}$

**方差**：$\text{Var}(X) = \frac{(b-a)^2}{12}$

#### 正态分布（高斯分布）

$$f_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

记为 $X \sim \mathcal{N}(\mu, \sigma^2)$

**期望**：$E[X] = \mu$

**方差**：$\text{Var}(X) = \sigma^2$

**性质**：
- 对称性：关于 $\mu$ 对称
- 68-95-99.7 规则：
  - $P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 68\%$
  - $P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 95\%$
  - $P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 99.7\%$

#### 指数分布

$$f_X(x) = \begin{cases} \lambda e^{-\lambda x}, & x \geq 0 \\ 0, & x < 0 \end{cases}$$

**期望**：$E[X] = \frac{1}{\lambda}$

**方差**：$\text{Var}(X) = \frac{1}{\lambda^2}$

**无记忆性**：$P(X > s + t | X > s) = P(X > t)$

#### Beta分布

$$f_X(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 \leq x \leq 1$$

其中 $B(\alpha, \beta)$ 是 Beta 函数。

**期望**：$E[X] = \frac{\alpha}{\alpha + \beta}$

**方差**：$\text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$

**应用**：建模概率、比例。

---

## 4. 多维随机变量

### 4.1 联合分布

**离散型**：

$$p_{X,Y}(x, y) = P(X = x, Y = y)$$

**连续型**：

$$f_{X,Y}(x, y) = \frac{\partial^2}{\partial x \partial y} F_{X,Y}(x, y)$$

### 4.2 边缘分布

**离散型**：

$$p_X(x) = \sum_{y} p_{X,Y}(x, y)$$

**连续型**：

$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) dy$$

### 4.3 条件分布

**离散型**：

$$p_{Y|X}(y|x) = \frac{p_{X,Y}(x, y)}{p_X(x)}$$

**连续型**：

$$f_{Y|X}(y|x) = \frac{f_{X,Y}(x, y)}{f_X(x)}$$

### 4.4 独立性

$X$ 和 $Y$ 独立，如果：

$$f_{X,Y}(x, y) = f_X(x) f_Y(y)$$

### 4.5 多元正态分布

$$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$$

$$f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\mathbf{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

其中：
- $\boldsymbol{\mu}$ 是均值向量
- $\mathbf{\Sigma}$ 是协方差矩阵

**性质**：
- 边缘分布仍是正态分布
- 条件分布仍是正态分布
- 线性组合仍是正态分布

---

## 5. 期望与方差

### 5.1 期望

**离散型**：

$$E[X] = \sum_{x} x p_X(x)$$

**连续型**：

$$E[X] = \int_{-\infty}^{\infty} x f_X(x) dx$$

**函数的期望**：

$$E[g(X)] = \int_{-\infty}^{\infty} g(x) f_X(x) dx$$

### 5.2 方差

**定义**：

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**标准差**：$\sigma_X = \sqrt{\text{Var}(X)}$

### 5.3 协方差与相关系数

**协方差**：

$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$$

**相关系数**：

$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

**性质**：
- $-1 \leq \rho_{X,Y} \leq 1$
- $\rho_{X,Y} = 1$：完全正相关
- $\rho_{X,Y} = -1$：完全负相关
- $\rho_{X,Y} = 0$：不相关（不一定独立）

### 5.4 期望与方差的性质

**期望的线性性**：

$$E[aX + bY + c] = aE[X] + bE[Y] + c$$

**方差的性质**：

$$\text{Var}(aX + b) = a^2 \text{Var}(X)$$

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

**协方差的性质**：

$$\text{Cov}(X, Y) = \text{Cov}(Y, X)$$

$$\text{Cov}(aX, bY) = ab \text{Cov}(X, Y)$$

### 5.5 矩

**k阶原点矩**：

$$\mu_k' = E[X^k]$$

**k阶中心矩**：

$$\mu_k = E[(X - E[X])^k]$$

**偏度**（Skewness）：

$$\gamma_1 = \frac{\mu_3}{\sigma^3}$$

**峰度**（Kurtosis）：

$$\gamma_2 = \frac{\mu_4}{\sigma^4} - 3$$

---

## 6. 在具身智能中的应用

### 6.1 传感器噪声建模

**高斯噪声**：大多数传感器噪声建模为高斯分布。

$$z = h(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, R)$$

### 6.2 状态估计

**卡尔曼滤波**：假设状态和观测都是高斯分布。

$$x_t \sim \mathcal{N}(\hat{x}_t, P_t)$$

$$z_t \sim \mathcal{N}(H x_t, R)$$

### 6.3 不确定性表示

**协方差矩阵**：表示状态估计的不确定性。

$$P = \begin{bmatrix} \sigma_x^2 & \sigma_{xy} & \sigma_{xz} \\ \sigma_{yx} & \sigma_y^2 & \sigma_{yz} \\ \sigma_{zx} & \sigma_{zy} & \sigma_z^2 \end{bmatrix}$$

### 6.4 鲁棒性分析

**异常检测**：使用统计方法检测异常值。

$$|z - \mu| > 3\sigma \Rightarrow \text{异常}$$

---

## 7. 相关论文

### 统计理论

1. **"Mathematical Statistics with Applications"** - Wackerly et al. (2014)
   - 数理统计经典教材

2. **"All of Statistics: A Concise Course in Statistical Inference"** - Wasserman (2004)
   - 统计推断简明教程

### 应用相关

3. **"Probabilistic Robotics"** - Thrun et al. (2005)
   - 概率机器人学

4. **"Gaussian Processes for Machine Learning"** - Rasmussen & Williams (2006)
   - 高斯过程

---

## 8. 实践练习

### 练习1：常见分布的可视化

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, beta, binom, poisson

# 连续分布
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 正态分布
x = np.linspace(-4, 4, 100)
axes[0, 0].plot(x, norm.pdf(x, 0, 1), 'b-', label='N(0,1)')
axes[0, 0].plot(x, norm.pdf(x, 0, 2), 'r--', label='N(0,2)')
axes[0, 0].set_title('Normal Distribution')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 指数分布
x = np.linspace(0, 5, 100)
axes[0, 1].plot(x, expon.pdf(x, scale=1), 'b-', label='λ=1')
axes[0, 1].plot(x, expon.pdf(x, scale=0.5), 'r--', label='λ=2')
axes[0, 1].set_title('Exponential Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Beta分布
x = np.linspace(0, 1, 100)
axes[0, 2].plot(x, beta.pdf(x, 2, 5), 'b-', label='α=2, β=5')
axes[0, 2].plot(x, beta.pdf(x, 5, 2), 'r--', label='α=5, β=2')
axes[0, 2].set_title('Beta Distribution')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 离散分布
# 二项分布
k = np.arange(0, 21)
axes[1, 0].bar(k, binom.pmf(k, 20, 0.3), alpha=0.7, label='n=20, p=0.3')
axes[1, 0].set_title('Binomial Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 泊松分布
k = np.arange(0, 20)
axes[1, 1].bar(k, poisson.pmf(k, 5), alpha=0.7, label='λ=5')
axes[1, 1].set_title('Poisson Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 正态分布的68-95-99.7规则
x = np.linspace(-4, 4, 100)
y = norm.pdf(x, 0, 1)
axes[1, 2].plot(x, y, 'b-', label='N(0,1)')
axes[1, 2].fill_between(x, y, where=(np.abs(x) <= 1), alpha=0.3, label='68%')
axes[1, 2].fill_between(x, y, where=(np.abs(x) <= 2), alpha=0.2, label='95%')
axes[1, 2].fill_between(x, y, where=(np.abs(x) <= 3), alpha=0.1, label='99.7%')
axes[1, 2].set_title('68-95-99.7 Rule')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 练习2：计算期望和方差

```python
import numpy as np
from scipy.stats import norm, binom, poisson

# 理论值
print("理论值:")
print(f"正态分布 N(2, 4):")
print(f"  期望: {norm.mean(2, 2):.4f}")
print(f"  方差: {norm.var(2, 2):.4f}")

print(f"\n二项分布 B(10, 0.3):")
print(f"  期望: {binom.mean(10, 0.3):.4f}")
print(f"  方差: {binom.var(10, 0.3):.4f}")

print(f"\n泊松分布 P(5):")
print(f"  期望: {poisson.mean(5):.4f}")
print(f"  方差: {poisson.var(5):.4f}")

# 蒙特卡洛估计
def monte_carlo_estimate(dist_func, n_samples=100000):
    """
    使用蒙特卡洛方法估计期望和方差
    """
    samples = dist_func(n_samples)
    mean_est = np.mean(samples)
    var_est = np.var(samples, ddof=1)
    return mean_est, var_est

print("\n蒙特卡洛估计 (n=100000):")

# 正态分布
mean_mc, var_mc = monte_carlo_estimate(lambda n: np.random.normal(2, 2, n))
print(f"正态分布 N(2, 4):")
print(f"  期望估计: {mean_mc:.4f}")
print(f"  方差估计: {var_mc:.4f}")

# 二项分布
mean_mc, var_mc = monte_carlo_estimate(lambda n: np.random.binomial(10, 0.3, n))
print(f"\n二项分布 B(10, 0.3):")
print(f"  期望估计: {mean_mc:.4f}")
print(f"  方差估计: {var_mc:.4f}")

# 泊松分布
mean_mc, var_mc = monte_carlo_estimate(lambda n: np.random.poisson(5, n))
print(f"\n泊松分布 P(5):")
print(f"  期望估计: {mean_mc:.4f}")
print(f"  方差估计: {var_mc:.4f}")
```

### 练习3：协方差与相关系数

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_correlated_data(n_samples=1000, rho=0.7):
    """
    生成相关数据
    """
    # 生成独立的标准正态分布
    z1 = np.random.randn(n_samples)
    z2 = np.random.randn(n_samples)
    
    # 生成相关的标准正态分布
    x1 = z1
    x2 = rho * z1 + np.sqrt(1 - rho**2) * z2
    
    return x1, x2

# 生成不同相关系数的数据
rhos = [-0.9, -0.5, 0, 0.5, 0.9]
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, rho in enumerate(rhos):
    x1, x2 = generate_correlated_data(500, rho)
    
    # 计算相关系数
    corr_coef = np.corrcoef(x1, x2)[0, 1]
    
    # 绘制散点图
    axes[i].scatter(x1, x2, alpha=0.5, s=10)
    axes[i].set_xlabel('X1')
    axes[i].set_ylabel('X2')
    axes[i].set_title(f'ρ = {rho}\n计算值: {corr_coef:.3f}')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_aspect('equal')

plt.tight_layout()
plt.show()

# 协方差矩阵
print("协方差矩阵示例:")
x1, x2 = generate_correlated_data(1000, 0.7)
data = np.column_stack([x1, x2])
cov_matrix = np.cov(data, rowvar=False)
print(cov_matrix)

print(f"\n相关系数矩阵:")
print(np.corrcoef(data, rowvar=False))
```

### 练习4：多元正态分布

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 定义多元正态分布
mean = np.array([0, 0])

# 不同的协方差矩阵
cov_matrices = [
    np.array([[1, 0], [0, 1]]),           # 独立
    np.array([[1, 0.8], [0.8, 1]]),       # 正相关
    np.array([[1, -0.8], [-0.8, 1]]),     # 负相关
    np.array([[2, 0], [0, 0.5]])          # 各向异性
]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

for i, cov in enumerate(cov_matrices):
    # 创建网格
    x, y = np.mgrid[-3:3:.1, -3:3:.1]
    pos = np.dstack((x, y))
    
    # 计算PDF
    rv = multivariate_normal(mean, cov)
    z = rv.pdf(pos)
    
    # 绘制等高线
    contour = axes[i].contourf(x, y, z, levels=20, cmap='viridis')
    axes[i].contour(x, y, z, levels=20, colors='black', alpha=0.3)
    axes[i].set_xlabel('X1')
    axes[i].set_ylabel('X2')
    axes[i].set_title(f'Covariance Matrix:\n{cov}')
    axes[i].set_aspect('equal')
    plt.colorbar(contour, ax=axes[i])

plt.tight_layout()
plt.show()

# 采样
print("多元正态分布采样:")
samples = multivariate_normal.rvs(mean=mean, cov=cov_matrices[1], size=1000)
print(f"样本均值: {np.mean(samples, axis=0)}")
print(f"样本协方差:\n{np.cov(samples, rowvar=False)}")
```

### 练习5：条件分布

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 二元正态分布
mean = np.array([0, 0])
cov = np.array([[1, 0.7], [0.7, 1]])

# 生成样本
samples = multivariate_normal.rvs(mean=mean, cov=cov, size=1000)

# 条件分布：给定 X1 = x1，求 X2 的分布
x1_condition = 1.0

# 选择 X1 接近 x1_condition 的样本
epsilon = 0.1
mask = np.abs(samples[:, 0] - x1_condition) < epsilon
x2_given_x1 = samples[mask, 1]

# 理论条件分布
# X2|X1 ~ N(μ2 + ρ(σ2/σ1)(x1-μ1), σ2²(1-ρ²))
rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
mu_cond = mean[1] + rho * np.sqrt(cov[1, 1] / cov[0, 0]) * (x1_condition - mean[0])
sigma_cond = np.sqrt(cov[1, 1] * (1 - rho**2))

print(f"条件分布 X2|X1={x1_condition}:")
print(f"  理论均值: {mu_cond:.4f}")
print(f"  理论标准差: {sigma_cond:.4f}")
print(f"  样本均值: {np.mean(x2_given_x1):.4f}")
print(f"  样本标准差: {np.std(x2_given_x1, ddof=1):.4f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 联合分布
axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
axes[0].axvline(x=x1_condition, color='red', linestyle='--', label=f'X1={x1_condition}')
axes[0].set_xlabel('X1')
axes[0].set_ylabel('X2')
axes[0].set_title('Joint Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 条件分布
axes[1].hist(x2_given_x1, bins=30, density=True, alpha=0.7, label='Sample')
x_range = np.linspace(x2_given_x1.min(), x2_given_x1.max(), 100)
axes[1].plot(x_range, norm.pdf(x_range, mu_cond, sigma_cond), 'r-', 
             label=f'Theoretical N({mu_cond:.2f}, {sigma_cond**2:.2f})')
axes[1].set_xlabel('X2')
axes[1].set_ylabel('Density')
axes[1].set_title(f'Conditional Distribution X2|X1={x1_condition}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

**下一步**：[贝叶斯推断](03-bayesian-inference.md)