# 3.5 蒙特卡洛方法

## 目录

- [1. 蒙特卡洛方法概述](#1-蒙特卡洛方法概述)
- [2. 随机数生成](#2-随机数生成)
- [3. 蒙特卡洛积分](#3-蒙特卡洛积分)
- [4. 重要性采样](#4-重要性采样)
- [5. 马尔可夫链蒙特卡洛（MCMC）](#5-马尔可夫链蒙特卡洛mcmc)
- [6. 在具身智能中的应用](#6-在具身智能中的应用)
- [7. 相关论文](#7-相关论文)
- [8. 实践练习](#8-实践练习)

---

## 1. 蒙特卡洛方法概述

### 1.1 基本思想

**蒙特卡洛方法**：通过随机采样解决计算问题。

**核心思想**：
- 用样本均值估计期望
- 用样本频率估计概率
- 适用于高维、复杂问题

### 1.2 大数定律

**强大数定律**：

$$\frac{1}{n} \sum_{i=1}^{n} X_i \xrightarrow{a.s.} E[X]$$

**弱大数定律**：

$$\frac{1}{n} \sum_{i=1}^{n} X_i \xrightarrow{p} E[X]$$

### 1.3 中心极限定理

$$\sqrt{n}\left(\frac{1}{n} \sum_{i=1}^{n} X_i - E[X]\right) \xrightarrow{d} \mathcal{N}(0, \text{Var}(X))$$

**意义**：样本均值的分布近似正态分布。

### 1.4 收敛速度

蒙特卡洛方法的收敛速度为 $O(1/\sqrt{n})$，与维度无关。

**优势**：适用于高维问题。

---

## 2. 随机数生成

### 2.1 均匀随机数

**线性同余生成器**：

$$X_{n+1} = (aX_n + c) \mod m$$

**Mersenne Twister**：常用的伪随机数生成器。

### 2.2 逆变换法

**原理**：如果 $U \sim \text{Uniform}(0, 1)$，则 $X = F^{-1}(U)$ 的分布为 $F$。

**步骤**：
1. 生成 $U \sim \text{Uniform}(0, 1)$
2. 计算 $X = F^{-1}(U)$

**示例**：指数分布

$$F(x) = 1 - e^{-\lambda x}$$

$$F^{-1}(u) = -\frac{\ln(1-u)}{\lambda}$$

### 2.3 拒绝采样

**步骤**：
1. 选择提议分布 $g(x)$ 和常数 $M$，使得 $f(x) \leq M g(x)$
2. 生成 $X \sim g(x)$ 和 $U \sim \text{Uniform}(0, 1)$
3. 如果 $U \leq \frac{f(X)}{M g(X)}$，接受 $X$；否则拒绝

**接受概率**：$1/M$

### 2.4 Box-Muller变换

**生成标准正态分布**：

1. 生成 $U_1, U_2 \sim \text{Uniform}(0, 1)$
2. 计算：
   $$Z_1 = \sqrt{-2\ln U_1} \cos(2\pi U_2)$$
   $$Z_2 = \sqrt{-2\ln U_1} \sin(2\pi U_2)$$

则 $Z_1, Z_2 \sim \mathcal{N}(0, 1)$

---

## 3. 蒙特卡洛积分

### 3.1 基本方法

**目标**：计算积分

$$I = \int_{\Omega} f(x) dx$$

**方法**：
1. 从均匀分布采样 $x_i \sim \text{Uniform}(\Omega)$
2. 计算样本均值

$$\hat{I} = \frac{|\Omega|}{n} \sum_{i=1}^{n} f(x_i)$$

其中 $|\Omega|$ 是积分区域的体积。

### 3.2 期望形式

$$I = \int f(x) dx = \int \frac{f(x)}{p(x)} p(x) dx = E\left[\frac{f(X)}{p(X)}\right]$$

**估计**：

$$\hat{I} = \frac{1}{n} \sum_{i=1}^{n} \frac{f(x_i)}{p(x_i)}$$

其中 $x_i \sim p(x)$。

### 3.3 方差

$$\text{Var}(\hat{I}) = \frac{1}{n} \text{Var}\left(\frac{f(X)}{p(X)}\right)$$

**标准误差**：

$$\text{SE}(\hat{I}) = \sqrt{\frac{\text{Var}(\hat{I})}{n}}$$

### 3.4 收敛性

**误差界**（Chebyshev不等式）：

$$P(|\hat{I} - I| \geq \epsilon) \leq \frac{\text{Var}(\hat{I})}{\epsilon^2}$$

**置信区间**（中心极限定理）：

$$\hat{I} \pm z_{\alpha/2} \cdot \text{SE}(\hat{I})$$

---

## 4. 重要性采样

### 4.1 基本思想

**目标**：减少估计方差。

**原理**：从重要区域（贡献大的区域）采样。

### 4.2 重要性采样估计

$$I = \int f(x) dx = \int \frac{f(x)}{q(x)} q(x) dx = E_q\left[\frac{f(X)}{q(X)}\right]$$

**估计**：

$$\hat{I} = \frac{1}{n} \sum_{i=1}^{n} \frac{f(x_i)}{q(x_i)}$$

其中 $x_i \sim q(x)$。

### 4.3 最优提议分布

**最优提议分布**：

$$q^*(x) \propto |f(x)|$$

**此时方差为零**。

### 4.4 自适应重要性采样

**步骤**：
1. 从初始分布采样
2. 根据样本更新提议分布
3. 重复

---

## 5. 马尔可夫链蒙特卡洛（MCMC）

### 5.1 基本思想

**目标**：从复杂分布 $p(x)$ 采样。

**方法**：构造马尔可夫链，使其平稳分布为 $p(x)$。

### 5.2 Metropolis-Hastings算法

**步骤**：
1. 从当前状态 $x_t$ 开始
2. 从提议分布 $q(x'|x_t)$ 采样候选 $x'$
3. 计算接受概率

$$\alpha = \min\left(1, \frac{p(x') q(x_t|x')}{p(x_t) q(x'|x_t)}\right)$$

4. 从 $\text{Uniform}(0, 1)$ 采样 $u$
5. 如果 $u \leq \alpha$，接受 $x_{t+1} = x'$；否则 $x_{t+1} = x_t$

### 5.3 Metropolis算法

**对称提议分布**：$q(x'|x_t) = q(x_t|x')$

**接受概率**：

$$\alpha = \min\left(1, \frac{p(x')}{p(x_t)}\right)$$

### 5.4 Gibbs采样

**步骤**：
1. 初始化 $x^{(0)} = (x_1^{(0)}, \ldots, x_d^{(0)})$
2. 对于 $t = 1, 2, \ldots$：
   - $x_1^{(t)} \sim p(x_1 | x_2^{(t-1)}, \ldots, x_d^{(t-1)})$
   - $x_2^{(t)} \sim p(x_2 | x_1^{(t)}, x_3^{(t-1)}, \ldots, x_d^{(t-1)})$
   - $\vdots$
   - $x_d^{(t)} \sim p(x_d | x_1^{(t)}, \ldots, x_{d-1}^{(t)})$

**优势**：不需要接受-拒绝步骤。

### 5.5 Hamiltonian Monte Carlo（HMC）

**基本思想**：引入动量变量，利用物理动力学提高采样效率。

**步骤**：
1. 从当前状态 $x$ 和动量 $p \sim \mathcal{N}(0, M)$ 开始
2. 模拟Hamiltonian动力学
3. Metropolis接受-拒绝步骤

**优势**：适用于高维问题。

### 5.6 收敛诊断

**方法**：
- 迹图（Trace plot）
- 自相关函数
- 有效样本大小（ESS）
- Gelman-Rubin统计量

---

## 6. 在具身智能中的应用

### 6.1 粒子滤波

**重要性采样**：

$$w_t^{(i)} \propto w_{t-1}^{(i)} \frac{p(z_t|x_t^{(i)})}{q(x_t^{(i)}|x_{t-1}^{(i)}, z_t)}$$

**重采样**：根据权重重新采样粒子。

### 6.2 SLAM

**FastSLAM**：使用粒子滤波估计机器人轨迹和地图。

**GMapping**：使用Rao-Blackwellized粒子滤波。

### 6.3 路径规划

**蒙特卡洛树搜索（MCTS）**：
- 选择
- 扩展
- 模拟
- 反向传播

### 6.4 不确定性传播

**蒙特卡洛方法**：通过采样传播不确定性。

**应用**：传感器融合、状态估计。

### 6.5 强化学习

**蒙特卡洛策略评估**：

$$V(s) = \frac{1}{N(s)} \sum_{t: s_t = s} G_t$$

其中 $G_t$ 是回报。

---

## 7. 相关论文

### 蒙特卡洛方法

1. **"Monte Carlo Methods"** - Hammersley & Handscomb (1964)
   - 蒙特卡洛方法经典教材

2. **"Monte Carlo Statistical Methods"** - Robert & Casella (2004)
   - 蒙特卡洛统计方法

### MCMC

3. **"Markov Chain Monte Carlo: Stochastic Simulation for Bayesian Inference"** - Gamerman & Lopes (2006)
   - MCMC方法

4. **"The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo"** - Hoffman & Gelman (2014)
   - NUTS采样器

### 应用相关

5. **"Probabilistic Robotics"** - Thrun et al. (2005)
   - 概率机器人学

6. **"Monte Carlo Tree Search: A Review"** - Browne et al. (2012)
   - MCTS综述

---

## 8. 实践练习

### 练习1：蒙特卡洛积分

```python
import numpy as np

def monte_carlo_integral(f, a, b, n_samples=100000):
    """
    蒙特卡洛积分
    """
    x = np.random.uniform(a, b, n_samples)
    fx = f(x)
    
    estimate = (b - a) * np.mean(fx)
    std_error = (b - a) * np.std(fx) / np.sqrt(n_samples)
    
    return estimate, std_error

# 示例1：计算 π
def estimate_pi(n_samples=100000):
    """
    估计 π
    """
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.mean(inside)
    
    return pi_estimate

print("估计 π:")
for n in [1000, 10000, 100000, 1000000]:
    pi_est = estimate_pi(n)
    error = abs(pi_est - np.pi)
    print(f"  n={n:7d}: π ≈ {pi_est:.6f}, 误差 = {error:.6f}")

# 示例2：计算积分
f = lambda x: np.sin(x)
a, b = 0, np.pi

true_value = 2  # ∫₀^π sin(x) dx = 2

print(f"\n计算积分 ∫₀^π sin(x) dx:")
for n in [1000, 10000, 100000]:
    estimate, se = monte_carlo_integral(f, a, b, n)
    error = abs(estimate - true_value)
    print(f"  n={n:6d}: 估计值 = {estimate:.6f} ± {se:.6f}, 误差 = {error:.6f}")

# 示例3：高维积分
def high_dim_integral(d, n_samples=100000):
    """
    计算高维积分 ∫[0,1]^d (x₁ + x₂ + ... + x_d) dx
    """
    samples = np.random.uniform(0, 1, (n_samples, d))
    fx = np.sum(samples, axis=1)
    
    estimate = np.mean(fx)
    true_value = d / 2  # 解析解
    
    return estimate, true_value

print(f"\n高维积分:")
for d in [1, 2, 5, 10, 20]:
    estimate, true_value = high_dim_integral(d)
    error = abs(estimate - true_value)
    print(f"  d={2d}: 估计值 = {estimate:.6f}, 真值 = {true_value:.6f}, 误差 = {error:.6f}")
```

### 练习2：重要性采样

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def importance_sampling(f, p, q, n_samples=10000):
    """
    重要性采样
    """
    samples = q.rvs(n_samples)
    weights = f(samples) * p.pdf(samples) / q.pdf(samples)
    
    estimate = np.mean(weights)
    std_error = np.std(weights) / np.sqrt(n_samples)
    
    return estimate, std_error

# 示例：估计尾部概率
def estimate_tail_probability(mu=0, sigma=1, threshold=3, n_samples=100000):
    """
    估计 P(X > threshold)，其中 X ~ N(mu, sigma^2)
    """
    # 目标分布
    p = norm(mu, sigma)
    
    # 提议分布：平移的正态分布
    q = norm(threshold, 1)
    
    # 函数
    f = lambda x: (x > threshold).astype(float)
    
    # 重要性采样
    estimate_is, se_is = importance_sampling(f, p, q, n_samples)
    
    # 简单蒙特卡洛
    samples_mc = p.rvs(n_samples)
    estimate_mc = np.mean(samples_mc > threshold)
    se_mc = np.std(samples_mc > threshold) / np.sqrt(n_samples)
    
    # 真值
    true_value = 1 - p.cdf(threshold)
    
    return {
        'importance_sampling': (estimate_is, se_is),
        'monte_carlo': (estimate_mc, se_mc),
        'true_value': true_value
    }

print("估计尾部概率 P(X > 3), X ~ N(0, 1):")
result = estimate_tail_probability()
print(f"  真值: {result['true_value']:.6e}")
print(f"  简单蒙特卡洛: {result['monte_carlo'][0]:.6e} ± {result['monte_carlo'][1]:.6e}")
print(f"  重要性采样: {result['importance_sampling'][0]:.6e} ± {result['importance_sampling'][1]:.6e}")

# 比较不同阈值
thresholds = [2, 3, 4, 5]
print(f"\n比较不同阈值:")
for threshold in thresholds:
    result = estimate_tail_probability(threshold=threshold)
    true_val = result['true_value']
    mc_est, mc_se = result['monte_carlo']
    is_est, is_se = result['importance_sampling']
    
    print(f"  阈值 = {threshold}:")
    print(f"    真值: {true_val:.6e}")
    print(f"    简单MC: {mc_est:.6e} ± {mc_se:.6e}, 相对误差 = {abs(mc_est-true_val)/true_val:.2%}")
    print(f"    重要性采样: {is_est:.6e} ± {is_se:.6e}, 相对误差 = {abs(is_est-true_val)/true_val:.2%}")
```

### 练习3：Metropolis-Hastings算法

```python
import numpy as np
import matplotlib.pyplot as plt

def metropolis_hastings(target_log_prob, initial_state, n_iter=10000, proposal_std=1.0):
    """
    Metropolis-Hastings算法
    """
    current_state = initial_state
    samples = [current_state]
    
    for _ in range(n_iter):
        # 提议
        proposed_state = current_state + np.random.normal(0, proposal_std)
        
        # 计算接受概率
        log_alpha = target_log_prob(proposed_state) - target_log_prob(current_state)
        alpha = min(1, np.exp(log_alpha))
        
        # 接受或拒绝
        if np.random.rand() < alpha:
            current_state = proposed_state
        
        samples.append(current_state)
    
    return np.array(samples)

# 示例：从正态分布采样
def target_log_prob_normal(x, mu=0, sigma=1):
    """
    目标分布的对数概率：N(mu, sigma^2)
    """
    return -0.5 * ((x - mu) / sigma)**2 - np.log(sigma * np.sqrt(2 * np.pi))

# 运行MCMC
mu, sigma = 2, 1.5
target_log_prob = lambda x: target_log_prob_normal(x, mu, sigma)

samples = metropolis_hastings(target_log_prob, initial_state=0, n_iter=10000, proposal_std=1.0)

# 去除burn-in
burn_in = 1000
samples = samples[burn_in:]

# 计算统计量
sample_mean = np.mean(samples)
sample_std = np.std(samples, ddof=1)

print(f"从 N({mu}, {sigma}²) 采样:")
print(f"  样本均值: {sample_mean:.4f} (真值: {mu})")
print(f"  样本标准差: {sample_std:.4f} (真值: {sigma})")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 迹图
axes[0].plot(samples[:1000], alpha=0.7)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Sample')
axes[0].set_title('Trace Plot (First 1000)')
axes[0].grid(True, alpha=0.3)

# 直方图
axes[1].hist(samples, bins=50, density=True, alpha=0.7, label='Samples')
x_range = np.linspace(samples.min(), samples.max(), 100)
axes[1].plot(x_range, norm.pdf(x_range, mu, sigma), 'r-', label='True Distribution')
axes[1].set_xlabel('x')
axes[1].set_ylabel('Density')
axes[1].set_title('Histogram vs True Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 自相关
from statsmodels.tsa.stattools import acf
acf_values = acf(samples, nlags=50)
axes[2].stem(acf_values)
axes[2].set_xlabel('Lag')
axes[2].set_ylabel('Autocorrelation')
axes[2].set_title('Autocorrelation Function')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 有效样本大小
ess = len(samples) / (1 + 2 * np.sum(acf_values[1:]))
print(f"  有效样本大小: {ess:.0f} / {len(samples)}")
```

### 练习4：Gibbs采样

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def gibbs_sampling(initial_state, n_iter=10000):
    """
    Gibbs采样：二元正态分布
    """
    x, y = initial_state
    samples = [[x, y]]
    
    # 二元正态分布的参数
    mu = np.array([0, 0])
    sigma = np.array([[1, 0.8], [0.8, 1]])
    
    # 条件分布参数
    rho = sigma[0, 1] / np.sqrt(sigma[0, 0] * sigma[1, 1])
    
    for _ in range(n_iter):
        # 采样 x | y
        mu_x_given_y = mu[0] + rho * (y - mu[1]) * np.sqrt(sigma[0, 0] / sigma[1, 1])
        sigma_x_given_y = sigma[0, 0] * (1 - rho**2)
        x = np.random.normal(mu_x_given_y, np.sqrt(sigma_x_given_y))
        
        # 采样 y | x
        mu_y_given_x = mu[1] + rho * (x - mu[0]) * np.sqrt(sigma[1, 1] / sigma[0, 0])
        sigma_y_given_x = sigma[1, 1] * (1 - rho**2)
        y = np.random.normal(mu_y_given_x, np.sqrt(sigma_y_given_x))
        
        samples.append([x, y])
    
    return np.array(samples)

# 运行Gibbs采样
samples = gibbs_sampling(initial_state=[0, 0], n_iter=10000)

# 去除burn-in
burn_in = 1000
samples = samples[burn_in:]

# 计算统计量
sample_mean = np.mean(samples, axis=0)
sample_cov = np.cov(samples, rowvar=False)

print("Gibbs采样：二元正态分布 N([0, 0], [[1, 0.8], [0.8, 1]])")
print(f"  样本均值: {sample_mean}")
print(f"  真实均值: [0, 0]")
print(f"  样本协方差:\n{sample_cov}")
print(f"  真实协方差:\n[[1, 0.8],\n [0.8, 1]]")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 散点图
axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
axes[0].set_xlabel('X1')
axes[0].set_ylabel('X2')
axes[0].set_title('Gibbs Samples')
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# 与真实分布比较
x, y = np.mgrid[-3:3:.1, -3:3:.1]
pos = np.dstack((x, y))
rv = multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]])
z = rv.pdf(pos)

axes[1].contour(x, y, z, levels=10, colors='r', alpha=0.7)
axes[1].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10)
axes[1].set_xlabel('X1')
axes[1].set_ylabel('X2')
axes[1].set_title('Samples vs True Distribution')
axes[1].grid(True, alpha=0.3)
axes[1].set_aspect('equal')

plt.tight_layout()
plt.show()
```

### 练习5：粒子滤波

```python
import numpy as np
import matplotlib.pyplot as plt

def particle_filter(n_particles=100, n_steps=100):
    """
    简单粒子滤波：一维运动模型
    """
    # 真实状态
    true_states = []
    x = 0
    for _ in range(n_steps):
        x = x + 0.5 + np.random.normal(0, 0.1)  # 运动模型
        true_states.append(x)
    true_states = np.array(true_states)
    
    # 观测
    observations = true_states + np.random.normal(0, 0.5, n_steps)
    
    # 粒子滤波
    particles = np.random.normal(0, 1, n_particles)
    estimates = []
    
    for t in range(n_steps):
        # 预测
        particles = particles + 0.5 + np.random.normal(0, 0.1, n_particles)
        
        # 更新权重
        weights = np.exp(-0.5 * ((observations[t] - particles) / 0.5)**2)
        weights = weights / np.sum(weights)
        
        # 重采样
        indices = np.random.choice(n_particles, n_particles, p=weights)
        particles = particles[indices]
        
        # 估计
        estimate = np.mean(particles)
        estimates.append(estimate)
    
    estimates = np.array(estimates)
    
    return true_states, observations, estimates

# 运行粒子滤波
true_states, observations, estimates = particle_filter(n_particles=100, n_steps=100)

# 计算误差
rmse = np.sqrt(np.mean((estimates - true_states)**2))
print(f"粒子滤波 RMSE: {rmse:.4f}")

# 可视化
plt.figure(figsize=(12, 5))

# 状态估计
plt.subplot(1, 2, 1)
plt.plot(true_states, 'k-', label='True State', linewidth=2)
plt.plot(observations, 'r.', label='Observations', alpha=0.5)
plt.plot(estimates, 'b--', label='Estimate', linewidth=2)
plt.xlabel('Time')
plt.ylabel('State')
plt.title('Particle Filter: State Estimation')
plt.legend()
plt.grid(True, alpha=0.3)

# 误差
plt.subplot(1, 2, 2)
plt.plot(estimates - true_states, 'b-')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Estimation Error')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

**概率论与统计部分完成！** 🎉

**下一步**：[数值计算](../numerical-computation/README.md)