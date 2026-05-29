# 3.3 贝叶斯推断

## 目录

- [1. 贝叶斯定理回顾](#1-贝叶斯定理回顾)
- [2. 贝叶斯推断框架](#2-贝叶斯推断框架)
- [3. 参数估计](#3-参数估计)
- [4. 贝叶斯模型选择](#4-贝叶斯模型选择)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 贝叶斯定理回顾

### 1.1 基本形式

$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

**术语**：
- $P(\theta)$：先验分布
- $P(D|\theta)$：似然函数
- $P(\theta|D)$：后验分布
- $P(D) = \int P(D|\theta)P(\theta)d\theta$：证据（边际似然）

### 1.2 贝叶斯推断的核心思想

**从先验到后验**：通过观测数据更新对参数的信念。

$$\text{先验} \xrightarrow{\text{数据}} \text{后验} \xrightarrow{\text{新数据}} \text{新后验}$$

---

## 2. 贝叶斯推断框架

### 2.1 推断流程

1. **建立模型**：选择似然函数 $P(D|\theta)$
2. **指定先验**：根据先验知识选择 $P(\theta)$
3. **收集数据**：观测数据 $D$
4. **计算后验**：应用贝叶斯定理
5. **进行推断**：使用后验分布进行预测或决策

### 2.2 共轭先验

**定义**：先验和后验属于同一分布族。

**优势**：后验分布有解析解。

**常见共轭对**：

| 似然 | 先验 | 后验 |
|------|------|------|
| 伯努利 | Beta | Beta |
| 二项 | Beta | Beta |
| 泊松 | Gamma | Gamma |
| 正态（已知方差） | 正态 | 正态 |
| 正态（未知均值和方差） | Normal-Inverse-Gamma | Normal-Inverse-Gamma |

### 2.3 Beta-Bernoulli 共轭

**似然**：$P(D|\theta) = \theta^s (1-\theta)^{n-s}$

其中 $s$ 是成功次数，$n$ 是总试验次数。

**先验**：$\theta \sim \text{Beta}(\alpha, \beta)$

$$P(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$$

**后验**：$\theta|D \sim \text{Beta}(\alpha + s, \beta + n - s)$

---

## 3. 参数估计

### 3.1 点估计

#### 最大后验估计（MAP）

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|D) = \arg\max_{\theta} P(D|\theta)P(\theta)$$

**与最大似然估计（MLE）的关系**：

$$\hat{\theta}_{MLE} = \arg\max_{\theta} P(D|\theta)$$

MAP 是正则化的 MLE。

#### 后验均值

$$\hat{\theta}_{mean} = E[\theta|D] = \int \theta P(\theta|D) d\theta$$

#### 后验中位数

$$\hat{\theta}_{median} = \text{median}(\theta|D)$$

### 3.2 区间估计

**可信区间（Credible Interval）**：

$$P(a \leq \theta \leq b | D) = 1 - \alpha$$

**最高密度区间（HDI）**：

包含最高概率密度的区间，长度最短。

### 3.3 预测分布

**后验预测分布**：

$$P(\tilde{D}|D) = \int P(\tilde{D}|\theta) P(\theta|D) d\theta$$

**意义**：考虑参数不确定性，对新数据的预测。

---

## 4. 贝叶斯模型选择

### 4.1 边际似然

$$P(D|M) = \int P(D|\theta, M) P(\theta|M) d\theta$$

**贝叶斯因子**：

$$BF_{12} = \frac{P(D|M_1)}{P(D|M_2)}$$

- $BF_{12} > 1$：支持模型 $M_1$
- $BF_{12} < 1$：支持模型 $M_2$

### 4.2 贝叶斯信息准则（BIC）

$$\text{BIC} = -2\ln P(D|\hat{\theta}_{MLE}) + k \ln n$$

其中 $k$ 是参数数量，$n$ 是样本数量。

**模型选择**：选择 BIC 最小的模型。

### 4.3 贝叶斯模型平均

$$P(\tilde{D}|D) = \sum_{M} P(\tilde{D}|D, M) P(M|D)$$

**优势**：考虑模型不确定性。

---

## 5. 在具身智能中的应用

### 5.1 卡尔曼滤波

**状态更新**：

$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t(z_t - H\hat{x}_{t|t-1})$$

**卡尔曼增益**：

$$K_t = P_{t|t-1}H^T(HP_{t|t-1}H^T + R)^{-1}$$

**协方差更新**：

$$P_{t|t} = (I - K_t H)P_{t|t-1}$$

**贝叶斯解释**：递归贝叶斯滤波在高斯线性模型下的特例。

### 5.2 粒子滤波

**重要性采样**：

$$w_t^{(i)} \propto w_{t-1}^{(i)} \frac{p(z_t|x_t^{(i)})}{q(x_t^{(i)}|x_{t-1}^{(i)}, z_t)}$$

**重采样**：根据权重重新采样粒子。

**优势**：适用于非线性非高斯系统。

### 5.3 SLAM

**贝叶斯SLAM**：

$$p(x_{1:t}, m | z_{1:t}, u_{1:t})$$

其中：
- $x_{1:t}$：机器人轨迹
- $m$：地图
- $z_{1:t}$：观测
- $u_{1:t}$：控制输入

### 5.4 不确定性量化

**后验分布**：提供完整的不确定性信息。

**决策理论**：基于后验分布进行最优决策。

---

## 6. 相关论文

### 贝叶斯方法

1. **"Bayesian Data Analysis"** - Gelman et al. (2013)
   - 贝叶斯数据分析权威教材

2. **"Pattern Recognition and Machine Learning"** - Bishop (2006)
   - 贝叶斯方法在模式识别中的应用

### 应用相关

3. **"Probabilistic Robotics"** - Thrun et al. (2005)
   - 概率机器人学

4. **"Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking"** - Arulampalam et al. (2002)
   - 粒子滤波综述

---

## 7. 实践练习

### 练习1：Beta-Bernoulli 共轭

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def beta_bernoulli_conjugate(alpha_prior, beta_prior, n_successes, n_failures):
    """
    Beta-Bernoulli 共轭
    """
    # 先验
    theta = np.linspace(0, 1, 1000)
    prior = beta.pdf(theta, alpha_prior, beta_prior)
    
    # 后验
    alpha_post = alpha_prior + n_successes
    beta_post = beta_prior + n_failures
    posterior = beta.pdf(theta, alpha_post, beta_post)
    
    # MAP估计
    theta_map = (alpha_post - 1) / (alpha_post + beta_post - 2)
    
    # 后验均值
    theta_mean = alpha_post / (alpha_post + beta_post)
    
    # 95%可信区间
    ci_low, ci_high = beta.interval(0.95, alpha_post, beta_post)
    
    return {
        'theta': theta,
        'prior': prior,
        'posterior': posterior,
        'theta_map': theta_map,
        'theta_mean': theta_mean,
        'ci': (ci_low, ci_high),
        'alpha_post': alpha_post,
        'beta_post': beta_post
    }

# 示例：硬币抛掷
alpha_prior = 1  # Beta(1,1) = 均匀分布
beta_prior = 1
n_successes = 7
n_failures = 3

result = beta_bernoulli_conjugate(alpha_prior, beta_prior, n_successes, n_failures)

print(f"先验: Beta({alpha_prior}, {beta_prior})")
print(f"数据: {n_successes} 次正面, {n_failures} 次反面")
print(f"后验: Beta({result['alpha_post']}, {result['beta_post']})")
print(f"MAP估计: {result['theta_map']:.4f}")
print(f"后验均值: {result['theta_mean']:.4f}")
print(f"95%可信区间: [{result['ci'][0]:.4f}, {result['ci'][1]:.4f}]")

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(result['theta'], result['prior'], 'b--', label='Prior')
plt.plot(result['theta'], result['posterior'], 'r-', label='Posterior', linewidth=2)
plt.axvline(result['theta_map'], color='g', linestyle=':', label=f'MAP={result['theta_map']:.3f}')
plt.axvline(result['theta_mean'], color='orange', linestyle=':', label=f'Mean={result['theta_mean']:.3f}')
plt.axvspan(result['ci'][0], result['ci'][1], alpha=0.2, color='red', label='95% CI')
plt.xlabel('θ')
plt.ylabel('Probability Density')
plt.title('Beta-Bernoulli Conjugate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 绷习2：序列贝叶斯更新

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def sequential_bayesian_update(alpha_prior, beta_prior, observations):
    """
    序列贝叶斯更新
    """
    theta = np.linspace(0, 1, 1000)
    
    # 初始化
    alphas = [alpha_prior]
    betas = [beta_prior]
    posteriors = [beta.pdf(theta, alpha_prior, beta_prior)]
    
    # 逐步更新
    alpha_current = alpha_prior
    beta_current = beta_prior
    
    for obs in observations:
        if obs == 1:  # 成功
            alpha_current += 1
        else:  # 失败
            beta_current += 1
        
        alphas.append(alpha_current)
        betas.append(beta_current)
        posteriors.append(beta.pdf(theta, alpha_current, beta_current))
    
    return theta, posteriors, alphas, betas

# 模拟抛硬币序列
np.random.seed(42)
true_theta = 0.7
n_flips = 20
observations = np.random.binomial(1, true_theta, size=n_flips)

print(f"真实概率: {true_theta}")
print(f"观测序列: {observations}")

# 序列更新
theta, posteriors, alphas, betas = sequential_bayesian_update(1, 1, observations)

# 可视化
fig, axes = plt.subplots(4, 5, figsize=(20, 12))
axes = axes.flatten()

for i in range(len(observations) + 1):
    ax = axes[i]
    ax.plot(theta, posteriors[i], 'b-', linewidth=2)
    ax.axvline(true_theta, color='r', linestyle='--', label='True θ')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(posteriors[i]) * 1.1)
    ax.set_xlabel('θ')
    ax.set_ylabel('Density')
    if i == 0:
        ax.set_title(f'Prior: Beta({alphas[i]}, {betas[i]})')
    else:
        ax.set_title(f'After {i} flips: Beta({alphas[i]}, {betas[i]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 最终结果
print(f"\n最终后验: Beta({alphas[-1]}, {betas[-1]})")
print(f"后验均值: {alphas[-1] / (alphas[-1] + betas[-1]):.4f}")
```

### 练习3：正态分布的贝叶斯推断

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def normal_bayesian_inference(data, mu_prior, sigma_prior, sigma_known):
    """
    正态分布的贝叶斯推断（已知方差）
    """
    n = len(data)
    sample_mean = np.mean(data)
    
    # 后验参数
    sigma_post = 1 / np.sqrt(1/sigma_prior**2 + n/sigma_known**2)
    mu_post = sigma_post**2 * (mu_prior/sigma_prior**2 + n*sample_mean/sigma_known**2)
    
    # 可视化
    x = np.linspace(mu_prior - 3*sigma_prior, mu_prior + 3*sigma_prior, 1000)
    
    prior = norm.pdf(x, mu_prior, sigma_prior)
    likelihood = norm.pdf(sample_mean, x, sigma_known/np.sqrt(n))
    likelihood = likelihood / np.max(likelihood) * np.max(prior)  # 缩放以便可视化
    posterior = norm.pdf(x, mu_post, sigma_post)
    
    return {
        'x': x,
        'prior': prior,
        'likelihood': likelihood,
        'posterior': posterior,
        'mu_post': mu_post,
        'sigma_post': sigma_post
    }

# 生成数据
np.random.seed(42)
true_mu = 5
true_sigma = 2
data = np.random.normal(true_mu, true_sigma, size=20)

# 先验
mu_prior = 0
sigma_prior = 3

# 贝叶斯推断
result = normal_bayesian_inference(data, mu_prior, sigma_prior, true_sigma)

print(f"真实均值: {true_mu}")
print(f"先验: N({mu_prior}, {sigma_prior}²)")
print(f"样本均值: {np.mean(data):.4f}")
print(f"后验: N({result['mu_post']:.4f}, {result['sigma_post']:.4f}²)")

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(result['x'], result['prior'], 'b--', label='Prior')
plt.plot(result['x'], result['likelihood'], 'g:', label='Likelihood (scaled)')
plt.plot(result['x'], result['posterior'], 'r-', label='Posterior', linewidth=2)
plt.axvline(true_mu, color='k', linestyle='--', label='True μ')
plt.xlabel('μ')
plt.ylabel('Density')
plt.title('Bayesian Inference for Normal Mean')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 练习4：贝叶斯模型选择

```python
import numpy as np
from scipy.stats import norm

def bayesian_factor(model1, model2, data):
    """
    计算贝叶斯因子
    """
    # 简化：使用网格近似计算边际似然
    mu_grid = np.linspace(-5, 5, 100)
    sigma_grid = np.linspace(0.1, 5, 50)
    
    def marginal_likelihood(model):
        log_marginal = 0
        for mu in mu_grid:
            for sigma in sigma_grid:
                # 先验（均匀分布）
                prior = 1 / (10 * 4.9)
                
                # 似然
                likelihood = np.prod(norm.pdf(data, mu, sigma))
                
                log_marginal += prior * likelihood
        
        return log_marginal * (10 / 100) * (4.9 / 50)
    
    ml1 = marginal_likelihood(model1)
    ml2 = marginal_likelihood(model2)
    
    bf = ml1 / ml2
    
    return bf

# 生成数据
np.random.seed(42)
data = np.random.normal(0, 1, size=20)

# 两个模型
model1 = {'name': 'N(0, 1)'}
model2 = {'name': 'N(1, 1)'}

# 计算贝叶斯因子（简化版）
print("贝叶斯模型选择:")
print(f"模型1: {model1['name']}")
print(f"模型2: {model2['name']}")
print(f"\n注意: 精确计算需要数值积分或MCMC")
```

### 练习5：后验预测分布

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom

def posterior_predictive(alpha_post, beta_post, n_new, n_samples=10000):
    """
    后验预测分布（Beta-Binomial）
    """
    # 从后验采样
    theta_samples = np.random.beta(alpha_post, beta_post, n_samples)
    
    # 计算预测分布
    predictive_samples = np.random.binomial(n_new, theta_samples)
    
    # 理论预测分布（Beta-Binomial）
    k = np.arange(0, n_new + 1)
    predictive_theoretical = binom.pmf(k, n_new, alpha_post / (alpha_post + beta_post))
    
    return predictive_samples, predictive_theoretical, k

# 后验参数
alpha_post = 8
beta_post = 4
n_new = 10

# 计算后验预测分布
samples, theoretical, k = posterior_predictive(alpha_post, beta_post, n_new)

print(f"后验: Beta({alpha_post}, {beta_post})")
print(f"预测: {n_new} 次试验中的成功次数")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 直方图
axes[0].hist(samples, bins=np.arange(-0.5, n_new + 1.5, 1), density=True, alpha=0.7, label='Samples')
axes[0].plot(k, theoretical, 'ro-', label='Theoretical', markersize=8)
axes[0].set_xlabel('Number of Successes')
axes[0].set_ylabel('Probability')
axes[0].set_title('Posterior Predictive Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 累积分布
axes[1].plot(k, np.cumsum(theoretical), 'ro-', markersize=8, label='Theoretical')
axes[1].plot(k, np.bincount(samples, minlength=n_new+1) / len(samples), 'bo--', label='Sample')
axes[1].set_xlabel('Number of Successes')
axes[1].set_ylabel('Cumulative Probability')
axes[1].set_title('Cumulative Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

**下一步**：[信息论](04-information-theory.md)