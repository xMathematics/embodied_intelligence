# 1.2 贝叶斯滤波框架

## 1. 概述

贝叶斯滤波是SLAM问题的核心方法论基础。它提供了一种概率框架，通过递归方式从带噪声的传感器数据中估计系统的状态。本章将详细介绍贝叶斯滤波的数学基础、推导过程及其在SLAM中的应用。

## 2. 贝叶斯滤波的基本原理

### 2.1 马尔可夫假设

贝叶斯滤波的核心是**马尔可夫假设（Markov Assumption）**：当前状态包含了所有预测未来所需的信息，给定当前状态，过去的状态和观测条件独立于未来的状态。

数学上，马尔可夫假设表示为：

$$P(\mathbf{x}_k \mid \mathbf{x}_{1:k-1}, \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k-1}) = P(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k)$$

$$P(\mathbf{z}_k \mid \mathbf{x}_{1:k}, \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k}) = P(\mathbf{z}_k \mid \mathbf{x}_k)$$

这意味着：
- 状态转移只依赖于上一个状态和控制输入（**状态转移概率**）
- 观测只依赖于当前状态（**观测概率**）

### 2.2 置信度（Belief）

置信度是系统状态的后验概率分布，是所有贝叶斯滤波方法的核心概念：

$$\text{Bel}(\mathbf{x}_k) = P(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k})$$

即给定到时刻 $k$ 的所有观测 $\mathbf{z}_{1:k}$ 和控制输入 $\mathbf{u}_{1:k}$，状态 $\mathbf{x}_k$ 的概率分布。

### 2.3 贝叶斯滤波的递归推导

#### 预测步骤（Prediction / A Priori）

假设在时刻 $k-1$ 我们已经有了后验置信度 $\text{Bel}(\mathbf{x}_{k-1})$，在获取新的控制输入 $\mathbf{u}_k$ 后，我们需要预测时刻 $k$ 的状态：

$$\overline{\text{Bel}}(\mathbf{x}_k) = P(\mathbf{x}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k})$$

根据全概率公式：

$$\overline{\text{Bel}}(\mathbf{x}_k) = \int P(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k) \cdot \text{Bel}(\mathbf{x}_{k-1}) \, d\mathbf{x}_{k-1}$$

这就是**Chapman-Kolmogorov方程**，它将先验置信度通过状态转移概率向前传播。

#### 更新步骤（Correction / Posterior）

当获得新的观测 $\mathbf{z}_k$ 后，我们使用贝叶斯公式更新置信度：

$$\text{Bel}(\mathbf{x}_k) = P(\mathbf{x}_k \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k}) = \frac{P(\mathbf{z}_k \mid \mathbf{x}_k) \cdot \overline{\text{Bel}}(\mathbf{x}_k)}{P(\mathbf{z}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k})}$$

其中归一化常数为：

$$\eta = P(\mathbf{z}_k \mid \mathbf{z}_{1:k-1}, \mathbf{u}_{1:k}) = \int P(\mathbf{z}_k \mid \mathbf{x}_k) \cdot \overline{\text{Bel}}(\mathbf{x}_k) \, d\mathbf{x}_k$$

因此：

$$\text{Bel}(\mathbf{x}_k) = \eta \cdot P(\mathbf{z}_k \mid \mathbf{x}_k) \cdot \overline{\text{Bel}}(\mathbf{x}_k)$$

### 2.4 贝叶斯滤波算法

```
算法：贝叶斯滤波
输入：Bel(x_{k-1}), u_k, z_k
输出：Bel(x_k)

1: function BAYES_FILTER(Bel(x_{k-1}), u_k, z_k)
2:     // 预测步骤
3:     for all x_k do
4:         Bel_bar(x_k) = ∫ P(x_k | x_{k-1}, u_k) · Bel(x_{k-1}) dx_{k-1}
5:     end for
6:     // 更新步骤
7:     for all x_k do
8:         Bel(x_k) = η · P(z_k | x_k) · Bel_bar(x_k)
9:     end for
10:    return Bel(x_k)
11: end function
```

## 3. 贝叶斯滤波在SLAM中的应用

### 3.1 SLAM中的联合状态

在SLAM中，状态空间包括机器人的位姿和环境中的所有特征：

$$\mathbf{y}_k = [\mathbf{x}_k^T, \mathbf{m}_1^T, \mathbf{m}_2^T, \ldots, \mathbf{m}_N^T]^T$$

这使得状态空间的维度随着地图特征数量的增加而增长（对于大场景可能达到百万级别），这也是为什么直接应用贝叶斯滤波到SLAM在计算上具有挑战性。

### 3.2 贝叶斯滤波的变体

根据对概率分布表示和积分计算方式的不同，贝叶斯滤波有多种具体实现：

| 方法 | 分布表示 | 积分方法 | 适用场景 |
|------|----------|----------|----------|
| 卡尔曼滤波(KF) | 高斯分布 | 解析解 | 线性系统、高斯噪声 |
| 扩展卡尔曼滤波(EKF) | 高斯分布(线性化) | 解析近似 | 弱非线性系统 |
| 无迹卡尔曼滤波(UKF) | 高斯分布(Sigma点) | 数值近似 | 中非线性系统 |
| 粒子滤波(PF) | 粒子集 | 蒙特卡洛采样 | 任意非线性系统 |
| 信息滤波(IF) | 高斯分布(信息形式) | 解析解 | 多机器人系统 |

### 3.3 卡尔曼滤波（Kalman Filter）

卡尔曼滤波是贝叶斯滤波在线性高斯系统下的最优解。假设：

- 状态转移和观测模型都是线性的
- 噪声服从高斯分布

**线性模型**：

$$\mathbf{x}_k = \mathbf{A}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k$$
$$\mathbf{z}_k = \mathbf{C}_k \mathbf{x}_k + \mathbf{v}_k$$

**预测步骤**：

$$\hat{\mathbf{x}}_{k|k-1} = \mathbf{A}_k \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k$$
$$\mathbf{P}_{k|k-1} = \mathbf{A}_k \mathbf{P}_{k-1|k-1} \mathbf{A}_k^T + \mathbf{Q}_k$$

**更新步骤**：

$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{C}_k^T (\mathbf{C}_k \mathbf{P}_{k|k-1} \mathbf{C}_k^T + \mathbf{R}_k)^{-1}$$
$$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \mathbf{C}_k \hat{\mathbf{x}}_{k|k-1})$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{C}_k) \mathbf{P}_{k|k-1}$$

### 3.4 高斯滤波的一般形式

对于非线性系统，高斯滤波使用高斯分布近似后验，但线性化方法不同：

**扩展卡尔曼滤波（EKF）**：使用一阶泰勒展开线性化

$$\mathbf{A}_k = \frac{\partial f}{\partial \mathbf{x}}\bigg|_{\hat{\mathbf{x}}_{k-1|k-1}}, \quad \mathbf{C}_k = \frac{\partial h}{\partial \mathbf{x}}\bigg|_{\hat{\mathbf{x}}_{k|k-1}}$$

**无迹卡尔曼滤波（UKF）**：使用确定性Sigma点传播

$$\boldsymbol{\mathcal{X}}_{k-1} = [\hat{\mathbf{x}}_{k-1}, \hat{\mathbf{x}}_{k-1} \pm \sqrt{(n+\lambda)\mathbf{P}_{k-1}}]$$

### 3.5 粒子滤波（Particle Filter）

粒子滤波使用一组加权粒子来表示任意分布：

$$\text{Bel}(\mathbf{x}_k) \approx \sum_{i=1}^{M} w_k^{[i]} \delta(\mathbf{x}_k - \mathbf{x}_k^{[i]})$$

其中 $w_k^{[i]}$ 是第 $i$ 个粒子的重要性权重，满足 $\sum_i w_k^{[i]} = 1$。

**重要性采样**：从提议分布 $q(\mathbf{x}_k \mid \mathbf{x}_{k-1}^{[i]}, \mathbf{z}_k, \mathbf{u}_k)$ 中采样新粒子

**权重更新**：

$$w_k^{[i]} = w_{k-1}^{[i]} \cdot \frac{P(\mathbf{z}_k \mid \mathbf{x}_k^{[i]}) P(\mathbf{x}_k^{[i]} \mid \mathbf{x}_{k-1}^{[i]}, \mathbf{u}_k)}{q(\mathbf{x}_k^{[i]} \mid \mathbf{x}_{k-1}^{[i]}, \mathbf{z}_k, \mathbf{u}_k)}$$

## 4. 参考文献

1. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
2. Doucet, A., de Freitas, N., & Gordon, N. (2001). *Sequential Monte Carlo Methods in Practice*. Springer.
3. Julier, S. J., & Uhlmann, J. K. (1997). New extension of the Kalman filter to nonlinear systems. *SPIE Proceedings*.
4. Arulampalam, M. S., Maskell, S., Gordon, N., & Clapp, T. (2002). A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking. *IEEE Transactions on Signal Processing*, 50(2), 174-188.
