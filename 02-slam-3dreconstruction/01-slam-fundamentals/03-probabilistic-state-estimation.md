# 1.3 概率状态估计

## 1. 概述

概率状态估计是SLAM和机器人学的核心理论支柱。它提供了一套数学工具，用于从带有噪声的传感器数据中推断系统的状态变量。本章将系统介绍各种状态估计方法及其在SLAM中的应用。

## 2. 点估计方法

### 2.1 最大似然估计（MLE）

最大似然估计寻找使观测数据的似然函数最大化的参数值：

$$\hat{\theta}_{\text{MLE}} = \arg\max_{\theta} P(\mathbf{z} \mid \theta)$$

在SLAM中，MLE通常用于位姿估计：给定观测和地图，寻找最可能的相机位姿。

### 2.2 最大后验估计（MAP）

MAP估计引入了参数先验：

$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} P(\theta \mid \mathbf{z}) = \arg\max_{\theta} P(\mathbf{z} \mid \theta) P(\theta)$$

在SLAM后端优化中，MAP估计是最常用的框架。整个SLAM问题可以看作一个MAP估计问题：

$$\{\mathbf{x}_{1:K}^*, \mathbf{m}^*\} = \arg\max_{\mathbf{x}_{1:K}, \mathbf{m}} P(\mathbf{x}_{1:K}, \mathbf{m} \mid \mathbf{z}_{1:K}, \mathbf{u}_{1:K})$$

### 2.3 最小均方误差估计（MMSE）

MMSE估计最小化估计值与真实值之间的均方误差：

$$\hat{\theta}_{\text{MMSE}} = \mathbb{E}[\theta \mid \mathbf{z}] = \int \theta \, P(\theta \mid \mathbf{z}) \, d\theta$$

对于高斯分布，MMSE估计等价于MAP估计。

## 3. 高斯分布与状态估计

### 3.1 多元高斯分布

多元高斯分布的概率密度函数：

$$P(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n |\mathbf{\Sigma}|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

其中 $\boldsymbol{\mu}$ 是均值向量，$\mathbf{\Sigma}$ 是协方差矩阵。

### 3.2 信息形式表示

高斯分布也可以用信息形式（canonical form）表示：

$$P(\mathbf{x}) = \eta \exp\left(-\frac{1}{2}\mathbf{x}^T \mathbf{\Omega} \mathbf{x} + \boldsymbol{\xi}^T \mathbf{x}\right)$$

其中信息矩阵 $\mathbf{\Omega} = \mathbf{\Sigma}^{-1}$，信息向量 $\boldsymbol{\xi} = \mathbf{\Sigma}^{-1} \boldsymbol{\mu}$。

**信息形式的优势**：
- 条件独立一目了然（$\mathbf{\Omega}_{ij} = 0$ 表示 $\mathbf{x}_i$ 和 $\mathbf{x}_j$ 条件独立）
- 融合多个独立观测只需相加：$\mathbf{\Omega} = \sum_i \mathbf{\Omega}_i$
- 在稀疏SLAM问题中特别高效

### 3.3 高斯分布的边缘化和条件化

**联合高斯分布**：

$$\begin{bmatrix} \mathbf{x}_a \\ \mathbf{x}_b \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \boldsymbol{\mu}_a \\ \boldsymbol{\mu}_b \end{bmatrix}, \begin{bmatrix} \mathbf{\Sigma}_{aa} & \mathbf{\Sigma}_{ab} \\ \mathbf{\Sigma}_{ba} & \mathbf{\Sigma}_{bb} \end{bmatrix}\right)$$

**边缘分布**：

$$P(\mathbf{x}_a) = \mathcal{N}(\boldsymbol{\mu}_a, \mathbf{\Sigma}_{aa})$$

**条件分布**：

$$P(\mathbf{x}_a \mid \mathbf{x}_b) = \mathcal{N}(\boldsymbol{\mu}_a + \mathbf{\Sigma}_{ab} \mathbf{\Sigma}_{bb}^{-1} (\mathbf{x}_b - \boldsymbol{\mu}_b), \mathbf{\Sigma}_{aa} - \mathbf{\Sigma}_{ab} \mathbf{\Sigma}_{bb}^{-1} \mathbf{\Sigma}_{ba})$$

## 4. 非线性最小二乘估计

### 4.1 问题形式

SLAM后端本质上是一个非线性最小二乘问题：

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_{i} \|\mathbf{e}_i(\mathbf{x})\|^2_{\mathbf{\Sigma}_i}$$

其中 $\|\mathbf{e}\|^2_{\mathbf{\Sigma}} = \mathbf{e}^T \mathbf{\Sigma}^{-1} \mathbf{e}$ 是马氏距离平方。

### 4.2 迭代求解

对于非线性问题，使用高斯-牛顿法迭代求解：

$$\mathbf{x}^{t+1} = \mathbf{x}^{t} + \Delta \mathbf{x}$$

其中增量通过求解正规方程得到：

$$(\mathbf{J}^T \mathbf{\Sigma}^{-1} \mathbf{J}) \Delta \mathbf{x} = -\mathbf{J}^T \mathbf{\Sigma}^{-1} \mathbf{e}(\mathbf{x}^t)$$

## 5. 误差传播

### 5.1 一阶误差传播

给定函数 $\mathbf{y} = f(\mathbf{x})$，其中 $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_x, \mathbf{\Sigma}_x)$：

$$\boldsymbol{\mu}_y \approx f(\boldsymbol{\mu}_x)$$
$$\mathbf{\Sigma}_y \approx \mathbf{J}_f \mathbf{\Sigma}_x \mathbf{J}_f^T$$

其中 $\mathbf{J}_f = \frac{\partial f}{\partial \mathbf{x}}\big|_{\boldsymbol{\mu}_x}$。

### 5.2 SLAM中的误差传播

在SLAM中，误差传播体现在多个方面：
- 位姿估计误差随运动累积
- 观测误差影响地图特征的精度
- 回环检测的误差传播

## 6. Cramer-Rao下界

CRLB给出了无偏估计器方差的理论下界：

$$\text{Cov}(\hat{\theta}) \geq \mathcal{I}(\theta)^{-1}$$

其中 $\mathcal{I}(\theta)$ 是Fisher信息矩阵：

$$\mathcal{I}(\theta)_{ij} = -\mathbb{E}\left[\frac{\partial^2 \log P(\mathbf{z} \mid \theta)}{\partial \theta_i \partial \theta_j}\right]$$

## 7. 鲁棒估计

### 7.1 M-估计

M-估计使用鲁棒损失函数替代平方损失：

$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_{i} \rho(\mathbf{e}_i(\mathbf{x}))$$

其中 $\rho$ 是鲁棒核函数。

### 7.2 常用鲁棒核函数

| 核函数 | 公式 | 特点 |
|--------|------|------|
| L2（平方） | $\rho(e) = e^2/2$ | 对异常值敏感 |
| Huber | $\rho(e) = \begin{cases} e^2/2 & \|e\| \leq \delta \\ \delta(\|e\| - \delta/2) & \text{其他} \end{cases}$ | 平滑过渡 |
| Cauchy | $\rho(e) = \frac{\delta^2}{2} \log(1 + (e/\delta)^2)$ | 强鲁棒性 |
| Tukey | $\rho(e) = \begin{cases} \frac{\delta^2}{6}[1-(1-(e/\delta)^2)^3] & \|e\| \leq \delta \\ \delta^2/6 & \text{其他} \end{cases}$ | 完全拒绝异常值 |

## 8. 参考文献

1. Kay, S. M. (1993). *Fundamentals of Statistical Signal Processing: Estimation Theory*. Prentice Hall.
2. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision*. Cambridge University Press.
3. Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). Bundle adjustment—a modern synthesis. *Vision Algorithms: Theory and Practice*.
4. Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley.
