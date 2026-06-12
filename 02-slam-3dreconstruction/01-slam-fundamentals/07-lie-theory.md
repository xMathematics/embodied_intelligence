# 1.7 李群与李代数

## 1. 概述

李群（Lie Group）和李代数（Lie Algebra）是现代SLAM后端优化的数学基础。它们提供了在流形（manifold）上进行优化的理论框架，尤其适用于处理旋转和位姿的表示与扰动。

## 2. SO(3)群与so(3)代数

### 2.1 SO(3)定义

$$\text{SO}(3) = \{\mathbf{R} \in \mathbb{R}^{3 \times 3} \mid \mathbf{R}^T \mathbf{R} = \mathbf{I}, \det(\mathbf{R}) = 1\}$$

### 2.2 so(3)代数

$$\mathfrak{so}(3) = \{\boldsymbol{\phi} = \boldsymbol{\omega}^\wedge \in \mathbb{R}^{3 \times 3} \mid \boldsymbol{\omega} \in \mathbb{R}^3\}$$

其中 $^\wedge$ 运算是反对称矩阵：

$$ \boldsymbol{\omega}^\wedge = \begin{bmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{bmatrix} $$

$^\vee$ 运算是逆操作：$(\boldsymbol{\omega}^\wedge)^\vee = \boldsymbol{\omega}$

### 2.3 指数映射（Exponential Map）

从 $\mathfrak{so}(3)$ 到 $\text{SO}(3)$ 的映射：

$$ \exp(\boldsymbol{\phi}^\wedge) = \mathbf{I} + \frac{\sin\|\boldsymbol{\phi}\|}{\|\boldsymbol{\phi}\|} \boldsymbol{\phi}^\wedge + \frac{1-\cos\|\boldsymbol{\phi}\|}{\|\boldsymbol{\phi}\|^2} (\boldsymbol{\phi}^\wedge)^2 $$

当 $\boldsymbol{\phi} = \theta \mathbf{n}$ 时，即罗德里格斯公式。

### 2.4 对数映射（Logarithmic Map）

从 $\text{SO}(3)$ 到 $\mathfrak{so}(3)$ 的映射：

$$ \theta = \arccos\left(\frac{\text{tr}(\mathbf{R}) - 1}{2}\right) $$
$$ \boldsymbol{\omega} = \frac{\theta}{2\sin\theta} \begin{bmatrix} R_{32} - R_{23} \\ R_{13} - R_{31} \\ R_{21} - R_{12} \end{bmatrix} $$

### 2.5 左扰动的雅可比

在SLAM优化中，我们通常需要求旋转对扰动的导数。左扰动模型是：

$$ \frac{\partial \mathbf{R} \mathbf{p}}{\partial \boldsymbol{\phi}} = -(\mathbf{R} \mathbf{p})^\wedge $$

更一般地，对于函数 $f(\mathbf{R})$，其关于左扰动 $\delta\boldsymbol{\phi}$ 的导数为：

$$ \frac{\partial f(\mathbf{R} \exp(\delta\boldsymbol{\phi}^\wedge))}{\partial \delta\boldsymbol{\phi}} \bigg|_{\delta\boldsymbol{\phi}=0} $$

## 3. SE(3)群与se(3)代数

### 3.1 SE(3)定义

$$\text{SE}(3) = \left\{ \mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} \in \mathbb{R}^{4 \times 4} \mid \mathbf{R} \in \text{SO}(3), \mathbf{t} \in \mathbb{R}^3 \right\}$$

### 3.2 se(3)代数

$$ \mathfrak{se}(3) = \left\{ \boldsymbol{\xi}^\wedge = \begin{bmatrix} \boldsymbol{\phi}^\wedge & \boldsymbol{\rho} \\ \mathbf{0}^T & 0 \end{bmatrix} \in \mathbb{R}^{4 \times 4} \mid \boldsymbol{\xi} = \begin{bmatrix} \boldsymbol{\rho} \\ \boldsymbol{\phi} \end{bmatrix} \in \mathbb{R}^6 \right\} $$

其中 $\boldsymbol{\rho}$ 是平移分量，$\boldsymbol{\phi}$ 是旋转分量。

### 3.3 SE(3)的指数映射

$$ \exp(\boldsymbol{\xi}^\wedge) = \begin{bmatrix} \exp(\boldsymbol{\phi}^\wedge) & \mathbf{J} \boldsymbol{\rho} \\ \mathbf{0}^T & 1 \end{bmatrix} $$

其中 $\mathbf{J}$ 是左雅可比矩阵：

$$ \mathbf{J} = \frac{\sin\theta}{\theta} \mathbf{I} + \left(1 - \frac{\sin\theta}{\theta}\right) \mathbf{n} \mathbf{n}^T + \frac{1 - \cos\theta}{\theta} \mathbf{n}^\wedge $$

### 3.4 SE(3)的对数映射

$$ \theta = \arccos\left(\frac{\text{tr}(\mathbf{R}) - 1}{2}\right) $$
$$ \boldsymbol{\phi} = \frac{\theta}{2\sin\theta} \begin{bmatrix} R_{32} - R_{23} \\ R_{13} - R_{31} \\ R_{21} - R_{12} \end{bmatrix} $$
$$ \boldsymbol{\rho} = \mathbf{J}^{-1} \mathbf{t} $$

## 4. BCH公式及其近似

### 4.1 BCH公式

Baker-Campbell-Hausdorff公式给出了两个指数映射的复合结果：

$$ \ln(\exp(\mathbf{A}) \exp(\mathbf{B})) = \mathbf{A} + \mathbf{B} + \frac{1}{2}[\mathbf{A}, \mathbf{B}] + \frac{1}{12}[\mathbf{A}, [\mathbf{A}, \mathbf{B}]] - \frac{1}{12}[\mathbf{B}, [\mathbf{A}, \mathbf{B}]] + \cdots $$

### 4.2 BCH线性近似

当 $\boldsymbol{\phi}_1$ 或 $\boldsymbol{\phi}_2$ 为小量时：

$$ \ln(\exp(\boldsymbol{\phi}_1^\wedge) \exp(\boldsymbol{\phi}_2^\wedge))^\vee \approx \begin{cases} \mathbf{J}_l(\boldsymbol{\phi}_2)^{-1} \boldsymbol{\phi}_1 + \boldsymbol{\phi}_2 & \text{当} \boldsymbol{\phi}_1 \text{为小量} \\ \boldsymbol{\phi}_1 + \mathbf{J}_r(\boldsymbol{\phi}_1)^{-1} \boldsymbol{\phi}_2 & \text{当} \boldsymbol{\phi}_2 \text{为小量} \end{cases} $$

其中左雅可比 $\mathbf{J}_l$ 为：

$$ \mathbf{J}_l = \frac{\sin\theta}{\theta} \mathbf{I} + \left(1 - \frac{\sin\theta}{\theta}\right) \mathbf{n} \mathbf{n}^T + \frac{1-\cos\theta}{\theta} \mathbf{n}^\wedge $$

## 5. 伴随表示

### 5.1 伴随矩阵

对于 $\mathbf{T} \in \text{SE}(3)$，其伴随矩阵 $\text{Ad}(\mathbf{T})$ 为：

$$ \text{Ad}(\mathbf{T}) = \begin{bmatrix} \mathbf{R} & \mathbf{t}^\wedge \mathbf{R} \\ \mathbf{0} & \mathbf{R} \end{bmatrix} \in \mathbb{R}^{6 \times 6} $$

### 5.2 伴随性质

$$ \mathbf{T} \exp(\boldsymbol{\xi}^\wedge) \mathbf{T}^{-1} = \exp((\text{Ad}(\mathbf{T}) \boldsymbol{\xi})^\wedge) $$

这一性质在SLAM优化中非常重要，它允许我们在不同坐标系之间变换扰动。

## 6. 在SLAM中的应用

### 6.1 无约束优化

李代数最重要的优势是实现无约束优化。优化变量在流形上的更新使用指数映射：

$$ \mathbf{T} \leftarrow \mathbf{T} \exp(\delta\boldsymbol{\xi}^\wedge) $$

这样，优化过程在 $\mathbb{R}^6$ 中执行，无需额外的约束维护。

### 6.2 位姿图的边误差

在位姿图优化中，两个位姿之间的误差定义为：

$$ \mathbf{e}_{ij} = \ln(\mathbf{T}_{ij}^{-1} \mathbf{T}_i^{-1} \mathbf{T}_j)^\vee $$

其雅可比矩阵可以通过伴随表示高效计算。

## 7. 参考文献

1. Solà, J., Deray, J., & Atchuthan, D. (2018). A micro Lie theory for state estimation in robotics. *arXiv preprint arXiv:1812.01537*.
2. Barfoot, T. D. (2017). *State Estimation for Robotics*. Cambridge University Press.
3. Chirikjian, G. S. (2011). *Stochastic Models, Information Theory, and Lie Groups, Volume 2*. Springer.
4. Murray, R. M., Li, Z., & Sastry, S. S. (1994). *A Mathematical Introduction to Robotic Manipulation*. CRC Press.
