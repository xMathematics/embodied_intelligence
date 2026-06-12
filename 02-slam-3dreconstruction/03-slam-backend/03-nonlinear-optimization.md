# 3.3 非线性优化基础

## 1. 概述

非线性最小二乘优化是现代SLAM后端优化的核心工具。本章详细介绍SLAM中使用的各种优化方法及其数学原理。

## 2. 最小二乘问题

### 2.1 问题定义

$$ \mathbf{x}^* = \arg\min_{\mathbf{x}} \frac{1}{2} \sum_{i=1}^{m} \|\mathbf{e}_i(\mathbf{x})\|^2 = \arg\min_{\mathbf{x}} \frac{1}{2} \|\mathbf{e}(\mathbf{x})\|^2 $$

其中 $\mathbf{e}_i(\mathbf{x})$ 是第 $i$ 个残差。

### 2.2 线性最小二乘

$$ \mathbf{x}^* = \arg\min_{\mathbf{x}} \|\mathbf{A} \mathbf{x} - \mathbf{b}\|^2 $$

**正规方程**：$\mathbf{A}^T \mathbf{A} \mathbf{x} = \mathbf{A}^T \mathbf{b}$

**SVD求解**：$\mathbf{x}^* = \mathbf{V} \mathbf{\Sigma}^{-1} \mathbf{U}^T \mathbf{b}$

## 3. 非线性最小二乘方法

### 3.1 高斯-牛顿法

在 $\mathbf{x}^t$ 处对残差线性化：

$$ \mathbf{e}(\mathbf{x}^t + \Delta \mathbf{x}) \approx \mathbf{e}(\mathbf{x}^t) + \mathbf{J} \Delta \mathbf{x} $$

代入目标函数：

$$ \frac{1}{2} \|\mathbf{e}(\mathbf{x}^t) + \mathbf{J} \Delta \mathbf{x}\|^2 = \frac{1}{2} \|\mathbf{e}\|^2 + \Delta \mathbf{x}^T \mathbf{J}^T \mathbf{e} + \frac{1}{2} \Delta \mathbf{x}^T \mathbf{J}^T \mathbf{J} \Delta \mathbf{x} $$

对 $\Delta \mathbf{x}$ 求导并令为0：

$$ (\mathbf{J}^T \mathbf{J}) \Delta \mathbf{x} = -\mathbf{J}^T \mathbf{e} $$

这就是**高斯-牛顿方程**。

### 3.2 Levenberg-Marquardt方法

L-M方法在高斯-牛顿中加入阻尼项，在梯度下降和高斯-牛顿之间自适应：

$$ (\mathbf{J}^T \mathbf{J} + \lambda \mathbf{D}) \Delta \mathbf{x} = -\mathbf{J}^T \mathbf{e} $$

其中 $\lambda \geq 0$ 是阻尼参数，$\mathbf{D}$ 通常取 $\text{diag}(\mathbf{J}^T \mathbf{J})$。

**阻尼调整策略**：

$$ \rho = \frac{\|\mathbf{e}(\mathbf{x})\|^2 - \|\mathbf{e}(\mathbf{x} + \Delta \mathbf{x})\|^2}{\|\mathbf{e}(\mathbf{x})\|^2 - \|\mathbf{e}(\mathbf{x}) + \mathbf{J} \Delta \mathbf{x}\|^2} $$

- $\rho > 0$：实际下降充分，减小 $\lambda$（更接近高斯-牛顿）
- $\rho \leq 0$：下降不足，增大 $\lambda$（更接近梯度下降）

### 3.3 Dog-Leg方法

Dog-Leg在梯度下降方向（Cauchy点）和高斯-牛顿方向之间插值，受信赖域约束：

$$ \Delta \mathbf{x}_{dl} = \begin{cases} -\frac{\tau}{\|\mathbf{g}\|} \mathbf{g} & \tau \leq \tau_1 \\ \mathbf{x}_{cp} + (\beta - 1)(\mathbf{x}_{gn} - \mathbf{x}_{cp}) & \text{otherwise} \end{cases} $$

## 4. 收敛性分析

### 4.1 收敛定理

高斯-牛顿法在以下条件下具有二次收敛性：
- 初始估计足够接近最优解
- 残差在最优解附近足够小

### 4.2 收敛速度

| 方法 | 收敛阶 | 每次迭代计算量 |
|------|--------|---------------|
| 梯度下降 | 线性 | O(N) |
| 高斯-牛顿 | 超线性-二次 | O(N³) |
| L-M | 超线性 | O(N³) |

## 5. 鲁棒优化

### 5.1 鲁棒核函数

使用M-estimator降低外点影响：

$$ \mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_i \rho(\|\mathbf{e}_i(\mathbf{x})\|) $$

### 5.2 等价加权

M-estimator可以通过迭代重加权最小二乘（IRLS）实现：

$$ \mathbf{x}^{t+1} = \arg\min_{\mathbf{x}} \sum_i w_i^t \|\mathbf{e}_i(\mathbf{x})\|^2 $$

其中 $w_i = \frac{\rho'(e_i)}{e_i}$。

## 6. 数值线性代数

### 6.1 Cholesky分解

对于对称正定矩阵 $\mathbf{A}$：

$$ \mathbf{A} = \mathbf{L} \mathbf{L}^T $$

求解 $\mathbf{A} \mathbf{x} = \mathbf{b}$ 分两步：
1. $\mathbf{L} \mathbf{y} = \mathbf{b}$（前向替换）
2. $\mathbf{L}^T \mathbf{x} = \mathbf{y}$（回代）

### 6.2 QR分解

$$ \mathbf{A} = \mathbf{Q} \mathbf{R} $$

求解最小二乘：$\mathbf{x} = \mathbf{R}^{-1} \mathbf{Q}^T \mathbf{b}$

数值上比正规方程更稳定。

## 7. 参考文献

1. Nocedal, J., & Wright, S. (2006). *Numerical Optimization* (2nd ed.). Springer.
2. Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). Bundle adjustment—a modern synthesis. *Vision Algorithms: Theory and Practice*.
3. Lourakis, M. I., & Argyros, A. A. (2005). Is Levenberg-Marquardt the most efficient optimization algorithm for implementing bundle adjustment? *ICCV*.
4. Madsen, K., Nielsen, H. B., & Tingleff, O. (2004). *Methods for Non-Linear Least Squares Problems* (2nd ed.). IMM, DTU.
