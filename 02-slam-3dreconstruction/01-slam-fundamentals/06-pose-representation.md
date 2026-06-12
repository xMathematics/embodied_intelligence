# 1.6 位姿表示

## 1. 概述

位姿（Pose）是描述物体在空间中位置和朝向的数学概念。在SLAM中，正确理解和选择位姿表示方法直接影响系统的精度、效率和鲁棒性。本章将详细介绍各种位姿表示方法及其在SLAM中的应用。

## 2. 旋转矩阵

### 2.1 定义

旋转矩阵 $\mathbf{R} \in \text{SO}(3)$ 是一个 $3 \times 3$ 的正交矩阵，满足：

$$ \mathbf{R}^T \mathbf{R} = \mathbf{I}, \quad \det(\mathbf{R}) = +1 $$

### 2.2 特殊正交群

$\text{SO}(3)$ 构成一个李群（Lie Group），具有以下性质：
- **封闭性**：$\mathbf{R}_1 \mathbf{R}_2 \in \text{SO}(3)$
- **结合律**：$(\mathbf{R}_1 \mathbf{R}_2) \mathbf{R}_3 = \mathbf{R}_1 (\mathbf{R}_2 \mathbf{R}_3)$
- **单位元**：$\mathbf{I}$
- **逆元**：$\mathbf{R}^{-1} = \mathbf{R}^T$

## 3. 变换矩阵

### 3.1 齐次变换矩阵

$\mathbf{T} \in \text{SE}(3)$ 是一个 $4 \times 4$ 的齐次变换矩阵：

$$ \mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} $$

### 3.2 逆变换

$$ \mathbf{T}^{-1} = \begin{bmatrix} \mathbf{R}^T & -\mathbf{R}^T \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix} $$

### 3.3 变换复合

$$ \mathbf{T}_{13} = \mathbf{T}_{12} \mathbf{T}_{23} $$

## 4. 欧拉角

### 4.1 定义

欧拉角将旋转分解为三个绕坐标轴的连续旋转。常见的顺序有：

- **ZYX（偏航-俯仰-横滚）**：
$$\mathbf{R} = \mathbf{R}_z(\psi) \mathbf{R}_y(\theta) \mathbf{R}_x(\phi)$$

### 4.2 万向锁（Gimbal Lock）

当 $\theta = \pm 90^\circ$ 时，第一轴和第三轴的旋转轴重合，丧失一个自由度。

### 4.3 优缺点

| 优点 | 缺点 |
|------|------|
| 直观易理解 | 存在万向锁问题 |
| 仅需3个参数 | 插值不自然 |
| 适合人机交互 | 不适合优化 |

## 5. 四元数

### 5.1 定义

四元数 $\mathbf{q} = q_w + q_x i + q_y j + q_z k$ 满足 $i^2 = j^2 = k^2 = ijk = -1$。

用向量表示：$\mathbf{q} = [q_w, q_x, q_y, q_z]^T$，满足 $q_w^2 + q_x^2 + q_y^2 + q_z^2 = 1$。

### 5.2 四元数乘法

$$ \mathbf{q}_1 \otimes \mathbf{q}_2 = \begin{bmatrix} q_{w1}q_{w2} - q_{x1}q_{x2} - q_{y1}q_{y2} - q_{z1}q_{z2} \\ q_{w1}q_{x2} + q_{x1}q_{w2} + q_{y1}q_{z2} - q_{z1}q_{y2} \\ q_{w1}q_{y2} - q_{x1}q_{z2} + q_{y1}q_{w2} + q_{z1}q_{x2} \\ q_{w1}q_{z2} + q_{x1}q_{y2} - q_{y1}q_{x2} + q_{z1}q_{w2} \end{bmatrix} $$

### 5.3 四元数与旋转矩阵的转换

将旋转矩阵转换为四元数：

$$ q_w = \frac{\sqrt{1 + R_{11} + R_{22} + R_{33}}}{2} $$
$$ q_x = \frac{R_{32} - R_{23}}{4 q_w} $$
$$ q_y = \frac{R_{13} - R_{31}}{4 q_w} $$
$$ q_z = \frac{R_{21} - R_{12}}{4 q_w} $$

将四元数转换为旋转矩阵：

$$ \mathbf{R} = \begin{bmatrix} 1-2(q_y^2+q_z^2) & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\ 2(q_xq_y + q_wq_z) & 1-2(q_x^2+q_z^2) & 2(q_yq_z - q_wq_x) \\ 2(q_xq_z - q_wq_y) & 2(q_yq_z + q_wq_x) & 1-2(q_x^2+q_y^2) \end{bmatrix} $$

### 5.4 四元数插值（SLERP）

$$ \text{SLERP}(\mathbf{q}_1, \mathbf{q}_2, t) = \frac{\sin[(1-t)\theta]}{\sin\theta} \mathbf{q}_1 + \frac{\sin(t\theta)}{\sin\theta} \mathbf{q}_2 $$

其中 $\theta = \arccos(\mathbf{q}_1 \cdot \mathbf{q}_2)$。

## 6. 轴角表示

### 6.1 定义

绕单位轴 $\mathbf{n}$ 旋转角度 $\theta$：

$$ \mathbf{R} = \cos\theta \mathbf{I} + \sin\theta [\mathbf{n}]_\times + (1-\cos\theta) \mathbf{n} \mathbf{n}^T $$

这即罗德里格斯公式（Rodrigues' Formula）。

### 6.2 指数映射

$$ \mathbf{R} = \exp([\boldsymbol{\omega}]_\times) = \mathbf{I} + \frac{\sin\theta}{\theta} [\boldsymbol{\omega}]_\times + \frac{1-\cos\theta}{\theta^2} [\boldsymbol{\omega}]_\times^2 $$

其中 $\boldsymbol{\omega} = \theta \mathbf{n}$，$\theta = \|\boldsymbol{\omega}\|$。

## 7. 各种表示方法的对比

| 方法 | 自由度 | 约束 | 奇异性 | 插值 | 计算效率 | 优化友好 |
|------|--------|------|--------|------|----------|----------|
| 旋转矩阵 | 9 | 6个约束 | 无 | 困难 | 高 | 否 |
| 欧拉角 | 3 | 无 | 万向锁 | 不平滑 | 高 | 否 |
| 四元数 | 4 | 单位范数 | 无 | 可SLERP | 高 | 是 |
| 轴角 | 3 | 无 | $\theta=0$ | 复杂 | 中 | 是 |
| 李代数 | 3 | 无 | 无 | 复杂 | 中 | 最佳 |

## 8. 在SLAM中的应用

SLAM系统通常使用多种表示方法：

- **优化过程中**：使用李代数（$\mathfrak{so}(3)$）的无约束表示
- **状态存储**：使用四元数 + 平移向量
- **可视化**：使用欧拉角
- **变换计算**：使用齐次变换矩阵

## 9. 参考文献

1. Shuster, M. D. (1993). A survey of attitude representations. *Journal of the Astronautical Sciences*, 41(4), 439-517.
2. Shoemake, K. (1985). Animating rotation with quaternion curves. *ACM SIGGRAPH*.
3. Solà, J., Deray, J., & Atchuthan, D. (2018). A micro Lie theory for state estimation in robotics. *arXiv preprint arXiv:1812.01537*.
4. Barfoot, T. D. (2017). *State Estimation for Robotics*. Cambridge University Press.
