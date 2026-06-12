# 2.5 运动估计

## 1. 概述

运动估计是视觉里程计的核心任务，根据特征匹配对估计相机在两帧之间的相对运动。本章详细介绍从两视图运动估计到多视图运动估计的各种方法。

## 2. 两视图运动估计

### 2.1 八点法（8-Point Algorithm）

八点法（Longuet-Higgins, 1981）使用8对匹配点估计基础矩阵。

**基本方程**：每对点 $(\mathbf{x}_1, \mathbf{x}_2)$ 提供一个线性约束：

$$ \mathbf{x}_2^T \mathbf{F} \mathbf{x}_1 = 0 $$

展开为：

$$ \begin{bmatrix} u_2 u_1 & u_2 v_1 & u_2 & v_2 u_1 & v_2 v_1 & v_2 & u_1 & v_1 & 1 \end{bmatrix} \mathbf{f} = 0 $$

其中 $\mathbf{f}$ 是 $\mathbf{F}$ 的向量化形式。

**归一化八点法**（Hartley, 1997）：
1. 对图像坐标进行归一化变换：$\tilde{\mathbf{x}}_i = \mathbf{T}_i \mathbf{x}_i$
2. 对归一化坐标使用八点法
3. 恢复：$\mathbf{F} = \mathbf{T}_2^T \mathbf{F}_{\text{norm}} \mathbf{T}_1$

归一化显著提高了算法的数值稳定性。

### 2.2 五点法（5-Point Algorithm）

五点法（Nistér, 2004）使用5对匹配点估计本质矩阵，利用 $\mathbf{E}$ 的5个自由度约束。

**约束条件**：
1. **极线约束**：$\mathbf{x}_2^T \mathbf{E} \mathbf{x}_1 = 0$（每个匹配提供1个方程）
2. **奇异值约束**：$\det(\mathbf{E}) = 0$（1个方程）
3. **Trace约束**：$\mathbf{E} \mathbf{E}^T \mathbf{E} - \frac{1}{2} \text{tr}(\mathbf{E} \mathbf{E}^T) \mathbf{E} = 0$（2个方程）

**求解方法**：使用Gröbner基将问题转化为多项式方程组求解，最多产生10个解。

### 2.3 七点法（7-Point Algorithm）

七点法使用7对点估计基础矩阵，利用$\mathbf{F}$的7个自由度。得到1维零空间：

$$ \mathbf{F} = \alpha \mathbf{F}_1 + (1-\alpha) \mathbf{F}_2 $$

通过 $\det(\mathbf{F}) = 0$ 求解 $\alpha$，最多3个实根。

### 2.4 PnP（Perspective-n-Point）

当已知3D点及其2D投影时，使用PnP算法估计相机位姿。

**P3P**：使用3对3D-2D对应点得到闭合形式解，最多4个解。
**EPnP**（Lepetit et al., 2009）：将3D点表示为4个控制点的加权和，$O(n)$ 复杂度。
**UPnP**（Kneip et al., 2014）：无需已知相机内参。
**DLS**（Hesch & Roumeliotis, 2011）：直接最小二乘PnP。

## 3. 三角化

### 3.1 线性三角化

给定两幅图像中的对应点和相机位姿，求解3D点坐标：

$$ \mathbf{x}_1 \times (\mathbf{P}_1 \mathbf{X}) = 0 $$
$$ \mathbf{x}_2 \times (\mathbf{P}_2 \mathbf{X}) = 0 $$

构造 $\mathbf{A} \mathbf{X} = \mathbf{0}$ 的线性系统，通过SVD求解。

### 3.2 最优三角化

Hartley & Sturm提出在矫正坐标系中进行最优三角化，最小化几何重投影误差。

### 3.3 三角化不确定性

三角化点的精度受基线长度和观测角度影响：

$$ \sigma_Z \propto \frac{Z^2}{fb} \sigma_d $$

其中 $Z$ 是深度，$f$ 是焦距，$b$ 是基线，$\sigma_d$ 是匹配误差。

## 4. PnP算法详解

### 4.1 EPnP改进算法

EPnP将世界坐标系中的3D点表示为4个虚拟控制点 $\mathbf{c}_j$ 的加权和：

$$ \mathbf{p}_i^w = \sum_{j=1}^{4} \alpha_{ij} \mathbf{c}_j^w, \quad \sum_{j=1}^{4} \alpha_{ij} = 1 $$

相机坐标系下的表示为：

$$ \mathbf{p}_i^c = \mathbf{R} \mathbf{p}_i^w + \mathbf{t} = \sum_{j=1}^{4} \alpha_{ij} \mathbf{c}_j^c $$

通过投影约束建立线性方程求解 $\mathbf{c}_j^c$，再恢复 $\mathbf{R}, \mathbf{t}$。

### 4.2 PnP-RANSAC

结合RANSAC的PnP算法：
1. 随机选择4对3D-2D匹配
2. 使用EPnP估计位姿
3. 计算所有匹配的重投影误差
4. 统计内点
5. 重复迭代，选择内点最多的解

## 5. 相对位姿估计对比

| 方法 | 最少点数 | 是否需内参 | 适用场景 |
|------|---------|-----------|----------|
| 八点法 | 8 | 不需要(F) | 初始位姿估计 |
| 五点法 | 5 | 需要(E) | 已标定相机 |
| 七点法 | 7 | 不需要(F) | 未标定相机 |
| P3P | 3 | 需要 | 已知3D点 |
| EPnP | 4 | 需要 | 已知3D点 (n≥4) |
| PnP+BA | 4 | 需要 | 高精度估计 |

## 6. 尺度问题

单目视觉存在**尺度模糊性**：
- 平移 $\mathbf{t}$ 和3D点坐标 $\mathbf{X}$ 同时乘以任意常数 $\lambda$ 不影响投影
- 需要额外信息（IMU、已知物体尺寸、双目基线）确定尺度

## 7. 参考文献

1. Hartley, R. (1997). In defense of the eight-point algorithm. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 19(6), 580-593.
2. Nistér, D. (2004). An efficient solution to the five-point relative pose problem. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 26(6), 756-770.
3. Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009). EPnP: An accurate O(n) solution to the PnP problem. *International Journal of Computer Vision*, 81(2), 155-166.
4. Kneip, L., Li, H., & Seo, Y. (2014). UPnP: An optimal O(n) solution to the absolute pose problem with universal applicability. *ECCV*.
