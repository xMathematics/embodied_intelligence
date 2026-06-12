# 3.5 因子图优化与图SLAM

## 1. 概述

因子图（Factor Graph）是一种概率图模型，它将SLAM问题分解为变量节点和因子节点的二部图。因子图优化是现代SLAM系统最主流的后端优化方法。

## 2. 因子图基础

### 2.1 定义

因子图是一个二部图 $\mathcal{G} = (\mathcal{X}, \mathcal{F}, \mathcal{E})$，其中：
- $\mathcal{X}$：变量节点（Variable Nodes），表示待估计的状态
- $\mathcal{F}$：因子节点（Factor Nodes），表示概率约束
- $\mathcal{E}$：边，连接因子和其涉及的变量

### 2.2 因子图分解

联合概率分布可以分解为因子的乘积：

$$ P(\mathbf{X} \mid \mathbf{Z}) \propto \prod_i f_i(\mathbf{X}_i) $$

其中 $\mathbf{X}_i$ 是因子 $f_i$ 涉及的变量子集。

### 2.3 SLAM中的因子类型

**先验因子**（Prior Factor）：
$$ f_{\text{prior}}(\mathbf{x}_0) = \exp\left(-\frac{1}{2}\|\mathbf{x}_0 - \boldsymbol{\mu}_0\|^2_{\mathbf{\Sigma}_0}\right) $$

**运动因子**（Motion/Odometry Factor）：
$$ f_{\text{odo}}(\mathbf{x}_k, \mathbf{x}_{k-1}) = \exp\left(-\frac{1}{2}\|g(\mathbf{x}_{k-1}, \mathbf{u}_k) - \mathbf{x}_k\|^2_{\mathbf{\Sigma}_u}\right) $$

**观测因子**（Measurement Factor）：
$$ f_{\text{obs}}(\mathbf{x}_k, \mathbf{m}_j) = \exp\left(-\frac{1}{2}\|h(\mathbf{x}_k, \mathbf{m}_j) - \mathbf{z}_{kj}\|^2_{\mathbf{\Sigma}_z}\right) $$

**回环因子**（Loop Closure Factor）：
$$ f_{\text{loop}}(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{1}{2}\|\ln(\mathbf{T}_{ij}^{-1} \mathbf{T}_i^{-1} \mathbf{T}_j)^\vee\|^2_{\mathbf{\Sigma}_{ij}}\right) \]

## 3. MAP估计与最小二乘

在因子图框架中，MAP估计等价于非线性最小二乘：

$$ \mathbf{X}^* = \arg\min_{\mathbf{X}} \sum_i \|\mathbf{e}_i(\mathbf{X}_i)\|^2_{\mathbf{\Sigma}_i} $$

其中 $\mathbf{e}_i = h_i(\mathbf{X}_i) - \mathbf{z}_i$ 是第 $i$ 个因子的残差。

## 4. 变量消元

### 4.1 消元过程

通过逐步消元变量，将因子图转换为贝叶斯网：

$$ P(\mathbf{X}) \propto \prod f_i(\mathbf{X}_i) $$

消元变量 $\mathbf{x}_1$：

$$ P(\mathbf{x}_1 \mid \mathbf{x}_2, \mathbf{x}_3) \propto \prod f_i(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3) $$

### 4.2 消元顺序

消元顺序对因子图优化的效率至关重要：
- COLAMD（Column Approximate Minimum Degree）
- METIS（基于图划分的排序）
- 自然顺序（时间顺序）

好的消元顺序可以最小化填充（fill-in）。

## 5. 因子图优化算法

### 5.1 高斯-牛顿法

在因子图上应用高斯-牛顿法：

$$ (\mathbf{J}^T \mathbf{\Sigma}^{-1} \mathbf{J}) \Delta \mathbf{X} = -\mathbf{J}^T \mathbf{\Sigma}^{-1} \mathbf{e} $$

### 5.2 增量式求解

在SLAM中，新的约束不断加入，增量求解避免重复计算所有历史数据。

## 6. 因子图与位姿图的关系

- **因子图**：包含所有变量（位姿和特征点）
- **位姿图**：特征点已被边缘化，只包含位姿变量

因子图 → 边缘化特征点 → 位姿图

## 7. 主要优化库

| 库 | 因子图支持 | 增量求解 | 特点 |
|---|-----------|---------|------|
| GTSAM | 是 | iSAM2 | 最完整的因子图库 |
| g2o | 部分 | 否 | 易于使用 |
| Ceres | 否(通用) | 否 | 灵活通用 |
| Vertigo | 是 | 否 | 鲁棒优化 |

## 8. 参考文献

1. Kschischang, F. R., Frey, B. J., & Loeliger, H.-A. (2001). Factor graphs and the sum-product algorithm. *IEEE Transactions on Information Theory*, 47(2), 498-519.
2. Dellaert, F., & Kaess, M. (2006). Square root SAM: Simultaneous localization and mapping via square root information smoothing. *The International Journal of Robotics Research*, 25(12), 1181-1203.
3. Dellaert, F. (2012). Factor graphs and GTSAM: A hands-on introduction. *Georgia Tech Technical Report*.
4. Kaess, M., Ranganathan, A., & Dellaert, F. (2008). iSAM: Incremental smoothing and mapping. *IEEE Transactions on Robotics*, 24(6), 1365-1378.
