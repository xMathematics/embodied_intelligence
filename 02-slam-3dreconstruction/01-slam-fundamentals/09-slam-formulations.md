# 1.9 SLAM问题形式化

## 1. 概述

SLAM问题有多种形式化方法，不同的形式化对应不同的求解策略。本章将系统介绍SLAM问题的各种数学形式化，从概率视角到优化视角，从滤波方法到图优化方法。

## 2. 概率形式化

### 2.1 全SLAM（Full SLAM）

全SLAM估计所有时刻的位姿和完整地图的后验分布：

$$ P(\mathbf{x}_{1:K}, \mathbf{m} \mid \mathbf{z}_{1:K}, \mathbf{u}_{1:K}) $$

利用条件独立性进行分解：

$$ P(\mathbf{x}_{1:K}, \mathbf{m} \mid \mathbf{z}_{1:K}, \mathbf{u}_{1:K}) \propto P(\mathbf{x}_0) \prod_{k=1}^{K} P(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \mathbf{u}_k) \prod_{k=1}^{K} \prod_{j} P(\mathbf{z}_{kj} \mid \mathbf{x}_k, \mathbf{m}_j) $$

### 2.2 在线SLAM（Online SLAM）

在线SLAM只估计当前位姿和地图：

$$ P(\mathbf{x}_k, \mathbf{m} \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k}) $$

通过贝叶斯滤波递归求解。

### 2.3 基于滤波的SLAM

**EKF-SLAM** 将位姿和所有特征点联合建模为高斯分布：

$$ \begin{bmatrix} \mathbf{x}_k \\ \mathbf{m} \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \hat{\mathbf{x}}_k \\ \hat{\mathbf{m}} \end{bmatrix}, \begin{bmatrix} \mathbf{\Sigma}_{xx} & \mathbf{\Sigma}_{xm} \\ \mathbf{\Sigma}_{mx} & \mathbf{\Sigma}_{mm} \end{bmatrix} \right) $$

**RBPF-SLAM**（Rao-Blackwellized Particle Filter SLAM，即FastSLAM）利用条件独立性：

$$ P(\mathbf{x}_{1:k}, \mathbf{m} \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k}) = P(\mathbf{m} \mid \mathbf{x}_{1:k}, \mathbf{z}_{1:k}) \cdot P(\mathbf{x}_{1:k} \mid \mathbf{z}_{1:k}, \mathbf{u}_{1:k}) $$

即：位姿使用粒子滤波估计，地图使用条件高斯估计（每个粒子有自己的地图）。

## 3. 基于图优化的形式化

### 3.1 图SLAM（Graph SLAM）

图SLAM将SLAM问题建模为图优化问题：

- **节点（Nodes）**：机器人的位姿和环境特征
- **边（Edges）**：观测约束和运动约束

优化目标（MAP估计）：

$$ \mathbf{X}^*, \mathbf{M}^* = \arg\min_{\mathbf{X}, \mathbf{M}} \left[ \sum_k \|\mathbf{e}_{u,k}\|^2_{\mathbf{\Sigma}_u} + \sum_{k,j} \|\mathbf{e}_{z,kj}\|^2_{\mathbf{\Sigma}_z} \right] $$

其中：

$$ \mathbf{e}_{u,k} = \mathbf{x}_k - f(\mathbf{x}_{k-1}, \mathbf{u}_k) $$
$$ \mathbf{e}_{z,kj} = \mathbf{z}_{kj} - h(\mathbf{x}_k, \mathbf{m}_j) $$

### 3.2 位姿图优化（Pose Graph Optimization）

当特征点被边缘化后，SLAM退化为位姿图优化：

$$ \mathbf{X}^* = \arg\min_{\mathbf{X}} \sum_{(i,j) \in \mathcal{E}} \|\ln(\mathbf{T}_{ij}^{-1} \mathbf{T}_i^{-1} \mathbf{T}_j)^\vee\|^2_{\mathbf{\Sigma}_{ij}} $$

### 3.3 因子图（Factor Graph）

因子图是图SLAM的更一般形式。它将联合概率分布分解为因子乘积：

$$ P(\mathbf{X} \mid \mathbf{Z}) \propto \prod_i f_i(\mathbf{X}_i) $$

其中每个因子 $f_i$ 对应一个约束，$\mathbf{X}_i$ 是因子涉及的变量子集。

## 4. 最大似然形式化

MLE-SLAM寻找最有可能解释观测数据的变量值：

$$ \{\mathbf{X}^*, \mathbf{M}^*\} = \arg\max_{\mathbf{X}, \mathbf{M}} P(\mathbf{Z} \mid \mathbf{X}, \mathbf{M}) $$

在高斯噪声假设下，这等价于最小二乘问题。

## 5. 不同形式化的联系

### 5.1 滤波 vs 优化

| 方面 | 滤波方法 | 优化方法 |
|------|----------|----------|
| 时间范围 | 递归（只保留当前状态） | 批处理（优化所有历史） |
| 计算复杂度 | O(N²) | O(N log N) |
| 重新线性化 | 仅一次 | 多次迭代 |
| 精度 | 较低（线性化误差累积） | 较高 |
| 大规模场景 | 不适合 | 适合 |
| 实时性 | 好 | 需要优化策略 |

### 5.2 一致性问题

滤波方法（特别是EKF-SLAM）存在一致性问题，因为：
1. 线性化在**当前估计**处进行
2. 但真实值偏离当前估计
3. 线性化误差导致协方差矩阵被低估

图优化方法没有这个问题，因为它在每次迭代中重新线性化。

### 5.3 从滤波到优化的演进

SLAM方法的发展可以看作从滤波到优化的演进：

```
EKF-SLAM → FastSLAM → GraphSLAM → iSAM → iSAM2 → ORB-SLAM
 滤波          混合      全优化     增量      贝叶斯树   实时优化
```

## 6. 参考文献

1. Thrun, S., & Montemerlo, M. (2006). The graph SLAM algorithm with applications to large-scale mapping of urban structures. *The International Journal of Robotics Research*, 25(5-6), 403-429.
2. Dellaert, F., & Kaess, M. (2006). Square root SAM: Simultaneous localization and mapping via square root information smoothing. *The International Journal of Robotics Research*, 25(12), 1181-1203.
3. Kaess, M., Johannsson, H., Roberts, R., Ila, V., Leonard, J. J., & Dellaert, F. (2012). iSAM2: Incremental smoothing and mapping using the Bayes tree. *The International Journal of Robotics Research*, 31(2), 216-235.
4. Grisetti, G., Kümmerle, R., Stachniss, C., & Burgard, W. (2010). A tutorial on graph-based SLAM. *IEEE Intelligent Transportation Systems Magazine*, 2(4), 31-43.
