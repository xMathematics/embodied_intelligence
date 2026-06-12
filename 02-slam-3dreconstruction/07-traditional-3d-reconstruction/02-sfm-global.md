# 7.2 全局式SfM

## 1. 概述

全局式SfM（Global SfM）同时估计所有相机的位姿，避免了增量式SfM的误差累积问题。它通过先估计全局旋转再估计全局平移的两步法实现。

## 2. 全局SfM原理

### 2.1 两步法

全局SfM的核心思路：

1. **全局旋转估计**：利用相对旋转估计所有相机的绝对旋转
2. **全局平移估计**：根据旋转估计全局平移和3D点位置

### 2.2 全局旋转估计

**旋转平均（Rotation Averaging）**：

给定相对旋转 $\mathbf{R}_{ij}$，求解绝对旋转 $\mathbf{R}_i$：

$$ \mathbf{R}_{ij} = \mathbf{R}_j \mathbf{R}_i^T $$

优化目标：

$$ \min_{\mathbf{R}_i} \sum_{(i,j) \in \mathcal{E}} \|\ln(\mathbf{R}_{ij}^{-1} \mathbf{R}_j \mathbf{R}_i^T)^\vee\|^2 $$

**求解方法**：
- **L1旋转平均**：鲁棒性更好
- **Weiszfeld算法**：迭代重加权
- **对偶四元数平均**

### 2.3 全局平移估计

**平移平均（Translation Averaging）**：

给定相对平移方向 $\mathbf{t}_{ij}$ 和旋转 $\mathbf{R}_i, \mathbf{R}_j$：

$$ \mathbf{R}_i^T (\mathbf{t}_j - \mathbf{t}_i) = d_{ij} \mathbf{t}_{ij} $$

其中 $d_{ij}$ 是未知的深度比例。

**线性方法**：
- 使用方向约束建立线性系统
- 最小化一致方向偏差

## 3. 全局SfM vs 增量SfM

| 特性 | 增量式 | 全局式 |
|------|--------|--------|
| 精度 | 高（经过BA充分优化） | 中（旋转平均引入误差） |
| 鲁棒性 | 差（误差累积） | 好（全局约束） |
| 计算效率 | 慢（增量BA） | 快（一次性求解） |
| 大规模 | 差 | 好 |
| 实现难度 | 中 | 高 |
| 处理外点 | 好（逐步筛选） | 难（一次性处理） |

## 4. 混合SfM

结合增量式和全局式的优点：
- 使用全局方法获得初始值
- 使用增量方法进行精化

**代表性方法**：
- **Sequential SfM with Global Initialization**
- **Hierarchical SfM**

## 5. 参考文献

1. Hartley, R., Trumpf, J., Dai, Y., & Li, H. (2013). Rotation averaging. *International Journal of Computer Vision*, 103(3), 267-305.
2. Wilson, K., & Snavely, N. (2014). Robust global translations with 1DSfM. *ECCV*.
3. Moulon, P., Monasse, P., & Marlet, R. (2013). Global fusion of relative motions for robust, accurate and scalable structure from motion. *ICCV*.
4. Cui, Z., & Tan, P. (2015). Global structure-from-motion with similarity averaging. *ICCV*.
