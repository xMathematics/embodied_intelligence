# 3.6 位姿图优化

## 1. 概述

位姿图优化（Pose Graph Optimization, PGO）是SLAM后端的重要方法，在特征点被边缘化后只保留位姿变量。PGO特别适用于大规模SLAM场景。

## 2. 位姿图模型

### 2.1 节点与边

- **节点**：机器人在各时刻的位姿 $\mathbf{T}_i \in \text{SE}(3)$
- **边**：位姿之间的约束（里程计约束、回环约束）

### 2.2 误差函数

两帧之间的相对位姿误差：

$$ \mathbf{e}_{ij} = \ln(\mathbf{T}_{ij}^{-1} \mathbf{T}_i^{-1} \mathbf{T}_j)^\vee \in \mathbb{R}^6 $$

其中 $\mathbf{T}_{ij}$ 是第 $i$ 帧到第 $j$ 帧的相对位姿测量。

### 2.3 目标函数

$$ \mathbf{T}^* = \arg\min_{\mathbf{T}} \sum_{(i,j) \in \mathcal{E}} \|\mathbf{e}_{ij}\|^2_{\mathbf{\Sigma}_{ij}} $$

## 3. 流形上的优化

### 3.1 SE(3)上的更新

在位姿图优化中，更新在流形上执行：

$$ \mathbf{T}_i \leftarrow \mathbf{T}_i \exp(\delta\boldsymbol{\xi}_i^\wedge) $$

### 3.2 线性化

误差在 $\mathbf{T}$ 处线性化：

$$ \mathbf{e}_{ij}(\mathbf{T} + \delta\boldsymbol{\xi}) \approx \mathbf{e}_{ij}(\mathbf{T}) + \mathbf{J}_{ij} \delta\boldsymbol{\xi} $$

其中 $\mathbf{J}_{ij}$ 是误差对位姿扰动的雅可比矩阵。

## 4. 鲁棒PGO

### 4.1 鲁棒核函数

$$ \mathbf{T}^* = \arg\min_{\mathbf{T}} \sum_{(i,j) \in \mathcal{E}} \rho(\|\mathbf{e}_{ij}\|_{\mathbf{\Sigma}_{ij}}) $$

### 4.2 Dynamic Covariance Scaling

DCS方法自适应调整回环边的协方差：

$$ \tilde{\mathbf{\Sigma}}_{ij} = \mathbf{\Sigma}_{ij} \cdot \max\left(1, \frac{\|\mathbf{e}_{ij}\|^2}{c^2}\right) $$

### 4.3 Switchable Constraints

引入开关变量 $s_{ij} \in [0,1]$ 控制每条边的权重：

$$ \mathbf{T}^* = \arg\min \sum_{(i,j)} s_{ij} \|\mathbf{e}_{ij}\|^2 + \Psi(s_{ij}) $$

## 5. 位姿图初始化

好的初始值对PGO收敛至关重要：

- **顺序初始化**：按时间顺序累积里程计
- **回环的初始猜测**：使用回环检测的相对位姿
- **分层优化**：先粗后细

## 6. 大规模PGO

| 算法 | 特点 | 复杂度 |
|------|------|--------|
| Gauss-Newton | 标准方法 | O(N³) |
| LM | 阻尼策略 | O(N³) |
| Toro | 树结构优化 | O(N log N) |
| HOG-Man | 层次优化 | O(N) |
| SE-Sync | 凸松弛 | O(N³) |

## 7. 参考文献

1. Grisetti, G., Kümmerle, R., Stachniss, C., & Burgard, W. (2010). A tutorial on graph-based SLAM. *IEEE Intelligent Transportation Systems Magazine*, 2(4), 31-43.
2. Dellaert, F., & Kaess, M. (2006). Square root SAM. *IJRR*, 25(12), 1181-1203.
3. Sünderhauf, N., & Protzel, P. (2012). Switchable constraints for robust pose graph SLAM. *IROS*.
4. Rosen, D. M., Carlone, L., Bandeira, A. S., & Leonard, J. J. (2017). SE-Sync: A certifiably correct algorithm for synchronization over the special Euclidean group. *The International Journal of Robotics Research*, 38(2-3), 95-125.
