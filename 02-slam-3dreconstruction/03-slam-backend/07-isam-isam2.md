# 3.7 iSAM与iSAM2

## 1. 概述

iSAM（Incremental Smoothing and Mapping）系列算法是SLAM后端增量式优化的里程碑工作。与批处理BA不同，iSAM支持在新观测到达时只更新受影响的部分，实现实时大规模SLAM优化。

## 2. iSAM 1.0

### 2.1 核心思想

iSAM将SLAM问题表示为平方根信息矩阵，使用QR分解维护矩阵的三角形式。

**信息矩阵的平方根形式**：

$$ \mathbf{J}^T \mathbf{J} = \mathbf{R}^T \mathbf{R} $$

其中 $\mathbf{R}$ 是上三角矩阵（平方根信息矩阵）。

### 2.2 增量QR分解

当新观测到达时（新行加入 $\mathbf{J}$），使用Givens旋转更新 $\mathbf{R}$：

$$ \begin{bmatrix} \mathbf{R} \\ \mathbf{A}_{\text{new}} \end{bmatrix} \xrightarrow{\text{Givens}} \begin{bmatrix} \mathbf{R}' \\ \mathbf{0} \end{bmatrix} $$

Givens旋转通过逐元素消零实现，只影响 $\mathbf{R}$ 的局部。

### 2.3 流体重新线性化

iSAM的重新线性化策略：
- 新观测：立即线性化
- 旧观测：定期重新线性化（当变量变化超过阈值时）

**重新线性化触发条件**：

$$ \|\mathbf{x}_i - \mathbf{x}_i^{\text{lin}}\| > \text{threshold} $$

### 2.4 局限性

- 重新线性化时需重建整个 $\mathbf{R}$ 矩阵
- 变量消元顺序无法动态调整
- 大规模问题仍有性能瓶颈

## 3. iSAM 2.0

### 3.1 贝叶斯树

iSAM2引入贝叶斯树（Bayes Tree）数据结构，将因子图转化为树形结构。

**从因子图到贝叶斯树**：

```
因子图 → 变量消元 → 贝叶斯网 → 贝叶斯树
```

### 3.2 贝叶斯树的性质

- **根到叶**：条件概率分解方向
- **团（Clique）**：树中的每个节点是一个团，包含一组变量
- **条件概率**：每个团代表条件概率 $P(\mathbf{x}_c \mid \mathbf{S}_c)$
- **分离子（Separator）**：$\mathbf{S}_c = \text{Clique}_c \cap \text{Parent}_c$

### 3.3 增量更新

当新因子加入时：

1. **识别影响区域**：找到贝叶斯树中受影响的团
2. **重建子树**：从影响区域重新构建因子图
3. **变量重排序**：在受影响区域内优化消元顺序
4. **重新构建**：将更新后的子树重新插入贝叶斯树

### 3.4 流体重新线性化

iSAM2只在以下条件触发重新线性化：

$$ \|\boldsymbol{\Delta}_c\| > \alpha \cdot \text{threshold} $$

其中 $\boldsymbol{\Delta}_c$ 是团 $c$ 中变量的更新大小。

### 3.5 优势

- **局部更新**：只更新受影响的部分
- **自适应排序**：每部分可独立优化消元顺序
- **实时性能**：大规模SLAM中的实时优化
- **内存高效**：树结构有更好的局部性

## 4. iSAM vs iSAM2对比

| 特性 | iSAM 1.0 | iSAM 2.0 |
|------|----------|----------|
| 核心数据结构 | 平方根信息矩阵 | 贝叶斯树 |
| 更新策略 | Givens旋转 | 子树重建 |
| 重新线性化 | 全局 | 局部 |
| 变量排序 | 固定 | 自适应 |
| 大规模性能 | 中 | 优 |
| 实现复杂度 | 中 | 高 |

## 5. iSAM2在SLAM中的应用

iSAM2被广泛应用于现代SLAM系统：

- **ORB-SLAM3**：在后端优化中使用类似增量BA策略
- **VINS-Mono**：滑动窗口非线性优化
- **GTSAM**：以iSAM2为核心求解器

## 6. 参考文献

1. Kaess, M., Ranganathan, A., & Dellaert, F. (2008). iSAM: Incremental smoothing and mapping. *IEEE Transactions on Robotics*, 24(6), 1365-1378.
2. Kaess, M., Johannsson, H., Roberts, R., Ila, V., Leonard, J. J., & Dellaert, F. (2012). iSAM2: Incremental smoothing and mapping using the Bayes tree. *The International Journal of Robotics Research*, 31(2), 216-235.
3. Dellaert, F., & Kaess, M. (2017). Factor graphs for robot perception. *Foundations and Trends in Robotics*, 6(1-2), 1-139.
