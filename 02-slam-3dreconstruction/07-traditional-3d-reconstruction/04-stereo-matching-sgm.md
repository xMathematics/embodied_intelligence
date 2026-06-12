# 7.4 立体匹配与SGM

## 1. 概述

立体匹配（Stereo Matching）是在双目图像中寻找对应点的过程，是双目深度估计的核心。SGM（Semi-Global Matching）是最经典且实用的立体匹配算法。

## 2. 立体匹配问题

### 2.1 问题定义

对于左右图像 $I_L$ 和 $I_R$，找到每个像素 $p$ 的视差 $d_p$：

$$ d_p = \arg\min_{d} C(p, d) $$

### 2.2 匹配代价

**绝对差（SAD）**：
$$ C_{\text{SAD}}(p, d) = \sum_{q \in W_p} |I_L(q) - I_R(q - d)| $$

**零均值归一化互相关（ZNCC）**：
$$ C_{\text{ZNCC}}(p, d) = \frac{\sum (I_L - \bar{I}_L)(I_R - \bar{I}_R)}{\sqrt{\sum (I_L - \bar{I}_L)^2 \sum (I_R - \bar{I}_R)^2}} $$

**Census变换**：
$$ C_{\text{Census}}(p, d) = \text{Ham}\left( \text{BitString}_L(p), \text{BitString}_R(p-d) \right) $$

## 3. SGM算法

### 3.1 能量函数

SGM最小化全局能量函数：

$$ E(D) = \sum_p C(p, D_p) + \sum_q P_1 \cdot T[|D_p - D_q| = 1] + \sum_q P_2 \cdot T[|D_p - D_q| > 1] $$

### 3.2 代价聚合

SGM通过多个方向的DP路径聚合代价：

$$ L_r(p, d) = C(p, d) + \min \begin{cases} L_r(p-r, d) \\ L_r(p-r, d-1) + P_1 \\ L_r(p-r, d+1) + P_1 \\ \min_i L_r(p-r, i) + P_2 \end{cases} - \min_k L_r(p-r, k) $$

**最终代价**：
$$ S(p, d) = \sum_r L_r(p, d) $$

### 3.3 后处理

- **左右一致性检查**：$|d_L(p) - d_R(p-d_L(p))| < 1$
- **唯一性检查**：最优视差与次优的比值
- **插值填充**：填补遮挡区域
- **中值滤波**：去除噪声

### 3.4 SGM变体

| 变体 | 改进 | 速度 |
|------|------|------|
| SGM | 原始 | 中 |
| rSGM | 缩减路径数 | 快 |
| tSGM | 阈值调整 | 灵活 |
| SGM-GPU | GPU并行 | 极快 |
| MGM | 多窗口 | 高精度 |

## 4. 学习型立体匹配

### 4.1 MC-CNN

使用CNN计算匹配代价（Zbontar & LeCun, 2015）

### 4.2 PSMNet

使用3D CNN构建匹配代价体（2018）

**架构**：
```
输入 → 特征提取(2D CNN) → 代价体 → 3D CNN正则化 → 视差回归
```

### 4.3 RAFT-Stereo

基于迭代更新的立体匹配（2021）

## 5. 立体匹配基准

| 数据集 | 场景 | 指标 | 描述 |
|--------|------|------|------|
| KITTI | 自动驾驶 | D1-all, D2-all | 室外道路场景 |
| Middlebury | 室内 | bad 2.0 | 高分辨率 |
| ETH3D | 室内/外 | bad 1.0 | 多种场景 |
| SceneFlow | 合成 | EPE, D1 | 大规模合成数据 |

## 6. 参考文献

1. Hirschmuller, H. (2008). Stereo processing by semiglobal matching and mutual information. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30(2), 328-341.
2. Scharstein, D., & Szeliski, R. (2002). A taxonomy and evaluation of dense two-frame stereo correspondence algorithms. *International Journal of Computer Vision*, 47(1), 7-42.
3. Chang, J.-R., & Chen, Y.-S. (2018). Pyramid stereo matching network. *CVPR*.
4. Lipson, L., Teed, Z., & Deng, J. (2021). RAFT-Stereo: Multilevel recurrent field transforms for stereo matching. *3DV*.
