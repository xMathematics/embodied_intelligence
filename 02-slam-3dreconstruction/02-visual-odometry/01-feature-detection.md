# 2.1 特征点检测

## 1. 概述

特征点检测是视觉SLAM前端的核心步骤之一。好的特征点需要在不同光照、视角和尺度下保持稳定。本章将系统介绍从传统角点检测到最新的学习型特征点检测方法。

## 2. 经典角点检测

### 2.1 Harris角点检测器

Harris角点（1988）基于局部自相关函数，通过计算图像窗口在各个方向移动时的灰度变化判断角点。

**自相关矩阵**：

$$ \mathbf{M} = \sum_{x,y} w(x,y) \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} $$

**角点响应函数**：

$$ R = \det(\mathbf{M}) - k \cdot \text{tr}(\mathbf{M})^2 = \lambda_1 \lambda_2 - k (\lambda_1 + \lambda_2)^2 $$

其中 $\lambda_1, \lambda_2$ 是 $\mathbf{M}$ 的特征值：
- $\lambda_1 \approx 0, \lambda_2 \approx 0$：平坦区域
- $\lambda_1 \gg 0, \lambda_2 \approx 0$：边缘
- $\lambda_1 \gg 0, \lambda_2 \gg 0$：角点

**改进：Shi-Tomasi角点**使用 $R = \min(\lambda_1, \lambda_2)$，在跟踪中表现更好。

### 2.2 FAST角点检测器

FAST（Features from Accelerated Segment Test, 2006）通过比较像素与周围圆环上的像素快速检测角点。

**判断准则**：在半径为3的Bresenham圆上的16个像素中，存在连续 $n$ 个像素与中心像素的亮度差异超过阈值 $t$。

**机器学习方法**：使用ID3决策树选择最优的像素比较顺序，进一步提高速度。

**非极大值抑制**：在角点响应较大的邻域内保留响应最大的角点。

## 3. 尺度不变特征

### 3.1 尺度空间理论

尺度空间通过高斯卷积构建：

$$ L(x, y, \sigma) = G(x, y, \sigma) * I(x, y) $$

其中 $G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} e^{-(x^2+y^2)/(2\sigma^2)}$。

### 3.2 高斯差分（DoG）

$$ D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma) $$

DoG是尺度归一化拉普拉斯（LoG）的近似，计算更高效。

**极值点检测**：在DoG金字塔中，每个像素与同一尺度的8个邻居以及上下相邻尺度的9×2个邻居比较。

### 3.3 Hessian仿射检测器

使用Hessian矩阵检测斑点（blob）结构：

$$ \mathcal{H} = \begin{bmatrix} L_{xx} & L_{xy} \\ L_{xy} & L_{yy} \end{bmatrix} $$

Hessian的行列式响应 $\det(\mathcal{H})$ 在斑点位置达到极值。

## 4. 二进制特征检测

### 4.1 ORB中的改进FAST

ORB在FAST基础上增加了：
1. **尺度金字塔**：在多个尺度上检测FAST角点
2. **方向计算**：使用灰度质心法（Intensity Centroid）
3. **Harris响应排序**：使用Harris角点响应排序，保留前N个

**灰度质心法**：

$$ m_{pq} = \sum_{x,y} x^p y^q I(x, y) $$
$$ C = \left(\frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}}\right) $$
$$ \theta = \text{atan2}(m_{01}, m_{10}) $$

### 4.2 BRISK

BRISK使用FAST检测器加上尺度空间定位，在多个尺度上检测角点。

### 4.3 AGAST

AGAST（Adaptive and Generic Accelerated Segment Test）是FAST的泛化，使用更通用的决策树结构，在不同类型图像上自动适应。

## 5. 学习型特征检测

### 5.1 SuperPoint

SuperPoint（CVPR 2018）使用深度学习进行特征点检测和描述：

- **自监督训练**：使用合成数据（MagicPoint）和Homography Adaptation
- **全卷积架构**：在单次前向传播中同时检测特征点和计算描述子
- **共享编码器**：特征点检测头和描述子头共享编码器特征

### 5.2 D2-Net

D2-Net（CVPR 2019）将检测和描述统一在单个网络中，特征点根据描述子的可区分性检测。

### 5.3 R2D2

R2D2（NeurIPS 2019）同时预测特征点的**可重复性**（repeatability）和**可靠性**（reliability），选择同时具有高分数的位置。

### 5.4 DISK

DISK（NeurIPS 2020）使用强化学习策略直接优化特征匹配的数量和质量。

## 6. 特征点检测对比

| 检测器 | 速度 | 重复性 | 精度 | 尺度不变 | 旋转不变 | 适用场景 |
|--------|------|--------|------|----------|----------|----------|
| Harris | 快 | 中 | 中 | 否 | 否 | 基本角点检测 |
| FAST | 非常快 | 中 | 中 | 否 | 否 | 实时应用 |
| SIFT | 慢 | 高 | 高 | 是 | 是 | 高精度匹配 |
| SURF | 中 | 高 | 高 | 是 | 是 | 精度与速度平衡 |
| ORB | 非常快 | 中 | 中 | 部分 | 是 | 实时SLAM |
| SuperPoint | GPU快 | 高 | 高 | 学习得到 | 学习得到 | 高质量匹配 |
| DISK | GPU快 | 高 | 非常高 | 学习得到 | 学习得到 | 最先进匹配 |

## 7. 参考文献

1. Harris, C., & Stephens, M. (1988). A combined corner and edge detector. *Alvey Vision Conference*.
2. Rosten, E., & Drummond, T. (2006). Machine learning for high-speed corner detection. *ECCV*.
3. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *IJCV*, 60(2), 91-110.
4. DeTone, D., Malisiewicz, T., & Rabinovich, A. (2018). SuperPoint: Self-supervised interest point detection and description. *CVPR Workshop*.
5. Dusmanu, M., et al. (2019). D2-Net: A trainable CNN for joint description and detection of local features. *CVPR*.
