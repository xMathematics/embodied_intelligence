# 2.9 光流估计

## 1. 概述

光流（Optical Flow）描述图像中像素的瞬时运动。在SLAM中，光流可用于帧间像素跟踪、直接法中的像素对应和场景流估计等。

## 2. 光流基本原理

### 2.1 亮度恒假设

$$ I(x, y, t) = I(x + dx, y + dy, t + dt) $$

泰勒展开并忽略高阶项：

$$ I_x u + I_y v + I_t = 0 $$

其中 $u = \frac{dx}{dt}$，$v = \frac{dy}{dt}$ 是光流速度，$I_x, I_y, I_t$ 是图像梯度。

这就是**光流约束方程**（OFCE），一个方程两个未知数，需要额外的约束条件。

### 2.2 孔径问题

孔径问题是光流估计的核心难题：在局部窗口内，只能观测到垂直于边缘方向的光流分量，无法确定沿边缘方向的分量。

## 3. Lucas-Kanade光流

### 3.1 基本方法

LK方法假设局部窗口 $W$ 内所有像素运动相同：

$$ \sum_{(x,y) \in W} (I_x u + I_y v + I_t)^2 \rightarrow \min $$

**求解**：

$$ \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} \sum I_x I_t \\ \sum I_y I_t \end{bmatrix} $$

### 3.2 金字塔LK

处理大位移运动：

1. 构建图像金字塔（从粗到细）
2. 在顶层金字塔估计光流
3. 将结果传播到下一层
4. 在每一层细化光流估计

### 3.3 反向合成算法

反向合成（Inverse Compositional）算法将角色互换，预计算参考图像的Hessian矩阵，大幅降低计算成本：

$$ \Delta \mathbf{p} = \mathbf{H}^{-1} \sum_{\mathbf{x}} \left[ \nabla I(\mathbf{W}(\mathbf{x}; \mathbf{0})) \frac{\partial \mathbf{W}}{\partial \mathbf{p}} \right]^T [I(\mathbf{x}) - I(\mathbf{W}(\mathbf{x}; \mathbf{p}))] $$

## 4. Horn-Schunck光流

### 4.1 全局变分方法

HS方法引入全局平滑约束：

$$ E = \iint (I_x u + I_y v + I_t)^2 dxdy + \lambda \iint (|\nabla u|^2 + |\nabla v|^2) dxdy $$

### 4.2 求解

使用欧拉-拉格朗日方程迭代求解：

$$ \begin{aligned} u^{k+1} &= \bar{u}^k - \frac{I_x (I_x \bar{u}^k + I_y \bar{v}^k + I_t)}{\alpha^2 + I_x^2 + I_y^2} \\ v^{k+1} &= \bar{v}^k - \frac{I_y (I_x \bar{u}^k + I_y \bar{v}^k + I_t)}{\alpha^2 + I_x^2 + I_y^2} \end{aligned} $$

## 5. 稠密光流

### 5.1 Farneback光流

使用多项式展开近似每个邻域的图像信号，通过多项式系数估计位移。

### 5.2 EpicFlow

结合稀疏匹配和变分优化：
1. 稀疏特征匹配
2. 使用Edge-aware插值生成稠密初始化
3. 变分精化

## 6. 学习型光流

### 6.1 FlowNet/FlowNet2

FlowNet（CVPR 2015）首次使用CNN端到端学习光流：
- **FlowNetS**：直接输入两帧堆叠图像，输出光流
- **FlowNetC**：分别编码后计算相关性

FlowNet2堆叠多个FlowNet子网络，显著提升性能。

### 6.2 RAFT

RAFT（Recurrent All-pairs Field Transforms, ECCV 2020）：

**关键创新**：
1. **4D相关体**：计算所有像素对之间的相关性
2. **循环更新**：使用GRU迭代细化光流
3. **高分辨率**：保持1/8分辨率的特征图

**性能**：在Sintel和KITTI数据集上达到SOTA，成为基准方法。

### 6.3 GMA（Global Motion Aggregation）

在RAFT基础上引入全局运动聚合，利用自注意力机制处理遮挡区域的运动估计。

## 7. 光流在SLAM中的应用

- **直接法SLAM**：DSO中用于建立像素对应
- **SVO**：用LK光流跟踪特征点
- **帧间预测**：为特征匹配提供初始位置
- **运动分割**：识别独立运动的物体
- **场景流**：光流+深度图变化估计3D运动

## 8. 光流方法对比

| 方法 | 类型 | 速度 | 精度 | 大位移 | 遮挡处理 |
|------|------|------|------|--------|----------|
| LK | 稀疏 | 极快 | 中 | 需金字塔 | 无 |
| HS | 稠密 | 慢 | 低 | 差 | 差 |
| Farneback | 稠密 | 快 | 中 | 中 | 无 |
| FlowNet2 | 稠密(学习) | GPU快 | 高 | 好 | 中 |
| RAFT | 稠密(学习) | GPU中 | 极高 | 好 | 好 |

## 9. 参考文献

1. Lucas, B. D., & Kanade, T. (1981). An iterative image registration technique with an application to stereo vision. *IJCAI*.
2. Horn, B. K., & Schunck, B. G. (1981). Determining optical flow. *Artificial Intelligence*, 17(1-3), 185-203.
3. Dosovitskiy, A., et al. (2015). FlowNet: Learning optical flow with convolutional networks. *ICCV*.
4. Teed, Z., & Deng, J. (2020). RAFT: Recurrent all-pairs field transforms for optical flow. *ECCV*.
