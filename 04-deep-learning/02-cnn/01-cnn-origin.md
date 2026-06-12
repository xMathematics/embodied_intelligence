# 2.1 卷积神经网络起源

## 1. 为什么提出CNN

### 1.1 历史背景

**1962年**：Hubel和Wiesel发现猫视觉皮层中的神经元对局部边缘敏感（感受野概念）。

**1980年**：福岛邦彦提出Neocognitron，是第一个受生物视觉启发的神经网络。

**1998年**：LeCun提出LeNet-5，用于手写数字识别，确立了CNN的基本架构：卷积→池化→全连接。

### 1.2 MLP的问题

**参数爆炸**：224×224×3图像在MLP第一层就有224×224×3×N个参数。

**空间信息丢失**：像素展开为向量后，2D结构信息丢失。

**平移不变性缺失**：同一物体在不同位置需要分别学习。

### 1.3 CNN的解决方案

**局部连接**：每个神经元只连接输入的局部区域（感受野）。

**权重共享**：同一卷积核在整个图像上滑动，参数共享。

**层次化特征**：浅层检测边缘→中层检测纹理→深层检测物体。

## 2. 核心原理

### 2.1 卷积操作

$$(I * K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) K(m, n)$$

### 2.2 关键参数

| 参数 | 作用 | 典型值 |
|------|------|--------|
| 卷积核大小(K) | 感受野大小 | 3×3, 5×7, 7×7 |
| 步长(S) | 控制输出分辨率 | 1, 2 |
| 填充(P) | 控制输出尺寸 | 'same', 'valid' |
| 通道(C) | 卷积核数量 | 64, 128, 256 |

### 2.3 池化层

**最大池化**：取窗口内最大值，提供平移不变性。

**平均池化**：取窗口内平均值，减少估计方差。

**全局平均池化**：全连接层的替代，大幅减少参数量。

## 3. 局限性

| 局限 | 原因 | 影响 |
|------|------|------|
| 局部感受野 | 卷积核尺寸有限 | 难以捕获全局依赖 |
| 固定权重结构 | 静态卷积核 | 无法适应输入内容 |
| 下采样信息损失 | 池化操作 | 丢失精细定位信息 |
| 平移等变≠旋转等变 | 卷积性质 | 需数据增强旋转 |

## 4. 改进方向

- **空洞卷积**：扩大感受野不损失分辨率（DeepLab）
- **可变形卷积**：自适应感受野形状
- **注意力增强CNN**：SE-Net, CBAM, Non-local
- **Vision Transformer**：全局自注意力替代卷积

## 5. 在具身智能中的应用

- **机器人视觉感知**：物体检测、分割、深度估计的骨干网络
- **端到端驾驶**：从相机图像直接预测控制命令
- **抓取检测**：CNN预测抓取位置和姿态
- **视觉伺服**：实时物体跟踪和控制

## 6. 参考文献

1. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
2. Hubel, D. H., & Wiesel, T. N. (1962). Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. *The Journal of Physiology*, 160(1), 106-154.
3. Fukushima, K. (1980). Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. *Biological Cybernetics*, 36(4), 193-202.
