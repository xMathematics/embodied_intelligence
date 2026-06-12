# 2.12 半直接法视觉里程计

## 1. 概述

半直接法视觉里程计（Semi-Direct Visual Odometry, SVO）结合了特征法和直接法的优点：使用直接法跟踪特征点，使用特征法优化位姿。这种方法在保持实时性的同时，提供了良好的鲁棒性。

## 2. SVO原理

### 2.1 算法流程

SVO采用三层架构：

1. **稀疏图像对齐**：使用直接法估计帧间相对位姿
2. **特征点对齐**：使用Lucas-Kanade跟踪特征点
3. **位姿和结构优化**：使用BA优化位姿和3D点

### 2.2 稀疏图像对齐

最小化像素块的光度误差：

$$ \mathbf{T}_{k,k-1}^* = \arg\min_{\mathbf{T}} \frac{1}{2} \sum_{i} \|I_k(\pi(\mathbf{T} \cdot \mathbf{p}_i)) - I_{k-1}(\pi(\mathbf{p}_i))\|^2 $$

其中 $\mathbf{p}_i$ 是上一帧中已三角化的3D点。

### 2.3 特征对齐

对每个特征点，使用反向合成LK算法在图像块中精确定位：

$$ \mathbf{x}_i' = \arg\min_{\mathbf{x}'} \frac{1}{2} \|I_k(\mathbf{x}') - I_{\text{ref}}(\pi(\mathbf{p}_i))\|^2 $$

### 2.4 位姿优化

最小化重投影误差：

$$ \mathbf{T}_{k,w}^* = \arg\min_{\mathbf{T}} \frac{1}{2} \sum_i \|\mathbf{x}_i' - \pi(\mathbf{T} \cdot \mathbf{p}_i)\|^2 $$

### 2.5 深度滤波

SVO使用概率深度滤波器估计新特征点的深度：

$$ P(d \mid \text{obs}) \propto P(\text{obs} \mid d) \cdot P(d) $$

深度用高斯+均匀分布混合建模，通过递归贝叶斯更新。

## 3. SVO 2.0改进

SVO 2.0（2017）的主要改进：
- **边缘对齐**：使用图像边缘特征增强
- **在线光度标定**：估计曝光时间和响应函数
- **鲁棒跟踪**：处理快速运动和遮挡
- **地图复用**：支持长时间运行

## 4. 混合方法总结

| 方法 | 跟踪 | 优化 | 代表性系统 |
|------|------|------|-----------|
| 全特征法 | 特征匹配 | BA | ORB-SLAM |
| 全直接法 | 光度误差 | 光度BA | DSO |
| 半直接法 | 直接法跟踪特征 | 重投影BA | SVO |

### 4.1 SVO的优势

- **极快**：每秒数百帧
- **低纹理鲁棒**：直接法跟踪不依赖特征
- **精度好**：重投影误差优化

### 4.2 SVO的不足

- **无回环检测**（原始版本）
- **缺乏全局优化**
- **长时间运行漂移显著**

## 5. 参考文献

1. Forster, C., Pizzoli, M., & Scaramuzza, D. (2014). SVO: Fast semi-direct monocular visual odometry. *ICRA*.
2. Forster, C., Zhang, Z., Gassner, M., Werlberger, M., & Scaramuzza, D. (2017). SVO: Semidirect visual odometry for monocular and multicamera systems. *IEEE Transactions on Robotics*, 33(2), 249-265.
3. Pizzoli, M., Forster, C., & Scaramuzza, D. (2014). REMODE: Probabilistic, monocular dense reconstruction in real time. *ICRA*.
