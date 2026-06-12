# 2.8 直接法

## 1. 概述

直接法（Direct Methods）不提取特征点，而是直接利用图像像素的亮度信息进行相机运动估计。相比于特征法，直接法可以利用图像的更多信息，在低纹理场景下具有更好的鲁棒性。

## 2. 直接法原理

### 2.1 亮度恒假设

同一空间点在不同视角下的图像亮度保持不变：

$$ I_1(\mathbf{x}_1) = I_2(\mathbf{x}_2) $$

其中 $\mathbf{x}_2 = \pi(\mathbf{T} \cdot \pi^{-1}(\mathbf{x}_1, d))$。

### 2.2 光度误差

$$ e_i = I_1(\mathbf{x}_i) - I_2(\mathbf{x}_i') $$

其中 $\mathbf{x}_i'$ 是源点根据当前位姿估计投影到目标图像的坐标。

### 2.3 位姿优化

最小化所有像素的光度误差：

$$ \mathbf{T}^* = \arg\min_{\mathbf{T}} \sum_{i} \|I_1(\mathbf{x}_i) - I_2(\pi(\mathbf{T} \cdot \pi^{-1}(\mathbf{x}_i, d_i)))\|^2 $$

## 3. 直接法分类

| 类型 | 使用像素数量 | 地图密度 | 代表系统 |
|------|-------------|----------|----------|
| 稀疏直接法 | 有信息量的像素(几百个) | 稀疏 | DSO |
| 半稠密直接法 | 梯度明显的像素(几万个) | 半稠密 | LSD-SLAM |
| 稠密直接法 | 所有像素(几十万) | 稠密 | DTAM |

## 4. 稀疏直接法：DSO

### 4.1 像素选择

DSO选择有信息量的像素参与优化：
- 梯度幅值大的像素
- 分布均匀（避免集中在图像局部）
- 非边缘区域（避免陷入模糊）

### 4.2 光度标定

DSO考虑了完整的光度模型：

$$ I(\mathbf{x}) = G \left( t \cdot B(\mathbf{x}) \cdot V(\mathbf{x}) \cdot \exp\left(-\sum_{j} \lambda_j \mathbf{b}_j(\mathbf{x})\right) \right) $$

其中 $G$ 是响应函数，$t$ 是曝光时间，$V$ 是渐晕，$B$ 是辐照度。

### 4.3 滑动窗口优化

DSO使用滑动窗口中的关键帧进行联合优化，窗口外旧状态被边缘化。

## 5. 半稠密直接法：LSD-SLAM

LSD-SLAM构建半稠密深度地图：

**深度估计**：使用立体匹配和深度滤波：

$$ P(d) = \mathcal{N}(\mu_d, \sigma_d^2) $$

**深度更新**：贝叶斯递归融合：

$$ P(d \mid \text{obs}) \propto P(\text{obs} \mid d) P(d) $$

## 6. 直接法中的雅可比计算

直接法的雅可比链式法则：

$$ \mathbf{J} = \frac{\partial I_2}{\partial \mathbf{x}'} \cdot \frac{\partial \mathbf{x}'}{\partial \mathbf{p}} \cdot \frac{\partial \mathbf{p}}{\partial \mathbf{T}} \cdot \frac{\partial \mathbf{T}}{\partial \boldsymbol{\xi}} $$

其中：
- $\frac{\partial I_2}{\partial \mathbf{x}'}$：图像梯度
- $\frac{\partial \mathbf{x}'}{\partial \mathbf{p}}$：投影几何的导数
- $\frac{\partial \mathbf{p}}{\partial \mathbf{T}}$：变换的导数
- $\frac{\partial \mathbf{T}}{\partial \boldsymbol{\xi}}$：李代数的导数

## 7. 直接法vs特征法对比

| 方面 | 直接法 | 特征法 |
|------|--------|--------|
| 信息利用 | 充分利用所有像素 | 只使用特征点 |
| 低纹理环境 | 较好 | 差 |
| 运动模糊 | 较鲁棒 | 容易丢失特征 |
| 光照变化 | 敏感 | 较鲁棒 |
| 大视角变化 | 差 | 较好 |
| 回环检测 | 不天然支持 | 天然支持 |
| 初始化 | 需要特殊处理 | 自然 |

## 8. 主要直接法系统

| 系统 | 年份 | 密度 | 特点 |
|------|------|------|------|
| DTAM | 2011 | 稠密 | 实时稠密SLAM，GPU加速 |
| LSD-SLAM | 2014 | 半稠密 | 大规模直接单目SLAM |
| DSO | 2016 | 稀疏 | 最精确的直接法之一 |
| SVO | 2014 | 稀疏 | 半直接法，极快 |
| Direct Sparse Mapping | 2019 | 稀疏 | 无回环的直接法建图 |

## 9. 参考文献

1. Engel, J., Schöps, T., & Cremers, D. (2014). LSD-SLAM: Large-scale direct monocular SLAM. *ECCV*.
2. Engel, J., Koltun, V., & Cremers, D. (2016). Direct sparse odometry. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 40(3), 611-625.
3. Newcombe, R. A., Lovegrove, S. J., & Davison, A. J. (2011). DTAM: Dense tracking and mapping. *ICCV*.
4. Forster, C., Pizzoli, M., & Scaramuzza, D. (2014). SVO: Fast semi-direct monocular visual odometry. *ICRA*.
