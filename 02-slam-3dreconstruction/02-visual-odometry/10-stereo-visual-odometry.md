# 2.10 双目视觉里程计

## 1. 概述

双目视觉里程计（Stereo Visual Odometry）使用双目相机估计相机运动。与单目相比，双目相机可以通过视差直接获得场景的绝对尺度信息，避免了单目SLAM的尺度模糊问题。

## 2. 双目相机原理

### 2.1 深度计算

双目相机的深度由基线 $b$ 和视差 $d$ 决定：

$$ Z = \frac{f \cdot b}{d} $$

其中 $f$ 是焦距，$b$ 是基线距离，$d = u_L - u_R$ 是视差。

### 2.2 深度精度

深度估计的精度与基线、焦距和视差精度有关：

$$ \sigma_Z = \frac{Z^2}{f \cdot b} \sigma_d $$

- 近距离：深度精度高
- 远距离：深度精度迅速下降
- 延长基线可提高远距离精度

## 3. 立体匹配

### 3.1 极线校正

将左右图像校正到同一平面上，使对应点处于同一水平线：

$$ \mathbf{H}_{\text{rect}} = \text{rectify}(\mathbf{K}_L, \mathbf{K}_R, \mathbf{R}, \mathbf{t}) $$

### 3.2 匹配方法

| 方法 | 原理 | 复杂度 | 精度 |
|------|------|--------|------|
| Block Matching (BM) | 固定窗口SAD/SSD | 低 | 低 |
| Semi-Global Matching (SGM) | 多方向DP聚合 | 中 | 高 |
| ELAS | 基于先验的快速匹配 | 中 | 高 |
| PSMNet | 3D CNN端到端学习 | GPU高 | 最高 |
| RAFT-Stereo | 迭代更新 | GPU高 | 最高 |

### 3.3 SGM算法

SGM通过多方向路径聚合实现高效立体匹配：

$$ L(\mathbf{p}, d) = C(\mathbf{p}, d) + \min \begin{cases} L(\mathbf{p}-\mathbf{r}, d) \\ L(\mathbf{p}-\mathbf{r}, d-1) + P_1 \\ L(\mathbf{p}-\mathbf{r}, d+1) + P_1 \\ \min_i L(\mathbf{p}-\mathbf{r}, i) + P_2 \end{cases} - \min_k L(\mathbf{p}-\mathbf{r}, k) $$

## 4. 双目里程计方法

### 4.1 特征法双目VO

- 在左右图像中提取特征点
- 通过立体匹配获得3D坐标
- 在帧间匹配中估计位姿（PnP或3D-3D）

### 4.2 直接法双目VO

- 使用左右图像的光度误差和时序光度误差
- Stereo DSO (2017)

### 4.3 双目ORB-SLAM

ORB-SLAM2/3的双目模式：
- 利用立体匹配计算特征点深度
- 在跟踪线程中使用3D-2D PnP
- 关键帧的立体BA优化

## 5. 双目VO vs 单目VO

| 方面 | 单目 | 双目 |
|------|------|------|
| 尺度 | 不可观测 | 可观测 |
| 初始化 | 需要特定运动 | 瞬时初始化 |
| 鲁棒性 | 纯旋转失效 | 对旋转鲁棒 |
| 范围 | 近~远 | 受基线限制 |
| 成本 | 低 | 高 |
| 校准 | 简单 | 需要立体标定 |

## 6. 代表性双目VO系统

| 系统 | 年份 | 方法 | 特点 |
|------|------|------|------|
| Stereo DSO | 2017 | 直接法 | 光度标定 |
| ORB-SLAM2/3 | 2017/2020 | 特征法 | 完整SLAM |
| ROVIOLI | 2018 | 特征法 | 视觉-惯性 |
| VINS-Fusion | 2019 | 特征法 | 多传感器 |
| Droid-SLAM | 2021 | 学习型 | RAFT深度 |

## 7. 参考文献

1. Engel, J., Koltun, V., & Cremers, D. (2017). Direct sparse odometry with stereo cameras. *ICCV*.
2. Mur-Artal, R., & Tardós, J. D. (2017). ORB-SLAM2: An open-source SLAM system for monocular, stereo, and RGB-D cameras. *IEEE Transactions on Robotics*.
3. Hirschmuller, H. (2008). Stereo processing by semiglobal matching and mutual information. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30(2), 328-341.
4. Scharstein, D., & Szeliski, R. (2002). A taxonomy and evaluation of dense two-frame stereo correspondence algorithms. *International Journal of Computer Vision*, 47(1), 7-42.
