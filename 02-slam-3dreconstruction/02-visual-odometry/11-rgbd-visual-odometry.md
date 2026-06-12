# 2.11 RGB-D视觉里程计

## 1. 概述

RGB-D相机同时提供彩色图像和深度图，使得视觉里程计可以直接使用3D几何信息。RGB-D VO将RGB信息（纹理）和深度信息（几何）结合进行位姿估计。

## 2. RGB-D相机技术

### 2.1 结构光法（Kinect v1）

- 投影红外编码散斑图案
- 通过图案变形计算深度
- 受环境光干扰大
- 室内有效距离0.5-5m

### 2.2 飞行时间法（Kinect v2/Azure Kinect）

- 发射调制红外光，测量飞行时间
- 受环境光干扰较小
- 有效距离0.5-6m（Azure Kinect可达0.25-5.5m）
- 存在多径干扰

### 2.3 立体匹配法（Intel RealSense）

- 使用双目红外相机
- 主动红外投影辅助纹理
- 室内外都可用
- 有效距离0.1-10m（取决于型号）

## 3. RGB-D里程计方法

### 3.1 基于RGB-D ICP

$$ \mathbf{T}^* = \arg\min_{\mathbf{T}} \sum_i \|\mathbf{T} \mathbf{p}_i - \mathbf{q}_i\|^2 $$

其中 $\mathbf{p}_i$ 和 $\mathbf{q}_i$ 分别是RGB-D图像中的3D点。

### 3.2 联合光度-几何误差

$$ \mathbf{T}^* = \arg\min_{\mathbf{T}} \left[ \sum_i \rho\left(\|I_1(\mathbf{x}_i) - I_2(\mathbf{x}_i')\|^2\right) + w \sum_j \rho\left(\|Z_1(\mathbf{x}_j) - Z_2(\mathbf{x}_j')\|^2\right) \right] $$

同时最小化光度误差和深度误差。

### 3.3 DVO（Dense Visual Odometry）

DVO使用稠密深度图进行帧间位姿估计：
- 使用所有像素建立数据关联
- 结合稳健的M-estimator
- 多分辨率优化策略

## 4. RGB-D数据的挑战

### 4.1 深度噪声

- **Range噪声**：随距离平方增长
- **边界噪声**：深度图边缘的飞点（Flying Pixels）
- **传感器干扰**：多Kinect互相干扰
- **缺失深度**：反射表面、远距离、强光

### 4.2 飞点（Flying Pixels）

在深度不连续处，测到的深度是前景和背景点的混合值。

## 5. RGB-D VO系统

| 系统 | 年份 | 特点 |
|------|------|------|
| KinectFusion | 2011 | 实时TSDF融合 |
| RGB-D SLAM | 2011 | 特征法+ICP |
| DVO | 2012 | 稠密直接法 |
| ElasticFusion | 2015 | 非刚体变形 |
| BundleFusion | 2017 | 在线BA优化 |
| Kintinuous | 2015 | 大规模重建 |

## 6. 参考文献

1. Steinbrücker, F., Sturm, J., & Cremers, D. (2011). Real-time visual odometry from dense RGB-D images. *ICCV Workshop*.
2. Kerl, C., Sturm, J., & Cremers, D. (2013). Robust odometry estimation for RGB-D cameras. *ICRA*.
3. Newcombe, R. A., et al. (2011). KinectFusion: Real-time dense surface mapping and tracking. *ISMAR*.
4. Dai, A., Nießner, M., Zollhöfer, M., Izadi, S., & Theobalt, C. (2017). BundleFusion: Real-time globally consistent 3D reconstruction using on-the-fly surface re-integration. *ACM Transactions on Graphics*, 36(4).
