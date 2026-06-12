# 4.1 传感器类型与标定

## 1. 概述

多传感器融合是提高SLAM系统鲁棒性和精度的重要方法。不同传感器各有优缺点，通过融合可以取长补短。本章介绍各类传感器特性及多传感器标定方法。

## 2. 传感器特性对比

| 传感器 | 精度 | 频率 | 长期稳定性 | 环境依赖 | 成本 |
|--------|------|------|-----------|---------|------|
| 相机 | 中 | 30-60Hz | 好 | 光照 | 低 |
| IMU | 短期高/长期漂移 | 100-1000Hz | 差(漂移) | 无 | 低 |
| LiDAR | 高 | 10-20Hz | 好 | 天气 | 高 |
| GPS | 米级 | 1-10Hz | 好 | 信号 | 低 |
| 轮式编码器 | 中 | 10-1000Hz | 好(距离) | 地面 | 低 |

### 2.1 相机

- **优点**：信息丰富，成本低
- **缺点**：受光照影响，弱纹理失效
- **关键参数**：分辨率、帧率、视场角、动态范围

### 2.2 IMU

- **优点**：高频、不受环境干扰
- **缺点**：存在漂移，需要积分
- **关键参数**：加速度计量程、陀螺仪量程、噪声密度、零偏稳定性

### 2.3 LiDAR

- **优点**：精确的3D测量，不受光照影响
- **缺点**：受天气影响，稀疏，成本高
- **关键参数**：线数、视场角、测距范围、精度、回波次数

## 3. 传感器标定

### 3.1 相机内参标定

Zhang的标定法（1998）使用棋盘格：

1. 从不同角度拍摄棋盘格图像
2. 提取角点
3. 假设棋盘格平面，计算单应矩阵
4. 利用单应约束求解内参
5. 非线性优化细化参数

### 3.2 相机-IMU外参标定（Kalibr）

Kalibr是ETH开发的标定工具：

- 使用连续时间的B样条表示轨迹
- 同时估计时间偏移和空间外参
- 需要充分激励的运动

**标定状态**：
$$ \boldsymbol{\theta} = \{\mathbf{T}_{IC}, \mathbf{t}_d, \mathbf{b}_g, \mathbf{b}_a, \mathbf{K}\} $$

### 3.3 相机-LiDAR标定

- **基于标定板**：在LiDAR点云中检测标定板平面
- **基于互信息**：最大化LiDAR反射率和图像强度的互信息
- **基于边缘对齐**：对齐LiDAR边缘和图像边缘

### 3.4 LiDAR-IMU标定

- 利用LiDAR扫描匹配估计运动
- 与IMU测量对齐
- 估计旋转和平移外参

## 4. 时间同步

多传感器融合的关键前提是时间同步：

- **硬同步**：硬件触发信号同步
- **软同步**：基于时间戳插值对齐
- **时间偏移估计**：在线估计传感器间延迟

## 5. 参考文献

1. Zhang, Z. (2000). A flexible new technique for camera calibration. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), 1330-1334.
2. Furgale, P., Rehder, J., & Siegwart, R. (2013). Unified temporal and spatial calibration for multi-sensor systems. *IROS*.
3. Rehder, J., Nikolic, J., Schneider, T., Hinzmann, T., & Siegwart, R. (2016). Extending kalibr: Calibrating the extrinsics of multiple IMUs and of individual axes. *ICRA*.
4. Geiger, A., Moosmann, F., Car, Ö., & Schuster, B. (2012). Automatic camera and range sensor calibration using a single shot. *ICRA*.
