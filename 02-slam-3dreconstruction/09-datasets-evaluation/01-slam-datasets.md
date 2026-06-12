# 9.1 SLAM公开数据集

## 1. 概述

公开数据集对SLAM研究具有重要价值，它们提供标准化的测试环境和评估基准。

## 2. KITTI数据集

**典型场景**：自动驾驶
**网址**：http://www.cvlibs.net/datasets/kitti/

| 数据 | 详情 |
|------|------|
| 传感器 | 双目相机，Velodyne HDL-64E，GPS/IMU |
| 里程计序列 | 11个训练，11个测试 |
| 场景 | 城市、乡村、高速公路 |
| 总距离 | 39.2km |
| 数据量 | 约23,000帧 |

**评估指标**：相对平移误差(%)、相对旋转误差(°/m)

## 3. TUM RGB-D数据集

**典型场景**：室内RGB-D SLAM
**网址**：https://vision.in.tum.de/data/datasets/rgbd-dataset

| 序列类型 | 场景 | 特点 |
|----------|------|------|
| fr1/desk | 办公桌场景 | 运动速度快 |
| fr2/xyz | XYZ运动 | 运动规律 |
| fr3/office | 办公室场景 | 大场景 |
| fr3/walk | 行走 | 动态人物 |

**评估指标**：ATE（绝对轨迹误差），RPE（相对位姿误差）

## 4. EuRoC MAV

**典型场景**：无人机视觉-惯性
**网址**：https://projects.asl.ethz.ch/datasets/

| 序列 | 难度 | 特点 |
|------|------|------|
| MH_01 | 简单 | 机器间 |
| MH_04 | 困难 | 机器间+快速运动 |
| V2_03 | 困难 | 室内+快速旋转 |

**传感器**：双目相机，IMU (ADIS16448, BMI160)
**真值**：Vicon运动捕捉或Leica激光跟踪仪

## 5. 其他数据集

### 5.1 SLAM数据集

| 数据集 | 年份 | 传感器 | 场景 |
|--------|------|--------|------|
| New College | 2009 | 双目+激光 | 校园 |
| RAWSEEDS | 2009 | 多传感器 | 室内/外 |
| Aqualoc | 2022 | 视觉 | 水下 |
| SubT | 2022 | 多传感器 | 地下 |

### 5.2 VIO数据集

| 数据集 | 年份 | 传感器 | 特点 |
|--------|------|--------|------|
| PennCOSYVIO | 2017 | 多视觉+IMU | 室内/外 |
| ADVIO | 2018 | 手机传感器 | 日常 |
| TUM-VI | 2018 | 鱼眼+IMU | 快速运动 |
| UZH-FPV | 2020 | 事件相机+标准相机 | 高速 |

## 6. 数据集的挑战

- **真值精度**：不同数据集的真值精度差异大
- **场景多样性**：现有数据集覆盖的多样性不足
- **标注成本**：高质量真值的获取成本高
- **传感器差异**：不同数据集的传感器配置不同

## 7. 参考文献

1. Geiger, A., Lenz, P., & Urtasun, R. (2012). Are we ready for autonomous driving? The KITTI vision benchmark suite. *CVPR*.
2. Sturm, J., Engelhard, N., Endres, F., Burgard, W., & Cremers, D. (2012). A benchmark for the evaluation of RGB-D SLAM systems. *IROS*.
3. Burri, M., et al. (2016). The EuRoC micro aerial vehicle datasets. *The International Journal of Robotics Research*, 35(10), 1157-1163.
4. Schubert, D., et al. (2019). The TUM VI benchmark for evaluating visual-inertial odometry. *IROS*.
