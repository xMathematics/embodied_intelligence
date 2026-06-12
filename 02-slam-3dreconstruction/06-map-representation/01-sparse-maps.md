# 6.1 稀疏地图

## 1. 概述

稀疏地图只保留环境中的关键特征点，不存储完整的环境几何信息。它是视觉SLAM中最常用的地图表示方法，具有计算效率和内存占用小的优势。

## 2. 特征点地图

### 2.1 表示方法

每个特征点包含：
- **3D坐标**：$\mathbf{p} \in \mathbb{R}^3$
- **描述子**：用于重识别的特征向量
- **协方差**：$\mathbf{\Sigma}_p \in \mathbb{R}^{3 \times 3}$
- **观测信息**：首次观测到的关键帧ID等

### 2.2 关键帧地图

关键帧是经过筛选的代表性帧，包含：
- 该帧的相机位姿 $\mathbf{T}_{kw}$
- 该帧观测到的所有特征点
- 该帧的BoW向量

### 2.3 共视图（Covisibility Graph）

共视图连接共享共同特征点的关键帧：

```
关键帧A ──共享10个点── 关键帧B
   │                      │
共享5个点              共享8个点
   │                      │
关键帧C ──共享3个点── 关键帧D
```

共视图对高效的数据关联和局部优化至关重要。

## 3. 关键帧选择策略

好的关键帧选择需要在精度和效率之间平衡：

| 策略 | 条件 | 优点 | 缺点 |
|------|------|------|------|
| 距离阈值 | 位移超过阈值 | 简单 | 针对静止 |
| 视差阈值 | 视角变化超过阈值 | 精度好 | 忽略平移 |
| 共视率阈值 | 共同特征点比例低于阈值 | 信息量保证 | 计算复杂 |

ORB-SLAM的策略：插入新关键帧，如果距上一个关键帧有足够视差且跟踪质量好。

## 4. 稀疏地图的优势与局限

**优势**：
- 内存占用小（适合大规模环境）
- 优化效率高（少量变量）
- 适合回环检测（BoW向量）
- 快速数据关联

**局限**：
- 只能用于定位，不能用于导航避障
- 无法进行稠密重建
- 低纹理场景效果差
- 不适合交互任务

## 5. 代表性系统

| 系统 | 地图特点 | 特征类型 |
|------|----------|----------|
| ORB-SLAM | 稀疏ORB特征地图 | ORB |
| PTAM | 稀疏特征地图 | FAST/Shi-Tomasi |
| VINS-Mono | 稀疏特征地图 | KLT跟踪点 |
| DSO | 稀疏直接法地图 | 梯度强的像素 |

## 6. 参考文献

1. Mur-Artal, R., Montiel, J. M. M., & Tardós, J. D. (2015). ORB-SLAM: A versatile and accurate monocular SLAM system. *IEEE Transactions on Robotics*, 31(5), 1147-1163.
2. Klein, G., & Murray, D. (2007). Parallel tracking and mapping for small AR workspaces. *ISMAR*.
3. Strasdat, H., Montiel, J. M. M., & Davison, A. J. (2010). Scale drift-aware large scale monocular SLAM. *RSS*.
