# 4.2 紧耦合与松耦合

## 1. 概述

多传感器融合有两种基本范式：松耦合和紧耦合。两种方法在精度、鲁棒性和计算复杂度方面各有优劣。

## 2. 松耦合

### 2.1 定义

松耦合（Loose Coupling）将每个传感器独立处理，然后将各自的估计结果融合：

```
相机 → VO算法 → 位姿估计 ┐
                          ├→ 融合(KF/优化) → 最终位姿
IMU → 预积分/INS → 位姿估计 ┘
```

### 2.2 优点

- **模块化**：各传感器独立，可单独开发和调试
- **计算量小**：协方差融合计算量小
- **易于实现**：各子系统可独立开发和替换

### 2.3 缺点

- **信息损失**：中间估计丢失了底层观测信息
- **精度受限**：无法利用传感器间互补性校正
- **时序问题**：各传感器估计频率不同

## 3. 紧耦合

### 3.1 定义

紧耦合（Tight Coupling）在原始观测层面进行融合：

```
相机特征 ┐
          ├→ 联合优化 → 最终状态
IMU测量 ──┘
```

### 3.2 优点

- **信息最大保留**：直接使用原始观测
- **更高精度**：传感器之间可以互相校正
- **更好鲁棒性**：单传感器失效时仍可用
- **可处理退化解**：如相机纯旋转时IMU提供约束

### 3.3 缺点

- **计算量大**：状态维度高
- **实现复杂**：需联合处理所有传感器
- **耦合度高**：修改需考虑全部传感器

## 4. 对比

| 特性 | 松耦合 | 紧耦合 |
|------|--------|--------|
| 精度 | 中 | 高 |
| 计算量 | 低 | 高 |
| 鲁棒性 | 中 | 高 |
| 实现难度 | 低 | 高 |
| 信息利用率 | 低 | 高 |
| 退化处理 | 差 | 好 |
| 代表系统 | 简单KF融合 | VINS-Mono, MSCKF |

## 5. 混合方法

实际系统通常采用混合策略：
- **部分紧耦合**：对关键传感器紧耦合，其他松耦合
- **自适应耦合**：根据场景在松紧耦合间切换
- **分层融合**：低层紧耦合，高层松耦合

## 6. 参考文献

1. Weiss, S., & Siegwart, R. (2011). Real-time metric state estimation for modular vision-inertial systems. *ICRA*.
2. Leutenegger, S., Lynen, S., Bosse, M., Siegwart, R., & Furgale, P. (2015). Keyframe-based visual-inertial odometry using nonlinear optimization. *The International Journal of Robotics Research*, 34(3), 314-334.
3. Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A robust and versatile monocular visual-inertial state estimator. *IEEE Transactions on Robotics*, 34(4), 1004-1020.
