# 11.4 主动SLAM与探索

## 1. 概述

主动SLAM（Active SLAM）在SLAM过程中主动控制机器人的运动，以优化定位精度和地图构建质量。与传统被动SLAM不同，主动SLAM考虑"如何移动"以最大化SLAM性能和信息获取。

## 2. 主动SLAM问题形式化

### 2.1 问题定义

主动SLAM的优化目标：

$$ \mathbf{u}_{1:T}^* = \arg\max_{\mathbf{u}_{1:T}} \mathbb{E}\left[ \underbrace{I(\mathbf{x}_{1:T}, \mathbf{m}; \mathbf{z}_{1:T})}_{\text{信息增益}} - \underbrace{C(\mathbf{u}_{1:T})}_{\text{运动代价}} \right] $$

其中：
- $\mathbf{u}_{1:T}$：控制输入序列
- $I$：信息增益（定位精度、地图质量）
- $C$：运动代价（能耗、时间、风险）

### 2.2 探索与利用的权衡

主动SLAM需要平衡两个目标：

| 目标 | 描述 | 策略 |
|------|------|------|
| **探索** | 探索未访问区域 | 前往未知区域 |
| **利用** | 优化已知区域的地图 | 重新访问已建图区域 |
| **回环** | 检测回环消除漂移 | 规划回到已访问位置 |

**权衡方法**：
- **ε-贪婪**：以概率ε探索，1-ε利用
- **置信度边界**：选择信息增益+不确定性的上界
- **Pareto最优**：多目标优化

## 3. 信息论方法

### 3.1 信息增益

信息增益衡量SLAM不确定性的减少：

$$ I(\mathbf{x}, \mathbf{m}; \mathbf{z}) = H(\mathbf{x}, \mathbf{m}) - H(\mathbf{x}, \mathbf{m} \mid \mathbf{z}) $$

- $H$：信息熵
- 降低不确定性最大的行动是最优的

### 3.2 互信息方法

**基于互信息的探索**：

$$ \text{MI}(\mathbf{x}, \mathbf{m}) = \iint P(\mathbf{x}, \mathbf{m}) \log \frac{P(\mathbf{x}, \mathbf{m})}{P(\mathbf{x})P(\mathbf{m})} d\mathbf{x} d\mathbf{m} $$

高互信息区域 → 当前SLAM最不确定的区域 → 优先探索

### 3.3 不确定性量化

**定位不确定性**：协方差矩阵的迹/行列式

$$ \text{Uncertainty}_{\text{pose}} = \text{tr}(\mathbf{\Sigma}_{\text{pose}}) $$

**地图不确定性**：
$$ \text{Uncertainty}_{\text{map}} = \sum_i \text{tr}(\mathbf{\Sigma}_{\text{landmark}_i}) $$

## 4. 前沿探索

### 4.1 基本概念

前沿（Frontier）是已知自由空间和未知空间之间的边界。前沿探索驱动机器人向未知区域移动。

**前沿检测**：
1. 在占据网格中识别边界
2. 对边界点聚类
3. 计算每个前沿中心作为候选目标

**前沿选择**：
$$ \text{Frontier}^* = \arg\max_{f \in \mathcal{F}} \left[ \frac{\text{Information}(f)}{\text{Cost}(f)} \right] $$

### 4.2 改进的前沿探索

**基于信息的前沿**：考虑探索前沿的预期信息增益

**拓扑前沿**：基于拓扑图的前沿检测

**语义前沿**：结合语义信息的前沿选择（优先探索门、走廊等有语义价值的区域）

## 5. 闭环规划

### 5.1 闭环的必要性

漂移累积是SLAM的根本问题，闭环是消除漂移的关键。主动SLAM应有意识规划回环路径。

**闭环收益**：
- 定位误差的全局校正
- 地图一致性的提升
- 增加系统鲁棒性

### 5.2 闭环规划方法

**基于期望误差的方法**：
1. 预测未来漂移趋势
2. 规划回环路径降低总误差
3. 在探索和闭环间优化平衡

**回环候选评估**：

$$ \text{LoopValue}(l) = \frac{\text{ErrorReduction}(l)}{\text{PathCost}(l)} $$

### 5.3 回环触发策略

| 策略 | 描述 |
|------|------|
| **漂移触发** | 当估计不确定性超过阈值时触发 |
| **距离触发** | 距离上次闭环超过阈值时触发 |
| **信息触发** | 当前往区域的信息增益大时先探索后闭环 |

## 6. 协同探索

### 6.1 多机器人主动SLAM

**集中式探索规划**：
- 中心服务器分配探索任务
- 全局优化分配方案
- 避免重复探索

**分布式探索**：
- 各机器人独立决策
- 通过通信交换地图信息
- 协商探索分配

### 6.2 通信约束

在通信受限环境中的主动SLAM：
- 考虑通信覆盖范围规划
- 在信息增益和通信质量间权衡
- 在无明显通信时自主探索

## 7. 深度主动SLAM

### 7.1 强化学习方法

使用强化学习学习主动SLAM策略：

**状态**：SLAM状态（位姿、地图、不确定性）
**动作**：控制目标（方向、速度）
**奖励**：信息增益 - 运动代价

**DQN-ActiveSLAM**：使用深度Q网络学习主动探索策略。

### 7.2 预测性主动SLAM

利用深度学习预测未来观测的信息量：

1. 从历史数据学习预测模型
2. 对未访问区域预测信息含量
3. 引导探索到高价值区域

## 8. 性能评估

| 指标 | 描述 |
|------|------|
| **建图覆盖率** | 已建图面积/总面积 |
| **定位精度** | ATE/RPE |
| **探索效率** | 单位时间建图面积 |
| **闭环频率** | 每km闭环次数 |
| **能量效率** | 每焦耳建图面积 |

## 9. 参考文献

1. Chen, C., et al. (2022). Active SLAM: A review. *IEEE Access*, 10, 21536-21556.
2. Stachniss, C., Grisetti, G., & Burgard, W. (2005). Information gain-based exploration using Rao-Blackwellized particle filters. *RSS*.
3. Yamauchi, B. (1997). A frontier-based approach for autonomous exploration. *IEEE International Symposium on Computational Intelligence in Robotics and Automation*.
4. Kim, A., & Eustice, R. M. (2015). Active visual SLAM for robotic area coverage: Theory and experiment. *The International Journal of Robotics Research*, 34(4-5), 457-475.
5. Devin, C., et al. (2018). Deep active SLAM. *IROS*.
