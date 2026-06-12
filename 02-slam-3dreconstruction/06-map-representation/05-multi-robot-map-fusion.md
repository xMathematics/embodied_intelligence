# 6.5 多机器人地图融合

## 1. 概述

多机器人SLAM（Multi-Robot SLAM）涉及多个机器人协作探索环境并构建地图。地图融合是多机器人SLAM的核心问题之一。

## 2. 多机器人SLAM框架

### 2.1 集中式 vs 分布式

| 特性 | 集中式 | 分布式 |
|------|--------|--------|
| 计算中心 | 中心服务器 | 各机器人 |
| 通信需求 | 高 | 中 |
| 容错性 | 差 | 好 |
| 可扩展性 | 差 | 好 |
| 一致性 | 容易保证 | 需要一致算法 |
| 实时性 | 延迟高 | 好 |

### 2.2 通信方式

- **广播**：所有信息广播到所有机器人
- **点对点**：无线网络直接通信
- **间歇式**：当机器人相遇时交换信息

## 3. 地图合并

### 3.1 已知相对位姿

如果机器人知道相互间的相对位姿，地图合并就是简单的坐标系变换：

$$ \mathbf{p}^{\text{global}} = \mathbf{T}_{ri}^{\text{global}} \cdot \mathbf{p}^{\text{local}} $$

### 3.2 未知相对位姿

这是更一般的挑战。方法包括：

**基于地图匹配**：
- 寻找重叠区域
- 使用ICP或特征匹配对齐地图
- 验证对齐质量

**基于相遇检测**：
- 当机器人相遇时共享地图
- 计算相对位姿
- 传播相对位姿约束

**基于分布式优化**：
- 每个机器人维护局部地图
- 交换关键信息
- 分布式位姿图优化

### 3.3 DDF-SAM（2012）

分布式分布式因子图SLAM（DDF-SAM）：
1. 每个机器人维护自己的因子图
2. 将关键帧的边际信息发送给中心
3. 中心合并全局信息
4. 反馈全局信息到各机器人

## 4. 一致性维护

### 4.1 地图数据关联

- 同一特征被多个机器人观测到的关联
- 使用特征描述子或语义标签
- 几何一致性检查

### 4.2 冲突解决

当不同机器人地图冲突时：
- 使用RANSAC寻找一致性变换
- 基于置信度决定保留哪个信息
- 使用鲁棒优化方法处理异常

## 5. 代表性系统

| 系统 | 框架 | 特点 |
|------|------|------|
| C2TAM | 集中式 | 多机器人协同SLAM |
| DDF-SAM | 分布式 | 因子图框架 |
| Swarm-SLAM | 分布式 | 群体机器人SLAM |
| DOOR-SLAM | 分布式 | 关键帧交换 |
| Kimera-Multi | 分布式 | 度量-语义多机器人SLAM |

## 6. 参考文献

1. Cunningham, A., Paluri, M., & Dellaert, F. (2012). DDF-SAM: Fully distributed SLAM using constrained factor graphs. *IROS*.
2. Lajoie, P. Y., Ramtoula, B., Chang, Y., Carlone, L., & Beltrame, G. (2020). DOOR-SLAM: Distributed, online, and outlier resilient SLAM for robotic teams. *IEEE Robotics and Automation Letters*, 5(2), 1656-1663.
3. Tian, Y., Chang, Y., Arias, F. H., Nieto-Granda, C., How, J. P., & Carlone, L. (2022). Kimera-Multi: Robust, distributed, dense metric-semantic SLAM for multi-robot systems. *IEEE Transactions on Robotics*, 38(4), 2022-2038.
