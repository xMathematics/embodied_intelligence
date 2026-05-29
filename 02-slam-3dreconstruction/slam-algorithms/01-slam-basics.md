# 2.1 SLAM算法基础

## 目录

- [1. SLAM问题定义](#1-slam问题定义)
- [2. 滤波方法](#2-滤波方法)
- [3. 优化方法](#3-优化方法)
- [4. 关键技术](#4-关键技术)
- [5. 重要论文详解](#5-重要论文详解)
- [6. 实践练习](#6-实践练习)

---

## 1. SLAM问题定义

### 1.1 问题描述

**同时定位与地图构建（SLAM）**：机器人在未知环境中移动时，同时估计自身位姿和构建环境地图。

**数学模型**：

$$\begin{cases}
\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k, \mathbf{w}_k) \\
\mathbf{z}_k = h(\mathbf{x}_k, \mathbf{y}_k, \mathbf{v}_k)
\end{cases}$$

其中：
- $\mathbf{x}_k$：机器人在时刻k的位姿
- $\mathbf{u}_k$：控制输入
- $\mathbf{w}_k$：过程噪声
- $\mathbf{z}_k$：观测
- $\mathbf{y}_k$：环境特征
- $\mathbf{v}_k$：观测噪声

### 1.2 SLAM的分类

| 分类标准 | 类型 | 说明 |
|---------|------|------|
| **传感器** | 视觉SLAM | 使用相机作为主要传感器 |
| | 激光SLAM | 使用激光雷达 |
| | 融合SLAM | 多传感器融合 |
| **方法** | 滤波SLAM | 使用卡尔曼滤波等方法 |
| | 优化SLAM | 使用图优化等方法 |
| **地图类型** | 稀疏SLAM | 只保留关键特征点 |
| | 稠密SLAM | 重建完整环境 |
| **相机数量** | 单目SLAM | 单相机 |
| | 双目SLAM | 双相机 |
| | RGB-D SLAM | RGB-D相机 |

---

## 2. 滤波方法

### 2.1 扩展卡尔曼滤波（EKF-SLAM）

**核心思想**：使用卡尔曼滤波同时估计机器人位姿和地图特征。

**状态向量**：

$$\mathbf{x} = [\mathbf{x}_{robot}^T, \mathbf{y}_1^T, \mathbf{y}_2^T, \ldots, \mathbf{y}_n^T]^T$$

**预测步骤**：

$$\hat{\mathbf{x}}_k^- = f(\hat{\mathbf{x}}_{k-1}^+, \mathbf{u}_k)$$
$$\mathbf{P}_k^- = \mathbf{F}_k \mathbf{P}_{k-1}^+ \mathbf{F}_k^T + \mathbf{Q}_k$$

**更新步骤**：

$$\mathbf{K}_k = \mathbf{P}_k^- \mathbf{H}_k^T (\mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R}_k)^{-1}$$
$$\hat{\mathbf{x}}_k^+ = \hat{\mathbf{x}}_k^- + \mathbf{K}_k (\mathbf{z}_k - h(\hat{\mathbf{x}}_k^-))$$
$$\mathbf{P}_k^+ = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_k^-$$

**缺点**：
- 计算复杂度随特征数量平方增长
- 线性化误差累积
- 难以处理大规模场景

### 2.2 粒子滤波（PF-SLAM）

**核心思想**：使用粒子集表示后验分布。

**算法流程**：

1. **采样**：从先验分布采样粒子
2. **权重更新**：根据观测更新粒子权重
3. **重采样**：根据权重重新采样粒子
4. **状态估计**：计算加权平均

**优点**：
- 可以处理非线性非高斯系统
- 无需线性化

**缺点**：
- 粒子退化问题
- 计算量大

### 2.3 信息滤波

**核心思想**：在信息空间中进行滤波，将协方差矩阵转换为信息矩阵。

**信息矩阵**：

$$\mathbf{\Omega} = \mathbf{P}^{-1}$$
$$\mathbf{\xi} = \mathbf{P}^{-1} \mathbf{x}$$

**优点**：
- 信息可以增量更新
- 适合分布式SLAM

---

## 3. 优化方法

### 3.1 光束法平差（Bundle Adjustment）

**核心思想**：同时优化所有相机位姿和3D点位置。

**目标函数**：

$$\min \sum_{i=1}^{m} \sum_{j=1}^{n} \| \mathbf{z}_{ij} - \pi(\mathbf{x}_i, \mathbf{y}_j) \|^2$$

其中 $\pi$ 是投影函数。

**求解方法**：
- 高斯-牛顿法
- Levenberg-Marquardt法

### 3.2 因子图优化

**核心思想**：将SLAM问题建模为因子图，使用图优化求解。

**因子类型**：
- **位姿因子**：描述相邻位姿之间的约束
- **观测因子**：描述位姿与特征之间的约束
- **回环因子**：描述回环检测带来的约束

**求解方法**：
- iSAM：增量平滑与建图
- iSAM2：改进的增量方法

### 3.3 位姿图优化

**核心思想**：只优化相机位姿，将地图作为常数。

**目标函数**：

$$\min \sum_{(i,j) \in \mathcal{E}} \| \Delta \mathbf{x}_{ij} - \mathbf{x}_j \oplus (-\mathbf{x}_i) \|^2$$

其中 $\oplus$ 是位姿复合操作。

---

## 4. 关键技术

### 4.1 关键帧选择

**策略**：
- 基于距离的选择
- 基于信息增益的选择
- 基于视差的选择

### 4.2 回环检测

**方法**：
- 词袋模型（Bag of Words）
- 深度学习方法

### 4.3 闭环融合

**方法**：
- 全局BA
- 位姿图优化
- 增量优化

---

## 5. 重要论文详解

### 5.1 A Solution to the Simultaneous Localization and Map Building (SLAM) Problem (1995)

**作者**：J. J. Leonard, H. F. Durrant-Whyte

**发表会议**：ICRA

**问题背景**：
- 早期SLAM研究主要关注定位或地图构建单一问题
- 需要一个统一的框架来同时解决这两个问题

**核心贡献**：
1. **EKF-SLAM框架**：首次提出使用扩展卡尔曼滤波同时估计机器人位姿和环境特征
2. **联合状态估计**：将机器人位姿和地图特征统一到一个状态向量中
3. **收敛性证明**：证明了EKF-SLAM在理论上是收敛的

**为什么重要**：
- 奠定了现代SLAM的理论基础
- 开创了同时定位与地图构建的研究方向
- 为后续的滤波SLAM方法提供了框架

### 5.2 Bundle Adjustment - A Modern Synthesis (2000)

**作者**：Triggs, McLauchlan, Hartley, Fitzgibbon

**发表会议**：ICCV Workshop

**详细分析**：[查看论文详解](../papers/bundle-adjustment.md)

**问题背景**：
- 光束法平差是多视图几何中的核心问题
- 需要一篇综述性文章来整理BA的理论和方法

**核心贡献**：
1. **统一框架**：将BA问题统一到最小二乘框架下
2. **高效求解**：介绍了各种高效的求解方法
3. **实现细节**：提供了详细的实现建议

**为什么重要**：
- 成为BA领域的经典综述
- 为后续的优化SLAM方法提供了理论基础
- 影响了ORB-SLAM等系统的设计

### 5.3 iSAM: Incremental Smoothing and Mapping (2008)

**作者**：Michael Kaess, Ananth Ranganathan, Frank Dellaert

**发表会议**：IEEE Transactions on Robotics

**问题背景**：
- 传统BA需要对整个问题进行优化，计算量随规模增长
- 需要一种增量的优化方法

**核心贡献**：
1. **增量优化**：只更新受影响的部分
2. **贝叶斯树**：使用贝叶斯树表示因子图
3. **高效更新**：利用矩阵稀疏性进行高效更新

**为什么重要**：
- 开创了增量优化的先河
- 为实时SLAM提供了高效的优化方法
- 影响了后续的iSAM2和其他增量优化方法

### 5.4 Probabilistic Robotics (2005)

**作者**：Sebastian Thrun, Wolfram Burgard, Dieter Fox

**发表形式**：书籍

**问题背景**：
- 需要一本系统讲解概率机器人学的教科书
- 整合当时分散的研究成果

**核心贡献**：
1. **概率框架**：将机器人学问题统一到概率框架下
2. **算法覆盖**：涵盖了SLAM、定位、建图等多个方面
3. **实践指导**：提供了大量的实现建议

**为什么重要**：
- 成为机器人学领域的经典教材
- 对SLAM研究产生了深远影响
- 被广泛用于研究生教学

---

## 6. 实践练习

### 练习1：理解EKF-SLAM

```python
import numpy as np

class EKF_SLAM:
    def __init__(self, initial_pose, initial_covariance):
        self.x = initial_pose  # 状态向量 [x, y, theta, landmarks...]
        self.P = initial_covariance  # 协方差矩阵
    
    def predict(self, u, Q):
        """预测步骤"""
        # 简化的运动模型
        x, y, theta = self.x[:3]
        v, omega = u
        
        dt = 1.0  # 时间步长
        new_x = x + v * np.cos(theta) * dt
        new_y = y + v * np.sin(theta) * dt
        new_theta = theta + omega * dt
        
        # 更新状态
        self.x[:3] = [new_x, new_y, new_theta]
        
        # 更新协方差（简化版本）
        F = np.eye(len(self.x))
        F[0, 2] = -v * np.sin(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        
        self.P = F @ self.P @ F.T + Q
    
    def update(self, z, R, landmark_idx):
        """更新步骤"""
        # 简化的观测模型
        x, y, theta = self.x[:3]
        landmark_x, landmark_y = self.x[3 + 2*landmark_idx : 3 + 2*landmark_idx + 2]
        
        # 预测观测
        dx = landmark_x - x
        dy = landmark_y - y
        predicted_range = np.sqrt(dx**2 + dy**2)
        predicted_bearing = np.arctan2(dy, dx) - theta
        
        # 观测雅可比矩阵
        H = np.zeros((2, len(self.x)))
        H[0, 0] = -dx / predicted_range
        H[0, 1] = -dy / predicted_range
        H[0, 3 + 2*landmark_idx] = dx / predicted_range
        H[0, 3 + 2*landmark_idx + 1] = dy / predicted_range
        
        H[1, 0] = dy / predicted_range**2
        H[1, 1] = -dx / predicted_range**2
        H[1, 2] = -1
        H[1, 3 + 2*landmark_idx] = -dy / predicted_range**2
        H[1, 3 + 2*landmark_idx + 1] = dx / predicted_range**2
        
        # 卡尔曼增益
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)
        
        # 更新状态和协方差
        innovation = np.array([z[0] - predicted_range, z[1] - predicted_bearing])
        self.x += K @ innovation
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P

# 示例使用
initial_pose = np.array([0.0, 0.0, 0.0])
initial_covariance = np.eye(3) * 0.1
ekf_slam = EKF_SLAM(initial_pose, initial_covariance)

# 模拟运动
u = np.array([1.0, 0.1])  # 速度和角速度
Q = np.eye(3) * 0.01  # 过程噪声协方差
ekf_slam.predict(u, Q)

print("预测后的状态:", ekf_slam.x)
print("预测后的协方差:", ekf_slam.P)
```

### 练习2：理解因子图优化

```python
import numpy as np
import matplotlib.pyplot as plt

class FactorGraph:
    def __init__(self):
        self.nodes = {}  # 节点（位姿和特征）
        self.factors = []  # 因子（约束）
    
    def add_node(self, node_id, value):
        self.nodes[node_id] = value
    
    def add_factor(self, node_ids, error_func, jacobian_func):
        self.factors.append({
            'nodes': node_ids,
            'error': error_func,
            'jacobian': jacobian_func
        })
    
    def optimize(self, max_iter=100, tolerance=1e-6):
        """高斯-牛顿法优化"""
        for _ in range(max_iter):
            # 构建增量方程
            n = sum(len(v) for v in self.nodes.values())
            H = np.zeros((n, n))
            b = np.zeros(n)
            
            for factor in self.factors:
                # 获取节点值
                values = [self.nodes[node_id] for node_id in factor['nodes']]
                
                # 计算误差和雅可比
                e = factor['error'](*values)
                J = factor['jacobian'](*values)
                
                # 累积到H和b
                idx = 0
                for i, node_id in enumerate(factor['nodes']):
                    node_size = len(self.nodes[node_id])
                    for j, node_id2 in enumerate(factor['nodes']):
                        node_size2 = len(self.nodes[node_id2])
                        H[idx:idx+node_size, idx+sum(len(self.nodes[k]) for k in factor['nodes'][:j]):
                            idx+sum(len(self.nodes[k]) for k in factor['nodes'][:j])+node_size2] += J[i*node_size:(i+1)*node_size, j*node_size2:(j+1)*node_size2] @ J[i*node_size:(i+1)*node_size, j*node_size2:(j+1)*node_size2].T
                    b[idx:idx+node_size] += J[i*node_size:(i+1)*node_size] @ e
                    idx += node_size
            
            # 求解增量
            delta = np.linalg.inv(H) @ b
            
            # 更新节点
            idx = 0
            for node_id in self.nodes:
                node_size = len(self.nodes[node_id])
                self.nodes[node_id] += delta[idx:idx+node_size]
                idx += node_size
            
            # 检查收敛
            if np.linalg.norm(delta) < tolerance:
                break

# 示例使用
fg = FactorGraph()
fg.add_node('pose0', np.array([0.0, 0.0]))
fg.add_node('pose1', np.array([1.0, 0.0]))

# 添加位姿约束
def error_func(x0, x1):
    return x1 - x0 - np.array([1.0, 0.0])

def jacobian_func(x0, x1):
    return np.array([[-np.eye(2), np.eye(2)]])

fg.add_factor(['pose0', 'pose1'], error_func, jacobian_func)

fg.optimize()
print("优化后的节点值:", fg.nodes)
```

---

**下一步**：[主流SLAM系统详解](2.2-mainstream-slam-systems.md)