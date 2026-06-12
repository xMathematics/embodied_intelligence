# 6.3 TSDF与ESDF

## 1. 概述

截断符号距离函数（Truncated Signed Distance Function, TSDF）和欧几里得符号距离函数（Euclidean Signed Distance Function, ESDF）是3D重建和机器人导航中重要的隐式地图表示方法。

## 2. TSDF（截断符号距离函数）

### 2.1 定义

TSDF在体素网格中存储每个点到最近表面的有符号距离，并截断到一定范围：

$$ \text{TSDF}(\mathbf{v}) = \max(-1, \min(1, \frac{\text{SDF}(\mathbf{v})}{t})) $$

其中 $t$ 是截断阈值，SDF 是真实符号距离。

- **正值**：体素在表面前方
- **负值**：体素在表面后方
- **零值**：体素位于表面（零水平集）

### 2.2 融合更新

多帧TSDF的加权融合：

$$ \text{TSDF}_{\text{fused}}(\mathbf{v}) = \frac{\sum_{i} w_i(\mathbf{v}) \cdot \text{TSDF}_i(\mathbf{v})}{\sum_{i} w_i(\mathbf{v})} $$

权重通常与测量角度有关：$w_i = \cos(\theta)$ 或 $w_i \propto 1/\sigma^2$。

### 2.3 KinectFusion中的TSDF

KinectFusion（Newcombe et al., 2011）使用TSDF实现实时RGB-D重建：

1. **深度图转换**：将深度图转换为3D点云
2. **相机跟踪**：ICP配准
3. **TSDF融合**：将新深度融合到全局TSDF
4. **表面提取**：从TSDF提取三角网格（Raycasting或Marching Cubes）

**硬件实现**：使用GPU并行处理体素，实现实时性能。

### 2.4 Voxel Hashing

为了支持大规模场景，使用哈希表管理稀疏体素块：

- **空间哈希**：$h(x, y, z) = (x \cdot p_1 \oplus y \cdot p_2 \oplus z \cdot p_3) \mod N$
- **体素块**：$8^3$ 或 $16^3$ 体素为一个块
- **动态分配**：只在表面附近分配体素

**代表性系统**：
- **InfiniTAM**：开源的大规模TSDF融合
- **BundleFusion**：全局一致的TSDF融合
- **Voxblox**：用于导航的TSDF

## 3. ESDF（欧几里得符号距离函数）

### 3.1 定义

ESDF存储每个体素到最近障碍物的欧几里得距离。与TSDF不同，ESDF在自由空间中也有定义。

### 3.2 ESDF在导航中的应用

ESDF在机器人运动规划中至关重要：
- 碰撞检查：$d(\mathbf{p}) > \text{安全半径}$
- 梯度信息：$\nabla \text{ESDF}(\mathbf{p})$ 用于轨迹优化
- 势场导航

### 3.3 ESDF生成

**增量ESDF更新（Voxblox）**：

从TSDF生成ESDF：
1. 从TSDF零水平集提取障碍物
2. 使用BFS/Dijkstra传播距离
3. 更新局部受影响的体素

**批处理方法**：
- 使用快速行进法（FMM）
- 使用距离变换算法（Meijster et al.）

## 4. TSDF vs ESDF

| 特性 | TSDF | ESDF |
|------|------|------|
| 表面表示 | 隐式（零水平集） | 通过距离梯度 |
| 自由空间距离 | 无 | 有 |
| 更新速度 | 快（局部） | 慢（全局传播） |
| 主要用途 | 重建 | 导航 |
| 内存 | 表面附近 | 全局 |

## 5. 参考文献

1. Newcombe, R. A., et al. (2011). KinectFusion: Real-time dense surface mapping and tracking. *ISMAR*.
2. Curless, B., & Levoy, M. (1996). A volumetric method for building complex models from range images. *SIGGRAPH*.
3. Oleynikova, H., Taylor, Z., Fehr, M., Siegwart, R., & Nieto, J. (2017). Voxblox: Incremental 3D Euclidean signed distance fields for on-board MAV planning. *IROS*.
4. Nießner, M., Zollhöfer, M., Izadi, S., & Stamminger, M. (2013). Real-time 3D reconstruction at scale using voxel hashing. *ACM Transactions on Graphics*, 32(6).
