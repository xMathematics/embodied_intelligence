# 6.2 稠密地图

## 1. 概述

稠密地图包含环境的完整几何信息，适用于导航、避障和交互任务。稠密地图可以有多种表示形式，包括点云、网格、体素等。

## 2. 点云地图

### 2.1 无组织点云

最基本的稠密地图形式，由大量3D点组成。

**特点**：
- 直接存储每个点的 $(x, y, z)$ 坐标
- 可带颜色 $(r, g, b)$ 或反射强度
- 无拓扑结构
- 内存占用大

**降采样**：使用体素网格滤波

```python
# 体素降采样
voxel_size = 0.05  # 5cm体素
voxel_grid = {key: centroid for key, centroid in points_by_voxel}
```

### 2.2 有组织点云

将点云按传感器扫描线组织（如LiDAR点云），支持快速邻域查询。

## 3. 网格地图

### 3.1 2D占据网格

$$ \text{Occupancy Grid Map: } m = \{P(\text{occupied} \mid \mathbf{z}_{1:k}, \mathbf{x}_{1:k})\} $$

每个网格单元独立维护被占据的概率。

**更新**：使用反演测量模型（Inverse Sensor Model）

$$ l_i = \log\frac{P(m_i = 1 \mid \mathbf{z}_{1:k})}{P(m_i = 0 \mid \mathbf{z}_{1:k})} $$

### 3.2 3D体素网格

将3D空间划分为规则体素网格：
- 支持直接索引访问
- 内存占用与体积成正比
- 适合室内小场景

## 4. OctoMap

### 4.1 八叉树结构

OctoMap（Hornung et al., 2013）使用八叉树（Octree）表示占据地图：

**特点**：
- **自适应分辨率**：占据区域精细，空闲区域粗糙
- **内存高效**：仅展开需要精细表示的节点
- **概率更新**：支持动态环境

**节点类型**：
- 内部节点：包含8个子节点
- 叶子节点：存储占据概率

**概率更新**：

$$ P(n \mid \mathbf{z}_{1:k}) = \left[1 + \frac{1-P(n \mid \mathbf{z}_{k})}{P(n \mid \mathbf{z}_{k})} \frac{1-P(n \mid \mathbf{z}_{1:k-1})}{P(n \mid \mathbf{z}_{1:k-1})} \frac{P(n)}{1-P(n)}\right]^{-1} $$

### 4.2 八叉树遍历

```python
def update_octree(node, point, origin):
    if node.is_leaf():
        node.update_probability()
    else:
        child = node.get_child_containing(point)
        update_octree(child, point, origin)
        node.update_from_children()
```

## 5. 参考文献

1. Hornung, A., Wurm, K. M., Bennewitz, M., Stachniss, C., & Burgard, W. (2013). OctoMap: An efficient probabilistic 3D mapping framework based on octrees. *Autonomous Robots*, 34(3), 189-206.
2. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
3. Hähnel, D., Burgard, W., & Thrun, S. (2003). Learning compact 3D models of indoor and outdoor environments with a mobile robot. *Robotics and Autonomous Systems*, 44(1), 15-27.
