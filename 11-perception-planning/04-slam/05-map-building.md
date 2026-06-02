# 4.5 地图构建

## 目录

- [1. 地图表示方法](#1-地图表示方法)
- [2. 栅格地图](#2-栅格地图)
- [3. 点云地图](#3-点云地图)
- [4. 语义地图](#4-语义地图)
- [5. 拓扑地图与混合地图](#5-拓扑地图与混合地图)
- [6. 地图优化与压缩](#6-地图优化与压缩)
- [7. 实践练习](#7-实践练习)

---

## 1. 地图表示方法

### 1.1 为什么需要地图

**问题的提出**

SLAM系统不仅需要估计机器人位姿，还需要构建环境地图。地图在机器人系统中承担多种重要角色：

1. **定位**：地图为机器人提供参考，实现全局定位
2. **导航**：路径规划需要地图信息
3. **避障**：实时避障需要知道障碍物位置
4. **人机交互**：人类需要理解机器人所在环境
5. **长期任务**：多会话SLAM需要地图复用

**地图表示的核心挑战**：
- 精度 vs 存储效率的权衡
- 静态假设 vs 动态环境
- 全局一致性 vs 局部更新效率

### 1.2 地图分类体系

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2

class MapTypes:
    """
    地图类型分类
    
    论文核心思想（Thrun, 2002; Cadena et al., 2016）:
    地图表示的选择取决于应用场景和任务需求。
    没有一种地图表示适用于所有场景。
    """
    
    @staticmethod
    def sparse_map():
        """
        稀疏地图
        
        问题提出：
        稠密地图存储和计算开销大，对于某些任务（如定位）
        只需要关键特征点即可。
        
        解决方案：
        只存储特征点和路标，不存储完整几何信息。
        
        优点：
        - 存储空间小
        - 计算效率高
        - 适合定位和回环检测
        
        缺点：
        - 无法用于导航和避障
        - 缺乏环境几何信息
        
        代表系统：ORB-SLAM、PTAM
        """
        return {
            "name": "稀疏地图",
            "elements": ["特征点", "路标", "关键帧位姿"],
            "size": "小 (MB级别)",
            "use_case": ["定位", "回环检测", "AR应用"],
            "examples": ["ORB-SLAM地图", "PTAM地图"],
            "storage": "特征点坐标 + 描述子 + 关键帧位姿"
        }
    
    @staticmethod
    def dense_map():
        """
        稠密地图
        
        问题提出：
        稀疏地图缺乏几何信息，无法支持导航、避障等任务。
        
        解决方案：
        存储环境的完整几何信息（每个像素/体素）。
        
        优点：
        - 信息完整
        - 可用于重建、导航、避障
        - 可视化效果好
        
        缺点：
        - 存储空间大
        - 计算开销高
        - 需要GPU加速
        
        代表系统：DTAM、ElasticFusion、InfiniTAM
        """
        return {
            "name": "稠密地图",
            "elements": ["像素/体素", "表面", "体积"],
            "size": "大 (GB级别)",
            "use_case": ["重建", "导航", "避障", "VR/AR"],
            "examples": ["TSDF地图", "Surfel地图", "占据栅格"],
            "storage": "每个体素的占据/距离/颜色信息"
        }
    
    @staticmethod
    def metric_map():
        """
        度量地图
        
        精确表示环境的几何信息，使用全局坐标系。
        """
        return {
            "name": "度量地图",
            "coordinate": "全局坐标系",
            "properties": ["精确距离", "绝对位置", "几何形状"],
            "examples": ["栅格地图", "点云地图", "TSDF地图"],
            "applications": ["路径规划", "精确导航", "操作任务"]
        }
    
    @staticmethod
    def topological_map():
        """
        拓扑地图
        
        问题提出：
        度量地图在大规模环境下存储和规划效率低。
        
        解决方案：
        用图结构表示环境，节点表示地点，边表示连通性。
        
        优点：
        - 适合大规模环境
        - 规划效率高
        - 符合人类认知
        
        缺点：
        - 缺乏精确几何信息
        - 拓扑节点识别困难
        
        代表系统：FAB-MAP、SeqSLAM
        """
        return {
            "name": "拓扑地图",
            "elements": ["节点(地点)", "边(路径)"],
            "properties": ["连通性", "层次结构"],
            "examples": ["导航图", "场景图", "语义拓扑图"],
            "applications": ["全局规划", "粗定位", "场景理解"]
        }
    
    @staticmethod
    def semantic_map():
        """
        语义地图
        
        问题提出：
        纯几何地图缺乏对环境的理解，无法支持高层任务。
        
        解决方案：
        在几何地图上叠加语义信息（物体类别、属性等）。
        
        优点：
        - 支持高层任务规划
        - 人机交互友好
        - 动态物体识别
        
        缺点：
        - 需要深度学习模型
        - 语义标注开销大
        - 类别定义依赖先验
        
        代表系统：SemanticFusion、Kimera
        """
        return {
            "name": "语义地图",
            "elements": ["几何", "语义标签", "物体实例"],
            "properties": ["可理解性", "层次性"],
            "examples": ["语义点云", "语义网格", "场景图"],
            "applications": ["任务规划", "人机交互", "智能导航"]
        }
```

### 1.3 地图表示的选择策略

```python
class MapSelectionStrategy:
    """地图选择策略"""
    
    @staticmethod
    def select_map_type(task_requirements):
        """
        根据任务需求选择地图类型
        
        参数:
            task_requirements: {
                'need_navigation': bool,
                'need_collision_avoidance': bool,
                'scale': str,  # 'small', 'medium', 'large'
                'computation_budget': str,  # 'low', 'medium', 'high'
                'need_semantic': bool
            }
        """
        req = task_requirements
        
        if req.get('need_semantic'):
            return "语义地图 + 几何地图"
        
        if req.get('scale') == 'large' and not req.get('need_collision_avoidance'):
            return "拓扑地图 + 稀疏度量地图"
        
        if req.get('need_collision_avoidance') or req.get('need_navigation'):
            if req.get('computation_budget') == 'high':
                return "稠密地图 (TSDF/点云)"
            else:
                return "栅格地图"
        
        return "稀疏地图"
    
    @staticmethod
    def comparison_table():
        """地图类型对比表"""
        return {
            "稀疏地图": {
                "存储": "★★★★★",
                "定位精度": "★★★★☆",
                "导航支持": "★☆☆☆☆",
                "重建质量": "★☆☆☆☆",
                "计算效率": "★★★★★"
            },
            "栅格地图": {
                "存储": "★★★☆☆",
                "定位精度": "★★★☆☆",
                "导航支持": "★★★★☆",
                "重建质量": "★★☆☆☆",
                "计算效率": "★★★★☆"
            },
            "点云地图": {
                "存储": "★★☆☆☆",
                "定位精度": "★★★★★",
                "导航支持": "★★★★☆",
                "重建质量": "★★★★☆",
                "计算效率": "★★☆☆☆"
            },
            "TSDF地图": {
                "存储": "★★★☆☆",
                "定位精度": "★★★★☆",
                "导航支持": "★★★★★",
                "重建质量": "★★★★★",
                "计算效率": "★★☆☆☆"
            },
            "语义地图": {
                "存储": "★★☆☆☆",
                "定位精度": "★★★☆☆",
                "导航支持": "★★★★★",
                "重建质量": "★★★★★",
                "计算效率": "★☆☆☆☆"
            }
        }
```

---

## 2. 栅格地图

### 2.1 占据栅格地图原理

**论文核心思想（Elfes, 1989; Moravec, 1988）**

占据栅格地图将环境离散化为网格，每个格子存储被占据的概率。使用概率对数优势（log-odds）表示，便于增量更新。

```python
class OccupancyGridMap:
    """
    占据栅格地图
    
    论文核心思想（Elfes, 1989）:
    使用概率对数优势表示占据状态，实现高效的贝叶斯更新。
    
    数学推导：
    设P(occ|z)为观测到z时格子被占据的概率
    对数优势: l = log(P(occ|z) / (1 - P(occ|z)))
    
    更新公式: l_t = l_{t-1} + l_{sensor}
    
    其中l_{sensor} = log(P(occ|z) / (1 - P(occ|z))) - log(P(occ) / (1 - P(occ)))
    
    优点：
    1. 增量更新高效（加法而非乘法）
    2. 数值稳定性好
    3. 可以表示未知区域（l=0）
    
    缺点：
    1. 分辨率固定
    2. 内存消耗随环境尺寸平方增长
    3. 不考虑传感器模型不确定性
    """
    
    def __init__(self, resolution=0.05, width=100, height=100, 
                 prob_occ=0.7, prob_free=0.3, prob_prior=0.5):
        """
        初始化栅格地图
        
        参数:
            resolution: 栅格分辨率（米/格）
            width, height: 栅格尺寸
            prob_occ: 观测到占据时的概率
            prob_free: 观测到空闲时的概率
            prob_prior: 先验概率
        """
        self.resolution = resolution
        self.width = width
        self.height = height
        
        # 原点在地图中心
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2
        
        # 概率对数优势
        self.log_odds = np.zeros((height, width))
        
        # 概率参数
        self.prob_occ = prob_occ
        self.prob_free = prob_free
        self.prob_prior = prob_prior
        
        # 计算对数优势值
        self.l_occ = np.log(prob_occ / (1 - prob_occ))
        self.l_free = np.log(prob_free / (1 - prob_free))
        self.l_prior = np.log(prob_prior / (1 - prob_prior))
        
        # 设置截断值防止数值溢出
        self.log_odds_min = -10.0
        self.log_odds_max = 10.0
    
    def world_to_grid(self, x, y):
        """世界坐标转栅格坐标"""
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy
    
    def grid_to_world(self, gx, gy):
        """栅格坐标转世界坐标（格子中心）"""
        x = gx * self.resolution + self.origin_x + self.resolution / 2
        y = gy * self.resolution + self.origin_y + self.resolution / 2
        return x, y
    
    def is_inside(self, gx, gy):
        """检查坐标是否在地图范围内"""
        return 0 <= gx < self.width and 0 <= gy < self.height
    
    def update_cell(self, gx, gy, is_occupied):
        """
        更新单个栅格
        
        参数:
            gx, gy: 栅格坐标
            is_occupied: True表示占据，False表示空闲
        """
        if not self.is_inside(gx, gy):
            return
        
        # 对数优势更新
        if is_occupied:
            self.log_odds[gy, gx] += self.l_occ - self.l_prior
        else:
            self.log_odds[gy, gx] += self.l_free - self.l_prior
        
        # 截断防止溢出
        self.log_odds[gy, gx] = np.clip(
            self.log_odds[gy, gx], 
            self.log_odds_min, 
            self.log_odds_max
        )
    
    def get_probability(self, gx, gy):
        """获取占据概率"""
        if not self.is_inside(gx, gy):
            return self.prob_prior
        
        odds = np.exp(self.log_odds[gy, gx])
        return odds / (odds + 1)
    
    def get_probability_map(self):
        """获取概率地图"""
        odds = np.exp(self.log_odds)
        return odds / (odds + 1)
    
    def get_occupancy_status(self, gx, gy, threshold=0.5):
        """
        获取占据状态
        
        返回: 'occupied', 'free', 或 'unknown'
        """
        if not self.is_inside(gx, gy):
            return 'unknown'
        
        prob = self.get_probability(gx, gy)
        
        if prob > threshold + 0.1:
            return 'occupied'
        elif prob < threshold - 0.1:
            return 'free'
        else:
            return 'unknown'
    
    def update_scan(self, sensor_pose, ranges, angles, max_range=10.0):
        """
        更新激光扫描
        
        参数:
            sensor_pose: (x, y, theta) 传感器位姿
            ranges: 距离数组
            angles: 角度数组（相对于传感器）
            max_range: 最大有效距离
        """
        sx, sy, st = sensor_pose
        
        for r, a in zip(ranges, angles):
            # 跳过无效测量
            if r <= 0 or r > max_range:
                continue
            
            # 计算终点全局坐标
            a_global = st + a
            end_x = sx + r * np.cos(a_global)
            end_y = sy + r * np.sin(a_global)
            
            # Bresenham射线算法更新空闲格子
            self._update_ray(sx, sy, end_x, end_y, r)
    
    def _update_ray(self, x0, y0, x1, y1, range_val):
        """
        使用Bresenham算法更新射线
        
        射线上的格子标记为空闲，终点标记为占据
        """
        # 转换为栅格坐标
        x0g, y0g = self.world_to_grid(x0, y0)
        x1g, y1g = self.world_to_grid(x1, y1)
        
        # Bresenham算法
        dx = abs(x1g - x0g)
        dy = abs(y1g - y0g)
        sx = 1 if x0g < x1g else -1
        sy = 1 if y0g < y1g else -1
        err = dx - dy
        
        x, y = x0g, y0g
        
        while True:
            # 检查是否到达终点
            if x == x1g and y == y1g:
                # 终点：占据
                self.update_cell(x, y, True)
                break
            else:
                # 路径：空闲
                self.update_cell(x, y, False)
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            
            # 安全检查
            if not self.is_inside(x, y):
                break
    
    def inflate_obstacles(self, inflation_radius):
        """
        膨胀障碍物
        
        用于路径规划时考虑机器人尺寸
        """
        from scipy.ndimage import binary_dilation
        
        # 获取占据地图
        occ_prob = self.get_probability_map()
        occupied = (occ_prob > 0.5).astype(np.uint8)
        
        # 膨胀
        radius_pixels = int(inflation_radius / self.resolution)
        structure = np.ones((2*radius_pixels+1, 2*radius_pixels+1))
        inflated = binary_dilation(occupied, structure)
        
        # 更新地图
        for y in range(self.height):
            for x in range(self.width):
                if inflated[y, x] and not occupied[y, x]:
                    self.log_odds[y, x] = self.l_occ


class AdaptiveOccupancyGrid:
    """
    自适应分辨率栅格地图
    
    问题提出：
    固定分辨率栅格地图在空旷区域浪费内存，
    在复杂区域分辨率不足。
    
    解决方案：
    使用四叉树/八叉树实现自适应分辨率。
    
    代表实现：OctoMap
    """
    
    def __init__(self, min_resolution=0.05, max_resolution=0.5):
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.cells = {}  # (x, y, level) -> log_odds
    
    def get_cell_resolution(self, level):
        """获取指定层级的分辨率"""
        return self.max_resolution / (2 ** level)
    
    def update_point(self, x, y, occupied, level=0):
        """递归更新点"""
        resolution = self.get_cell_resolution(level)
        
        # 检查是否需要细分
        if resolution > self.min_resolution and level < 5:
            # 细分到下一级
            self.update_point(x, y, occupied, level + 1)
        else:
            # 更新当前格子
            key = (int(x / resolution), int(y / resolution), level)
            if key not in self.cells:
                self.cells[key] = 0.0
            
            if occupied:
                self.cells[key] += 0.5
            else:
                self.cells[key] -= 0.3
```

### 2.2 距离地图与ESDF

**论文核心思想（Oleynikova et al., 2017）**

欧氏距离场（ESDF）存储每个点到最近障碍物的距离，对路径规划至关重要。

```python
class EuclideanDistanceField:
    """
    欧氏距离场 (ESDF)
    
    论文核心思想（Oleynikova et al., 2017）:
    ESDF存储每个自由点到最近障碍物的欧氏距离，
    用于快速碰撞检测和梯度计算。
    
    构建方法：
    1. 从占据栅格提取表面点
    2. 对每个自由点计算到所有表面的距离
    3. 或使用快速距离变换算法
    
    应用：
    - 碰撞检测
    - 路径规划（CHOMP、STOMP）
    - 梯度计算
    """
    
    def __init__(self, occupancy_map):
        self.occ_map = occupancy_map
        self.distance = None
        self.gradient = None
    
    def compute_distance_transform(self):
        """
        计算欧氏距离变换
        
        使用scipy的distance_transform_edt
        """
        from scipy.ndimage import distance_transform_edt
        
        # 获取占据概率地图
        prob_map = self.occ_map.get_probability_map()
        
        # 二值化：占据概率>0.5视为障碍物
        occupied = (prob_map > 0.5).astype(np.uint8)
        
        # 计算到最近0的距离（即到障碍物的距离）
        self.distance = distance_transform_edt(1 - occupied)
        
        # 转换到世界坐标尺度
        self.distance *= self.occ_map.resolution
        
        return self.distance
    
    def compute_gradient(self):
        """
        计算距离梯度
        
        梯度指向距离增加最快的方向（远离障碍物）
        """
        if self.distance is None:
            self.compute_distance_transform()
        
        # 使用numpy的梯度函数
        gy, gx = np.gradient(self.distance)
        
        self.gradient = np.stack([gx, gy], axis=-1)
        return self.gradient
    
    def get_distance_at(self, x, y):
        """获取指定位置的距离"""
        if self.distance is None:
            return None
        
        gx, gy = self.occ_map.world_to_grid(x, y)
        
        if not self.occ_map.is_inside(gx, gy):
            return float('inf')
        
        return self.distance[gy, gx]
    
    def check_collision(self, x, y, robot_radius):
        """碰撞检测"""
        dist = self.get_distance_at(x, y)
        return dist is not None and dist < robot_radius
    
    def get_nearest_obstacle(self, x, y):
        """获取最近障碍物位置"""
        # 简化实现
        gx, gy = self.occ_map.world_to_grid(x, y)
        
        if not self.occ_map.is_inside(gx, gy):
            return None
        
        # 在局部窗口内搜索
        search_radius = int(2.0 / self.occ_map.resolution)
        min_dist = float('inf')
        nearest = None
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx, ny = gx + dx, gy + dy
                
                if self.occ_map.is_inside(nx, ny):
                    if self.occ_map.get_probability(nx, ny) > 0.5:
                        ox, oy = self.occ_map.grid_to_world(nx, ny)
                        dist = np.sqrt((ox - x)**2 + (oy - y)**2)
                        
                        if dist < min_dist:
                            min_dist = dist
                            nearest = (ox, oy)
        
        return nearest


class SignedDistanceField:
    """
    有符号距离场 (SDF)
    
    正值：到最近障碍物的距离（自由空间）
    负值：到最近自由空间的距离（障碍物内部）
    零：表面
    """
    
    def __init__(self, grid_size, resolution):
        self.grid_size = grid_size
        self.resolution = resolution
        self.sdf = np.zeros(grid_size)
    
    def from_point_cloud(self, points, normals=None):
        """
        从点云构建SDF
        
        参数:
            points: 表面点
            normals: 法向量（用于确定符号）
        """
        # 简化的SDF构建
        # 实际使用TSDF或其他方法
        pass
    
    def interpolate(self, point):
        """三线性插值获取SDF值"""
        # 简化的插值
        x, y, z = point / self.resolution
        
        # 获取周围8个格子的值
        x0, y0, z0 = int(x), int(y), int(z)
        
        # 边界检查
        if not (0 <= x0 < self.grid_size[0] - 1 and
                0 <= y0 < self.grid_size[1] - 1 and
                0 <= z0 < self.grid_size[2] - 1):
            return 0.0
        
        # 三线性插值
        dx, dy, dz = x - x0, y - y0, z - z0
        
        c000 = self.sdf[x0, y0, z0]
        c001 = self.sdf[x0, y0, z0+1]
        c010 = self.sdf[x0, y0+1, z0]
        c011 = self.sdf[x0, y0+1, z0+1]
        c100 = self.sdf[x0+1, y0, z0]
        c101 = self.sdf[x0+1, y0, z0+1]
        c110 = self.sdf[x0+1, y0+1, z0]
        c111 = self.sdf[x0+1, y0+1, z0+1]
        
        c00 = c000 * (1 - dx) + c100 * dx
        c01 = c001 * (1 - dx) + c101 * dx
        c10 = c010 * (1 - dx) + c110 * dx
        c11 = c011 * (1 - dx) + c111 * dx
        
        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy
        
        return c0 * (1 - dz) + c1 * dz
```

---

## 3. 点云地图

### 3.1 TSDF地图原理

**论文核心思想（Curless & Levoy, 1996; Newcombe et al., 2011）**

截断有符号距离函数（TSDF）是稠密重建的主流方法，被KinectFusion等系统采用。

```python
class TSDFMap:
    """
    TSDF地图 (Truncated Signed Distance Function)
    
    论文核心思想（Curless & Levoy, 1996; Newcombe et al., 2011）:
    在每个体素存储到最近表面的截断有符号距离。
    正值表示在表面前方，负值表示在后方，零表示表面。
    
    数学定义：
    TSDF(x) = sign * min(|D(x)|/trunc, 1)
    其中D(x)是到最近表面的距离
    
    更新公式（加权平均）：
    D_new = (D_old * W_old + D_obs * W_obs) / (W_old + W_obs)
    W_new = min(W_old + W_obs, W_max)
    
    优点：
    1. 隐式表面表示
    2. 抗噪声能力强
    3. 可提取高质量表面
    
    缺点：
    1. 内存消耗大
    2. 需要已知截断距离
    3. 对透明/反光表面敏感
    
    代表系统：KinectFusion, InfiniTAM, Voxblox
    """
    
    def __init__(self, voxel_size=0.05, truncation_distance=0.3, max_weight=100):
        """
        初始化TSDF地图
        
        参数:
            voxel_size: 体素大小（米）
            truncation_distance: 截断距离
            max_weight: 最大权重
        """
        self.voxel_size = voxel_size
        self.truncation_distance = truncation_distance
        self.max_weight = max_weight
        
        # 使用字典存储稀疏体素 (x, y, z) -> (tsdf_value, weight, color)
        self.voxels = {}
        
        # 边界框
        self.bounds_min = np.array([float('inf')] * 3)
        self.bounds_max = np.array([float('-inf')] * 3)
    
    def to_voxel_coord(self, point):
        """点转体素坐标（整数索引）"""
        return tuple(np.floor(point / self.voxel_size).astype(int))
    
    def voxel_to_world(self, voxel_coord):
        """体素坐标转世界坐标（体素中心）"""
        return (np.array(voxel_coord) + 0.5) * self.voxel_size
    
    def integrate_scan(self, pointcloud, sensor_pose, colors=None):
        """
        集成深度扫描
        
        参数:
            pointcloud: 点云 (N, 3)
            sensor_pose: 传感器位姿 (4, 4)
            colors: 颜色 (N, 3)，可选
        """
        # 传感器原点在全局坐标系
        origin = sensor_pose[:3, 3]
        
        # 变换点云到世界坐标系
        R = sensor_pose[:3, :3]
        points_world = (R @ pointcloud.T).T + origin
        
        # 更新边界框
        self.bounds_min = np.minimum(self.bounds_min, points_world.min(axis=0))
        self.bounds_max = np.maximum(self.bounds_max, points_world.max(axis=0))
        
        # 对每个点，沿光线采样并更新TSDF
        for i, point in enumerate(points_world):
            # 计算到点的向量
            vec = point - origin
            dist_to_point = np.linalg.norm(vec)
            
            if dist_to_point < 1e-6:
                continue
            
            # 光线方向
            direction = vec / dist_to_point
            
            # 沿光线采样（从传感器到截断距离外）
            start_dist = max(0, dist_to_point - self.truncation_distance)
            end_dist = dist_to_point + self.truncation_distance
            
            step = self.voxel_size / 2
            for d in np.arange(start_dist, end_dist, step):
                # 采样点
                sample = origin + direction * d
                voxel_idx = self.to_voxel_coord(sample)
                
                # 计算TSDF值
                # 正值：在表面前方（传感器侧）
                # 负值：在表面后方
                sdf_val = dist_to_point - d
                
                # 截断
                if abs(sdf_val) > self.truncation_distance:
                    continue
                
                tsdf_val = sdf_val / self.truncation_distance
                tsdf_val = max(-1.0, min(1.0, tsdf_val))
                
                # 权重（距离越远权重越低）
                weight = 1.0 - abs(tsdf_val)
                
                # 更新体素
                self._update_voxel(voxel_idx, tsdf_val, weight, 
                                   colors[i] if colors is not None else None)
    
    def _update_voxel(self, voxel_idx, tsdf_val, weight, color):
        """更新单个体素"""
        if voxel_idx in self.voxels:
            old_tsdf, old_weight, old_color = self.voxels[voxel_idx]
            
            # 加权平均
            new_weight = min(old_weight + weight, self.max_weight)
            new_tsdf = (old_tsdf * old_weight + tsdf_val * weight) / new_weight
            
            # 颜色更新
            if color is not None and old_color is not None:
                new_color = (old_color * old_weight + color * weight) / new_weight
            else:
                new_color = color if color is not None else old_color
            
            self.voxels[voxel_idx] = (new_tsdf, new_weight, new_color)
        else:
            self.voxels[voxel_idx] = (tsdf_val, weight, color)
    
    def extract_surface(self, method='marching_cubes'):
        """
        提取表面
        
        方法:
            'marching_cubes': 移动立方体算法
            'zero_crossing': 零交叉点提取
        """
        if method == 'zero_crossing':
            return self._extract_surface_zero_crossing()
        else:
            # 需要完整的体素网格才能使用marching cubes
            return self._extract_surface_marching_cubes()
    
    def _extract_surface_zero_crossing(self):
        """零交叉点提取（简化版）"""
        surface_points = []
        surface_colors = []
        
        for idx, (tsdf_val, weight, color) in self.voxels.items():
            # 找零交叉点（TSDF接近0且权重大）
            if abs(tsdf_val) < 0.1 and weight > 2:
                point = self.voxel_to_world(idx)
                surface_points.append(point)
                if color is not None:
                    surface_colors.append(color)
        
        result = {'points': np.array(surface_points)}
        if surface_colors:
            result['colors'] = np.array(surface_colors)
        
        return result
    
    def _extract_surface_marching_cubes(self):
        """
        移动立方体算法提取表面
        
        论文：Lorensen & Cline, 1987
        """
        try:
            from skimage import measure
        except ImportError:
            print("需要安装scikit-image: pip install scikit-image")
            return self._extract_surface_zero_crossing()
        
        # 构建密集网格
        # 注意：对于大场景，这会消耗大量内存
        min_idx = np.floor(self.bounds_min / self.voxel_size).astype(int)
        max_idx = np.ceil(self.bounds_max / self.voxel_size).astype(int)
        
        grid_size = max_idx - min_idx + 1
        
        # 创建体积数组
        volume = np.ones(grid_size)  # 默认占据
        
        for idx, (tsdf_val, weight, _) in self.voxels.items():
            grid_idx = tuple(np.array(idx) - min_idx)
            
            if all(0 <= i < s for i, s in zip(grid_idx, grid_size)):
                volume[grid_idx] = tsdf_val
        
        # 移动立方体
        verts, faces, normals, values = measure.marching_cubes(
            volume, level=0, spacing=(self.voxel_size,) * 3
        )
        
        # 调整坐标
        verts += min_idx * self.voxel_size
        
        return {
            'vertices': verts,
            'faces': faces,
            'normals': normals
        }
    
    def get_tsdf_at(self, point):
        """获取指定点的TSDF值（三线性插值）"""
        voxel_coord = point / self.voxel_size
        
        # 获取周围8个体素
        x0, y0, z0 = np.floor(voxel_coord).astype(int)
        dx, dy, dz = voxel_coord - np.array([x0, y0, z0])
        
        # 收集8个角点的值
        values = []
        for dz_i in [0, 1]:
            for dy_i in [0, 1]:
                for dx_i in [0, 1]:
                    idx = (x0 + dx_i, y0 + dy_i, z0 + dz_i)
                    if idx in self.voxels:
                        values.append(self.voxels[idx][0])
                    else:
                        values.append(1.0)  # 未知区域视为前方
        
        # 三线性插值
        c000, c100, c010, c110, c001, c101, c011, c111 = values
        
        c00 = c000 * (1 - dx) + c100 * dx
        c01 = c001 * (1 - dx) + c101 * dx
        c10 = c010 * (1 - dx) + c110 * dx
        c11 = c011 * (1 - dx) + c111 * dx
        
        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy
        
        return c0 * (1 - dz) + c1 * dz
    
    def fuse(self, other_tsdf):
        """融合另一个TSDF地图"""
        for idx, (tsdf_val, weight, color) in other_tsdf.voxels.items():
            self._update_voxel(idx, tsdf_val, weight, color)


class VoxelHashing:
    """
    体素哈希（Niessner et al., 2013）
    
    问题提出：
    稠密TSDF内存消耗大，但表面附近的数据才有意义。
    
    解决方案：
    使用哈希表只存储表面附近的体素。
    
    优点：
    - 内存效率高
    - 支持大规模场景
    - 实时性能好
    
    代表系统：InfiniTAM, VoxelHashing
    """
    
    def __init__(self, voxel_size=0.01, truncation_distance=0.03):
        self.voxel_size = voxel_size
        self.truncation_distance = truncation_distance
        
        # 哈希表
        self.hash_table = {}
        self.block_size = 8  # 每个块包含8x8x8个体素
    
    def hash_function(self, block_pos):
        """哈希函数"""
        # 简单的哈希
        p = block_pos
        return (p[0] * 73856093) ^ (p[1] * 19349663) ^ (p[2] * 83492791)
    
    def allocate_block(self, block_pos):
        """分配块"""
        hash_val = self.hash_function(block_pos)
        
        if hash_val not in self.hash_table:
            # 创建新块
            self.hash_table[hash_val] = {
                'position': block_pos,
                'voxels': np.zeros((self.block_size,) * 3, dtype=np.float32),
                'weights': np.zeros((self.block_size,) * 3, dtype=np.float32)
            }
        
        return self.hash_table[hash_val]


class SemanticMap:
    """
    语义地图
    
    问题提出：
    传统几何地图缺乏对环境高层语义信息的理解，无法区分不同物体类别，
    难以支持高级任务规划（如"去厨房拿杯子"）。
    
    解决方案：
    将语义分割/检测结果与几何地图融合，构建包含物体类别、实例ID、
    语义关系的地图表示。
    
    核心论文：
    - SemanticFusion (McCormac et al., 2017): 将CNN语义分割与SLAM融合
    - Kimera (Rosinol et al., 2020): 度量-语义SLAM系统
    - PanopticFusion (Narita et al., 2019): 全景分割融合
    - MaskFusion (Runz et al., 2018): 实例级语义SLAM
    
    优点：
    - 支持高层语义查询（如"找到所有椅子"）
    - 提升数据关联鲁棒性
    - 支持任务规划和导航
    
    缺点：
    - 语义分割计算开销大
    - 语义标签噪声影响地图质量
    - 动态物体处理困难
    """
    
    def __init__(self, num_classes=40, voxel_size=0.05):
        self.num_classes = num_classes
        self.voxel_size = voxel_size
        
        # 语义体素地图: (x,y,z) -> {class_probs, instance_id, color}
        self.semantic_voxels = {}
        
        # 物体实例数据库
        self.instances = {}  # instance_id -> {class_label, voxels, centroid, bbox}
        self.next_instance_id = 1
        
        # 语义标签映射
        self.class_names = {}
        
        # 贝叶斯融合参数
        self.prior_prob = 1.0 / num_classes
    
    def integrate_semantic_frame(self, depth, semantic_labels, instance_labels, 
                                  camera_pose, camera_intrinsics):
        """
        融合一帧语义分割结果
        
        论文核心思想（SemanticFusion）：
        使用贝叶斯更新融合多帧语义观测，提高标签一致性。
        
        Args:
            depth: 深度图
            semantic_labels: 语义标签图 (H, W)
            instance_labels: 实例标签图 (H, W)
            camera_pose: 相机位姿 (4, 4)
            camera_intrinsics: 相机内参 (3, 3)
        """
        height, width = depth.shape
        
        # 反投影到3D
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        # 相机坐标系到世界坐标系
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        
        for v in range(0, height, 4):  # 降采样加速
            for u in range(0, width, 4):
                z = depth[v, u]
                if z <= 0 or z > 10.0:
                    continue
                
                # 反投影
                x_cam = (u - cx) * z / fx
                y_cam = (v - cy) * z / fy
                z_cam = z
                
                # 变换到世界坐标
                point_cam = np.array([x_cam, y_cam, z_cam])
                point_world = R @ point_cam + t
                
                # 体素索引
                voxel_idx = tuple((point_world / self.voxel_size).astype(int))
                
                # 获取语义标签
                semantic_class = semantic_labels[v, u]
                instance_id = instance_labels[v, u]
                
                # 更新语义体素
                self._update_semantic_voxel(voxel_idx, semantic_class, instance_id)
    
    def _update_semantic_voxel(self, voxel_idx, semantic_class, instance_id):
        """
        贝叶斯语义更新
        
        论文核心思想：
        使用对数几率（log-odds）进行多帧语义融合，
        类似于占据栅格的概率更新。
        """
        if voxel_idx not in self.semantic_voxels:
            # 初始化
            self.semantic_voxels[voxel_idx] = {
                'class_probs': np.ones(self.num_classes) * self.prior_prob,
                'instance_id': 0,
                'observation_count': 0,
                'color': np.zeros(3)
            }
        
        voxel = self.semantic_voxels[voxel_idx]
        
        # 观测似然（简化版，实际使用CNN输出概率）
        likelihood = np.ones(self.num_classes) * 0.01
        likelihood[semantic_class] = 0.99
        
        # 贝叶斯更新
        prior = voxel['class_probs']
        posterior = prior * likelihood
        posterior /= posterior.sum()  # 归一化
        
        voxel['class_probs'] = posterior
        voxel['observation_count'] += 1
        
        # 更新实例ID（使用最大概率的实例）
        if instance_id > 0:
            if voxel['instance_id'] == 0:
                voxel['instance_id'] = instance_id
                self._add_voxel_to_instance(instance_id, semantic_class, voxel_idx)
            elif voxel['instance_id'] != instance_id:
                # 实例冲突，选择观测次数更多的
                pass
    
    def _add_voxel_to_instance(self, instance_id, class_label, voxel_idx):
        """将体素添加到对应实例"""
        if instance_id not in self.instances:
            self.instances[instance_id] = {
                'class_label': class_label,
                'voxels': set(),
                'centroid': np.zeros(3),
                'bbox_min': np.array([float('inf')] * 3),
                'bbox_max': np.array([float('-inf')] * 3)
            }
        
        instance = self.instances[instance_id]
        instance['voxels'].add(voxel_idx)
        
        # 更新边界框
        point = np.array(voxel_idx) * self.voxel_size
        instance['bbox_min'] = np.minimum(instance['bbox_min'], point)
        instance['bbox_max'] = np.maximum(instance['bbox_max'], point)
    
    def get_semantic_label(self, voxel_idx):
        """获取体素的语义标签（最大概率类别）"""
        if voxel_idx not in self.semantic_voxels:
            return -1, 0.0
        
        probs = self.semantic_voxels[voxel_idx]['class_probs']
        label = np.argmax(probs)
        confidence = probs[label]
        return label, confidence
    
    def query_objects_by_class(self, class_label):
        """查询特定类别的所有物体实例"""
        results = []
        for instance_id, instance in self.instances.items():
            if instance['class_label'] == class_label:
                results.append({
                    'instance_id': instance_id,
                    'centroid': instance['centroid'],
                    'bbox': (instance['bbox_min'], instance['bbox_max']),
                    'voxel_count': len(instance['voxels'])
                })
        return results
    
    def get_nearest_object(self, class_label, query_position):
        """找到距离查询位置最近的特定类别物体"""
        objects = self.query_objects_by_class(class_label)
        if not objects:
            return None
        
        min_dist = float('inf')
        nearest = None
        for obj in objects:
            dist = np.linalg.norm(obj['centroid'] - query_position)
            if dist < min_dist:
                min_dist = dist
                nearest = obj
        
        return nearest


class TopologicalMap:
    """
    拓扑地图
    
    问题提出：
    度量地图（栅格/点云）数据量大，路径规划计算复杂度高，
    且难以表达环境的抽象结构（房间、走廊、门等）。
    
    解决方案：
    构建拓扑图表示环境，节点表示关键位置/区域，
    边表示可达性关系。
    
    核心论文：
    - Topological Mapping (Kuipers et al., 2004): 空间语义层次结构
    - Pose Graph SLAM (Grisetti et al., 2010): 位姿图优化
    - Place Recognition (Cummins & Newman, 2008): FAB-MAP
    - Experience Mapping (Churchill & Newman, 2013): 经验地图
    
    优点：
    - 路径规划高效（图搜索 vs 栅格搜索）
    - 支持高层导航指令
    - 存储空间小
    
    缺点：
    - 缺乏精确几何信息
    - 拓扑边构建困难
    - 回环检测依赖外观
    """
    
    def __init__(self):
        # 图结构
        self.nodes = {}  # node_id -> {pose, descriptor, type, neighbors}
        self.edges = {}  # (node1, node2) -> {weight, type, traversability}
        
        # 节点计数
        self.next_node_id = 0
        
        # 空间索引
        self.spatial_index = {}
        
        # 回环检测
        self.place_recognizer = PlaceRecognizer()
    
    def add_node(self, pose, descriptor=None, node_type='place'):
        """
        添加拓扑节点
        
        Args:
            pose: 节点位姿 (x, y, theta) 或 (x, y, z, qx, qy, qz, qw)
            descriptor: 位置描述子（用于回环检测）
            node_type: 节点类型（place, room, corridor, door等）
        
        Returns:
            node_id: 新节点ID
        """
        node_id = self.next_node_id
        self.next_node_id += 1
        
        self.nodes[node_id] = {
            'pose': np.array(pose),
            'descriptor': descriptor,
            'type': node_type,
            'neighbors': set(),
            'creation_time': time.time(),
            'visit_count': 0
        }
        
        return node_id
    
    def add_edge(self, node1, node2, weight=None, edge_type='traversable'):
        """添加拓扑边"""
        if node1 not in self.nodes or node2 not in self.nodes:
            return False
        
        # 计算权重（欧氏距离）
        if weight is None:
            pose1 = self.nodes[node1]['pose']
            pose2 = self.nodes[node2]['pose']
            weight = np.linalg.norm(pose1[:2] - pose2[:2])  # 2D距离
        
        self.edges[(node1, node2)] = {
            'weight': weight,
            'type': edge_type,
            'traversability': 1.0
        }
        
        # 更新邻居关系
        self.nodes[node1]['neighbors'].add(node2)
        self.nodes[node2]['neighbors'].add(node1)
        
        return True
    
    def detect_loop_closure(self, current_descriptor, threshold=0.8):
        """
        回环检测
        
        论文核心思想（FAB-MAP）：
        使用词袋模型（Bag-of-Words）进行外观-based位置识别，
        通过Chow-Liu树建模特征相关性。
        
        Args:
            current_descriptor: 当前帧描述子
            threshold: 相似度阈值
        
        Returns:
            matched_node_id: 匹配节点ID，若无匹配返回None
        """
        best_match = None
        best_score = 0.0
        
        for node_id, node in self.nodes.items():
            if node['descriptor'] is None:
                continue
            
            # 计算相似度
            score = self._compute_similarity(current_descriptor, node['descriptor'])
            
            if score > threshold and score > best_score:
                best_score = score
                best_match = node_id
        
        return best_match, best_score
    
    def _compute_similarity(self, desc1, desc2):
        """计算描述子相似度"""
        # 余弦相似度
        return np.dot(desc1, desc2) / (np.linalg.norm(desc1) * np.linalg.norm(desc2))
    
    def find_path(self, start_node, goal_node):
        """
        拓扑路径规划（A*算法）
        
        相比栅格地图的优势：
        - 搜索空间小（节点数 << 栅格数）
        - 计算速度快
        - 路径更平滑
        """
        if start_node not in self.nodes or goal_node not in self.nodes:
            return None
        
        # A*算法
        open_set = [(0, start_node)]
        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_node:
                return self._reconstruct_path(came_from, current)
            
            for neighbor in self.nodes[current]['neighbors']:
                edge_weight = self.edges.get((current, neighbor), 
                                              self.edges.get((neighbor, current), 
                                                            {'weight': 1.0}))['weight']
                tentative_g = g_score[current] + edge_weight
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # 无路径
    
    def _heuristic(self, node1, node2):
        """启发式函数（欧氏距离）"""
        pose1 = self.nodes[node1]['pose']
        pose2 = self.nodes[node2]['pose']
        return np.linalg.norm(pose1[:2] - pose2[:2])
    
    def _reconstruct_path(self, came_from, current):
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


class PlaceRecognizer:
    """
    位置识别（回环检测）
    
    问题提出：
    SLAM中累积误差需要回环检测来校正，但如何可靠地识别
    曾经访问过的位置是一个挑战（感知混淆问题）。
    
    核心论文：
    - FAB-MAP (Cummins & Newman, 2008): 概率位置识别
    - DBoW2 (Galvez-Lopez & Tardos, 2012): 词袋模型
    - NetVLAD (Arandjelovic et al., 2016): 深度学习描述子
    - PointNetVLAD (Angelo et al., 2018): 点云位置识别
    """
    
    def __init__(self, method='dbow'):
        self.method = method
        
        # 数据库
        self.descriptors = []
        self.positions = []
        
        if method == 'dbow':
            # DBoW2词袋模型
            self.vocabulary = self._load_vocabulary()
        elif method == 'netvlad':
            # NetVLAD深度学习模型
            self.model = self._load_netvlad_model()
    
    def _load_vocabulary(self):
        """加载预训练词袋词典"""
        # 实际实现需要加载ORB或SIFT特征词典
        return None
    
    def _load_netvlad_model(self):
        """加载NetVLAD模型"""
        # 实际实现需要加载预训练网络
        return None
    
    def compute_descriptor(self, image):
        """计算图像描述子"""
        if self.method == 'dbow':
            return self._compute_bow_descriptor(image)
        elif self.method == 'netvlad':
            return self._compute_netvlad_descriptor(image)
    
    def _compute_bow_descriptor(self, image):
        """计算词袋描述子"""
        # 提取特征
        # orb = cv2.ORB_create()
        # keypoints, descriptors = orb.detectAndCompute(image, None)
        # 量化到词袋
        # ...
        return np.random.rand(128)  # 占位
    
    def _compute_netvlad_descriptor(self, image):
        """计算NetVLAD描述子"""
        # 使用CNN提取特征
        # ...
        return np.random.rand(4096)  # 占位
    
    def query(self, descriptor, threshold=0.8):
        """
        查询数据库
        
        Returns:
            match_idx: 匹配索引
            confidence: 置信度
        """
        if len(self.descriptors) == 0:
            return -1, 0.0
        
        # 计算与所有历史描述子的相似度
        similarities = []
        for desc in self.descriptors:
            sim = np.dot(descriptor, desc) / (np.linalg.norm(descriptor) * np.linalg.norm(desc))
            similarities.append(sim)
        
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        
        if best_sim > threshold:
            return best_idx, best_sim
        return -1, 0.0
    
    def add_to_database(self, descriptor, position):
        """添加到数据库"""
        self.descriptors.append(descriptor)
        self.positions.append(position)


class ElevationMap:
    """
    高程地图（2.5D地图）
    
    问题提出：
    地面机器人导航需要地形信息，全3D地图计算开销大。
    
    解决方案：
    使用2.5D表示（x,y位置存储高度和方差），
    平衡几何表达能力与计算效率。
    
    核心论文：
    - Elevation Mapping (Fankhauser et al., 2014): 概率高程地图
    - Grid Map (Fankhauser et al., 2016): 通用网格地图库
    
    优点：
    - 适合地面机器人
    - 支持地形 traversability 分析
    - 计算效率高
    
    缺点：
    - 无法表示垂直结构
    - 不支持悬空物体
    """
    
    def __init__(self, resolution=0.1, width=100, height=100):
        self.resolution = resolution
        self.width = width
        self.height = height
        
        # 高程数据
        self.elevation = np.zeros((height, width))
        self.variance = np.ones((height, width)) * 1000  # 大方差表示未知
        self.valid = np.zeros((height, width), dtype=bool)
        
        # 原点
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2
    
    def update(self, point_cloud, sensor_variance=0.01):
        """
        更新高程地图
        
        使用卡尔曼滤波融合多帧观测
        """
        for point in point_cloud:
            x, y, z = point
            
            # 转换到地图坐标
            mx = int((x - self.origin_x) / self.resolution)
            my = int((y - self.origin_y) / self.resolution)
            
            if 0 <= mx < self.width and 0 <= my < self.height:
                if not self.valid[my, mx]:
                    # 首次观测
                    self.elevation[my, mx] = z
                    self.variance[my, mx] = sensor_variance
                    self.valid[my, mx] = True
                else:
                    # 卡尔曼更新
                    prior_var = self.variance[my, mx]
                    new_var = (prior_var * sensor_variance) / (prior_var + sensor_variance)
                    new_elev = (self.elevation[my, mx] * sensor_variance + z * prior_var) / \
                               (prior_var + sensor_variance)
                    
                    self.elevation[my, mx] = new_elev
                    self.variance[my, mx] = new_var
    
    def get_traversability(self, x, y, robot_radius=0.3):
        """
        计算地形 traversability
        
        基于局部高程变化判断机器人能否通过
        """
        mx = int((x - self.origin_x) / self.resolution)
        my = int((y - self.origin_y) / self.resolution)
        
        radius_cells = int(robot_radius / self.resolution)
        
        elevations = []
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                nx, ny = mx + dx, my + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.valid[ny, nx]:
                    elevations.append(self.elevation[ny, nx])
        
        if len(elevations) < 4:
            return 0.0  # 未知区域不可通行
        
        # 计算高程变化
        elev_range = max(elevations) - min(elevations)
        
        # Traversability分数（0-1，1表示完全可通行）
        if elev_range < 0.05:
            return 1.0
        elif elev_range < 0.1:
            return 0.5
        else:
            return 0.0


class MultiLayeredMap:
    """
    多层地图表示
    
    问题提出：
    单一地图表示难以同时满足定位、规划、语义理解等多种需求。
    
    解决方案：
    构建多层地图，每层针对不同任务优化。
    
    典型层次：
    - 几何层：原始传感器数据（点云、网格）
    - 语义层：物体类别和实例
    - 拓扑层：高层结构关系
    -  traversability层：导航可行性
    
    代表系统：
    - Kimera (Rosinol et al., 2020)
    - Hydra (Hughes et al., 2022)
    """
    
    def __init__(self):
        # 几何层
        self.tsdf_layer = TSDFMap()
        self.occupancy_layer = OccupancyGridMap()
        
        # 语义层
        self.semantic_layer = SemanticMap()
        
        # 拓扑层
        self.topological_layer = TopologicalMap()
        
        # 高程层（地面机器人）
        self.elevation_layer = ElevationMap()
        
        # 层间变换
        self.layer_transforms = {}
    
    def integrate_observation(self, observation, sensor_type='rgbd'):
        """
        融合观测到多层地图
        
        Args:
            observation: 传感器观测数据
            sensor_type: 传感器类型
        """
        if sensor_type == 'rgbd':
            # 更新几何层
            self.tsdf_layer.integrate_scan(
                observation['pointcloud'],
                observation['pose'],
                observation.get('colors')
            )
            
            # 更新语义层
            if 'semantic_labels' in observation:
                self.semantic_layer.integrate_semantic_frame(
                    observation['depth'],
                    observation['semantic_labels'],
                    observation.get('instance_labels', np.zeros_like(observation['semantic_labels'])),
                    observation['pose'],
                    observation['intrinsics']
                )
        
        elif sensor_type == 'lidar':
            # 更新占据栅格
            self.occupancy_layer.update_scan(
                observation['pose'][:2, 3],  # 简化
                observation['ranges'],
                observation['angles']
            )
            
            # 更新高程地图
            self.elevation_layer.update(observation['pointcloud'])
    
    def query(self, query_type, **kwargs):
        """
        多层查询接口
        
        Examples:
            # 几何查询
            map.query('geometry', position=[x,y,z], radius=1.0)
            
            # 语义查询
            map.query('semantic', class_label='chair')
            
            # 拓扑查询
            map.query('topology', start='room_A', goal='room_B')
        """
        if query_type == 'geometry':
            return self._query_geometry(**kwargs)
        elif query_type == 'semantic':
            return self._query_semantic(**kwargs)
        elif query_type == 'topology':
            return self._query_topology(**kwargs)
        elif query_type == 'traversability':
            return self._query_traversability(**kwargs)
    
    def _query_geometry(self, position, radius):
        """查询局部几何"""
        # 从TSDF地图提取表面
        pass
    
    def _query_semantic(self, class_label):
        """查询语义物体"""
        return self.semantic_layer.query_objects_by_class(class_label)
    
    def _query_topology(self, start, goal):
        """查询拓扑路径"""
        return self.topological_layer.find_path(start, goal)
    
    def _query_traversability(self, position):
        """查询 traversability"""
        x, y = position[:2]
        return self.elevation_layer.get_traversability(x, y)


# ==================== 地图构建前沿方向 ====================

"""
## 地图表示的前沿研究方向

### 1. 神经隐式地图（Neural Implicit Maps）

**问题提出：**
传统显式地图（栅格、点云）存储效率低，难以表达精细几何。

**解决方案：**
使用神经网络隐式表示场景，如NeRF、Instant-NGP。

**核心论文：**
- NeRF (Mildenhall et al., 2020): 神经辐射场
- Instant-NGP (Müller et al., 2022): 多分辨率哈希编码
- iMAP (Sucar et al., 2021): 隐式神经SLAM
- NICE-SLAM (Zhu et al., 2022): 神经隐式可编码SLAM

**优点：**
- 内存效率高
- 连续表示支持任意分辨率
- 可渲染高质量图像

**缺点：**
- 训练时间长
- 难以编辑和更新
- 泛化能力有限

### 2. 混合表示地图

**问题提出：**
单一表示难以平衡效率与表达能力。

**解决方案：**
结合显式和隐式表示的优势。

**代表工作：**
- Kimera: 度量-语义混合地图
- Hydra: 场景图+体素地图
- NG-Mapping: 神经-几何混合

### 3. 动态环境地图

**问题提出：**
传统SLAM假设静态环境，难以处理动态物体。

**解决方案：**
- 检测并剔除动态物体
- 建模动态物体运动
- 维护动态地图层

**核心论文：**
- DynaSLAM (Bescos et al., 2018)
- VDO-SLAM (Zhang et al., 2020)
- EM-Fusion (Strecke & Alcantarilla, 2019)

### 4. 开放词汇语义地图

**问题提出：**
传统语义地图受限于预定义类别。

**解决方案：**
结合视觉-语言模型（CLIP）实现开放词汇识别。

**核心论文：**
- CLIP-Fields (Shafiullah et al., 2022)
- LERF (Kerr et al., 2023)
- NLMap (Huang et al., 2023)

**应用：**
- 自然语言查询（"找到可以放杯子的家具"）
- 零样本语义分割

### 5. 终身地图（Lifelong Mapping）

**问题提出：**
环境随时间变化，地图需要持续更新。

**挑战：**
- 场景变化检测
- 地图版本管理
- 长期定位鲁棒性

**核心论文：**
- Experience Mapping (Churchill & Newman, 2013)
- Summarization (Dymczyk et al., 2015)
- Map Management (Krajnik et al., 2017)

### 6. 多智能体协同建图

**问题提出：**
单机器人建图效率低，多机器人可加速探索。

**挑战：**
- 分布式数据融合
- 通信受限下的协同
- 地图合并与对齐

**核心论文：**
- CCM-SLAM (Schmuck & Chli, 2019)
- DOOR-SLAM (Lajoie et al., 2020)
- Kimera-Multi (Reinke et al., 2022)

### 7. 不确定性量化

**问题提出：**
地图估计的不确定性对规划和决策至关重要。

**方法：**
- 贝叶斯深度学习
- 蒙特卡洛Dropout
- 集成方法

**应用：**
- 主动探索
- 安全导航
- 人机交互
"""

# ==================== 总结 ====================

"""
地图构建是SLAM系统的核心输出，直接影响下游任务（导航、规划、操作）的性能。

**关键权衡：**
1. 精度 vs 效率：高精度地图需要更多的计算和存储资源
2. 几何 vs 语义：纯几何地图缺乏语义理解，语义地图计算开销大
3. 局部 vs 全局：局部地图细节丰富，全局地图需要层次化表示
4. 静态 vs 动态：静态假设简化问题，动态建模更贴近现实

**未来方向：**
- 神经隐式表示将成为主流
- 开放词汇语义理解
- 多模态融合（视觉+激光+语言）
- 终身学习与自适应
- 多智能体协同
"""
