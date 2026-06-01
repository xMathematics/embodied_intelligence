# 4.5 地图构建

## 目录

- [1. 地图表示方法](#1-地图表示方法)
- [2. 栅格地图](#2-栅格地图)
- [3. 点云地图](#3-点云地图)
- [4. 语义地图](#4-语义地图)
- [5. 实践练习](#5-实践练习)

---

## 1. 地图表示方法

### 1.1 地图分类

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2

class MapTypes:
    """地图类型"""
    
    @staticmethod
    def sparse_map():
        """稀疏地图"""
        return {
            "name": "稀疏地图",
            "elements": ["特征点", "路标"],
            "size": "小",
            "use_case": ["定位", "回环检测"]
        }
    
    @staticmethod
    def dense_map():
        """稠密地图"""
        return {
            "name": "稠密地图",
            "elements": ["像素/体素"],
            "size": "大",
            "use_case": ["重建", "导航", "避障"]
        }
    
    @staticmethod
    def metric_map():
        """度量地图"""
        return {
            "name": "度量地图",
            "coordinate": "全局坐标系",
            "examples": ["栅格地图", "点云地图"]
        }
    
    @staticmethod
    def topological_map():
        """拓扑地图"""
        return {
            "name": "拓扑地图",
            "elements": ["节点", "边"],
            "examples": ["导航图"]
        }
```

---

## 2. 栅格地图

### 2.1 占据栅格地图

```python
class OccupancyGridMap:
    """占据栅格地图"""
    
    def __init__(self, resolution=0.05, width=100, height=100):
        self.resolution = resolution
        self.width = width
        self.height = height
        
        # 原点在地图中心
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2
        
        # 概率对数优势
        self.log_odds = np.zeros((height, width))
        
        # 初始概率
        self.prob_occ = 0.7
        self.prob_free = 0.3
        self.prob_prior = 0.5
        
        self.l_occ = np.log(self.prob_occ / (1 - self.prob_occ))
        self.l_free = np.log(self.prob_free / (1 - self.prob_free))
        self.l_prior = np.log(self.prob_prior / (1 - self.prob_prior))
    
    def world_to_grid(self, x, y):
        """世界坐标转栅格坐标"""
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy
    
    def grid_to_world(self, gx, gy):
        """栅格坐标转世界坐标"""
        x = gx * self.resolution + self.origin_x + self.resolution / 2
        y = gy * self.resolution + self.origin_y + self.resolution / 2
        return x, y
    
    def update_cell(self, gx, gy, is_occupied):
        """更新栅格"""
        if 0 <= gx < self.width and 0 <= gy < self.height:
            if is_occupied:
                self.log_odds[gy, gx] += self.l_occ
            else:
                self.log_odds[gy, gx] += self.l_free
    
    def get_probability(self, gx, gy):
        """获取占据概率"""
        if 0 <= gx < self.width and 0 <= gy < self.height:
            odds = np.exp(self.log_odds[gy, gx])
            return odds / (odds + 1)
        return 0.5
    
    def get_probability_map(self):
        """获取概率地图"""
        odds = np.exp(self.log_odds)
        return odds / (odds + 1)
    
    def update_scan(self, sensor_pose, ranges, angles):
        """
        更新扫描
        
        参数:
            sensor_pose: (x, y, theta)
            ranges: 距离
            angles: 角度
        """
        sx, sy, st = sensor_pose
        
        for r, a in zip(ranges, angles):
            if r <= 0:
                continue
            
            # 全局坐标
            a_global = st + a
            end_x = sx + r * np.cos(a_global)
            end_y = sy + r * np.sin(a_global)
            
            # 终点：占据
            gx, gy = self.world_to_grid(end_x, end_y)
            self.update_cell(gx, gy, True)
            
            # 射线路径：空闲
            step = self.resolution
            for d in np.arange(0, r, step):
                x = sx + d * np.cos(a_global)
                y = sy + d * np.sin(a_global)
                gx, gy = self.world_to_grid(x, y)
                self.update_cell(gx, gy, False)
```

### 2.2 距离地图

```python
class DistanceMap:
    """距离地图"""
    
    def __init__(self, occupancy_map):
        self.occ_map = occupancy_map
        
        # 距离
        self.distance = None
    
    def compute_distance_transform(self):
        """距离变换"""
        prob_map = self.occ_map.get_probability_map()
        
        # 二值化
        binary = (prob_map > 0.5).astype(np.uint8)
        
        # 距离变换
        from scipy.ndimage import distance_transform_edt
        
        self.distance = distance_transform_edt(1 - binary)
        return self.distance
```

---

## 3. 点云地图

### 3.1 TSDF地图

```python
class TSDFMap:
    """TSDF地图 (Truncated Signed Distance Function)"""
    
    def __init__(self, voxel_size=0.05, truncation_distance=0.3):
        self.voxel_size = voxel_size
        self.truncation_distance = truncation_distance
        
        # 使用字典存储稀疏体素
        self.voxels = {}
    
    def to_voxel_coord(self, point):
        """点转体素坐标"""
        return tuple(np.floor(point / self.voxel_size).astype(int))
    
    def integrate_scan(self, pointcloud, sensor_pose):
        """
        集成扫描
        
        参数:
            pointcloud: 点云
            sensor_pose: 传感器位姿
        """
        # 变换点云到世界坐标系
        R = sensor_pose[:3, :3]
        t = sensor_pose[:3, 3]
        points_world = (R @ pointcloud.T).T + t
        
        # 原点
        origin_world = t
        
        for point in points_world:
            # 计算距离
            vec = point - origin_world
            dist = np.linalg.norm(vec)
            
            # 沿光线采样
            step = self.voxel_size / 2
            for d in np.arange(max(0, dist - self.truncation_distance), 
                               dist + self.truncation_distance, step):
                sample = origin_world + vec * (d / dist)
                voxel_idx = self.to_voxel_coord(sample)
                
                # 计算TSDF值
                tsdf_val = (dist - d) / self.truncation_distance
                tsdf_val = max(-1.0, min(1.0, tsdf_val))
                
                # 更新
                if voxel_idx in self.voxels:
                    # 权重平均
                    old_val, old_weight = self.voxels[voxel_idx]
                    new_weight = old_weight + 1
                    new_val = (old_val * old_weight + tsdf_val) / new_weight
                    self.voxels[voxel_idx] = (new_val, new_weight)
                else:
                    self.voxels[voxel_idx] = (tsdf_val, 1.0)
    
    def extract_surface(self):
        """提取表面"""
        surface_points = []
        
        for idx, (tsdf_val, weight) in self.voxels.items():
            # 找零交叉点
            if abs(tsdf_val) < 0.1 and weight > 2:
                # 计算中心点
                point = (np.array(idx) + 0.5) * self.voxel_size
                surface_points.append(point)
        
        return np.array(surface_points)
```

### 3.2 八叉树地图

```python
class OctreeNode:
    """八叉树节点"""
    
    def __init__(self, center, size, depth=0, max_depth=5):
        self.center = np.array(center)
        self.size = size
        self.depth = depth
        self.max_depth = max_depth
        
        # 子节点
        self.children = None
        
        # 占据概率
        self.probability = 0.5
        
        # 点计数
        self.num_points = 0
    
    def is_leaf(self):
        """是否是叶子节点"""
        return self.children is None
    
    def split(self):
        """分裂"""
        if self.depth >= self.max_depth:
            return
        
        half_size = self.size / 2
        self.children = []
        
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    child_center = self.center + np.array([dx, dy, dz]) * (half_size / 2)
                    child = OctreeNode(child_center, half_size, self.depth + 1, self.max_depth)
                    self.children.append(child)
    
    def update_point(self, point):
        """更新点"""
        # 检查是否在内部
        half_size = self.size / 2
        bounds = [
            self.center[0] - half_size, self.center[0] + half_size,
            self.center[1] - half_size, self.center[1] + half_size,
            self.center[2] - half_size, self.center[2] + half_size
        ]
        
        if not (bounds[0] <= point[0] < bounds[1] and
                bounds[2] <= point[1] < bounds[3] and
                bounds[4] <= point[2] < bounds[5]):
            return
        
        self.num_points += 1
        
        # 更新概率
        # 简化的更新
        self.probability = min(0.95, self.probability + 0.1)
        
        if self.depth < self.max_depth and self.num_points > 10:
            if self.is_leaf():
                self.split()
            
            # 递归更新子节点
            if not self.is_leaf():
                for child in self.children:
                    child.update_point(point)

class OctreeMap:
    """八叉树地图"""
    
    def __init__(self, center=(0,0,0), size=20, max_depth=5):
        self.root = OctreeNode(center, size, 0, max_depth)
    
    def insert_points(self, points):
        """插入点"""
        for point in points:
            self.root.update_point(point)
```

---

## 4. 语义地图

### 4.1 语义信息

```python
class SemanticMap:
    """语义地图"""
    
    def __init__(self):
        self.object_database = []
        self.class_names = [
            "unknown", "table", "chair", "sofa", "bed", 
            "door", "window", "plant", "tv", "computer"
        ]
    
    def add_object(self, pose, bounding_box, class_id, confidence=0.9):
        """添加物体"""
        obj = {
            'id': len(self.object_database),
            'pose': pose.copy(),
            'bounding_box': bounding_box,
            'class_id': class_id,
            'class_name': self.class_names[class_id] if class_id < len(self.class_names) else "unknown",
            'confidence': confidence,
            'observations': 1
        }
        self.object_database.append(obj)
    
    def associate_objects(self, new_objects):
        """关联物体"""
        # 简单的关联：距离最近
        for new_obj in new_objects:
            best_match = None
            best_dist = float('inf')
            
            for existing in self.object_database:
                dist = np.linalg.norm(new_obj['pose'][:3, 3] - existing['pose'][:3, 3])
                if dist < best_dist and dist < 0.5:
                    best_dist = dist
                    best_match = existing
            
            if best_match:
                # 更新
                best_match['observations'] += 1
                best_match['confidence'] = min(1.0, best_match['confidence'] + 0.1)
            else:
                self.object_database.append(new_obj)
```

### 4.2 语义SLAM

```python
class SemanticSLAM:
    """语义SLAM"""
    
    def __init__(self):
        self.geometric_map = None
        self.semantic_map = SemanticMap()
        
        # 物体检测
        self.detector = None
        
        # 实例分割
        self.segmentor = None
    
    def process_frame(self, img, depth=None):
        """处理帧"""
        # 1. 检测物体
        detections = self.detect_objects(img)
        
        # 2. 估计位姿
        pose = self.estimate_pose(img, depth)
        
        # 3. 生成语义观测
        semantic_observations = []
        for det in detections:
            # 估计物体位置
            obj_pose = self.estimate_object_pose(det, depth, pose)
            semantic_observations.append(obj_pose)
        
        # 4. 关联并更新语义地图
        self.semantic_map.associate_objects(semantic_observations)
        
        return pose
    
    def detect_objects(self, img):
        """检测物体"""
        return []
    
    def estimate_pose(self, img, depth):
        """估计位姿"""
        return np.eye(4)
    
    def estimate_object_pose(self, det, depth, robot_pose):
        """估计物体位姿"""
        return np.eye(4)
```

---

## 5. 实践练习

### 练习1：占据栅格地图

```python
def exercise_occupancy_grid():
    """占据栅格地图练习"""
    print("=== 占据栅格地图练习 ===")
    
    # 创建地图
    grid_map = OccupancyGridMap(resolution=0.1, width=100, height=100)
    
    # 模拟扫描
    sensor_pose = (0, 0, 0)
    
    # 360度激光
    num_beams = 180
    angles = np.linspace(-np.pi, np.pi, num_beams)
    
    # 模拟矩形房间
    room_size = 3
    
    ranges = []
    for a in angles:
        x = room_size * np.cos(a)
        y = room_size * np.sin(a)
        r = np.sqrt(x**2 + y**2)
        
        # 添加噪声
        r += np.random.randn() * 0.05
        
        ranges.append(r)
    
    # 更新地图
    grid_map.update_scan(sensor_pose, ranges, angles)
    
    # 获取概率地图
    prob_map = grid_map.get_probability_map()
    
    # 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(prob_map, cmap='gray')
    plt.title('Occupancy Grid Map')
    plt.xlabel('X (grid)')
    plt.ylabel('Y (grid)')
    plt.savefig('occupancy_grid.png')
    print("地图已保存到 occupancy_grid.png")

# exercise_occupancy_grid()
```

### 练习2：地图类型对比

```python
def exercise_map_comparison():
    """地图对比练习"""
    print("=== 地图类型对比 ===")
    
    types = [
        MapTypes.sparse_map(),
        MapTypes.dense_map(),
        MapTypes.metric_map(),
        MapTypes.topological_map()
    ]
    
    for t in types:
        print(f"\n{t['name']}:")
        for k, v in t.items():
            if k != 'name':
                print(f"  {k}: {v}")

# exercise_map_comparison()
```

### 练习3：TSDF基本概念

```python
def exercise_tsdf():
    """TSDF练习"""
    print("=== TSDF练习 ===")
    
    # 创建TSDF地图
    tsdf = TSDFMap(voxel_size=0.1, truncation_distance=0.3)
    
    # 模拟一个平面
    num_points = 1000
    xx, yy = np.meshgrid(np.linspace(-2, 2, 32), np.linspace(-2, 2, 32))
    zz = np.ones_like(xx)
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # 传感器在原点
    sensor_pose = np.eye(4)
    sensor_pose[2, 3] = -1.5
    
    # 集成
    tsdf.integrate_scan(points, sensor_pose)
    
    print(f"TSDF体素数量: {len(tsdf.voxels)}")
    
    # 提取表面
    surface = tsdf.extract_surface()
    print(f"提取表面点数: {len(surface)}")
    
    print("\n注: 真实TSDF请使用Open3D、Fusion或Voxblox等库")

# exercise_tsdf()
```

---

恭喜完成第四部分！现在你已经了解了SLAM的各个方面。

## 参考文献

1. Hornung, A., et al. (2013). OctoMap: An Efficient Probabilistic 3D Mapping Framework Based on Octrees.
2. Curless, B., & Levoy, M. (1996). A Volumetric Method for Building Complex Models from Range Images.
3. Bowman, S. L., et al. (2017). Probabilistic Data Association for Semantic SLAM.
4. McCormac, J., et al. (2017). SemanticFusion: Dense 3D Semantic Mapping with Convolutional Neural Networks.
5. Elfes, A. (1989). Occupancy Grids: A Probabilistic Framework for Robot Perception and Navigation.
