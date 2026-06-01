# 4.3 激光SLAM

## 目录

- [1. 激光SLAM简介](#1-激光slam简介)
- [2. 点云配准方法](#2-点云配准方法)
- [3. 经典激光SLAM系统](#3-经典激光slam系统)
- [4. 激光+惯性融合](#4-激光惯性融合)
- [5. 实践练习](#5-实践练习)

---

## 1. 激光SLAM简介

### 1.1 激光传感器

```python
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2

class LidarSensor:
    """激光传感器"""
    
    def __init__(self, num_channels=64, points_per_channel=1024,
                 fov_up=20, fov_down=-24.8, max_range=100):
        self.num_channels = num_channels
        self.points_per_channel = points_per_channel
        self.fov_up = np.radians(fov_up)
        self.fov_down = np.radians(fov_down)
        self.max_range = max_range
    
    def simulate_scan(self, scene_func):
        """
        模拟一次激光扫描
        
        scene_func: 场景函数，输入点返回是否被占用
        """
        points = []
        
        for channel in range(self.num_channels):
            # 垂直角度
            vert_angle = self.fov_down + (self.fov_up - self.fov_down) * channel / (self.num_channels - 1)
            
            for i in range(self.points_per_channel):
                # 水平角度
                horiz_angle = 2 * np.pi * i / self.points_per_channel
                
                # 方向向量
                dir_vec = np.array([
                    np.cos(vert_angle) * np.sin(horiz_angle),
                    np.cos(vert_angle) * np.cos(horiz_angle),
                    np.sin(vert_angle)
                ])
                
                # 光线投射
                for dist in np.linspace(0, self.max_range, 1000):
                    point = dir_vec * dist
                    if scene_func(point):
                        points.append(point)
                        break
        
        return np.array(points)
    
    def to_range_image(self, points):
        """转距离图像"""
        range_img = np.zeros((self.num_channels, self.points_per_channel))
        
        for point in points:
            r = np.linalg.norm(point)
            if r < 1e-6:
                continue
            
            # 计算角度
            theta = np.arctan2(point[0], point[1])
            phi = np.arcsin(point[2] / r)
            
            # 像素坐标
            u = int((theta + np.pi) / (2 * np.pi) * (self.points_per_channel - 1))
            v = int((phi - self.fov_down) / (self.fov_up - self.fov_down) * (self.num_channels - 1))
            
            if 0 <= u < self.points_per_channel and 0 <= v < self.num_channels:
                if range_img[v, u] == 0 or r < range_img[v, u]:
                    range_img[v, u] = r
        
        return range_img
```

### 1.2 激光SLAM vs 视觉SLAM

```python
class LidarVSVisual:
    """激光 vs 视觉"""
    
    @staticmethod
    def compare():
        """对比"""
        comparison = {
            "精度": {
                "激光": "高",
                "视觉": "中"
            },
            "光照影响": {
                "激光": "无",
                "视觉": "有"
            },
            "纹理信息": {
                "激光": "无",
                "视觉": "有"
            },
            "成本": {
                "激光": "高",
                "视觉": "低"
            },
            "计算量": {
                "激光": "中等",
                "视觉": "较高"
            }
        }
        
        return comparison
    
    @staticmethod
    def scenarios():
        """适用场景"""
        return {
            "激光": ["室外", "夜间", "大场景"],
            "视觉": ["室内", "AR/VR", "小规模"]
        }
```

---

## 2. 点云配准方法

### 2.1 ICP

```python
class ICP:
    """迭代最近点"""
    
    def __init__(self, max_iterations=50, tolerance=1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def find_closest_points(self, source, target):
        """找最近点"""
        from scipy.spatial import KDTree
        
        tree = KDTree(target)
        distances, indices = tree.query(source)
        
        return source, target[indices]
    
    def compute_transform(self, source, target):
        """计算R和t"""
        # 去中心
        source_centered = source - np.mean(source, axis=0)
        target_centered = target - np.mean(target, axis=0)
        
        # SVD
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        
        R = Vt.T @ U.T
        
        # 处理反射
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = np.mean(target, axis=0) - R @ np.mean(source, axis=0)
        
        return R, t
    
    def register(self, source, target, R_init=None, t_init=None):
        """配准"""
        if R_init is None:
            R = np.eye(3)
        else:
            R = R_init.copy()
        
        if t_init is None:
            t = np.zeros(3)
        else:
            t = t_init.copy()
        
        errors = []
        last_error = float('inf')
        
        for i in range(self.max_iterations):
            # 变换源点云
            source_transformed = (R @ source.T).T + t
            
            # 找对应
            src_corr, target_corr = self.find_closest_points(source_transformed, target)
            
            # 计算变换
            delta_R, delta_t = self.compute_transform(src_corr, target_corr)
            
            # 更新
            R = delta_R @ R
            t = delta_R @ t + delta_t
            
            # 计算误差
            error = np.mean(np.linalg.norm(
                (R @ source.T).T + t - target_corr, axis=1
            ))
            errors.append(error)
            
            # 收敛检查
            if abs(last_error - error) < self.tolerance:
                break
            
            last_error = error
        
        return R, t, errors
```

### 2.2 NDT

```python
class NDT:
    """正态分布变换"""
    
    def __init__(self, voxel_size=0.5):
        self.voxel_size = voxel_size
        self.grid = {}
    
    def build_grid(self, points):
        """建立网格"""
        for point in points:
            # 计算voxel索引
            idx = tuple(np.floor(point / self.voxel_size).astype(int))
            
            if idx not in self.grid:
                self.grid[idx] = {
                    'points': [],
                    'mean': None,
                    'cov': None
                }
            
            self.grid[idx]['points'].append(point)
        
        # 计算均值和协方差
        for idx in self.grid:
            pts = np.array(self.grid[idx]['points'])
            if len(pts) >= 3:
                self.grid[idx]['mean'] = np.mean(pts, axis=0)
                self.grid[idx]['cov'] = np.cov(pts.T)
    
    def compute_score(self, point, pose):
        """计算配准分数"""
        # 变换点
        point_tf = pose[:3, :3] @ point + pose[:3, 3]
        
        # 找voxel
        idx = tuple(np.floor(point_tf / self.voxel_size).astype(int))
        
        if idx not in self.grid or self.grid[idx]['mean'] is None:
            return 0.0
        
        # 计算似然
        delta = point_tf - self.grid[idx]['mean']
        cov = self.grid[idx]['cov']
        
        try:
            score = np.exp(-0.5 * delta @ np.linalg.inv(cov) @ delta)
        except:
            score = 0.0
        
        return score
    
    def register(self, source, target, init_pose=None):
        """配准"""
        # 建立target的NDT
        self.build_grid(target)
        
        if init_pose is None:
            init_pose = np.eye(4)
        
        # 使用优化方法优化pose
        # 这里简化
        return init_pose
```

### 2.3 ICP变种

```python
class ICPVariants:
    """ICP变种"""
    
    @staticmethod
    def point_to_plane(source, target, normal_target):
        """点到平面ICP"""
        # 简化
        return np.eye(3), np.zeros(3)
    
    @staticmethod
    def plane_to_plane(source, target, source_normals, target_normals):
        """平面对平面ICP"""
        return np.eye(3), np.zeros(3)
    
    @staticmethod
    def generalized_icp(source, target):
        """广义ICP"""
        return np.eye(3), np.zeros(3)
```

---

## 3. 经典激光SLAM系统

### 3.1 LOAM

```python
class LOAM:
    """LOAM (Lidar Odometry and Mapping)"""
    
    def __init__(self):
        self.scan_registered = None
        self.map = None
        
        # 特征提取
        self.corner_points = []
        self.surface_points = []
    
    def extract_features(self, scan):
        """提取特征点"""
        # 计算曲率
        curvature = self._compute_curvature(scan)
        
        # 选择角点和面点
        corner_idx = []
        surface_idx = []
        
        # 简化版
        sorted_indices = np.argsort(curvature)
        
        corner_idx = sorted_indices[-500:]
        surface_idx = sorted_indices[:5000]
        
        return scan[corner_idx], scan[surface_idx]
    
    def _compute_curvature(self, scan):
        """计算曲率"""
        curvature = np.zeros(len(scan))
        
        for i in range(5, len(scan) - 5):
            diff = -10 * scan[i]
            for j in range(1, 6):
                diff += scan[i - j]
                diff += scan[i + j]
            
            curvature[i] = np.linalg.norm(diff) ** 2
        
        return curvature
    
    def odometry(self, curr_scan, last_scan):
        """激光里程计"""
        # 提取特征
        curr_corner, curr_surface = self.extract_features(curr_scan)
        last_corner, last_surface = self.extract_features(last_scan)
        
        # 匹配并计算运动
        # 简化
        R, t = np.eye(3), np.zeros(3)
        
        return R, t
    
    def mapping(self, scan, pose):
        """建图"""
        # 将点云加入地图
        pass
```

### 3.2 Cartographer

```python
class Cartographer:
    """Cartographer"""
    
    def __init__(self, resolution=0.05):
        self.submaps = []
        self.current_submap = None
        self.resolution = resolution
        
        # Scan matching
        self.ceres_solver = None
    
    def add_scan(self, scan, pose):
        """添加扫描"""
        if self.current_submap is None:
            self.current_submap = {
                'points': [],
                'pose': pose.copy()
            }
        
        # 加入当前子图
        self.current_submap['points'].append({
            'points': scan,
            'pose': pose.copy()
        })
        
        # 检查是否应该创建新子图
        if self._should_create_new_submap():
            self.submaps.append(self.current_submap)
            self.current_submap = {
                'points': [],
                'pose': pose.copy()
            }
    
    def _should_create_new_submap(self):
        """是否应该创建新子图"""
        if not self.current_submap:
            return True
        return len(self.current_submap['points']) > 100
    
    def scan_match(self, scan, estimate_pose):
        """扫描匹配"""
        # Ceres优化
        # 简化
        return estimate_pose
    
    def detect_loop_closure(self):
        """回环检测"""
        pass
    
    def optimize_pose_graph(self):
        """位姿图优化"""
        pass
```

### 3.3 GMapping

```python
class GMapping:
    """GMapping (Grid-based FastSLAM)"""
    
    def __init__(self, num_particles=30, map_resolution=0.05):
        self.num_particles = num_particles
        self.particles = []
        self.map_resolution = map_resolution
        
        # 初始化粒子
        for i in range(num_particles):
            self.particles.append({
                'pose': np.eye(4),
                'map': None,
                'weight': 1.0 / num_particles
            })
    
    def update_odometry(self, odometry):
        """里程计更新"""
        for p in self.particles:
            # 添加噪声
            noise = np.random.randn(3) * 0.1
            p['pose'][:3, 3] += noise
    
    def update_scan(self, scan):
        """扫描更新"""
        for p in self.particles:
            # 计算观测似然
            likelihood = self._compute_likelihood(scan, p['pose'], p['map'])
            p['weight'] *= likelihood
        
        # 归一化权重
        total_weight = sum(p['weight'] for p in self.particles)
        for p in self.particles:
            p['weight'] /= total_weight
        
        # 重采样
        self._resample()
    
    def _compute_likelihood(self, scan, pose, grid_map):
        """计算似然"""
        # 简化
        return 1.0
    
    def _resample(self):
        """重采样"""
        # 低方差重采样
        pass
```

---

## 4. 激光+惯性融合

### 4.1 LIO-SAM

```python
class LIOSAM:
    """LIO-SAM"""
    
    def __init__(self):
        self.feature_extractor = None
        self.imu_preintegrator = None
        self.map_optimization = None
        
        # 缓存
        self.pointclouds = []
        self.imu_data = []
        
        # 因子图
        self.graph = None
    
    def imu_callback(self, imu_msg):
        """IMU回调"""
        # 预积分
        self.imu_preintegrator.integrate(imu_msg)
    
    def lidar_callback(self, pc_msg):
        """激光回调"""
        # 1. 特征提取
        corner, surface = self._extract_features(pc_msg)
        
        # 2. IMU运动补偿
        pc_compensated = self._motion_compensation(corner, surface)
        
        # 3. Scan to map配准
        pose = self._scan_to_map(pc_compensated)
        
        # 4. 添加到因子图
        self._add_to_graph(pose)
        
        # 5. 回环检测
        if self._detect_loop():
            self._optimize()
    
    def _extract_features(self, pc):
        """特征提取"""
        # 类似LOAM
        return [], []
    
    def _motion_compensation(self, corner, surface):
        """运动补偿"""
        return corner, surface
    
    def _scan_to_map(self, pc):
        """Scan to map"""
        return np.eye(4)
    
    def _detect_loop(self):
        """回环检测"""
        return False
    
    def _optimize(self):
        """优化"""
        pass
```

### 4.2 FAST-LIO

```python
class FASTLIO:
    """FAST-LIO"""
    
    def __init__(self):
        self.ikd_tree = None
        self.imu_state = None
        
        # 状态估计
        self.current_pose = np.eye(4)
    
    def update_imu(self, data):
        """IMU更新"""
        # 反向传播
        pass
    
    def update_lidar(self, points):
        """激光更新"""
        # 1. 体素滤波
        # 2. 构建IKD-tree
        # 3. 迭代卡尔曼滤波更新
        pass
```

---

## 5. 实践练习

### 练习1：ICP配准

```python
def exercise_icp():
    """ICP配准练习"""
    print("=== ICP配准练习 ===")
    
    # 生成源点云
    np.random.seed(42)
    source = np.random.randn(1000, 3) * 3
    
    # 变换
    R_true = np.array([
        [np.cos(0.3), -np.sin(0.3), 0],
        [np.sin(0.3), np.cos(0.3), 0],
        [0, 0, 1]
    ])
    t_true = np.array([1.0, 2.0, 0.5])
    target = (R_true @ source.T).T + t_true + np.random.randn(1000, 3) * 0.02
    
    # ICP
    icp = ICP(max_iterations=50)
    R, t, errors = icp.register(source, target)
    
    print(f"真实旋转:\n{R_true}")
    print(f"估计旋转:\n{R}")
    print(f"旋转误差: {np.linalg.norm(R - R_true):.6f}")
    print()
    
    print(f"真实平移: {t_true}")
    print(f"估计平移: {t}")
    print(f"平移误差: {np.linalg.norm(t - t_true):.6f}")
    print()
    
    # 绘制误差
    plt.figure(figsize=(10, 4))
    plt.plot(errors)
    plt.xlabel('迭代')
    plt.ylabel('误差')
    plt.title('ICP收敛')
    plt.grid(True)
    plt.savefig('icp_errors.png')
    print("误差曲线已保存")

# exercise_icp()
```

### 练习2：激光SLAM系统对比

```python
def exercise_lidar_slam_comparison():
    """激光SLAM系统对比"""
    print("=== 激光SLAM系统对比 ===")
    
    systems = [
        {
            "name": "GMapping",
            "year": 2007,
            "method": "粒子滤波",
            "map": "栅格地图",
            "sensor": "2D激光",
            "pros": ["计算量小", "经典"],
            "cons": ["粒子退化", "只2D"]
        },
        {
            "name": "LOAM",
            "year": 2014,
            "method": "特征匹配",
            "map": "点云地图",
            "sensor": "3D激光",
            "pros": ["精度高", "实时"],
            "cons": ["无回环检测"]
        },
        {
            "name": "Cartographer",
            "year": 2016,
            "method": "Submap + Ceres",
            "map": "混合地图",
            "sensor": "2D/3D",
            "pros": ["工程化好", "稳定"],
            "cons": ["计算量大"]
        },
        {
            "name": "LIO-SAM",
            "year": 2020,
            "method": "因子图",
            "map": "点云地图",
            "sensor": "3D激光+IMU",
            "pros": ["融合IMU", "回环检测"],
            "cons": ["复杂"]
        }
    ]
    
    for s in systems:
        print(f"{s['name']} ({s['year']})")
        print(f"  方法: {s['method']}")
        print(f"  地图: {s['map']}")
        print(f"  传感器: {s['sensor']}")
        print(f"  优点: {', '.join(s['pros'])}")
        print(f"  缺点: {', '.join(s['cons'])}")
        print()

# exercise_lidar_slam_comparison()
```

### 练习3：点云可视化

```python
def exercise_pointcloud():
    """点云练习"""
    print("=== 点云练习 ===")
    
    # 生成点云
    np.random.seed(42)
    
    # 平面1
    xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    zz = np.zeros_like(xx)
    plane1 = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # 平面2
    xx2, yy2 = np.meshgrid(np.linspace(-2, 2, 40), np.linspace(-2, 2, 40))
    zz2 = np.ones_like(xx2) * 3
    plane2 = np.stack([xx2, yy2, zz2], axis=-1).reshape(-1, 3)
    
    points = np.vstack([plane1, plane2])
    points += np.random.randn(*points.shape) * 0.02
    
    print(f"点云数量: {len(points)}")
    
    # Open3D可视化
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        
        o3d.io.write_point_cloud("example_pointcloud.pcd", pcd)
        print("点云已保存到 example_pointcloud.pcd")
    except:
        print("Open3D未安装")

# exercise_pointcloud()
```

---

**下一节**：[多模态SLAM](04-multimodal-slam.md)

---

## 参考文献

1. Zhang, J., & Singh, S. (2014). LOAM: Lidar Odometry and Mapping in Real-Time.
2. Hess, W., et al. (2016). Real-Time Loop Closure in 2D LIDAR SLAM.
3. Grisetti, G., et al. (2007). Improved Techniques for Grid Mapping with Rao-Blackwellized Particle Filters.
4. Shan, T., et al. (2020). LIO-SAM: Tightly-coupled LIDAR-Inertial Odometry via Smoothing and Mapping.
5. Xu, W., & Zhang, F. (2021). FAST-LIO: A Fast, Robust LiDAR-Inertial Odometry Package by Tightly-Coupled Iterated Kalman Filter.
