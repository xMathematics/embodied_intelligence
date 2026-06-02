# 4.3 激光SLAM

## 目录

- [1. 激光SLAM简介](#1-激光slam简介)
- [2. 点云配准方法](#2-点云配准方法)
- [3. 经典激光SLAM系统](#3-经典激光slam系统)
- [4. 激光+惯性融合](#4-激光惯性融合)
- [5. 实践练习](#5-实践练习)

---

## 1. 激光SLAM简介

### 1.1 什么是激光SLAM

**激光SLAM (LiDAR SLAM)** 使用激光雷达作为主要传感器，通过点云配准同时估计机器人运动和构建环境地图。

**问题提出**：
- 视觉SLAM在光照变化、纹理缺失场景下表现不佳
- 激光雷达提供精确的距离测量，不受光照影响
- 如何高效处理大规模点云数据？

**解决方案**：
- 点云配准算法（ICP、NDT）
- 特征提取与匹配
- 多传感器融合

### 1.2 激光传感器类型

```python
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2

class LidarSensor:
    """激光传感器"""
    
    def __init__(self, num_channels=64, points_per_channel=1024,
                 fov_up=20, fov_down=-24.8, max_range=100):
        """
        参数:
            num_channels: 线数（如64线、32线、16线）
            points_per_channel: 每线点数
            fov_up: 上视场角（度）
            fov_down: 下视场角（度）
            max_range: 最大测距（米）
        """
        self.num_channels = num_channels
        self.points_per_channel = points_per_channel
        self.fov_up = np.radians(fov_up)
        self.fov_down = np.radians(fov_down)
        self.max_range = max_range
    
    def simulate_scan(self, scene_func):
        """
        模拟一次激光扫描
        
        参数:
            scene_func: 场景函数，输入点返回是否被占用
        
        返回:
            points: 扫描点云 (N x 3)
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
        """
        转距离图像
        
        参数:
            points: 点云 (N x 3)
        
        返回:
            range_img: 距离图像
        """
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


class LidarTypes:
    """激光雷达类型"""
    
    @staticmethod
    def mechanical_lidar():
        """
        机械旋转式激光雷达
        
        **核心思想**：
        通过机械旋转实现360度扫描。
        
        **问题提出**：
        如何获得全向环境感知？
        
        **解决方案**：
        - 电机驱动激光发射器和接收器旋转
        - 多线激光同时扫描不同高度
        
        **优缺点**：
        - 优点：360度视野、精度高、测距远
        - 缺点：机械部件易磨损、成本高、体积大
        
        **代表产品**：Velodyne HDL-64E、Ouster OS1、RoboSense RS-LiDAR
        
        **论文核心思想**：
        Velodyne公司2005年首次推出64线激光雷达，
        通过垂直排列64组激光发射/接收对，实现高密度点云采集。
        """
        return {
            "name": "机械旋转式",
            "examples": ["Velodyne HDL-64E", "Ouster OS1", "RoboSense RS-LiDAR"],
            "fov": "360°水平，±15-30°垂直",
            "range": "100-200m",
            "points_per_second": "0.6M-2.2M",
            "pros": ["360度视野", "精度高", "测距远"],
            "cons": ["机械磨损", "成本高", "体积大"],
            "key_papers": [
                "Velodyne (2005) - 64线激光雷达技术"
            ]
        }
    
    @staticmethod
    def mems_lidar():
        """
        MEMS固态激光雷达
        
        **核心思想**：
        使用微机电系统（MEMS）微镜偏转激光束。
        
        **问题提出**：
        如何减小激光雷达体积并提高可靠性？
        
        **解决方案**：
        - 微机电系统控制微镜偏转
        - 无需机械旋转部件
        
        **优缺点**：
        - 优点：体积小、可靠性高、成本低
        - 缺点：视场角有限、扫描速度受限
        
        **代表产品**：Quanergy S3、Velodyne Velabit
        
        **论文核心思想**：
        MEMS技术将机械部件微型化，
        通过微镜的快速振动实现激光束扫描。
        """
        return {
            "name": "MEMS固态",
            "examples": ["Quanergy S3", "Velodyne Velabit"],
            "fov": "120°水平，25°垂直",
            "range": "50-150m",
            "points_per_second": "0.1M-0.5M",
            "pros": ["体积小", "可靠性高", "成本低"],
            "cons": ["视场角有限", "扫描速度受限"],
            "key_papers": [
                "MEMS-based LiDAR scanning technology"
            ]
        }
    
    @staticmethod
    def flash_lidar():
        """
        Flash固态激光雷达
        
        **核心思想**：
        一次性照亮整个场景，无需扫描。
        
        **问题提出**：
        如何消除扫描延迟，实现瞬时成像？
        
        **解决方案**：
        - 大面积激光脉冲照亮场景
        - 传感器阵列同时接收反射光
        
        **优缺点**：
        - 优点：无扫描延迟、高帧率、抗振动
        - 缺点：有效距离短、功耗大
        
        **代表产品**：Advanced Scientific Concepts
        
        **论文核心思想**：
        Flash LiDAR借鉴相机原理，
        通过单次曝光获取整个场景的深度信息。
        """
        return {
            "name": "Flash固态",
            "examples": ["Advanced Scientific Concepts"],
            "fov": "45°x45°",
            "range": "10-50m",
            "points_per_second": "0.5M-1M",
            "pros": ["无扫描延迟", "高帧率", "抗振动"],
            "cons": ["有效距离短", "功耗大"],
            "key_papers": [
                "Flash LiDAR technology"
            ]
        }
    
    @staticmethod
    def opa_lidar():
        """
        OPA光学相控阵激光雷达
        
        **核心思想**：
        使用光学相控阵技术控制激光束方向。
        
        **问题提出**：
        如何实现纯固态、高可靠性的激光雷达？
        
        **解决方案**：
        - 光学相控阵通过相位差控制光束方向
        - 无机械运动部件
        
        **优缺点**：
        - 优点：纯固态、扫描速度快、抗干扰
        - 缺点：旁瓣干扰、技术不成熟
        
        **代表产品**：Lumotive
        
        **论文核心思想**：
        OPA借鉴雷达相控阵技术，
        通过控制光波相位实现光束扫描。
        """
        return {
            "name": "OPA光学相控阵",
            "examples": ["Lumotive"],
            "fov": "120°水平，15°垂直",
            "range": "50-100m",
            "points_per_second": "待提升",
            "pros": ["纯固态", "扫描速度快", "抗干扰"],
            "cons": ["旁瓣干扰", "技术不成熟"],
            "key_papers": [
                "Optical Phased Array technology"
            ]
        }
```

### 1.3 激光SLAM vs 视觉SLAM

```python
class LidarVSVisual:
    """激光 vs 视觉对比"""
    
    @staticmethod
    def compare():
        """
        激光SLAM与视觉SLAM对比
        
        **激光SLAM优势**：
        - 不受光照影响（白天/夜间均可工作）
        - 测距精度高（厘米级）
        - 360度视野（机械式）
        - 可直接获取3D结构
        
        **视觉SLAM优势**：
        - 成本低（相机便宜）
        - 信息丰富（颜色、纹理）
        - 体积小、功耗低
        - 可用于场景理解
        
        **适用场景**：
        - 激光SLAM：自动驾驶、机器人导航、测绘
        - 视觉SLAM：AR/VR、室内导航、无人机
        """
        comparison = {
            "精度": {
                "激光": "厘米级（1-3cm）",
                "视觉": "分米级（5-20cm）",
                "winner": "激光"
            },
            "光照鲁棒性": {
                "激光": "非常好（主动光源）",
                "视觉": "差（依赖环境光）",
                "winner": "激光"
            },
            "成本": {
                "激光": "高（数千-数万美元）",
                "视觉": "低（数十-数百美元）",
                "winner": "视觉"
            },
            "视野": {
                "激光": "360度（机械式）",
                "视觉": "有限（通常<120度）",
                "winner": "激光"
            },
            "语义信息": {
                "激光": "无（仅几何）",
                "视觉": "丰富（颜色、纹理）",
                "winner": "视觉"
            },
            "功耗": {
                "激光": "高（10-30W）",
                "视觉": "低（1-5W）",
                "winner": "视觉"
            },
            "体积": {
                "激光": "大",
                "视觉": "小",
                "winner": "视觉"
            },
            "动态物体": {
                "激光": "较好",
                "视觉": "较差",
                "winner": "激光"
            }
        }
        return comparison
```

---

## 2. 点云配准方法

### 2.1 ICP (Iterative Closest Point)

**问题提出**：
有两帧点云，如何找到它们之间的最优刚体变换？

**解决方案**：
1. 找到源点云中每个点在目标点云中的最近点
2. 计算使对应点距离最小的刚体变换
3. 迭代直到收敛

**优缺点**：
- 优点：精度高、收敛快（局部最优）
- 缺点：需要好的初始值、对噪声敏感、可能陷入局部最优

**论文核心思想**：
Besl & McKay (1992) 提出了ICP算法，
通过迭代最近点匹配和最小二乘优化实现点云配准。

```python
class ICP:
    """ICP点云配准算法"""
    
    def __init__(self, max_iterations=50, tolerance=1e-6, 
                 max_correspondence_distance=1.0):
        """
        参数:
            max_iterations: 最大迭代次数
            tolerance: 收敛阈值
            max_correspondence_distance: 最大对应点距离
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
    
    def register(self, source, target, init_transform=None):
        """
        ICP配准
        
        参数:
            source: 源点云 (N x 3)
            target: 目标点云 (M x 3)
            init_transform: 初始变换 (4 x 4)
        
        返回:
            transform: 最终变换 (4 x 4)
            fitness: 拟合度
            rmse: 均方根误差
        """
        if init_transform is None:
            transform = np.eye(4)
        else:
            transform = init_transform.copy()
        
        prev_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # 1. 变换源点云
            source_transformed = self._transform_points(source, transform)
            
            # 2. 寻找最近点对应
            correspondences = self._find_correspondences(
                source_transformed, target
            )
            
            # 3. 计算最优变换
            delta_transform = self._compute_optimal_transform(
                source, target, correspondences
            )
            
            # 4. 更新变换
            transform = delta_transform @ transform
            
            # 5. 计算误差
            error = self._compute_error(source_transformed, target, correspondences)
            
            # 6. 检查收敛
            if abs(prev_error - error) < self.tolerance:
                break
            
            prev_error = error
        
        # 计算拟合度
        fitness = len(correspondences) / len(source)
        rmse = np.sqrt(error)
        
        return transform, fitness, rmse
    
    def _transform_points(self, points, transform):
        """变换点云"""
        points_h = np.hstack([points, np.ones((len(points), 1))])
        transformed = (transform @ points_h.T).T
        return transformed[:, :3]
    
    def _find_correspondences(self, source, target):
        """
        寻找最近点对应
        
        使用KD树加速最近邻搜索
        """
        from scipy.spatial import KDTree
        
        tree = KDTree(target)
        distances, indices = tree.query(source)
        
        # 过滤距离过大的对应
        valid = distances < self.max_correspondence_distance
        correspondences = [(i, indices[i]) for i in range(len(source)) if valid[i]]
        
        return correspondences
    
    def _compute_optimal_transform(self, source, target, correspondences):
        """
        计算最优刚体变换（SVD方法）
        
        基于Kabsch算法
        """
        if len(correspondences) < 3:
            return np.eye(4)
        
        # 提取对应点
        src_points = np.array([source[i] for i, _ in correspondences])
        tgt_points = np.array([target[j] for _, j in correspondences])
        
        # 计算质心
        src_centroid = np.mean(src_points, axis=0)
        tgt_centroid = np.mean(tgt_points, axis=0)
        
        # 中心化
        src_centered = src_points - src_centroid
        tgt_centered = tgt_points - tgt_centroid
        
        # 计算协方差矩阵
        H = src_centered.T @ tgt_centered
        
        # SVD分解
        U, S, Vt = np.linalg.svd(H)
        
        # 计算旋转
        R = Vt.T @ U.T
        
        # 处理反射情况
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 计算平移
        t = tgt_centroid - R @ src_centroid
        
        # 构建变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        return transform
    
    def _compute_error(self, source, target, correspondences):
        """计算配准误差"""
        if len(correspondences) == 0:
            return float('inf')
        
        errors = []
        for i, j in correspondences:
            error = np.linalg.norm(source[i] - target[j])
            errors.append(error ** 2)
        
        return np.mean(errors)


class ICPVariants:
    """ICP变体"""
    
    @staticmethod
    def point_to_point():
        """
        点到点ICP（标准ICP）
        
        **核心思想**：
        最小化对应点之间的欧氏距离。
        
        **目标函数**：
        $$E = \\sum_{i} \\|R p_i + t - q_i\\|^2$$
        
        **优缺点**：
        - 优点：简单、计算快
        - 缺点：不考虑表面几何
        """
        return {
            "name": "点到点ICP",
            "error_metric": "欧氏距离",
            "convergence": "快",
            "accuracy": "一般"
        }
    
    @staticmethod
    def point_to_plane():
        """
        点到平面ICP
        
        **核心思想**：
        最小化点到目标点云局部平面的距离。
        
        **问题提出**：
        点到点ICP收敛慢，如何利用表面法向信息？
        
        **解决方案**：
        计算目标点云的表面法向，
        最小化点到平面的距离而非点到点距离。
        
        **目标函数**：
        $$E = \\sum_{i} ((R p_i + t - q_i) \\cdot n_i)^2$$
        
        其中 $n_i$ 是目标点 $q_i$ 处的法向。
        
        **优缺点**：
        - 优点：收敛快、精度高
        - 缺点：需要计算法向、对噪声敏感
        
        **论文核心思想**：
        Chen & Medioni (1992) 提出了点到平面ICP，
        通过利用表面法向信息加速收敛。
        """
        return {
            "name": "点到平面ICP",
            "error_metric": "点到平面距离",
            "convergence": "快",
            "accuracy": "高",
            "key_papers": [
                "Chen & Medioni (1992) - Object modelling by registration of multiple range images"
            ]
        }
    
    @staticmethod
    def generalized_icp():
        """
        广义ICP (GICP)
        
        **核心思想**：
        统一点到点和点到平面ICP的概率框架。
        
        **问题提出**：
        如何在一个框架中处理不同几何特征？
        
        **解决方案**：
        将点云建模为高斯分布，
        通过协方差矩阵编码局部几何。
        
        **目标函数**：
        $$E = \\sum_{i} d_i^T (C_i^s + R C_i^t R^T)^{-1} d_i$$
        
        其中 $C_i^s$ 和 $C_i^t$ 是源点和目标点的协方差矩阵。
        
        **优缺点**：
        - 优点：统一框架、自适应几何
        - 缺点：计算协方差开销大
        
        **论文核心思想**：
        Segal et al. (2009) 提出了GICP，
        通过概率框架统一了点云配准方法。
        """
        return {
            "name": "广义ICP (GICP)",
            "error_metric": "马氏距离",
            "convergence": "快",
            "accuracy": "高",
            "key_papers": [
                "Segal et al. (2009) - Generalized-ICP"
            ]
        }
    
    @staticmethod
    def trimmed_icp():
        """
        Trimmed ICP (TrICP)
        
        **核心思想**：
        只使用部分最佳对应点进行配准。
        
        **问题提出**：
        点云存在大量噪声和外点时如何提高鲁棒性？
        
        **解决方案**：
        只保留距离最小的部分对应点（如50%），
        忽略可能的外点。
        
        **优缺点**：
        - 优点：鲁棒性好、抗外点
        - 缺点：需要调参（trim比例）
        
        **论文核心思想**：
        Chetverikov et al. (2002) 提出了TrICP，
        通过截断最小二乘提高鲁棒性。
        """
        return {
            "name": "Trimmed ICP",
            "error_metric": "截断最小二乘",
            "convergence": "中等",
            "accuracy": "高",
            "robustness": "好",
            "key_papers": [
                "Chetverikov et al. (2002) - The Trimmed Iterative Closest Point algorithm"
            ]
        }
```

### 2.2 NDT (Normal Distributions Transform)

**问题提出**：
ICP需要计算最近点对应，计算量大。如何提高配准效率？

**解决方案**：
将点云离散化为体素网格，
每个体素用高斯分布表示，
最大化源点云在目标分布中的似然。

**优缺点**：
- 优点：不需要显式对应关系、对初始值不敏感、适合大场景
- 缺点：需要调参（体素大小）、精度略低于ICP

**论文核心思想**：
Biber & Straßer (2003) 提出了NDT算法，
通过概率分布表示点云，避免了耗时的最近邻搜索。

```python
class NDT:
    """NDT点云配准算法"""
    
    def __init__(self, resolution=1.0, max_iterations=50, tolerance=1e-6):
        """
        参数:
            resolution: 体素分辨率
            max_iterations: 最大迭代次数
            tolerance: 收敛阈值
        """
        self.resolution = resolution
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # NDT地图
        self.target_ndt = None
    
    def build_ndt_map(self, points):
        """
        构建NDT地图
        
        参数:
            points: 点云 (N x 3)
        
        返回:
            ndt_map: NDT地图 {voxel_idx: (mean, cov)}
        """
        # 计算体素索引
        voxel_indices = np.floor(points / self.resolution).astype(int)
        
        # 按体素分组
        ndt_map = {}
        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in ndt_map:
                ndt_map[key] = []
            ndt_map[key].append(points[i])
        
        # 计算每个体素的高斯分布
        ndt_distributions = {}
        for key, pts in ndt_map.items():
            if len(pts) < 3:  # 至少需要3个点
                continue
            
            pts_array = np.array(pts)
            mean = np.mean(pts_array, axis=0)
            cov = np.cov(pts_array.T)
            
            # 正则化协方差矩阵
            cov += np.eye(3) * 1e-3
            
            ndt_distributions[key] = (mean, cov)
        
        return ndt_distributions
    
    def register(self, source, target, init_transform=None):
        """
        NDT配准
        
        参数:
            source: 源点云 (N x 3)
            target: 目标点云 (M x 3)
            init_transform: 初始变换 (4 x 4)
        
        返回:
            transform: 最终变换 (4 x 4)
            score: 配准分数
        """
        # 构建目标NDT地图
        self.target_ndt = self.build_ndt_map(target)
        
        if init_transform is None:
            transform = np.eye(4)
        else:
            transform = init_transform.copy()
        
        prev_score = -float('inf')
        
        for iteration in range(self.max_iterations):
            # 变换源点云
            source_transformed = self._transform_points(source, transform)
            
            # 计算得分和梯度
            score, gradient = self._compute_score_and_gradient(source_transformed)
            
            # 使用牛顿法更新
            # 这里简化为梯度下降
            step_size = 0.1
            delta = step_size * gradient
            
            # 更新变换
            delta_transform = self._exp_map(delta)
            transform = delta_transform @ transform
            
            # 检查收敛
            if abs(score - prev_score) < self.tolerance:
                break
            
            prev_score = score
        
        return transform, prev_score
    
    def _transform_points(self, points, transform):
        """变换点云"""
        points_h = np.hstack([points, np.ones((len(points), 1))])
        transformed = (transform @ points_h.T).T
        return transformed[:, :3]
    
    def _compute_score_and_gradient(self, points):
        """
        计算NDT得分和梯度
        
        得分：点云在目标分布中的对数似然
        """
        score = 0
        gradient = np.zeros(6)  # 6自由度
        
        for point in points:
            # 找到对应的体素
            voxel_idx = tuple(np.floor(point / self.resolution).astype(int))
            
            if voxel_idx not in self.target_ndt:
                continue
            
            mean, cov = self.target_ndt[voxel_idx]
            
            # 计算马氏距离
            diff = point - mean
            try:
                cov_inv = np.linalg.inv(cov)
                mahalanobis = diff.T @ cov_inv @ diff
                
                # 高斯似然
                likelihood = np.exp(-0.5 * mahalanobis)
                score += likelihood
                
                # 计算梯度（简化）
                # 实际实现需要推导完整的雅可比
                
            except np.linalg.LinAlgError:
                continue
        
        return score, gradient
    
    def _exp_map(self, xi):
        """指数映射：李代数到李群"""
        # 简化的实现
        transform = np.eye(4)
        transform[:3, 3] = xi[:3]
        return transform
```

### 2.3 特征-based配准

**问题提出**：
ICP和NDT需要密集点云，计算量大。如何加速配准？

**解决方案**：
提取点云特征（如边缘、平面），
先进行特征匹配获得初始变换，
再用ICP/NDT精化。

```python
class PointCloudFeatures:
    """点云特征提取"""
    
    def __init__(self):
        pass
    
    def extract_edge_features(self, points, k=10):
        """
        提取边缘特征
        
        基于曲率：高曲率点为边缘点
        
        参数:
            points: 点云 (N x 3)
            k: 近邻数量
        
        返回:
            edge_points: 边缘点索引
        """
        from scipy.spatial import KDTree
        
        tree = KDTree(points)
        curvatures = []
        
        for i, point in enumerate(points):
            # 找到k近邻
            _, indices = tree.query(point, k=k+1)
            neighbors = points[indices[1:]]  # 排除自己
            
            # 计算协方差矩阵
            centered = neighbors - np.mean(neighbors, axis=0)
            cov = centered.T @ centered / len(neighbors)
            
            # 特征值分解
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)
            
            # 曲率 = 最小特征值 / 特征值和
            curvature = eigenvalues[0] / np.sum(eigenvalues)
            curvatures.append(curvature)
        
        curvatures = np.array(curvatures)
        
        # 选择高曲率点作为边缘点
        threshold = np.percentile(curvatures, 80)
        edge_points = np.where(curvatures > threshold)[0]
        
        return edge_points
    
    def extract_planar_features(self, points, k=10):
        """
        提取平面特征
        
        基于曲率：低曲率点为平面点
        
        参数:
            points: 点云 (N x 3)
            k: 近邻数量
        
        返回:
            planar_points: 平面点索引
        """
        from scipy.spatial import KDTree
        
        tree = KDTree(points)
        curvatures = []
        
        for i, point in enumerate(points):
            _, indices = tree.query(point, k=k+1)
            neighbors = points[indices[1:]]
            
            centered = neighbors - np.mean(neighbors, axis=0)
            cov = centered.T @ centered / len(neighbors)
            
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)
            
            curvature = eigenvalues[0] / np.sum(eigenvalues)
            curvatures.append(curvature)
        
        curvatures = np.array(curvatures)
        
        # 选择低曲率点作为平面点
        threshold = np.percentile(curvatures, 30)
        planar_points = np.where(curvatures < threshold)[0]
        
        return planar_points
    
    def compute_fpfh(self, points, normals, radius=0.5):
        """
        计算FPFH (Fast Point Feature Histograms)
        
        参数:
            points: 点云 (N x 3)
            normals: 法向 (N x 3)
            radius: 邻域半径
        
        返回:
            fpfh: FPFH特征 (N x 33)
        """
        # 简化的FPFH实现
        # 实际实现参考Rusu et al. (2009)
        
        from scipy.spatial import KDTree
        
        tree = KDTree(points)
        fpfh = []
        
        for i, point in enumerate(points):
            # 找到半径内的近邻
            indices = tree.query_ball_point(point, radius)
            
            if len(indices) < 3:
                fpfh.append(np.zeros(33))
                continue
            
            # 计算PFH（简化）
            pfh = self._compute_pfh(point, points[indices], 
                                   normals[i], normals[indices])
            
            fpfh.append(pfh)
        
        return np.array(fpfh)
    
    def _compute_pfh(self, point, neighbors, normal, neighbor_normals):
        """计算点特征直方图（简化）"""
        # 简化的实现
        return np.zeros(33)


class FeatureBasedRegistration:
    """基于特征的配准"""
    
    def __init__(self):
        self.feature_extractor = PointCloudFeatures()
    
    def register(self, source, target):
        """
        基于特征的配准
        
        1. 提取特征
        2. 特征匹配
        3. RANSAC估计初始变换
        4. ICP精化
        """
        # 1. 提取边缘特征
        source_edges = self.feature_extractor.extract_edge_features(source)
        target_edges = self.feature_extractor.extract_edge_features(target)
        
        print(f"源点云边缘点: {len(source_edges)}")
        print(f"目标点云边缘点: {len(target_edges)}")
        
        # 2. 特征匹配（简化）
        # 实际实现需要描述子匹配
        
        # 3. RANSAC估计初始变换
        # 4. ICP精化
        icp = ICP()
        transform, fitness, rmse = icp.register(source, target)
        
        return transform, fitness, rmse
```

---

## 3. 经典激光SLAM系统

### 3.1 LOAM (Lidar Odometry and Mapping)

**论文**：Zhang, J., & Singh, S. (2014). LOAM: Lidar Odometry and Mapping in Real-time.

**核心思想**：
- 高频里程计 + 低频建图
- 提取边缘和平面特征
- 特征到边缘/平面的距离最小化

**系统架构**：
1. **激光里程计**：高频（10Hz）估计位姿
2. **激光建图**：低频（1Hz）优化地图

**创新点**：
- 特征提取减少计算量
- 双线程架构平衡实时性和精度
- 适用于旋转式激光雷达

**局限性**：
- 需要特征丰富的环境
- 对快速旋转敏感
- 无回环检测

```python
class LOAM:
    """LOAM系统架构"""
    
    def __init__(self, scan_period=0.1):
        """
        参数:
            scan_period: 扫描周期（秒）
        """
        self.scan_period = scan_period
        
        # 特征提取
        self.feature_extractor = LOAMFeatureExtractor()
        
        # 里程计线程
        self.lidar_odometry = LidarOdometry()
        
        # 建图线程
        self.lidar_mapping = LidarMapping()
        
        # 状态
        self.pose = np.eye(4)
        self.trajectory = [self.pose.copy()]
    
    def process_scan(self, points):
        """
        处理一帧激光扫描
        
        参数:
            points: 点云 (N x 3)
        """
        # 1. 特征提取
        edge_features, planar_features = self.feature_extractor.extract(points)
        
        # 2. 激光里程计（高频）
        odometry_pose = self.lidar_odometry.estimate(
            edge_features, planar_features, self.pose
        )
        
        # 3. 激光建图（低频，每10帧）
        if len(self.trajectory) % 10 == 0:
            mapping_pose = self.lidar_mapping.optimize(
                edge_features, planar_features, odometry_pose
            )
            self.pose = mapping_pose
        else:
            self.pose = odometry_pose
        
        self.trajectory.append(self.pose.copy())
        
        return self.pose


class LOAMFeatureExtractor:
    """LOAM特征提取器"""
    
    def __init__(self, num_edge=2, num_planar=4):
        """
        参数:
            num_edge: 每扫描线提取的边缘特征数
            num_planar: 每扫描线提取的平面特征数
        """
        self.num_edge = num_edge
        self.num_planar = num_planar
    
    def extract(self, points):
        """
        提取边缘和平面特征
        
        参数:
            points: 点云 (N x 3)
        
        返回:
            edge_features: 边缘特征
            planar_features: 平面特征
        """
        # 计算曲率
        curvatures = self._compute_curvature(points)
        
        # 分类特征
        edge_features = []
        planar_features = []
        
        # 按扫描线分组（假设已知）
        # 简化实现：全局选择
        
        # 边缘特征：高曲率
        edge_indices = np.argsort(curvatures)[-self.num_edge * 100:]
        edge_features = points[edge_indices]
        
        # 平面特征：低曲率
        planar_indices = np.argsort(curvatures)[:self.num_planar * 100]
        planar_features = points[planar_indices]
        
        return edge_features, planar_features
    
    def _compute_curvature(self, points, k=5):
        """计算曲率"""
        from scipy.spatial import KDTree
        
        tree = KDTree(points)
        curvatures = []
        
        for point in points:
            _, indices = tree.query(point, k=k+1)
            neighbors = points[indices[1:]]
            
            centered = neighbors - np.mean(neighbors, axis=0)
            cov = centered.T @ centered / len(neighbors)
            
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)
            
            curvature = eigenvalues[0] / np.sum(eigenvalues)
            curvatures.append(curvature)
        
        return np.array(curvatures)


class LidarOdometry:
    """激光里程计"""
    
    def __init__(self):
        self.last_edge_features = None
        self.last_planar_features = None
    
    def estimate(self, edge_features, planar_features, initial_pose):
        """
        估计位姿
        
        最小化特征到对应边缘/平面的距离
        """
        if self.last_edge_features is None:
            self.last_edge_features = edge_features
            self.last_planar_features = planar_features
            return initial_pose
        
        # 使用点到线/面距离优化位姿
        # 简化为ICP
        
        self.last_edge_features = edge_features
        self.last_planar_features = planar_features
        
        return initial_pose


class LidarMapping:
    """激光建图"""
    
    def __init__(self):
        self.global_map = []
    
    def optimize(self, edge_features, planar_features, initial_pose):
        """
        优化建图
        
        将特征匹配到全局地图
        """
        # 更新全局地图
        self.global_map.append(edge_features)
        
        # 位姿优化
        # 简化为返回初始位姿
        
        return initial_pose
```

### 3.2 LeGO-LOAM

**论文**：Shan, T., & Englot, B. (2018). LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain.

**核心思想**：
- 地面优化：分离地面点，利用地面约束
- 轻量级：减少计算量，适合嵌入式平台
- 回环检测：添加位姿图优化

**改进点**：
- 地面分割提高精度
- 两步LM优化加速
- 支持回环检测

```python
class LeGOLoam:
    """LeGO-LOAM系统"""
    
    def __init__(self):
        self.ground_segmentation = GroundSegmentation()
        self.feature_extraction = LeGOFeatureExtraction()
        self.lidar_odometry = LeGOOdometry()
        self.lidar_mapping = LeGOMapping()
        self.loop_closure = LeGOLoopClosure()
    
    def process_scan(self, points):
        """处理扫描"""
        # 1. 地面分割
        ground_points, obstacle_points = self.ground_segmentation.segment(points)
        
        # 2. 特征提取
        features = self.feature_extraction.extract(
            ground_points, obstacle_points
        )
        
        # 3. 里程计
        pose = self.lidar_odometry.estimate(features)
        
        # 4. 建图
        self.lidar_mapping.update(features, pose)
        
        # 5. 回环检测
        if self.loop_closure.detect(pose):
            self.lidar_mapping.optimize_pose_graph()
        
        return pose


class GroundSegmentation:
    """地面分割"""
    
    def segment(self, points, sensor_height=1.7):
        """
        分割地面点
        
        基于RANSAC平面拟合
        """
        # 简化的地面分割
        # 假设地面在传感器下方一定高度
        
        ground_mask = points[:, 2] < -sensor_height + 0.5
        
        ground_points = points[ground_mask]
        obstacle_points = points[~ground_mask]
        
        return ground_points, obstacle_points
```

### 3.3 Cartographer

**论文**：Hess, W., et al. (2016). Real-Time Loop Closure in 2D LIDAR SLAM.

**核心思想**：
- 子地图：局部地图作为扫描匹配参考
- 回环检测：分支定界加速扫描匹配
- 位姿图优化：稀疏位姿调整（SPA）

**系统架构**：
1. **局部SLAM**：扫描匹配构建子地图
2. **全局SLAM**：回环检测和位姿图优化

**创新点**：
- 子地图减少漂移
- 分支定界加速回环检测
- 实时性能好

```python
class Cartographer:
    """Cartographer系统"""
    
    def __init__(self):
        self.local_slam = LocalSLAM()
        self.global_slam = GlobalSLAM()
        
        # 子地图
        self.submaps = []
        self.current_submap = None
    
    def process_scan(self, scan, pose_estimate):
        """
        处理扫描
        
        参数:
            scan: 激光扫描
            pose_estimate: 位姿估计
        """
        # 1. 局部SLAM
        pose = self.local_slam.match_scan(scan, pose_estimate)
        
        # 2. 插入子地图
        if self.current_submap is None or self.current_submap.is_finished():
            self.current_submap = SubMap()
            self.submaps.append(self.current_submap)
        
        self.current_submap.insert_scan(scan, pose)
        
        # 3. 全局SLAM（回环检测）
        self.global_slam.add_node(pose, scan)
        
        if self.global_slam.should_detect_loop():
            constraint = self.global_slam.detect_loop(self.submaps)
            if constraint:
                self.global_slam.add_constraint(constraint)
                self.global_slam.optimize()
        
        return pose


class SubMap:
    """子地图"""
    
    def __init__(self, max_scans=90):
        self.max_scans = max_scans
        self.num_scans = 0
        self.grid_map = None
    
    def insert_scan(self, scan, pose):
        """插入扫描"""
        self.num_scans += 1
        # 更新栅格地图
    
    def is_finished(self):
        """子地图是否完成"""
        return self.num_scans >= self.max_scans


class LocalSLAM:
    """局部SLAM"""
    
    def match_scan(self, scan, pose_estimate):
        """
        扫描匹配
        
        使用CSM (Correlation Scan Matching) 或 ICP
        """
        # 简化为返回估计位姿
        return pose_estimate


class GlobalSLAM:
    """全局SLAM"""
    
    def __init__(self):
        self.nodes = []
        self.constraints = []
        self.pose_graph = None
    
    def add_node(self, pose, scan):
        """添加节点"""
        self.nodes.append({'pose': pose, 'scan': scan})
    
    def should_detect_loop(self):
        """是否应该检测回环"""
        return len(self.nodes) % 10 == 0
    
    def detect_loop(self, submaps):
        """
        回环检测
        
        使用分支定界加速扫描匹配
        """
        # 简化的回环检测
        return None
    
    def add_constraint(self, constraint):
        """添加约束"""
        self.constraints.append(constraint)
    
    def optimize(self):
        """位姿图优化"""
        # 使用SPA (Sparse Pose Adjustment)
        pass
```

---

## 4. 激光+惯性融合

### 4.1 为什么需要融合IMU

**问题提出**：
- 激光雷达扫描频率有限（10-20Hz）
- 快速运动时扫描畸变严重
- 纯激光在特征缺失区域失效

**解决方案**：
融合IMU高频数据（100-1000Hz）：
- 去畸变：补偿运动畸变
- 预测：提供位姿初始估计
- 约束：增加运动模型约束

```python
class LidarIMUFusion:
    """激光+惯性融合"""
    
    def __init__(self):
        self.imu_integrator = IMUIntegrator()
        self.lidar_odometry = LidarOdometry()
        self.state_estimator = StateEstimator()
    
    def process_imu(self, accel, gyro, dt):
        """处理IMU数据"""
        self.imu_integrator.integrate(accel, gyro, dt)
    
    def process_lidar(self, points, timestamp):
        """处理激光数据"""
        # 1. 获取IMU预测
        predicted_pose = self.imu_integrator.get_prediction(timestamp)
        
        # 2. 去畸变
        undistorted_points = self._undistort(points, timestamp)
        
        # 3. 激光里程计
        lidar_pose = self.lidar_odometry.estimate(undistorted_points, predicted_pose)
        
        # 4. 融合
        fused_pose = self.state_estimator.fuse(lidar_pose, predicted_pose)
        
        # 5. 重置IMU积分器
        self.imu_integrator.reset(fused_pose)
        
        return fused_pose
    
    def _undistort(self, points, timestamp):
        """
        去畸变
        
        使用IMU数据补偿扫描期间的运动
        """
        # 简化的去畸变
        return points


class IMUIntegrator:
    """IMU积分器"""
    
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.eye(3)
        
        self.last_accel = None
        self.last_gyro = None
        self.last_time = None
    
    def integrate(self, accel, gyro, dt):
        """积分IMU数据"""
        if self.last_time is None:
            self.last_time = 0
            self.last_accel = accel
            self.last_gyro = gyro
            return
        
        # 中值积分
        avg_accel = (accel + self.last_accel) / 2
        avg_gyro = (gyro + self.last_gyro) / 2
        
        # 更新姿态（简化）
        angle = np.linalg.norm(avg_gyro) * dt
        if angle > 1e-6:
            axis = avg_gyro / np.linalg.norm(avg_gyro)
            delta_R = self._axis_angle_to_matrix(axis, angle)
            self.orientation = self.orientation @ delta_R
        
        # 更新速度
        global_accel = self.orientation @ avg_accel
        self.velocity += global_accel * dt
        
        # 更新位置
        self.position += self.velocity * dt
        
        self.last_accel = accel
        self.last_gyro = gyro
        self.last_time += dt
    
    def get_prediction(self, timestamp):
        """获取预测位姿"""
        pose = np.eye(4)
        pose[:3, :3] = self.orientation
        pose[:3, 3] = self.position
        return pose
    
    def reset(self, pose):
        """重置积分器"""
        self.position = pose[:3, 3]
        self.orientation = pose[:3, :3]
        self.velocity = np.zeros(3)
    
    def _axis_angle_to_matrix(self, axis, angle):
        """轴角到旋转矩阵（Rodrigues公式）"""
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R
```

### 4.2 LIO-SAM

**论文**：Shan, T., et al. (2020). LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping.

**核心思想**：
- 紧耦合：激光和IMU在因子图中联合优化
- 因子图：包含IMU预积分因子、激光里程计因子、GPS因子、回环因子
- iSAM2：增量平滑和建图

**系统架构**：
1. **IMU预积分**：高频预测和约束
2. **激光特征提取**：边缘和平面特征
3. **因子图优化**：多源信息融合

**创新点**：
- 紧耦合融合提高精度
- 因子图框架灵活扩展
- 支持GPS融合

```python
class LIOSAM:
    """LIO-SAM系统"""
    
    def __init__(self):
        self.imu_preintegration = IMUPreintegration()
        self.feature_extraction = LOAMFeatureExtractor()
        self.factor_graph = FactorGraph()
        self.isam2 = ISAM2()
    
    def process_imu(self, accel, gyro, timestamp):
        """处理IMU数据"""
        self.imu_preintegration.integrate(accel, gyro, timestamp)
    
    def process_lidar(self, points, timestamp):
        """处理激光数据"""
        # 1. 去畸变
        undistorted = self._undistort(points, timestamp)
        
        # 2. 特征提取
        edge_features, planar_features = self.feature_extraction.extract(undistorted)
        
        # 3. 激光匹配
        lidar_factor = self._create_lidar_factor(edge_features, planar_features)
        
        # 4. IMU因子
        imu_factor = self.imu_preintegration.get_factor()
        
        # 5. 添加到因子图
        self.factor_graph.add_factor(lidar_factor)
        self.factor_graph.add_factor(imu_factor)
        
        # 6. 优化
        result = self.isam2.optimize(self.factor_graph)
        
        # 7. 重置IMU预积分
        self.imu_preintegration.reset(result.get_pose())
        
        return result.get_pose()


class IMUPreintegration:
    """IMU预积分"""
    
    def __init__(self):
        self.delta_position = np.zeros(3)
        self.delta_velocity = np.zeros(3)
        self.delta_rotation = np.eye(3)
        
        self.covariance = np.eye(9)
        self.jacobian = np.eye(9)
    
    def integrate(self, accel, gyro, timestamp):
        """预积分"""
        # Forster et al. (2017) IMU预积分理论
        pass
    
    def get_factor(self):
        """获取IMU因子"""
        pass
    
    def reset(self, pose):
        """重置"""
        pass


class FactorGraph:
    """因子图"""
    
    def __init__(self):
        self.factors = []
        self.variables = []
    
    def add_factor(self, factor):
        """添加因子"""
        self.factors.append(factor)
    
    def add_variable(self, variable):
        """添加变量"""
        self.variables.append(variable)


class ISAM2:
    """iSAM2优化器"""
    
    def optimize(self, factor_graph):
        """增量优化"""
        # 简化的优化
        pass
```

---

## 5. 实践练习

### 练习1：ICP配准

```python
def exercise_icp_registration():
    """ICP配准练习"""
    print("=== ICP配准练习 ===")
    
    # 创建源点云（立方体）
    source = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float32)
    
    # 创建目标点云（添加变换）
    angle = np.radians(30)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    t = np.array([0.5, 0.3, 0.1])
    
    target = (R @ source.T).T + t
    
    # 添加噪声
    target += np.random.normal(0, 0.02, target.shape)
    
    # ICP配准
    icp = ICP(max_iterations=100)
    transform, fitness, rmse = icp.register(source, target)
    
    print(f"\n真实变换:")
    print(f"  旋转角度: 30°")
    print(f"  平移: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    
    print(f"\n估计变换:")
    print(f"  旋转矩阵:\n{transform[:3, :3]}")
    print(f"  平移: [{transform[0, 3]:.3f}, {transform[1, 3]:.3f}, {transform[2, 3]:.3f}]")
    
    print(f"\n配准结果:")
    print(f"  拟合度: {fitness:.3f}")
    print(f"  RMSE: {rmse:.4f} m")

# exercise_icp_registration()
```

### 练习2：点云特征提取

```python
def exercise_feature_extraction():
    """点云特征提取练习"""
    print("=== 点云特征提取练习 ===")
    
    # 创建模拟点云
    np.random.seed(42)
    
    # 平面点
    plane_points = np.random.rand(500, 3)
    plane_points[:, 2] = 0  # z=0平面
    
    # 边缘点
    edge_points = np.random.rand(100, 3)
    edge_points[:, 1] = 0
    edge_points[:, 2] = 0
    
    # 合并
    points = np.vstack([plane_points, edge_points])
    points += np.random.normal(0, 0.01, points.shape)
    
    # 特征提取
    extractor = PointCloudFeatures()
    
    edge_indices = extractor.extract_edge_features(points)
    planar_indices = extractor.extract_planar_features(points)
    
    print(f"总点数: {len(points)}")
    print(f"边缘点: {len(edge_indices)}")
    print(f"平面点: {len(planar_indices)}")
    
    # 可视化
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c='blue', s=1, alpha=0.5)
    ax1.set_title('原始点云')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[planar_indices, 0], 
               points[planar_indices, 1], 
               points[planar_indices, 2], 
               c='green', s=1, label='平面特征')
    ax2.scatter(points[edge_indices, 0], 
               points[edge_indices, 1], 
               points[edge_indices, 2], 
               c='red', s=5, label='边缘特征')
    ax2.set_title('特征提取')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('point_cloud_features.png', dpi=150)
    print("\n特征图已保存到 point_cloud_features.png")

# exercise_feature_extraction()
```

### 练习3：激光SLAM系统对比

```python
def exercise_lidar_slam_comparison():
    """激光SLAM系统对比"""
    print("=== 激光SLAM系统对比 ===\n")
    
    systems = [
        {
            "name": "LOAM",
            "year": 2014,
            "features": "边缘+平面",
            "threads": 2,
            "imu": False,
            "loop": False,
            "pros": ["实时性好", "精度高", "开创性工作"],
            "cons": ["无回环检测", "需要特征丰富环境"],
            "impact": "激光SLAM的里程碑"
        },
        {
            "name": "LeGO-LOAM",
            "year": 2018,
            "features": "地面优化+边缘/平面",
            "threads": 2,
            "imu": False,
            "loop": True,
            "pros": ["轻量级", "地面优化", "支持回环"],
            "cons": ["依赖地面检测", "室外效果一般"],
            "impact": "适合嵌入式平台"
        },
        {
            "name": "LIO-SAM",
            "year": 2020,
            "features": "紧耦合IMU+激光",
            "threads": 2,
            "imu": True,
            "loop": True,
            "pros": ["紧耦合精度高", "因子图框架", "支持GPS"],
            "cons": ["计算量大", "依赖IMU质量"],
            "impact": "当前主流激光惯性SLAM"
        },
        {
            "name": "Cartographer",
            "year": 2016,
            "features": "子地图+CSM",
            "threads": 2,
            "imu": True,
            "loop": True,
            "pros": ["回环检测强", "2D/3D支持", "实时性好"],
            "cons": ["2D效果优于3D", "参数调优复杂"],
            "impact": "工业级SLAM系统"
        },
        {
            "name": "FAST-LIO",
            "year": 2021,
            "features": "卡尔曼滤波+ikd-Tree",
            "threads": 1,
            "imu": True,
            "loop": False,
            "pros": ["速度极快", "精度高", "计算效率高"],
            "cons": ["无回环检测", "依赖IMU"],
            "impact": "最快的激光惯性里程计之一"
        }
    ]
    
    for i, s in enumerate(systems, 1):
        print(f"{i}. {s['name']} ({s['year']})")
        print(f"   特征: {s['features']}")
        print(f"   线程: {s['threads']}")
        print(f"   IMU支持: {'是' if s['imu'] else '否'}")
        print(f"   回环检测: {'是' if s['loop'] else '否'}")
        print(f"   优点: {', '.join(s['pros'])}")
        print(f"   缺点: {', '.join(s['cons'])}")
        print(f"   影响: {s['impact']}")
        print()

# exercise_lidar_slam_comparison()
```

### 练习4：激光雷达类型对比

```python
def exercise_lidar_types_comparison():
    """激光雷达类型对比"""
    print("=== 激光雷达类型对比 ===\n")
    
    types = LidarTypes
    
    lidars = [
        types.mechanical_lidar(),
        types.mems_lidar(),
        types.flash_lidar(),
        types.opa_lidar()
    ]
    
    print(f"{'类型':<15} {'视场角':<20} {'测距':<10} {'点数/秒':<15} {'优缺点'}")
    print("-" * 100)
    
    for lidar in lidars:
        pros_cons = f"优点: {', '.join(lidar['pros'][:2])}; 缺点: {', '.join(lidar['cons'][:2])}"
        print(f"{lidar['name']:<15} {lidar['fov']:<20} {lidar['range']:<10} {lidar['points_per_second']:<15} {pros_cons}")

# exercise_lidar_types_comparison()
```

---

## 6. 总结与展望

### 6.1 激光SLAM发展历程

| 时期 | 代表工作 | 主要特点 |
|------|----------|----------|
| 1992 | ICP算法 | 点云配准基础 |
| 2003 | NDT算法 | 概率配准方法 |
| 2014 | LOAM | 实时激光SLAM |
| 2016 | Cartographer | 子地图+回环检测 |
| 2018 | LeGO-LOAM | 地面优化 |
| 2020 | LIO-SAM | 紧耦合激光惯性融合 |
| 2021+ | FAST-LIO | 高效卡尔曼滤波 |

### 6.2 未来发展方向

1. **固态激光雷达SLAM**：适应固态雷达特性
2. **多激光雷达融合**：360度全覆盖
3. **深度学习增强**：点云特征学习
4. **语义激光SLAM**：结合语义信息
5. **终身学习**：持续适应环境变化

### 6.3 学习资源推荐

**开源代码**：
- LOAM: https://github.com/HKUST-Aerial-Robotics/A-LOAM
- LeGO-LOAM: https://github.com/RobustFieldAutonomyLab/LeGO-LOAM
- LIO-SAM: https://github.com/TixiaoShan/LIO-SAM
- Cartographer: https://github.com/cartographer-project
- FAST-LIO: https://github.com/hku-mars/FAST_LIO

**数据集**：
- KITTI: http://www.cvlibs.net/datasets/kitti/
- MulRan: https://sites.google.com/view/mulran-pr/dataset
- NCLT: http://robots.engin.umich.edu/nclt/
- Oxford Radar RobotCar: https://oxford-robotics-institute.github.io/radar-robotcar-dataset/

---

## 7. 参考文献

1. Besl, P. J., & McKay, N. D. (1992). A Method for Registration of 3-D Shapes. IEEE TPAMI.
2. Biber, P., & Straßer, W. (2003). The Normal Distributions Transform