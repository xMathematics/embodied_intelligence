# 2.4 LiDAR点云

## 目录

- [1. 引言](#1-引言)
- [2. LiDAR点云概述](#2-lidar点云概述)
- [3. 点云基础操作](#3-点云基础操作)
- [4. 点云配准](#4-点云配准)
- [5. 点云深度学习](#5-点云深度学习)
- [6. 实践练习](#6-实践练习)

---

## 1. 引言

### 1.1 LiDAR点云的重要性

**LiDAR (Light Detection and Ranging)** 通过激光测距获取高精度三维点云数据，是自动驾驶、机器人导航、三维重建等领域的核心传感器。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **自动驾驶** | 环境感知和障碍物检测 | 激光雷达车辆定位 |
| **机器人导航** | 同时定位与地图构建(SLAM) | 移动机器人避障 |
| **三维重建** | 场景数字化 | 文物保护、建筑建模 |
| **测绘** | 地形测量和绘制 | 地理信息系统(GIS) |

---

## 2. LiDAR点云概述

### 2.1 点云数据结构

```python
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

class PointCloud:
    def __init__(self, points=None, colors=None, normals=None):
        """
        点云类
        
        参数:
            points: Nx3 点坐标
            colors: Nx3 颜色
            normals: Nx3 法向量
        """
        self.points = points
        self.colors = colors
        self.normals = normals
    
    @classmethod
    def from_numpy(cls, points, colors=None, normals=None):
        """从NumPy数组创建点云"""
        return cls(points, colors, normals)
    
    @classmethod
    def from_open3d(cls, o3d_pcd):
        """从Open3D点云创建"""
        points = np.asarray(o3d_pcd.points)
        colors = np.asarray(o3d_pcd.colors) if o3d_pcd.has_colors() else None
        normals = np.asarray(o3d_pcd.normals) if o3d_pcd.has_normals() else None
        return cls(points, colors, normals)
    
    def to_open3d(self):
        """转换为Open3D点云"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        if self.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.colors)
        if self.normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(self.normals)
        return pcd
    
    def save(self, filename):
        """保存点云"""
        pcd = self.to_open3d()
        o3d.io.write_point_cloud(filename, pcd)
    
    @classmethod
    def load(cls, filename):
        """加载点云"""
        pcd = o3d.io.read_point_cloud(filename)
        return cls.from_open3d(pcd)
    
    def visualize(self):
        """可视化点云"""
        pcd = self.to_open3d()
        o3d.visualization.draw_geometries([pcd])
```

### 2.2 模拟LiDAR点云

```python
class LiDARSensor:
    def __init__(self, num_lines=64, points_per_line=1024,
                 fov_up=20, fov_down=-24.8, max_range=100):
        """
        模拟LiDAR传感器
        
        参数:
            num_lines: 线数
            points_per_line: 每线点数
            fov_up: 上视场角(度)
            fov_down: 下视场角(度)
            max_range: 最大测距(米)
        """
        self.num_lines = num_lines
        self.points_per_line = points_per_line
        self.fov_up = np.radians(fov_up)
        self.fov_down = np.radians(fov_down)
        self.max_range = max_range
    
    def generate_scan(self, scene_function):
        """
        生成一次LiDAR扫描
        
        参数:
            scene_function: 场景函数，输入(x,y,z)返回是否在场景内
        """
        points = []
        
        # 俯仰角
        pitch_angles = np.linspace(self.fov_down, self.fov_up, self.num_lines)
        
        # 方位角
        yaw_angles = np.linspace(0, 2 * np.pi, self.points_per_line, endpoint=False)
        
        for pitch in pitch_angles:
            for yaw in yaw_angles:
                # 射线方向
                direction = np.array([
                    np.cos(pitch) * np.sin(yaw),
                    np.cos(pitch) * np.cos(yaw),
                    np.sin(pitch)
                ])
                
                # 射线投射
                for depth in np.linspace(0, self.max_range, 1000):
                    point = direction * depth
                    if scene_function(point):
                        points.append(point)
                        break
        
        return PointCloud.from_numpy(np.array(points))
    
    def project_to_range_image(self, points):
        """
        将点云投影为距离图像
        
        返回:
            range_image: 距离图像
            point_indices: 对应点云索引
        """
        h = self.num_lines
        w = self.points_per_line
        
        range_image = np.zeros((h, w), dtype=np.float32)
        point_indices = -np.ones((h, w), dtype=np.int32)
        
        for i, point in enumerate(points):
            x, y, z = point
            
            # 计算球坐标
            r = np.linalg.norm(point)
            if r > self.max_range or r < 0.1:
                continue
            
            pitch = np.arcsin(z / r)
            yaw = np.arctan2(x, y)
            
            # 计算像素坐标
            u = 0.5 * (yaw / np.pi + 1.0)
            v = (pitch - self.fov_down) / (self.fov_up - self.fov_down)
            
            u_idx = int(u * w) % w
            v_idx = int(v * h)
            
            if 0 <= v_idx < h and 0 <= u_idx < w:
                if range_image[v_idx, u_idx] == 0 or r < range_image[v_idx, u_idx]:
                    range_image[v_idx, u_idx] = r
                    point_indices[v_idx, u_idx] = i
        
        return range_image, point_indices
```

---

## 3. 点云基础操作

### 3.1 滤波和降采样

```python
class PointCloudProcessor:
    def __init__(self):
        pass
    
    def statistical_outlier_removal(self, pcd, nb_neighbors=20, std_ratio=2.0):
        """
        统计滤波去除离群点
        
        参数:
            pcd: 点云
            nb_neighbors: 邻居数
            std_ratio: 标准差倍率
        """
        o3d_pcd = pcd.to_open3d()
        filtered, _ = o3d_pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return PointCloud.from_open3d(filtered)
    
    def voxel_downsample(self, pcd, voxel_size=0.05):
        """
        体素降采样
        
        参数:
            pcd: 点云
            voxel_size: 体素大小
        """
        o3d_pcd = pcd.to_open3d()
        downsampled = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)
        return PointCloud.from_open3d(downsampled)
    
    def passthrough_filter(self, pcd, axis='z', min_val=-1, max_val=1):
        """
        直通滤波
        
        参数:
            pcd: 点云
            axis: 过滤轴
            min_val, max_val: 最小/最大值
        """
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        mask = (pcd.points[:, axis_idx] >= min_val) & (pcd.points[:, axis_idx] <= max_val)
        
        filtered_points = pcd.points[mask]
        filtered_colors = pcd.colors[mask] if pcd.colors is not None else None
        filtered_normals = pcd.normals[mask] if pcd.normals is not None else None
        
        return PointCloud.from_numpy(filtered_points, filtered_colors, filtered_normals)
    
    def compute_normals(self, pcd, radius=0.1, max_nn=30):
        """
        计算法向量
        
        参数:
            pcd: 点云
            radius: 搜索半径
            max_nn: 最大邻居数
        """
        o3d_pcd = pcd.to_open3d()
        o3d_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )
        o3d_pcd.orient_normals_towards_camera_location()
        return PointCloud.from_open3d(o3d_pcd)
```

### 3.2 点云分割

```python
class PointCloudSegmentation:
    def __init__(self):
        pass
    
    def plane_segmentation(self, pcd, distance_threshold=0.01,
                           ransac_n=3, num_iterations=1000):
        """
        RANSAC平面分割
        
        参数:
            pcd: 点云
            distance_threshold: 距离阈值
            ransac_n: RANSAC采样点
            num_iterations: 迭代次数
        
        返回:
            plane_model: 平面模型
            inlier_indices: 内点索引
        """
        o3d_pcd = pcd.to_open3d()
        plane_model, inliers = o3d_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )
        
        inlier_cloud = o3d_pcd.select_by_index(inliers)
        outlier_cloud = o3d_pcd.select_by_index(inliers, invert=True)
        
        return (
            plane_model,
            PointCloud.from_open3d(inlier_cloud),
            PointCloud.from_open3d(outlier_cloud)
        )
    
    def euclidean_cluster(self, pcd, eps=0.02, min_points=10, max_points=1000):
        """
        欧式聚类
        
        参数:
            pcd: 点云
            eps: 搜索半径
            min_points: 最小点数
            max_points: 最大点数
        
        返回:
            clusters: 聚类列表
        """
        o3d_pcd = pcd.to_open3d()
        labels = np.array(o3d_pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False
        ))
        
        max_label = labels.max()
        clusters = []
        
        for i in range(max_label + 1):
            indices = np.where(labels == i)[0]
            if len(indices) > 0:
                cluster_pcd = o3d_pcd.select_by_index(indices)
                clusters.append(PointCloud.from_open3d(cluster_pcd))
        
        return clusters
    
    def ground_segmentation(self, pcd, height_threshold=0.2):
        """
        简单地面分割
        
        参数:
            pcd: 点云
            height_threshold: 高度阈值
        """
        # 假设z轴向上
        ground_mask = pcd.points[:, 2] < height_threshold
        
        ground_points = pcd.points[ground_mask]
        non_ground_points = pcd.points[~ground_mask]
        
        ground_colors = pcd.colors[ground_mask] if pcd.colors is not None else None
        non_ground_colors = pcd.colors[~ground_mask] if pcd.colors is not None else None
        
        ground = PointCloud.from_numpy(ground_points, ground_colors)
        non_ground = PointCloud.from_numpy(non_ground_points, non_ground_colors)
        
        return ground, non_ground
```

---

## 4. 点云配准

### 4.1 ICP配准

```python
class PointCloudRegistration:
    def __init__(self):
        pass
    
    def icp_registration(self, source_pcd, target_pcd,
                         init_transform=np.eye(4),
                         max_correspondence_distance=0.05,
                         max_iterations=50):
        """
        ICP配准
        
        参数:
            source_pcd: 源点云
            target_pcd: 目标点云
            init_transform: 初始变换
            max_correspondence_distance: 最大对应距离
            max_iterations: 最大迭代次数
        
        返回:
            transformation: 变换矩阵
            evaluation: 配准评估
        """
        source = source_pcd.to_open3d()
        target = target_pcd.to_open3d()
        
        # 计算法向量
        source.estimate_normals()
        target.estimate_normals()
        
        # ICP配准
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance,
            init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations
            )
        )
        
        return reg_p2p.transformation, reg_p2p.fitness
    
    def icp_point_to_plane(self, source_pcd, target_pcd,
                          init_transform=np.eye(4),
                          max_correspondence_distance=0.05,
                          max_iterations=50):
        """点到面ICP配准"""
        source = source_pcd.to_open3d()
        target = target_pcd.to_open3d()
        
        source.estimate_normals()
        target.estimate_normals()
        
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance,
            init_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations
            )
        )
        
        return reg_p2l.transformation, reg_p2l.fitness
    
    def ransac_feature_matching(self, source_pcd, target_pcd,
                                 voxel_size=0.05):
        """基于特征的RANSAC配准"""
        source = source_pcd.to_open3d()
        target = target_pcd.to_open3d()
        
        # 降采样
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # 计算法向量
        source_down.estimate_normals()
        target_down.estimate_normals()
        
        # 计算FPFH特征
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
        )
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
        )
        
        # RANSAC配准
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=voxel_size * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )
        
        return result.transformation, result.fitness
    
    def transform_point_cloud(self, pcd, transform):
        """应用变换到点云"""
        o3d_pcd = pcd.to_open3d()
        o3d_pcd.transform(transform)
        return PointCloud.from_open3d(o3d_pcd)
    
    def visualize_registration(self, source_pcd, target_pcd, transform):
        """可视化配准结果"""
        source = source_pcd.to_open3d()
        target = target_pcd.to_open3d()
        
        source.transform(transform)
        
        source.paint_uniform_color([1, 0.706, 0])
        target.paint_uniform_color([0, 0.651, 0.929])
        
        o3d.visualization.draw_geometries([source, target])
```

---

## 5. 点云深度学习

### 5.1 PointNet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """变换网络 T-Net"""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        """前向传播"""
        batch_size = x.size(0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # 初始化为单位矩阵
        iden = torch.eye(self.k, dtype=x.dtype, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        
        return x

class PointNet(nn.Module):
    """PointNet分类网络"""
    def __init__(self, num_classes=40):
        super().__init__()
        
        self.input_transform = TNet(k=3)
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        
        self.feature_transform = TNet(k=64)
        
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: (B, 3, N) 点云
        """
        batch_size = x.size(0)
        
        # 输入变换
        trans = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        # MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # 特征变换
        trans_feat = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)
        
        # MLP
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # 最大池化
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # FC
        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

# 测试
model = PointNet(num_classes=10)
x = torch.randn(32, 3, 1024)  # (batch, 3, num_points)
out = model(x)
print(f"输出形状: {out.shape}")
```

### 5.2 PointNet++

```python
class PointNetSetAbstraction(nn.Module):
    """PointNet++ 集抽象层"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        # MLP
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    
    def forward(self, xyz, points):
        """
        前向传播
        
        参数:
            xyz: (B, N, 3) 点坐标
            points: (B, C, N) 点特征
        
        返回:
            new_xyz: (B, npoint, 3) 采样点
            new_points: (B, mlp[-1], npoint) 聚合特征
        """
        # 简化实现：随机采样
        if self.group_all:
            idx = torch.arange(xyz.shape[1]).unsqueeze(0).repeat(xyz.shape[0], 1)
        else:
            idx = torch.randint(0, xyz.shape[1], (xyz.shape[0], self.npoint))
        
        new_xyz = xyz[torch.arange(xyz.shape[0]).unsqueeze(1), idx]
        
        # 简化MLP
        if points is not None:
            new_points = points[:, :, idx[0]]
            new_points = new_points.unsqueeze(-1)
            
            for i, (conv, bn) in enumerate(zip(self.mlp_convs, self.mlp_bns)):
                new_points = F.relu(bn(conv(new_points)))
            
            new_points = torch.max(new_points, -1)[0]
        else:
            new_points = None
        
        return new_xyz, new_points

class PointNet2Cls(nn.Module):
    """PointNet++ 分类网络"""
    def __init__(self, num_classes=40):
        super().__init__()
        
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=3, mlp=[64, 64, 128], group_all=False
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, xyz):
        """前向传播"""
        B, N, _ = xyz.shape
        
        # Set Abstraction
        l1_xyz, l1_points = self.sa1(xyz, xyz.transpose(1, 2))
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # FC
        x = l3_points.view(B, 1024)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

# 测试
model = PointNet2Cls(num_classes=10)
xyz = torch.randn(32, 1024, 3)  # (batch, num_points, 3)
out = model(xyz)
print(f"输出形状: {out.shape}")
```

---

## 6. 实践练习

### 练习1：点云基础操作

```python
def basic_point_cloud_demo():
    """点云基础操作演示"""
    # 生成随机点云
    np.random.seed(42)
    points = np.random.randn(1000, 3)
    
    # 创建点云
    pcd = PointCloud.from_numpy(points)
    print(f"原始点数: {len(pcd.points)}")
    
    # 降采样
    processor = PointCloudProcessor()
    pcd_down = processor.voxel_downsample(pcd, voxel_size=0.2)
    print(f"降采样后点数: {len(pcd_down.points)}")
    
    # 计算法向量
    pcd_normals = processor.compute_normals(pcd_down)
    
    # 保存
    pcd_normals.save("demo_pointcloud.pcd")
    
    print("点云保存成功!")

# basic_point_cloud_demo()
```

### 练习2：平面分割

```python
def plane_segmentation_demo():
    """平面分割演示"""
    # 生成带平面的点云
    np.random.seed(42)
    
    # 平面点
    xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    zz = np.zeros_like(xx)
    plane_points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    
    # 噪声点
    noise_points = np.random.randn(500, 3) * 0.5
    
    # 合并
    points = np.vstack([plane_points, noise_points])
    
    pcd = PointCloud.from_numpy(points)
    
    # RANSAC平面分割
    segmenter = PointCloudSegmentation()
    plane_model, inlier_cloud, outlier_cloud = segmenter.plane_segmentation(
        pcd, distance_threshold=0.05
    )
    
    print(f"平面模型: {plane_model}")
    print(f"平面点数: {len(inlier_cloud.points)}")
    print(f"离群点数: {len(outlier_cloud.points)}")

# plane_segmentation_demo()
```

### 练习3：点云配准

```python
def registration_demo():
    """点云配准演示"""
    # 生成源点云
    np.random.seed(42)
    source_points = np.random.randn(500, 3)
    
    # 生成目标点云（带变换）
    angle = np.radians(30)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
    t = np.array([0.5, 0.3, 0])
    
    target_points = source_points @ R.T + t
    target_points += np.random.randn(*target_points.shape) * 0.01
    
    source = PointCloud.from_numpy(source_points)
    target = PointCloud.from_numpy(target_points)
    
    # ICP配准
    registration = PointCloudRegistration()
    transform, fitness = registration.icp_registration(
        source, target,
        max_correspondence_distance=0.1,
        max_iterations=50
    )
    
    print(f"配准变换矩阵:\n{transform}")
    print(f"配准精度: {fitness:.4f}")

# registration_demo()
```

---

**下一节**：[多传感器融合](05-multi-sensor-fusion.md)

---

## 参考文献

1. Qi, C. R., et al. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.
2. Qi, C. R., et al. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space.
3. Besl, P. J., & McKay, N. D. (1992). A Method for Registration of 3-D Shapes.
4. Rusu, R. B., & Cousins, S. (2011). 3D is here: Point Cloud Library (PCL).
