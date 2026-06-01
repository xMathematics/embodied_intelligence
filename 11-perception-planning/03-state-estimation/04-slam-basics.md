# 3.4 SLAM基础

## 目录

- [1. 引言](#1-引言)
- [2. SLAM概述](#2-slam概述)
- [3. 前端跟踪](#3-前端跟踪)
- [4. 后端优化](#4-后端优化)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 SLAM的重要性

**SLAM (Simultaneous Localization and Mapping)** 同时定位与建图，是机器人在未知环境中自主导航的核心技术：一边跟踪自身位姿，一边构建环境地图。

### 1.2 SLAM分类

| 类型 | 说明 | 示例 |
|------|------|------|
| **视觉SLAM** | 使用相机 | ORB-SLAM、VINS-Mono |
| **激光SLAM** | 使用激光雷达 | Gmapping、LOAM、Cartographer |
| **多传感器SLAM** | 融合多种传感器 | LIO-SAM、LIOM |
| **语义SLAM** | 结合语义理解 | 语义SLAM系统 |

---

## 2. SLAM概述

### 2.1 SLAM系统架构

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

class Frame:
    """关键帧"""
    
    def __init__(self, idx, pose, keypoints=None, descriptors=None):
        self.idx = idx
        self.pose = pose  # 4x4变换矩阵
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.map_point_ids = []  # 关联的地图点


class MapPoint:
    """地图点"""
    
    def __init__(self, idx, position):
        self.idx = idx
        self.position = position  # 3D位置
        self.observations = []  # 观测到该点的帧ID
```

### 2.2 地图表示

```python
class Map:
    """地图类"""
    
    def __init__(self):
        self.frames = {}  # 帧ID -> Frame
        self.map_points = {}  # 地图点ID -> MapPoint
        self.next_frame_id = 0
        self.next_point_id = 0
    
    def add_frame(self, pose, keypoints=None, descriptors=None):
        """添加新帧"""
        frame = Frame(self.next_frame_id, pose, keypoints, descriptors)
        self.frames[self.next_frame_id] = frame
        self.next_frame_id += 1
        return frame
    
    def add_map_point(self, position):
        """添加地图点"""
        point = MapPoint(self.next_point_id, position)
        self.map_points[self.next_point_id] = point
        self.next_point_id += 1
        return point
    
    def get_all_points(self):
        """获取所有地图点"""
        points = [p.position for p in self.map_points.values()]
        return np.array(points) if points else np.empty((0, 3))
    
    def get_all_poses(self):
        """获取所有位姿"""
        poses = [f.pose for f in self.frames.values()]
        return np.array(poses) if poses else np.empty((0, 4, 4))
```

---

## 3. 前端跟踪

### 3.1 特征提取与匹配

```python
class Frontend:
    """SLAM前端"""
    
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        
        # 特征检测
        self.detector = cv2.ORB_create(nfeatures=1000)
        
        # 特征匹配
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
    
    def process_frame(self, img):
        """
        处理一帧图像
        
        参数:
            img: 输入图像
        
        返回:
            pose: 相对位姿
            matches: 特征匹配
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # 检测特征
        kp, des = self.detector.detectAndCompute(gray, None)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_des = des
            return np.eye(4), kp, kp
        
        # 匹配
        matches = self.matcher.match(self.prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 获取匹配点
        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
        
        # 计算本质矩阵
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.camera_matrix,
            cv2.RANSAC, 0.999, 1.0
        )
        
        # 恢复位姿
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
        
        # 构造变换矩阵
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten()
        
        # 更新
        self.prev_frame = gray
        self.prev_kp = kp
        self.prev_des = des
        
        return pose, pts1, pts2
```

### 3.2 三角化

```python
def triangulate_points(pts1, pts2, P1, P2):
    """
    三角化恢复3D点
    
    参数:
        pts1, pts2: Nx2 2D点
        P1, P2: 3x4 投影矩阵
    
    返回:
        points3d: Nx3 3D点
    """
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = pts4d[:3] / pts4d[3]
    return pts3d.T
```

### 3.3 回环检测

```python
class LoopDetector:
    """回环检测器"""
    
    def __init__(self):
        self.database = []
    
    def add_frame(self, descriptors):
        """添加帧到数据库"""
        self.database.append(descriptors)
    
    def detect_loop(self, descriptors, min_distance=0.7):
        """
        检测回环
        
        参数:
            descriptors: 当前帧描述符
            min_distance: 最小距离
        
        返回:
            loop_id: 回环帧ID，-1表示无回环
        """
        if len(self.database) < 3:
            return -1
        
        # 使用BOW或其他方法计算相似性
        # 这里简化实现
        return -1
```

---

## 4. 后端优化

### 4.1 位姿图优化

```python
class PoseGraph:
    """位姿图"""
    
    def __init__(self):
        self.nodes = {}  # 节点ID -> 位姿
        self.edges = []  # 约束边
    
    def add_node(self, node_id, pose):
        """添加节点"""
        self.nodes[node_id] = pose.copy()
    
    def add_edge(self, id1, id2, relative_pose, information=None):
        """
        添加边（约束）
        
        参数:
            id1, id2: 节点ID
            relative_pose: 相对位姿变换
            information: 信息矩阵
        """
        if information is None:
            information = np.eye(6)
        
        self.edges.append({
            'id1': id1,
            'id2': id2,
            'relative_pose': relative_pose.copy(),
            'information': information
        })
    
    def get_pose_matrix(self):
        """获取所有节点的位姿矩阵"""
        node_ids = sorted(self.nodes.keys())
        poses = []
        for idx in node_ids:
            poses.append(self.nodes[idx])
        return np.array(poses)
    
    def apply_updates(self, updates):
        """应用更新到节点"""
        for node_id, delta in updates.items():
            # 这里简化，实际需要SE(3)更新
            self.nodes[node_id] += delta
```

### 4.2 束调整 (Bundle Adjustment)

```python
class BundleAdjustment:
    """束调整"""
    
    def __init__(self):
        pass
    
    def reproject_point(self, point3d, pose, camera_matrix):
        """
        重投影3D点到图像平面
        
        参数:
            point3d: 3D点
            pose: 相机位姿
            camera_matrix: 内参
        
        返回:
            point2d: 2D点
        """
        # 变换到相机坐标系
        R = pose[:3, :3]
        t = pose[:3, 3]
        point_cam = R @ point3d + t
        
        # 投影
        point2d = camera_matrix @ point_cam
        point2d = point2d[:2] / point2d[2]
        
        return point2d
    
    def compute_reprojection_error(self, point3d, point2d, pose, camera_matrix):
        """计算重投影误差"""
        projected = self.reproject_point(point3d, pose, camera_matrix)
        error = projected - point2d
        return np.sum(error**2)
    
    def simple_ba(self, poses, points3d, measurements, camera_matrix, num_iterations=10):
        """
        简单的BA（示意用，实际用g2o或Ceres）
        
        参数:
            poses: 相机位姿列表
            points3d: 3D点列表
            measurements: 观测 (frame_id, point_id, point2d)
            camera_matrix: 内参
        
        返回:
            optimized_poses: 优化后的位姿
            optimized_points: 优化后的3D点
        """
        # 这里只是示意，实际需要用专门的优化库
        # 真实的BA使用g2o, Ceres, GTSAM等
        
        optimized_poses = poses.copy()
        optimized_points = points3d.copy()
        
        # 计算初始误差
        total_error = 0
        for frame_id, point_id, point2d in measurements:
            if frame_id < len(poses) and point_id < len(points3d):
                error = self.compute_reprojection_error(
                    points3d[point_id], point2d, poses[frame_id], camera_matrix
                )
                total_error += error
        
        print(f"初始总误差: {total_error:.4f}")
        
        return optimized_poses, optimized_points
```

### 4.3 简单的SLAM系统

```python
class SimpleSLAM:
    """简单的SLAM系统"""
    
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        
        self.frontend = Frontend(camera_matrix)
        self.map = Map()
        
        self.current_pose = np.eye(4)
        self.poses = [self.current_pose.copy()]
        
        # 初始帧
        self.map.add_frame(self.current_pose)
    
    def process_frame(self, img):
        """处理一帧图像"""
        # 前端
        relative_pose, pts1, pts2 = self.frontend.process_frame(img)
        
        # 更新位姿
        self.current_pose = self.current_pose @ relative_pose
        self.poses.append(self.current_pose.copy())
        
        # 添加关键帧（简化）
        keyframe = self.map.add_frame(self.current_pose)
        
        # 三角化（如果有足够的匹配）
        if len(pts1) > 10 and len(self.map.frames) > 1:
            # 获取前一帧位姿
            prev_frame_id = self.map.next_frame_id - 2
            prev_pose = self.map.frames[prev_frame_id].pose
            
            # 构建投影矩阵
            P1 = self.camera_matrix @ prev_pose[:3]
            P2 = self.camera_matrix @ self.current_pose[:3]
            
            # 三角化
            points3d = triangulate_points(pts1[:50], pts2[:50], P1, P2)
            
            # 添加地图点
            for p in points3d:
                if p[2] > 0:  # 在前方
                    mp = self.map.add_map_point(p)
                    keyframe.map_point_ids.append(mp.idx)
        
        return self.current_pose
    
    def get_trajectory(self):
        """获取轨迹"""
        return np.array(self.poses)
    
    def get_map_points(self):
        """获取地图点"""
        return self.map.get_all_points()
```

---

## 5. 实践练习

### 练习1：SLAM前端演示

```python
def exercise_slam_frontend():
    """SLAM前端练习"""
    print("=== SLAM前端练习 ===")
    
    # 相机内参
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])
    
    # 创建SLAM系统
    slam = SimpleSLAM(K)
    
    # 模拟图像序列
    num_frames = 20
    frames = []
    
    for i in range(num_frames):
        # 生成模拟图像
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 添加简单的运动
        if i > 0:
            dx = np.sin(i * 0.1) * 10
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            frame = cv2.warpAffine(frame, M, (640, 480))
        
        frames.append(frame)
    
    # 处理每帧
    for i, frame in enumerate(frames):
        pose = slam.process_frame(frame)
        position = pose[:3, 3]
        print(f"帧 {i}: 位置 = {position}")
    
    # 获取轨迹
    trajectory = slam.get_trajectory()
    map_points = slam.get_map_points()
    
    print(f"\n总帧数: {len(trajectory)}")
    print(f"地图点数: {len(map_points)}")
    
    # 绘制轨迹
    positions = trajectory[:, :3, 3]
    
    plt.figure(figsize=(10, 10))
    plt.plot(positions[:, 0], positions[:, 1], 'b-', label='轨迹')
    
    if len(map_points) > 0:
        plt.scatter(map_points[:, 0], map_points[:, 1], c='r', s=10, alpha=0.5, label='地图点')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('SLAM轨迹与地图')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('slam_trajectory.png')
    print("轨迹图已保存")

# exercise_slam_frontend()
```

### 练习2：位姿图构建

```python
def exercise_pose_graph():
    """位姿图练习"""
    print("=== 位姿图练习 ===")
    
    # 创建位姿图
    pose_graph = PoseGraph()
    
    # 添加节点（模拟轨迹）
    num_nodes = 10
    true_poses = []
    
    for i in range(num_nodes):
        # 模拟真实轨迹
        theta = i * 0.2
        x = np.cos(theta) * 5
        y = np.sin(theta) * 5
        
        pose = np.eye(4)
        pose[0, 3] = x
        pose[1, 3] = y
        true_poses.append(pose)
        
        # 添加噪声
        pose_noisy = pose.copy()
        pose_noisy[:3, 3] += np.random.randn(3) * 0.1
        pose_graph.add_node(i, pose_noisy)
    
    # 添加顺序边
    for i in range(num_nodes - 1):
        # 相对变换
        pose1 = true_poses[i]
        pose2 = true_poses[i + 1]
        
        # T_1to2 = T_1^{-1} @ T_2
        T1 = np.eye(4)
        T1[:3, :3] = pose1[:3, :3].T
        T1[:3, 3] = -pose1[:3, :3].T @ pose1[:3, 3]
        relative_pose = T1 @ pose2
        
        pose_graph.add_edge(i, i + 1, relative_pose)
    
    # 添加回环边
    if num_nodes > 3:
        T_first = true_poses[0]
        T_last = true_poses[-1]
        
        T1 = np.eye(4)
        T1[:3, :3] = T_first[:3, :3].T
        T1[:3, 3] = -T_first[:3, :3].T @ T_first[:3, 3]
        loop_pose = T1 @ T_last
        
        pose_graph.add_edge(0, num_nodes - 1, loop_pose)
    
    print(f"节点数: {len(pose_graph.nodes)}")
    print(f"边数: {len(pose_graph.edges)}")
    
    # 获取初始位姿
    initial_poses = pose_graph.get_pose_matrix()
    initial_positions = initial_poses[:, :3, 3]
    
    # 绘制初始和真实轨迹
    true_positions = np.array(true_poses)[:, :3, 3]
    
    plt.figure(figsize=(10, 10))
    plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='真实轨迹', linewidth=2)
    plt.plot(initial_positions[:, 0], initial_positions[:, 1], 'b--', label='初始估计')
    plt.scatter(initial_positions[:, 0], initial_positions[:, 1], c='b')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('位姿图初始状态')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('pose_graph_initial.png')
    print("位姿图已保存")

# exercise_pose_graph()
```

### 练习3：重投影误差计算

```python
def exercise_reprojection_error():
    """重投影误差练习"""
    print("=== 重投影误差练习 ===")
    
    K = np.array([
        [1000, 0, 500],
        [0, 1000, 375],
        [0, 0, 1]
    ])
    
    ba = BundleAdjustment()
    
    # 生成3D点
    num_points = 50
    points3d = np.random.randn(num_points, 3)
    points3d[:, 2] = np.abs(points3d[:, 2]) + 2  # 确保在前方
    
    # 真实位姿
    true_pose = np.eye(4)
    
    # 生成观测
    measurements = []
    for i, p in enumerate(points3d):
        p2d = ba.reproject_point(p, true_pose, K)
        measurements.append((0, i, p2d))
    
    # 计算带噪声的误差
    noise_levels = [0, 0.1, 0.5, 1.0, 2.0]
    
    print("\n噪声水平 (像素) | 平均重投影误差")
    print("-" * 40)
    
    for noise in noise_levels:
        total_error = 0
        for frame_id, point_id, point2d in measurements:
            # 添加噪声
            point2d_noisy = point2d + np.random.randn(2) * noise
            error = ba.compute_reprojection_error(points3d[point_id], point2d_noisy, true_pose, K)
            total_error += error
        
        avg_error = total_error / len(measurements)
        print(f"{noise:13.1f} | {avg_error:15.4f}")

# exercise_reprojection_error()
```

---

**下一节**：[多传感器融合](05-sensor-fusion.md)

---

## 参考文献

1. Durrant-Whyte, H., & Bailey, T. (2006). Simultaneous Localization and Mapping: Part I.
2. Mur-Artal, R., et al. (2015). ORB-SLAM: A Versatile and Accurate Monocular SLAM System.
3. Triggs, B., et al. (2000). Bundle Adjustment — A Modern Synthesis.
4. Grisetti, G., et al. (2010). A Tutorial on Graph-Based SLAM.
5. Zhang, J., & Singh, S. (2014). LOAM: Lidar Odometry and Mapping in Real-Time.
