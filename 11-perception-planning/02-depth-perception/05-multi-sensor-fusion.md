# 2.5 多传感器融合

## 目录

- [1. 引言](#1-引言)
- [2. 多传感器融合概述](#2-多传感器融合概述)
- [3. 传感器标定](#3-传感器标定)
- [4. 数据级融合](#4-数据级融合)
- [5. 特征级和决策级融合](#5-特征级和决策级融合)
- [6. 实践练习](#6-实践练习)

---

## 1. 引言

### 1.1 多传感器融合的重要性

**多传感器融合**将来自不同传感器（相机、LiDAR、IMU等）的信息进行整合，弥补单一传感器的缺陷，获得更准确、鲁棒的环境感知。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **自动驾驶** | 多模态环境感知 | 相机+LiDAR+RADAR融合 |
| **机器人导航** | 高精度定位 | LiDAR+IMU+GPS融合 |
| **AR/VR** | 精确姿态追踪 | 视觉+IMU融合 |
| **安防监控** | 全面目标检测 | 可见光+红外融合 |

---

## 2. 多传感器融合概述

### 2.1 融合层次

```python
import numpy as np
import cv2
import torch
import torch.nn as nn

class SensorFusionLevels:
    """多传感器融合层次"""
    
    @staticmethod
    def data_level_fusion(data_list):
        """
        数据级融合: 原始数据层面融合
        
        参数:
            data_list: 传感器数据列表
        """
        # 简单的加权平均
        weights = np.ones(len(data_list)) / len(data_list)
        fused = sum(w * d for w, d in zip(weights, data_list))
        return fused
    
    @staticmethod
    def feature_level_fusion(features_list):
        """
        特征级融合: 特征层面融合
        
        参数:
            features_list: 特征列表
        """
        # 拼接
        fused = np.concatenate(features_list, axis=-1)
        return fused
    
    @staticmethod
    def decision_level_fusion(decisions_list):
        """
        决策级融合: 决策层面融合
        
        参数:
            decisions_list: 决策列表
        """
        # 投票
        from collections import Counter
        votes = Counter(decisions_list)
        return votes.most_common(1)[0][0]
```

### 2.2 传感器类型

```python
class Sensor:
    """传感器基类"""
    def __init__(self, name, frequency):
        self.name = name
        self.frequency = frequency  # Hz
        self.data_buffer = []
    
    def add_data(self, data, timestamp):
        """添加数据到缓冲区"""
        self.data_buffer.append((timestamp, data))
    
    def get_data_at_time(self, timestamp, max_delay=0.1):
        """获取指定时间最近的数据"""
        best_data = None
        best_diff = float('inf')
        
        for t, d in reversed(self.data_buffer):
            diff = abs(t - timestamp)
            if diff < best_diff and diff <= max_delay:
                best_diff = diff
                best_data = d
            elif t < timestamp - max_delay:
                break
        
        return best_data

class Camera(Sensor):
    """相机传感器"""
    def __init__(self, name, frequency, image_size=(640, 480)):
        super().__init__(name, frequency)
        self.image_size = image_size

class LiDAR(Sensor):
    """LiDAR传感器"""
    def __init__(self, name, frequency, num_channels=64):
        super().__init__(name, frequency)
        self.num_channels = num_channels

class IMU(Sensor):
    """IMU传感器"""
    def __init__(self, name, frequency):
        super().__init__(name, frequency)
```

---

## 3. 传感器标定

### 3.1 相机内参标定

```python
class CameraCalibration:
    """相机标定"""
    
    def __init__(self, chessboard_size=(9, 6), square_size=0.025):
        """
        初始化标定
        
        参数:
            chessboard_size: 棋盘格尺寸
            square_size: 棋盘格大小(米)
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # 准备3D点
        self.obj_points = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.obj_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.obj_points *= square_size
    
    def calibrate(self, images):
        """
        标定相机
        
        参数:
            images: 棋盘格图像列表
        
        返回:
            camera_matrix: 内参矩阵
            dist_coeffs: 畸变系数
        """
        obj_points_list = []
        img_points_list = []
        
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 查找角点
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                obj_points_list.append(self.obj_points)
                
                # 亚像素细化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points_list.append(corners_refined)
        
        # 标定
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points_list, img_points_list, images[0].shape[:2], None, None
        )
        
        return camera_matrix, dist_coeffs
    
    def undistort_image(self, img, camera_matrix, dist_coeffs):
        """图像去畸变"""
        h, w = img.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # 裁剪
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
```

### 3.2 相机-LiDAR外参标定

```python
class CameraLiDARCalibration:
    """相机-LiDAR外参标定"""
    
    def __init__(self):
        self.R = np.eye(3)  # 旋转矩阵
        self.t = np.zeros(3)  # 平移向量
    
    def project_lidar_to_image(self, lidar_points, camera_matrix):
        """
        将LiDAR点投影到图像
        
        参数:
            lidar_points: Nx3 LiDAR点
            camera_matrix: 3x3 相机内参
        
        返回:
            image_points: Nx2 图像点
        """
        # 变换到相机坐标系
        points_cam = (self.R @ lidar_points.T).T + self.t
        
        # 投影到图像
        points_2d = camera_matrix @ points_cam.T
        points_2d = points_2d[:2] / points_2d[2]
        
        return points_2d.T
    
    def calibrate_from_correspondences(self, lidar_points, image_points, camera_matrix):
        """
        从对应点对标定外参
        
        参数:
            lidar_points: Nx3 LiDAR点
            image_points: Nx2 图像点
            camera_matrix: 3x3 相机内参
        
        返回:
            R, t: 旋转和平移
        """
        # 使用PnP求解
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            lidar_points, image_points, camera_matrix, None
        )
        
        # 转换为旋转矩阵
        self.R, _ = cv2.Rodrigues(rvec)
        self.t = tvec.flatten()
        
        return self.R, self.t
    
    def visualize_projection(self, img, lidar_points, camera_matrix):
        """可视化LiDAR点投影"""
        # 过滤前方点
        points_cam = (self.R @ lidar_points.T).T + self.t
        mask = points_cam[:, 2] > 0
        lidar_points_front = lidar_points[mask]
        
        # 投影
        points_2d = self.project_lidar_to_image(lidar_points_front, camera_matrix)
        
        # 深度着色
        depths = np.linalg.norm(lidar_points_front, axis=1)
        depth_normalized = (depths - depths.min()) / (depths.max() - depths.min())
        colors = plt.cm.jet(depth_normalized)[:, :3] * 255
        colors = colors.astype(np.uint8)
        
        # 绘制
        img_copy = img.copy()
        h, w = img.shape[:2]
        
        for i, (p, c) in enumerate(zip(points_2d, colors)):
            x, y = int(p[0]), int(p[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img_copy, (x, y), 2, color=(int(c[2]), int(c[1]), int(c[0])), thickness=-1)
        
        return img_copy
```

---

## 4. 数据级融合

### 4.1 RGB-D融合

```python
class RGBDFusion:
    """RGB-D数据融合"""
    
    def __init__(self):
        pass
    
    def depth_to_3d(self, depth_map, camera_matrix):
        """
        将深度图转换为点云
        
        参数:
            depth_map: HxW 深度图
            camera_matrix: 3x3 相机内参
        
        返回:
            point_cloud: Nx3 点云
        """
        h, w = depth_map.shape
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        # 像素坐标
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 3D坐标
        z = depth_map
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points = np.stack([x, y, z], axis=-1)
        points = points.reshape(-1, 3)
        
        # 过滤无效深度
        valid = (z > 0).flatten()
        points = points[valid]
        
        return points
    
    def fuse_rgb_depth(self, rgb_img, depth_map, camera_matrix):
        """
        融合RGB和深度
        
        参数:
            rgb_img: HxWx3 RGB图像
            depth_map: HxW 深度图
            camera_matrix: 3x3 相机内参
        
        返回:
            colored_point_cloud: Nx6 (x,y,z,r,g,b)
        """
        points = self.depth_to_3d(depth_map, camera_matrix)
        
        # 获取颜色
        colors = rgb_img.reshape(-1, 3)
        valid = (depth_map > 0).flatten()
        colors = colors[valid]
        
        fused = np.concatenate([points, colors], axis=-1)
        
        return fused
    
    def bilateral_filter_depth(self, depth_map, rgb_img, sigma_color=0.1, sigma_space=10):
        """
        RGB引导的双边滤波
        
        参数:
            depth_map: HxW 深度图
            rgb_img: HxWx3 RGB图像
            sigma_color: 颜色方差
            sigma_space: 空间方差
        
        返回:
            filtered_depth: 滤波后深度图
        """
        h, w = depth_map.shape
        filtered = np.zeros_like(depth_map)
        
        # 空间高斯
        k = max(int(sigma_space * 2) + 1, 5)
        space = cv2.getGaussianKernel(k, sigma_space)
        space_kernel = space @ space.T
        
        # 颜色高斯
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        
        # 滤波
        for y in range(h):
            for x in range(w):
                if depth_map[y, x] == 0:
                    continue
                
                # 邻域
                y_min = max(0, y - k // 2)
                y_max = min(h, y + k // 2 + 1)
                x_min = max(0, x - k // 2)
                x_max = min(w, x + k // 2 + 1)
                
                depth_patch = depth_map[y_min:y_max, x_min:x_max]
                color_patch = gray[y_min:y_max, x_min:x_max]
                
                # 颜色权重
                color_diff = np.abs(color_patch - gray[y, x])
                color_weight = np.exp(-color_diff ** 2 / (2 * sigma_color ** 2))
                
                # 空间权重
                sy = y_min - (y - k // 2)
                ey = y_max - (y - k // 2)
                sx = x_min - (x - k // 2)
                ex = x_max - (x - k // 2)
                space_weight = space_kernel[sy:ey, sx:ex]
                
                # 融合权重
                weight = color_weight * space_weight
                weight = weight * (depth_patch > 0)
                
                if weight.sum() > 0:
                    filtered[y, x] = (depth_patch * weight).sum() / weight.sum()
        
        return filtered
```

### 4.2 LiDAR-Camera深度融合

```python
class LiDARCameraFusion:
    """LiDAR-Camera融合"""
    
    def __init__(self):
        pass
    
    def project_lidar_to_depth(self, lidar_points, camera_matrix, R, t, img_size):
        """
        将LiDAR点投影为深度图
        
        参数:
            lidar_points: Nx3 LiDAR点
            camera_matrix: 3x3 相机内参
            R, t: 外参
            img_size: (H, W)
        
        返回:
            depth_map: HxW 深度图
        """
        h, w = img_size
        depth_map = np.zeros((h, w), dtype=np.float32)
        
        # 变换到相机坐标系
        points_cam = (R @ lidar_points.T).T + t
        
        # 过滤前方点
        mask = points_cam[:, 2] > 0
        points_cam = points_cam[mask]
        
        # 投影
        points_2d = camera_matrix @ points_cam.T
        points_2d = points_2d[:2] / points_2d[2]
        depths = points_cam[:, 2]
        
        # 填充深度图
        for i in range(len(points_2d)):
            x, y = int(points_2d[0, i]), int(points_2d[1, i])
            if 0 <= x < w and 0 <= y < h:
                if depth_map[y, x] == 0 or depths[i] < depth_map[y, x]:
                    depth_map[y, x] = depths[i]
        
        return depth_map
    
    def fuse_lidar_camera_depth(self, lidar_depth, mono_depth,
                                confidence_map=None, alpha=0.7):
        """
        融合LiDAR和单目深度
        
        参数:
            lidar_depth: LiDAR深度图
            mono_depth: 单目深度图
            confidence_map: 置信图
            alpha: 融合权重
        
        返回:
            fused_depth: 融合深度图
        """
        fused = np.zeros_like(lidar_depth)
        
        # LiDAR有效区域
        lidar_mask = lidar_depth > 0
        
        # 融合
        fused[lidar_mask] = alpha * lidar_depth[lidar_mask] + (1 - alpha) * mono_depth[lidar_mask]
        
        # 单目填充
        mono_mask = ~lidar_mask
        fused[mono_mask] = mono_depth[mono_mask]
        
        return fused
```

---

## 5. 特征级和决策级融合

### 5.1 深度融合网络

```python
class EarlyFusionNet(nn.Module):
    """早期融合网络（数据级）"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # RGB分支
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 深度分支
        self.depth_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 32 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, rgb, depth):
        """前向传播"""
        feat_rgb = self.rgb_conv(rgb)
        feat_depth = self.depth_conv(depth)
        
        # 拼接融合
        fused = torch.cat([feat_rgb, feat_depth], dim=1)
        fused = fused.view(fused.size(0), -1)
        
        out = self.classifier(fused)
        
        return out

class LateFusionNet(nn.Module):
    """晚期融合网络（决策级）"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # RGB分支
        self.rgb_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
        # 深度分支
        self.depth_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, rgb, depth):
        """前向传播"""
        out_rgb = self.rgb_branch(rgb)
        out_depth = self.depth_branch(depth)
        
        # 平均融合
        fused = (out_rgb + out_depth) / 2
        
        return fused

class AttentionFusionNet(nn.Module):
    """注意力融合网络"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # RGB特征
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 深度特征
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 注意力模块
        self.attention = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, rgb, depth):
        """前向传播"""
        feat_rgb = self.rgb_encoder(rgb)
        feat_depth = self.depth_encoder(depth)
        
        # 全局池化
        rgb_gap = torch.mean(feat_rgb, dim=[2, 3])
        depth_gap = torch.mean(feat_depth, dim=[2, 3])
        
        # 计算注意力权重
        combined = torch.cat([rgb_gap, depth_gap], dim=1)
        weights = self.attention(combined)
        
        # 加权融合
        weighted_rgb = feat_rgb * weights[:, 0].view(-1, 1, 1, 1)
        weighted_depth = feat_depth * weights[:, 1].view(-1, 1, 1, 1)
        fused = weighted_rgb + weighted_depth
        
        fused = fused.view(fused.size(0), -1)
        out = self.classifier(fused)
        
        return out
```

### 5.2 卡尔曼滤波融合

```python
class KalmanFilter:
    """卡尔曼滤波器用于传感器融合"""
    
    def __init__(self, dim_state=6, dim_obs=3):
        """
        初始化卡尔曼滤波器
        
        参数:
            dim_state: 状态维度
            dim_obs: 观测维度
        """
        self.dim_state = dim_state
        self.dim_obs = dim_obs
        
        # 状态向量 (x, y, z, vx, vy, vz)
        self.x = np.zeros(dim_state)
        
        # 状态协方差
        self.P = np.eye(dim_state) * 1.0
        
        # 状态转移矩阵
        self.F = np.eye(dim_state)
        for i in range(dim_obs):
            self.F[i, i + dim_obs] = 1.0
        
        # 观测矩阵
        self.H = np.zeros((dim_obs, dim_state))
        for i in range(dim_obs):
            self.H[i, i] = 1.0
        
        # 过程噪声
        self.Q = np.eye(dim_state) * 0.01
        
        # 观测噪声
        self.R = np.eye(dim_obs) * 0.1
    
    def predict(self, dt):
        """
        预测步骤
        
        参数:
            dt: 时间步长
        """
        # 更新状态转移矩阵
        F = self.F.copy()
        for i in range(self.dim_obs):
            F[i, i + self.dim_obs] = dt
        
        # 预测
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, z):
        """
        更新步骤
        
        参数:
            z: 观测值
        """
        # 卡尔曼增益
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_state) - K @ self.H) @ self.P
    
    def get_state(self):
        """获取当前状态"""
        return self.x.copy()

class MultiSensorFusion:
    """多传感器融合系统"""
    
    def __init__(self):
        self.kf = KalmanFilter()
    
    def fuse(self, sensor_data_list, dt):
        """
        融合多传感器数据
        
        参数:
            sensor_data_list: 传感器数据列表 [(z, R), ...]
            dt: 时间步长
        
        返回:
            fused_state: 融合状态
        """
        # 预测
        self.kf.predict(dt)
        
        # 逐个更新
        for z, R in sensor_data_list:
            self.kf.R = R
            self.kf.update(z)
        
        return self.kf.get_state()
```

---

## 6. 实践练习

### 练习1：相机内参标定

```python
def camera_calibration_demo():
    """相机标定演示"""
    # 生成模拟棋盘格图像
    chessboard_size = (9, 6)
    square_size = 0.025
    
    calibrator = CameraCalibration(chessboard_size, square_size)
    
    # 生成模拟图像（实际应用中使用真实图像）
    print("相机标定演示:")
    print("  棋盘格大小:", chessboard_size)
    print("  棋盘格方块尺寸:", square_size)
    print("  注: 在实际应用中需要收集多张不同角度的棋盘格图像")

# camera_calibration_demo()
```

### 练习2：LiDAR-相机投影

```python
def lidar_camera_projection_demo():
    """LiDAR-相机投影演示"""
    # 模拟参数
    camera_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])
    
    # 模拟外参
    R = np.eye(3)
    t = np.array([0.1, 0, 0])
    
    # 生成LiDAR点
    lidar_points = np.random.randn(1000, 3) * 5
    lidar_points[:, 2] = np.abs(lidar_points[:, 2]) + 1
    
    # 标定
    calib = CameraLiDARCalibration()
    calib.R = R
    calib.t = t
    
    # 投影
    points_2d = calib.project_lidar_to_image(lidar_points, camera_matrix)
    
    print("LiDAR-相机投影演示:")
    print(f"  投影点数: {len(points_2d)}")
    print(f"  相机矩阵:\n{camera_matrix}")

# lidar_camera_projection_demo()
```

### 练习3：卡尔曼滤波融合

```python
def kalman_filter_demo():
    """卡尔曼滤波融合演示"""
    # 创建滤波器
    kf = KalmanFilter()
    
    # 模拟数据
    num_steps = 100
    dt = 0.1
    
    # 真实轨迹
    true_states = []
    for i in range(num_steps):
        x = np.sin(i * 0.1) * 2
        y = np.cos(i * 0.1) * 2
        z = 0
        vx = np.cos(i * 0.1) * 0.2
        vy = -np.sin(i * 0.1) * 0.2
        vz = 0
        true_states.append([x, y, z, vx, vy, vz])
    
    true_states = np.array(true_states)
    
    # 模拟观测（带噪声）
    obs1 = true_states[:, :3] + np.random.randn(num_steps, 3) * 0.1
    obs2 = true_states[:, :3] + np.random.randn(num_steps, 3) * 0.05
    
    # 融合
    fusion = MultiSensorFusion()
    filtered_states = []
    
    for i in range(num_steps):
        sensor_data = [
            (obs1[i], np.eye(3) * 0.1),
            (obs2[i], np.eye(3) * 0.05)
        ]
        state = fusion.fuse(sensor_data, dt)
        filtered_states.append(state)
    
    filtered_states = np.array(filtered_states)
    
    # 计算误差
    error1 = np.mean(np.linalg.norm(obs1 - true_states[:, :3], axis=1))
    error2 = np.mean(np.linalg.norm(obs2 - true_states[:, :3], axis=1))
    error_filtered = np.mean(np.linalg.norm(filtered_states[:, :3] - true_states[:, :3], axis=1))
    
    print("卡尔曼滤波融合演示:")
    print(f"  传感器1平均误差: {error1:.4f}")
    print(f"  传感器2平均误差: {error2:.4f}")
    print(f"  融合后平均误差: {error_filtered:.4f}")

# kalman_filter_demo()
```

---

**恭喜！完成了第二部分：深度感知的学习！**

---

## 参考文献

1. Das, A., et al. (2016). Deep Attention Fusion for RGB-D Scene Recognition.
2. Durrant-Whyte, H. F., et al. (1988). Multisensor Data Fusion.
3. Ku, J., et al. (2018). Joint 3D Proposal Generation and Object Detection from View Aggregation.
4. Qi, C. R., et al. (2018). Frustum PointNets for 3D Object Detection from RGB-D Data.
5. Thrun, S., et al. (2005). Probabilistic Robotics.
