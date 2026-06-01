# 4.4 多模态SLAM

## 目录

- [1. 多模态SLAM简介](#1-多模态slam简介)
- [2. 视觉惯性SLAM](#2-视觉惯性slam)
- [3. 激光视觉融合](#3-激光视觉融合)
- [4. 更多传感器融合](#4-更多传感器融合)
- [5. 实践练习](#5-实践练习)

---

## 1. 多模态SLAM简介

### 1.1 多模态的优势

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2

class SensorComparison:
    """传感器对比"""
    
    @staticmethod
    def sensors():
        """传感器特性"""
        return {
            "camera": {
                "pros": ["纹理丰富", "语义信息", "成本低"],
                "cons": ["光照敏感", "尺度问题"]
            },
            "lidar": {
                "pros": ["高精度距离", "不受光照", "3D形状"],
                "cons": ["成本高", "无纹理"]
            },
            "imu": {
                "pros": ["高频", "不受光照"],
                "cons": ["漂移", "需要初始化"]
            },
            "gps": {
                "pros": ["绝对定位", "低成本"],
                "cons": ["更新慢", "室内失效"]
            }
        }
    
    @staticmethod
    def fusion_basics():
        """融合基础"""
        methods = {
            "loosely_coupled": "松耦合",
            "tightly_coupled": "紧耦合",
            "filter_based": "滤波",
            "optimization_based": "优化"
        }
        return methods
```

---

## 2. 视觉惯性SLAM

### 2.1 IMU预积分

```python
class IMUPreintegrator:
    """IMU预积分"""
    
    def __init__(self, acc_noise=0.01, gyro_noise=0.001):
        self.acc_noise = acc_noise
        self.gyro_noise = gyro_noise
        
        # 预积分量
        self.delta_rot = np.eye(3)
        self.delta_vel = np.zeros(3)
        self.delta_pos = np.zeros(3)
        
        # 偏置
        self.b_acc = np.zeros(3)
        self.b_gyro = np.zeros(3)
        
        self.dt_total = 0
    
    def integrate(self, acc, gyro, dt):
        """积分IMU数据"""
        # 减去偏置
        acc_corrected = acc - self.b_acc
        gyro_corrected = gyro - self.b_gyro
        
        # 更新旋转
        theta = gyro_corrected * dt
        theta_norm = np.linalg.norm(theta)
        
        if theta_norm > 1e-10:
            theta_unit = theta / theta_norm
            K = np.array([
                [0, -theta_unit[2], theta_unit[1]],
                [theta_unit[2], 0, -theta_unit[0]],
                [-theta_unit[1], theta_unit[0], 0]
            ])
            delta_R = np.eye(3) + np.sin(theta_norm) * K + (1 - np.cos(theta_norm)) * (K @ K)
            self.delta_rot = self.delta_rot @ delta_R
        
        # 更新速度和位置
        self.delta_vel += self.delta_rot @ acc_corrected * dt
        self.delta_pos += self.delta_vel * dt
        self.delta_pos += 0.5 * self.delta_rot @ acc_corrected * dt * dt
        
        self.dt_total += dt
    
    def propagate(self, R_i, v_i, p_i, g):
        """
        从i时刻传播到j时刻
        
        返回 R_j, v_j, p_j
        """
        R_j = R_i @ self.delta_rot
        v_j = v_i + R_i @ self.delta_vel + g * self.dt_total
        p_j = p_i + v_i * self.dt_total + 0.5 * g * self.dt_total ** 2 + R_i @ self.delta_pos
        
        return R_j, v_j, p_j
    
    def reset(self):
        """重置"""
        self.delta_rot = np.eye(3)
        self.delta_vel = np.zeros(3)
        self.delta_pos = np.zeros(3)
        self.dt_total = 0
```

### 2.2 VINS-Mono

```python
class VINSMono:
    """简化的VINS-Mono"""
    
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        
        # 前端
        self.feature_tracker = None
        self.imu_preintegrator = IMUPreintegrator()
        
        # 后端
        self.graph = None
        self.keyframes = []
        self.landmarks = {}
        
        # 初始化
        self.initialized = False
    
    def process_imu(self, acc, gyro, dt):
        """处理IMU数据"""
        self.imu_preintegrator.integrate(acc, gyro, dt)
    
    def process_image(self, img):
        """处理图像"""
        # 1. 特征跟踪
        features = self._track_features(img)
        
        if not self.initialized:
            # 初始化
            self._initialize(features)
        else:
            # 正常跟踪
            self._vio_update(features)
    
    def _track_features(self, img):
        """特征跟踪"""
        return []
    
    def _initialize(self, features):
        """初始化"""
        # 纯视觉SfM
        # 陀螺仪零偏估计
        # 速度重力初始化
        pass
    
    def _vio_update(self, features):
        """VIO更新"""
        # 1. 添加视觉约束
        # 2. 添加IMU约束
        # 3. 优化
        # 4. 边缘化
        pass
    
    def add_keyframe(self, img, pose, features):
        """添加关键帧"""
        pass
```

### 2.3 MSF和MSCKF

```python
class MSCKF:
    """MSCKF (Multi-State Constraint Kalman Filter)"""
    
    def __init__(self):
        # 状态
        self.state = {
            'p': np.zeros(3),
            'v': np.zeros(3),
            'q': np.array([1, 0, 0, 0]),
            'b_a': np.zeros(3),
            'b_g': np.zeros(3)
        }
        
        # 相机状态（滑动窗口）
        self.camera_states = []
        
        # 协方差
        self.P = np.eye(15)
    
    def predict(self, acc, gyro, dt):
        """预测"""
        # 传播状态
        pass
    
    def update(self, observations):
        """更新"""
        # 构建观测
        # EKF更新
        pass
```

---

## 3. 激光视觉融合

### 3.1 外参标定

```python
class CameraLidarCalibration:
    """相机-激光标定"""
    
    def __init__(self):
        self.R_cl = np.eye(3)
        self.t_cl = np.zeros(3)
    
    def calibrate(self, chessboard_3d, chessboard_img, lidar_points):
        """
        标定
        
        参数:
            chessboard_3d: 棋盘格3D点
            chessboard_img: 含棋盘格的图像
            lidar_points: 激光点云
        """
        # 1. 检测图像中的棋盘格
        # 2. 检测激光中的棋盘格
        # 3. PnP/ICP求外参
        return self.R_cl, self.t_cl
    
    def project_lidar_to_image(self, points, img_shape):
        """激光点投影到图像"""
        # 变换到相机坐标系
        points_cam = (self.R_cl @ points.T).T + self.t_cl
        
        # 投影
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
        points_2d = (K @ points_cam.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:]
        
        # 筛选可见的
        mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_shape[1]) & \
               (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_shape[0]) & \
               (points_cam[:, 2] > 0)
        
        return points_2d[mask], points_cam[mask, 2]
```

### 3.2 激光视觉SLAM系统

```python
class LidarVisualSLAM:
    """激光视觉SLAM"""
    
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        
        # 激光里程计
        self.lidar_odometry = None
        
        # 视觉里程计
        self.visual_odometry = None
        
        # 融合
        self.fusion = None
        
        # 地图
        self.map = None
    
    def process_image(self, img):
        """处理图像"""
        # 视觉处理
        pass
    
    def process_lidar(self, points):
        """处理激光"""
        # 激光处理
        pass
    
    def fuse(self):
        """融合"""
        # 紧耦合或松耦合
        pass
```

---

## 4. 更多传感器融合

### 4.1 GPS+SLAM

```python
class GPSFusion:
    """GPS融合"""
    
    def __init__(self):
        # 状态
        self.pose = np.eye(4)
        self.cov = np.eye(6)
        
        # GPS到局部坐标的转换
        self.gps_origin = None
    
    def set_gps_origin(self, gps_origin):
        """设置GPS原点"""
        self.gps_origin = gps_origin
    
    def gps_to_local(self, gps_data):
        """GPS数据转局部坐标"""
        if self.gps_origin is None:
            return None
        
        # 简化的转换
        lat, lon, alt = gps_data
        lat0, lon0, alt0 = self.gps_origin
        
        x = (lat - lat0) * 111320
        y = (lon - lon0) * 111320 * np.cos(np.radians(lat0))
        z = alt - alt0
        
        return np.array([x, y, z])
    
    def update_gps(self, gps_data, noise_cov):
        """GPS更新"""
        local_point = self.gps_to_local(gps_data)
        
        if local_point is not None:
            # 卡尔曼更新
            # 这里简化
            pass
```

### 4.2 轮速计融合

```python
class WheelOdometer:
    """轮速计"""
    
    def __init__(self, wheel_base=1.0):
        self.wheel_base = wheel_base
        
        # 状态
        self.x = 0
        self.y = 0
        self.theta = 0
    
    def update(self, v_left, v_right, dt):
        """更新"""
        v = (v_left + v_right) / 2
        omega = (v_right - v_left) / self.wheel_base
        
        # 积分
        self.theta += omega * dt
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
    
    def get_pose(self):
        """获取位姿"""
        pose = np.eye(4)
        pose[0, 0] = np.cos(self.theta)
        pose[0, 1] = -np.sin(self.theta)
        pose[1, 0] = np.sin(self.theta)
        pose[1, 1] = np.cos(self.theta)
        pose[0, 3] = self.x
        pose[1, 3] = self.y
        return pose
```

### 4.3 因子图融合

```python
class FactorGraphFusion:
    """因子图融合"""
    
    def __init__(self):
        # 因子图
        self.factors = []
        self.values = {}
        
        # 下一个索引
        self.next_idx = 0
    
    def add_pose_prior(self, idx, prior_pose, info):
        """添加先验"""
        self.factors.append({
            'type': 'prior',
            'keys': [idx],
            'prior': prior_pose,
            'info': info
        })
    
    def add_pose_factor(self, idx1, idx2, rel_pose, info):
        """添加相对位姿因子"""
        self.factors.append({
            'type': 'between',
            'keys': [idx1, idx2],
            'relative_pose': rel_pose,
            'info': info
        })
    
    def add_imu_factor(self, idx1, idx2, preint, info):
        """添加IMU因子"""
        self.factors.append({
            'type': 'imu',
            'keys': [idx1, idx2],
            'preintegration': preint,
            'info': info
        })
    
    def optimize(self):
        """优化"""
        # 这里简化，真实用GTSAM或g2o
        pass
```

---

## 5. 实践练习

### 练习1：IMU预积分

```python
def exercise_imu_preintegration():
    """IMU预积分练习"""
    print("=== IMU预积分练习 ===")
    
    dt = 0.01
    g = np.array([0, 0, -9.81])
    
    # 真实运动
    num_steps = 1000
    true_poses = []
    true_vels = []
    
    # 初始状态
    true_pose = np.eye(4)
    true_vel = np.zeros(3)
    
    true_poses.append(true_pose.copy())
    true_vels.append(true_vel.copy())
    
    # IMU测量
    acc_measurements = []
    gyro_measurements = []
    
    for i in range(num_steps):
        # 模拟运动
        omega_true = np.array([0.01 * np.sin(i * 0.01), 0.01 * np.cos(i * 0.01), 0.1])
        acc_true = np.array([0.1 * np.cos(i * 0.02), 0.1 * np.sin(i * 0.02), 0.0]) + g
        
        # 添加噪声
        gyro_noisy = omega_true + np.random.randn(3) * 0.001
        acc_noisy = acc_true + np.random.randn(3) * 0.01
        
        gyro_measurements.append(gyro_noisy)
        acc_measurements.append(acc_noisy)
        
        # 更新真实状态
        theta = omega_true * dt
        theta_norm = np.linalg.norm(theta)
        if theta_norm > 1e-10:
            theta_unit = theta / theta_norm
            K = np.array([
                [0, -theta_unit[2], theta_unit[1]],
                [theta_unit[2], 0, -theta_unit[0]],
                [-theta_unit[1], theta_unit[0], 0]
            ])
            delta_R = np.eye(3) + np.sin(theta_norm) * K + (1 - np.cos(theta_norm)) * (K @ K)
            true_pose[:3, :3] = true_pose[:3, :3] @ delta_R
        
        true_vel += (acc_true - g) * dt
        true_pose[:3, 3] += true_vel * dt
        
        true_poses.append(true_pose.copy())
        true_vels.append(true_vel.copy())
    
    # 预积分
    integrator = IMUPreintegrator()
    
    for i in range(num_steps):
        integrator.integrate(acc_measurements[i], gyro_measurements[i], dt)
    
    # 传播
    R_i = true_poses[0][:3, :3]
    v_i = true_vels[0]
    p_i = true_poses[0][:3, 3]
    
    R_j, v_j, p_j = integrator.propagate(R_i, v_i, p_i, g)
    
    # 比较
    true_R_j = true_poses[-1][:3, :3]
    true_v_j = true_vels[-1]
    true_p_j = true_poses[-1][:3, 3]
    
    print(f"真实最终位置: {true_p_j}")
    print(f"估计最终位置: {p_j}")
    print(f"位置误差: {np.linalg.norm(p_j - true_p_j):.6f}")
    print()

# exercise_imu_preintegration()
```

### 练习2：多模态系统对比

```python
def exercise_multimodal_comparison():
    """多模态系统对比"""
    print("=== 多模态系统对比 ===")
    
    systems = [
        {
            "name": "VINS-Mono",
            "sensors": "相机 + IMU",
            "method": "紧耦合优化",
            "pros": ["精度高", "尺度可观"],
            "cons": ["复杂", "需要标定"]
        },
        {
            "name": "LIO-SAM",
            "sensors": "激光 + IMU",
            "method": "因子图",
            "pros": ["稳定", "精度高"],
            "cons": ["计算量大"]
        },
        {
            "name": "LVI-SAM",
            "sensors": "激光 + 视觉 + IMU",
            "method": "紧耦合",
            "pros": ["完整", "鲁棒"],
            "cons": ["非常复杂"]
        }
    ]
    
    for s in systems:
        print(f"{s['name']}")
        print(f"  传感器: {s['sensors']}")
        print(f"  方法: {s['method']}")
        print(f"  优点: {', '.join(s['pros'])}")
        print(f"  缺点: {', '.join(s['cons'])}")
        print()

# exercise_multimodal_comparison()
```

### 练习3：因子图简单示例

```python
def exercise_factor_graph():
    """因子图示例"""
    print("=== 因子图示例 ===")
    
    # 创建因子图
    graph = FactorGraphFusion()
    
    # 添加先验
    prior_pose = np.eye(4)
    graph.add_pose_prior(0, prior_pose, np.eye(6))
    
    # 添加相对约束
    for i in range(5):
        rel_pose = np.eye(4)
        rel_pose[0, 3] = 1.0
        
        info = np.eye(6) * 100
        graph.add_pose_factor(i, i + 1, rel_pose, info)
    
    print(f"节点数: {len(graph.values)}")
    print(f"因子数: {len(graph.factors)}")
    print()
    print("注: 实际优化请使用GTSAM或g2o库")

# exercise_factor_graph()
```

---

**下一节**：[地图构建](05-map-building.md)

---

## 参考文献

1. Qin, T., et al. (2018). VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator.
2. Mourikis, A. I., & Roumeliotis, S. I. (2007). A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation.
3. Shan, T., et al. (2020). LIO-SAM: Tightly-coupled LIDAR-Inertial Odometry via Smoothing and Mapping.
4. Shan, T., et al. (2021). LVI-SAM: Tightly-coupled Lidar-Visual-Inertial Odometry via Smoothing and Mapping.
5. Forster, C., et al. (2017). IMU Preintegration on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation.
