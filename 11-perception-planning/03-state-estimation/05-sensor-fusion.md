# 3.5 多传感器融合

## 目录

- [1. 引言](#1-引言)
- [2. 传感器融合方法](#2-传感器融合方法)
- [3. IMU预积分](#3-imu预积分)
- [4. 视觉-惯性融合](#4-视觉-惯性融合)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 多传感器融合的重要性

**多传感器融合**结合多种传感器（相机、IMU、激光雷达、GPS等）的优点，提供更鲁棒、更精确的状态估计。

### 1.2 常见传感器

| 传感器 | 优点 | 缺点 |
|--------|------|------|
| **相机** | 纹理信息丰富、成本低 | 易受光照影响 |
| **IMU** | 高频、不受光照影响 | 有漂移 |
| **LiDAR** | 高精度深度 | 成本高、数据稀疏 |
| **GPS** | 绝对定位 | 室内失效、更新率低 |

---

## 2. 传感器融合方法

### 2.1 传感器模型

```python
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Sensor(ABC):
    """传感器基类"""
    
    @abstractmethod
    def predict(self, state, dt):
        """预测"""
        pass
    
    @abstractmethod
    def update(self, state, measurement):
        """更新"""
        pass

class IMUModel:
    """IMU模型"""
    
    def __init__(self, sigma_acc=0.01, sigma_gyro=0.001):
        self.sigma_acc = sigma_acc
        self.sigma_gyro = sigma_gyro
    
    def integrate(self, state, acc, gyro, dt):
        """积分IMU测量"""
        R = state[:3, :3]
        p = state[:3, 3]
        v = state[:3, 3] if state.shape == (4, 4) else np.zeros(3)
        
        # 更新旋转
        theta = gyro * dt
        dR = np.eye(3)
        theta_norm = np.linalg.norm(theta)
        if theta_norm > 1e-10:
            theta_unit = theta / theta_norm
            K = np.array([
                [0, -theta_unit[2], theta_unit[1]],
                [theta_unit[2], 0, -theta_unit[0]],
                [-theta_unit[1], theta_unit[0], 0]
            ])
            dR = np.eye(3) + np.sin(theta_norm) * K + (1 - np.cos(theta_norm)) * (K @ K
        
        new_state = np.eye(4)
        new_state[:3, :3] = R @ dR
        new_state[:3, 3] = p + v * dt
        
        return new_state
```

### 2.2 松耦合 vs 紧耦合

```python
class LooselyCoupledFusion:
    """松耦合融合"""
    
    def __init__(self):
        self.state = np.eye(4)
        self.covariance = np.eye(6)
    
    def update_imu(self, acc, gyro, dt):
        """IMU更新（预测）"""
        # 这里简化实现
        pass
    
    def update_vision(self, pose_measurement, cov):
        """视觉更新（校正）"""
        # 这里简化实现
        pass

class TightlyCoupledFusion:
    """紧耦合融合"""
    
    def __init__(self):
        self.imu_states = []
        self.vision_measurements = []
    
    def add_imu(self, acc, gyro, dt):
        """添加IMU数据"""
        self.imu_states.append((acc, gyro, dt))
    
    def add_vision(self, features, pose):
        """添加视觉数据"""
        self.vision_measurements.append((features, pose))
    
    def optimize(self):
        """联合优化"""
        # 使用g2o等库进行BA或图优化
        pass
```

---

## 3. IMU预积分

### 3.1 预积分实现

```python
class IMUPreintegrator:
    """IMU预积分器"""
    
    def __init__(self, acc_noise=0.01, gyro_noise=0.001):
        self.dt_sum = 0
        self.delta_p = np.zeros(3)
        self.delta_v = np.zeros(3)
        self.delta_R = np.eye(3)
        
        self.acc_noise = acc_noise
        self.gyro_noise = gyro_noise
    
    def reset(self):
        """重置"""
        self.dt_sum = 0
        self.delta_p = np.zeros(3)
        self.delta_v = np.zeros(3)
        self.delta_R = np.eye(3)
    
    def integrate(self, acc, gyro, dt, gravity=np.array([0, 0, 9.81])):
        """
        预积分IMU测量
        
        参数:
            acc: 加速度测量
            gyro: 角速度测量
            dt: 时间间隔
            gravity: 重力向量
        """
        # 更新旋转
        theta = gyro * dt
        theta_norm = np.linalg.norm(theta)
        
        if theta_norm > 1e-10:
            theta_unit = theta / theta_norm
            K = np.array([
                [0, -theta_unit[2], theta_unit[1]],
                [theta_unit[2], 0, -theta_unit[0]],
                [-theta_unit[1], theta_unit[0], 0]
            ])
            dR = np.eye(3) + np.sin(theta_norm) * K + (1 - np.cos(theta_norm)) * (K @ K)
            self.delta_R = self.delta_R @ dR
        
        # 更新速度和位置
        dv = (acc - gravity) * dt
        self.delta_v += self.delta_R @ dv
        self.delta_p += self.delta_v * dt + 0.5 * self.delta_R @ dv * dt
        
        self.dt_sum += dt
    
    def get_delta(self):
        """获取预积分量"""
        return self.delta_R, self.delta_v, self.delta_p, self.dt_sum
    
    def propagate(self, R_i, v_i, p_i, gravity=np.array([0, 0, 9.81])):
        """
        传播到j时刻的状态
        
        参数:
            R_i, v_i, p_i: i时刻旋转、速度、位置
        """
        R_j = R_i @ self.delta_R
        v_j = v_i + R_i @ self.delta_v + gravity * self.dt_sum
        p_j = p_i + v_i * self.dt_sum + 0.5 * gravity * self.dt_sum**2 + R_i @ self.delta_p
        
        return R_j, v_j, p_j
```

### 3.2 IMU仿真

```python
def simulate_imu_trajectory():
    """仿真IMU数据"""
    dt = 0.01
    num_steps = 1000
    
    # 真实轨迹
    true_poses = []
    true_vels = []
    
    # IMU测量
    acc_measurements = []
    gyro_measurements = []
    
    # 初始状态
    pose = np.eye(4)
    vel = np.zeros(3)
    
    for i in range(num_steps):
        true_poses.append(pose.copy())
        true_vels.append(vel.copy())
        
        # 模拟运动
        t = i * dt
        acc_true = np.array([np.sin(t * 0.5), 0, 0])
        gyro_true = np.array([0, 0, 0.1 * np.sin(t * 0.3))
        
        # 添加噪声
        acc_noisy = acc_true + np.random.randn(3) * 0.01
        gyro_noisy = gyro_true + np.random.randn(3) * 0.001
        
        acc_measurements.append(acc_noisy)
        gyro_measurements.append(gyro_noisy)
        
        # 更新真实状态
        theta = gyro_true * dt
        theta_norm = np.linalg.norm(theta)
        
        if theta_norm > 1e-10:
            theta_unit = theta / theta_norm
            K = np.array([
                [0, -theta_unit[2], theta_unit[1]],
                [theta_unit[2], 0, -theta_unit[0]],
                [-theta_unit[1], theta_unit[0], 0
            ])
            dR = np.eye(3) + np.sin(theta_norm) * K + (1 - np.cos(theta_norm)) * (K @ K)
            pose[:3, :3] = pose[:3, :3] @ dR
        
        pose[:3, 3] += vel * dt + 0.5 * acc_true * dt**2
        vel += acc_true * dt
    
    return np.array(true_poses), np.array(true_vels), np.array(acc_measurements), np.array(gyro_measurements), dt
```

---

## 4. 视觉-惯性融合

### 4.1 MSF (Multi-State Constraint Kalman Filter

```python
class MSF:
    """简化版MSF-like滤波器"""
    
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        
        # 状态: [p, v, q, b_g, b_a]
        self.state = np.zeros(16)
        self.state[6:10] = [1, 0, 0, 0]  # 初始四元数
        
        # 协方差
        self.covariance = np.eye(16) * 0.01
        
        # 预积分器
        self.integrator = IMUPreintegrator()
    
    def predict(self, acc, gyro, dt):
        """预测步"""
        # 简化的预测
        self.integrator.integrate(acc, gyro, dt)
        
        # 更新协方差
        # 这里省略详细的协方差传播
    
    def update_vision(self, pose_measurement, pose_covariance):
        """视觉更新"""
        # 简化的视觉更新
        pass
    
    def get_pose(self):
        """获取当前位姿"""
        p = self.state[:3]
        q = self.state[6:10]
        
        # 四元数转旋转矩阵
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y
        ])
        
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = p
        
        return pose
```

### 4.2 VINS-Mono框架

```python
class VINSLike:
    """简化的VINS类系统"""
    
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        
        # 关键帧
        self.keyframes = []
        self.map_points = {}
        
        # 预积分
        self.integrator = IMUPreintegrator()
        
        # 边缘化
        self.marginalizer = None
    
    def process_imu(self, acc, gyro, dt):
        """处理IMU数据"""
        self.integrator.integrate(acc, gyro, dt)
    
    def process_image(self, img, timestamp):
        """处理图像"""
        # 1. 特征检测和跟踪
        # 2. 如果是关键帧
        # 3. 进行优化
        pass
    
    def optimize(self):
        """滑动窗口优化"""
        # 使用g2o进行视觉惯性BA
        pass
```

---

## 5. 实践练习

### 练习1：IMU积分

```python
def exercise_imu_integration():
    """IMU积分练习"""
    print("=== IMU积分练习 ===")
    
    # 仿真数据
    true_poses, true_vels, accs, gyros, dt = simulate_imu_trajectory()
    
    # 初始化
    pose_est = np.eye(4)
    vel_est = np.zeros(3)
    
    # 估计轨迹
    est_poses = []
    
    # 预积分
    integrator = IMUPreintegrator()
    
    for i in range(len(accs)):
        est_poses.append(pose_est.copy())
        
        # 积分
        R_est, v_est, p_est = integrator.propagate(
            true_poses[i][:3, :3], vel_est, pose_est[:3, 3]
        )
        
        # 更新
        pose_est[:3, :3] = R_est
        pose_est[:3, 3] = p_est
        vel_est = v_est
    
    # 绘制
    true_positions = true_poses[:, :3, 3]
    est_positions = np.array(est_poses)[:, :3, 3]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='真实', linewidth=2)
    plt.plot(est_positions[:, 0], est_positions[:, 1], 'b--', label='估计')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('轨迹对比')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    errors = np.linalg.norm(est_positions - true_positions, axis=1)
    plt.plot(errors)
    plt.xlabel('时间步')
    plt.ylabel('位置误差 (m)')
    plt.title('积分误差')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('imu_integration.png')
    print("IMU积分演示完成")

# exercise_imu_integration()
```

### 练习2：传感器融合演示

```python
def exercise_sensor_fusion():
    """传感器融合练习"""
    print("=== 传感器融合练习 ===")
    
    # 模拟IMU + 视觉更新频率
    num_imu_steps = 100
    num_vision_updates = 10
    
    # 真实轨迹
    true_poses = []
    for i in range(num_imu_steps):
        t = i * 0.01
        pose = np.eye(4)
        pose[0, 3] = np.sin(t) * 5
        pose[1, 3] = np.cos(t) * 5
        true_poses.append(pose)
    
    true_poses = np.array(true_poses)
    
    # 模拟IMU（仅位置
    imu_poses = []
    for i in range(num_imu_steps):
        imu_poses.append(true_poses[i])
    
    # 模拟视觉（低频、有噪声
    vision_poses = []
    vision_timesteps = np.linspace(0, num_imu_steps-1, num_vision_updates, dtype=int)
    
    for t in vision_timesteps:
        pose = true_poses[t].copy()
        pose[:3, 3] += np.random.randn(3) * 0.2
        vision_poses.append((t, pose))
    
    # 简单的互补滤波
    fused_poses = []
    current_pose = np.eye(4)
    
    vision_idx = 0
    for i in range(num_imu_steps):
        # IMU预测
        if i < num_imu_steps - 1:
            current_pose = true_poses[i+1]
        
        # 视觉更新
        if vision_idx < len(vision_timesteps) and i == vision_timesteps[vision_idx]:
            t_vision, pose_vision = vision_poses[vision_idx]
            alpha = 0.5
            current_pose[:3, 3] = (1 - alpha) * current_pose[:3, 3] + alpha * pose_vision[:3, 3]
            vision_idx += 1
        
        fused_poses.append(current_pose.copy())
    
    # 绘制
    true_positions = true_poses[:, :3, 3]
    fused_positions = np.array(fused_poses)[:, :3, 3]
    
    # 计算误差
    fused_errors = np.linalg.norm(fused_positions - true_positions, axis=1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='真实', linewidth=2)
    plt.plot(fused_positions[:, 0], fused_positions[:, 1], 'r--', label='融合')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('融合轨迹')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(fused_errors, label='融合误差')
    plt.xlabel('时间步')
    plt.ylabel('位置误差 (m)')
    plt.title('融合误差')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sensor_fusion.png')
    print(f"平均误差: {np.mean(fused_errors):.4f} m")

# exercise_sensor_fusion()
```

### 练习3：预积分

```python
def exercise_preintegration():
    """预积分练习"""
    print("=== IMU预积分练习 ===")
    
    # 仿真数据
    true_poses, true_vels, accs, gyros, dt = simulate_imu_trajectory()
    
    # 预积分
    integrator = IMUPreintegrator()
    
    # 每100帧重置一次
    reset_interval = 100
    
    preint_poses = []
    
    current_pose = np.eye(4)
    
    for i in range(len(true_poses)):
        if i % reset_interval == 0:
            integrator.reset()
            start_pose = true_poses[i]
            start_vel = true_vels[i]
        
        if i > 0:
            integrator.integrate(accs[i-1], gyros[i-1], dt)
        
        if i % reset_interval == reset_interval - 1 or i == len(true_poses) - 1:
            R_est, v_est, p_est = integrator.propagate(
                start_pose[:3, :3], start_vel, start_pose[:3, 3]
            )
            pose_est = np.eye(4)
            pose_est[:3, :3] = R_est
            pose_est[:3, 3] = p_est
            preint_poses.append(pose_est)
    
    # 计算误差
    errors = []
    for i, pose_est in enumerate(preint_poses):
        true_idx = (i + 1) * reset_interval - 1
        true_idx = min(true_idx, len(true_poses) - 1)
        
        error = np.linalg.norm(pose_est[:3, 3] - true_poses[true_idx][:3, 3])
        errors.append(error)
    
    print(f"平均预积分误差: {np.mean(errors):.4f} m")
    
    plt.figure(figsize=(10, 5))
    plt.plot(errors, 'o-')
    plt.xlabel('段数')
    plt.ylabel('位置误差 (m)')
    plt.title('预积分误差')
    plt.grid(True)
    plt.savefig('preintegration_error.png')

# exercise_preintegration()
```

---

恭喜你完成了第三部分的学习！你现在掌握了从滤波方法、视觉里程计、位姿估计、SLAM基础和多传感器融合等核心技术。

## 参考文献

1. Bloesch, M., et al. (2017). Robust Visual Inertial Odometry Using a Direct EKF-Based Approach.
2. Qin, T., et al. (2018). VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator.
3. Forster, C., et al. (2017). IMU Preintegration on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation.
4. Mourikis, A. I., & Roumeliotis, S. I. (2007). A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation.
