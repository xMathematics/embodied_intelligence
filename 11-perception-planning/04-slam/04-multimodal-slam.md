# 4.4 多模态SLAM

## 目录

- [1. 多模态SLAM简介](#1-多模态slam简介)
- [2. 视觉惯性SLAM](#2-视觉惯性slam)
- [3. 激光视觉融合](#3-激光视觉融合)
- [4. 更多传感器融合](#4-更多传感器融合)
- [5. 经典系统深度解析](#5-经典系统深度解析)
- [6. 挑战与前沿方向](#6-挑战与前沿方向)
- [7. 实践练习](#7-实践练习)

---

## 1. 多模态SLAM简介

### 1.1 为什么需要多模态SLAM

**问题的提出**

单模态SLAM系统在实际应用中面临诸多挑战：

1. **视觉SLAM的局限性**：
   - 光照变化导致特征跟踪失败
   - 纹理缺失区域（白墙、天空）无法提取特征
   - 动态物体干扰
   - 尺度不确定性

2. **激光SLAM的局限性**：
   - 长走廊、隧道等几何退化环境
   - 成本高昂
   - 无法获取纹理和语义信息

3. **IMU的局限性**：
   - 积分漂移
   - 需要初始化
   - 无法提供绝对位置

**核心思想**：通过融合多种传感器的互补特性，构建更鲁棒、更精确的SLAM系统。

### 1.2 传感器特性对比

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2

class SensorComparison:
    """传感器对比分析"""
    
    @staticmethod
    def sensors():
        """传感器特性对比"""
        return {
            "camera": {
                "pros": ["纹理丰富", "语义信息", "成本低", "重量轻"],
                "cons": ["光照敏感", "尺度问题", "计算量大", "动态场景困难"],
                "frequency": "30-60 Hz",
                "range": "几米到几十米",
                "accuracy": "像素级"
            },
            "lidar": {
                "pros": ["高精度距离", "不受光照影响", "3D形状信息", "360°视野"],
                "cons": ["成本高", "无纹理信息", "雨雪雾影响", "重量大"],
                "frequency": "10-20 Hz",
                "range": "几十米到几百米",
                "accuracy": "厘米级"
            },
            "imu": {
                "pros": ["高频(100-1000Hz)", "不受光照影响", "短时精度高", "体积小"],
                "cons": ["长期漂移", "需要初始化", "噪声敏感", "温度漂移"],
                "frequency": "100-1000 Hz",
                "range": "无限制",
                "accuracy": "随时间漂移"
            },
            "gps": {
                "pros": ["绝对定位", "低成本", "全球覆盖", "无累积误差"],
                "cons": ["更新慢(1-10Hz)", "室内失效", "多路径效应", "精度有限"],
                "frequency": "1-10 Hz",
                "range": "全球",
                "accuracy": "米级(民用)"
            },
            "wheel_odometer": {
                "pros": ["简单可靠", "低成本", "实时性好"],
                "cons": ["打滑误差", "轮胎变形", "地面依赖"],
                "frequency": "50-100 Hz",
                "range": "局部",
                "accuracy": "随距离漂移"
            }
        }
    
    @staticmethod
    def fusion_strategies():
        """融合策略"""
        return {
            "loosely_coupled": {
                "description": "松耦合：各传感器独立估计，结果融合",
                "pros": ["实现简单", "模块化", "容错性好"],
                "cons": ["信息损失", "精度受限"]
            },
            "tightly_coupled": {
                "description": "紧耦合：原始测量联合优化",
                "pros": ["精度高", "信息利用充分"],
                "cons": ["复杂度高", "标定要求高"]
            },
            "filter_based": {
                "description": "滤波方法：EKF、UKF、粒子滤波",
                "pros": ["实时性好", "理论成熟"],
                "cons": ["线性化误差", "状态维度限制"]
            },
            "optimization_based": {
                "description": "优化方法：图优化、BA",
                "pros": ["精度高", "可处理大规模问题"],
                "cons": ["计算量大", "延迟较高"]
            }
        }
```

### 1.3 融合架构分类

**松耦合 vs 紧耦合**

```python
class FusionArchitecture:
    """融合架构对比"""
    
    @staticmethod
    def loosely_coupled_fusion():
        """
        松耦合融合
        
        问题提出：
        各传感器独立处理后再融合，实现简单但信息损失大
        
        解决方案：
        每个传感器模块输出位姿估计，通过卡尔曼滤波或加权平均融合
        
        优点：
        - 模块化设计，易于实现和维护
        - 单个传感器失效不会导致系统崩溃
        - 计算量相对较小
        
        缺点：
        - 信息损失：原始测量信息未充分利用
        - 精度上限受限
        - 时间同步要求高
        
        代表系统：ETH的MSF (Multi-Sensor Fusion)
        """
        architecture = {
            "modules": {
                "visual_odometry": "独立视觉里程计",
                "lidar_odometry": "独立激光里程计", 
                "imu_processor": "IMU积分器",
                "fusion_core": "融合核心(EKF)"
            },
            "data_flow": [
                "传感器数据 -> 各里程计模块",
                "各里程计 -> 输出6DOF位姿",
                "位姿 + 协方差 -> 融合核心",
                "融合核心 -> 统一状态估计"
            ]
        }
        return architecture
    
    @staticmethod
    def tightly_coupled_fusion():
        """
        紧耦合融合
        
        问题提出：
        松耦合融合存在信息损失，无法达到最优估计
        
        解决方案：
        将各传感器的原始测量统一到一个优化框架中联合估计
        
        优点：
        - 信息利用充分，精度更高
        - 可以处理传感器部分失效情况
        - 统一的误差模型
        
        缺点：
        - 实现复杂
        - 计算量大
        - 标定要求高
        - 一个传感器的问题可能影响整体
        
        代表系统：VINS-Mono, LIO-SAM, LVI-SAM
        """
        architecture = {
            "modules": {
                "frontend": "特征提取与跟踪",
                "preintegration": "IMU预积分",
                "backend": "联合优化器"
            },
            "data_flow": [
                "原始测量 -> 特征提取",
                "IMU数据 -> 预积分",
                "特征 + IMU约束 -> 联合优化",
                "优化器 -> 统一状态估计"
            ],
            "optimization": {
                "variables": ["位姿", "速度", "IMU偏置", "特征点深度"],
                "constraints": ["重投影误差", "IMU误差", "先验约束"]
            }
        }
        return architecture
```

---

## 2. 视觉惯性SLAM

### 2.1 IMU预积分的数学原理

**问题的提出**

传统IMU积分存在以下问题：
1. 每次优化后位姿变化，需要重新积分
2. 计算效率低
3. 数值稳定性差

**解决方案：IMU预积分（Forster et al., 2017）**

核心思想：将IMU积分表示为相对运动，与绝对位姿解耦。

```python
class IMUPreintegrator:
    """
    IMU预积分实现
    
    论文核心思想（Forster et al., RSS 2015; TRO 2017）：
    将IMU积分从世界坐标系转换到局部坐标系，使得积分结果
    与初始位姿无关，优化时无需重新积分。
    
    数学推导：
    给定IMU测量：
        a_t = R_t^T(a_t^w - g) + b_a + n_a
        ω_t = ω_t^w + b_g + n_g
    
    预积分量定义（从i时刻到j时刻）：
        ΔR_ij = ∏(k=i to j-1) Exp((ω_k - b_g)Δt)
        Δv_ij = Σ(k=i to j-1) ΔR_ik(a_k - b_a)Δt
        Δp_ij = Σ(k=i to j-1) [v_kΔt + 1/2ΔR_ik(a_k - b_a)Δt²]
    
    优点：
    1. 与初始位姿解耦，优化时无需重新积分
    2. 可以计算关于偏置的雅可比，支持偏置更新
    3. 协方差传播封闭形式
    
    缺点：
    1. 数学推导复杂
    2. 需要处理流形上的优化
    """
    
    def __init__(self, acc_noise=0.01, gyro_noise=0.001, 
                 acc_bias_noise=0.0001, gyro_bias_noise=0.00001):
        self.acc_noise = acc_noise
        self.gyro_noise = gyro_noise
        self.acc_bias_noise = acc_bias_noise
        self.gyro_bias_noise = gyro_bias_noise
        
        # 预积分量
        self.delta_rot = np.eye(3)
        self.delta_vel = np.zeros(3)
        self.delta_pos = np.zeros(3)
        
        # 偏置
        self.b_acc = np.zeros(3)
        self.b_gyro = np.zeros(3)
        
        # 雅可比矩阵（用于偏置更新）
        self.J_rot_bias = np.zeros((3, 3))
        self.J_vel_bias = np.zeros((3, 3))
        self.J_pos_bias = np.zeros((3, 3))
        
        # 协方差
        self.covariance = np.zeros((9, 9))
        
        self.dt_total = 0
        self.measurements = []
    
    def integrate(self, acc, gyro, dt):
        """
        积分IMU数据
        
        使用一阶近似（Forster et al. 方法）
        """
        # 减去偏置
        acc_corrected = acc - self.b_acc
        gyro_corrected = gyro - self.b_gyro
        
        # 保存测量
        self.measurements.append((acc, gyro, dt))
        
        # 更新旋转（使用指数映射）
        theta = gyro_corrected * dt
        theta_norm = np.linalg.norm(theta)
        
        if theta_norm > 1e-10:
            theta_unit = theta / theta_norm
            K = self._skew_symmetric(theta_unit)
            delta_R = np.eye(3) + np.sin(theta_norm) * K + \
                     (1 - np.cos(theta_norm)) * (K @ K)
            
            # 更新旋转雅可比
            self.J_rot_bias -= self.delta_rot @ self._skew_symmetric(acc_corrected) * dt
            
            self.delta_rot = self.delta_rot @ delta_R
        
        # 更新速度
        self.delta_vel += self.delta_rot @ acc_corrected * dt
        self.J_vel_bias += self.delta_rot * dt
        
        # 更新位置
        self.delta_pos += self.delta_vel * dt + 0.5 * self.delta_rot @ acc_corrected * dt * dt
        self.J_pos_bias += 0.5 * self.delta_rot * dt * dt
        
        # 更新协方差（简化版本）
        self._update_covariance(acc_corrected, gyro_corrected, dt)
        
        self.dt_total += dt
    
    def _skew_symmetric(self, v):
        """构造反对称矩阵"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
    def _update_covariance(self, acc, gyro, dt):
        """更新协方差矩阵"""
        # 简化的协方差传播
        noise_acc = self.acc_noise ** 2
        noise_gyro = self.gyro_noise ** 2
        
        # 这里使用简化的噪声模型
        # 实际实现需要考虑完整的噪声传播
        F = np.eye(9)
        F[3:6, 0:3] = -self.delta_rot @ self._skew_symmetric(acc) * dt
        F[6:9, 0:3] = -0.5 * self.delta_rot @ self._skew_symmetric(acc) * dt * dt
        F[6:9, 3:6] = dt * np.eye(3)
        
        Q = np.zeros((9, 9))
        Q[0:3, 0:3] = noise_gyro * dt * dt * np.eye(3)
        Q[3:6, 3:6] = noise_acc * dt * dt * np.eye(3)
        Q[6:9, 6:9] = 0.25 * noise_acc * dt**4 * np.eye(3)
        
        self.covariance = F @ self.covariance @ F.T + Q
    
    def propagate(self, R_i, v_i, p_i, g):
        """
        从i时刻传播到j时刻
        
        公式：
        R_j = R_i @ ΔR_ij
        v_j = v_i + R_i @ Δv_ij + g * Δt
        p_j = p_i + v_i * Δt + 0.5 * g * Δt² + R_i @ Δp_ij
        """
        R_j = R_i @ self.delta_rot
        v_j = v_i + R_i @ self.delta_vel + g * self.dt_total
        p_j = p_i + v_i * self.dt_total + \
              0.5 * g * self.dt_total ** 2 + R_i @ self.delta_pos
        
        return R_j, v_j, p_j
    
    def correct_bias(self, delta_b_acc, delta_b_gyro):
        """
        根据偏置变化修正预积分结果
        
        使用一阶近似更新，避免重新积分
        """
        # 更新预积分量
        self.delta_rot = self.delta_rot @ self._exp_map(
            self.J_rot_bias @ delta_b_gyro
        )
        self.delta_vel += self.J_vel_bias @ delta_b_acc
        self.delta_pos += self.J_pos_bias @ delta_b_acc
        
        # 更新偏置
        self.b_acc += delta_b_acc
        self.b_gyro += delta_b_gyro
    
    def _exp_map(self, theta):
        """指数映射"""
        theta_norm = np.linalg.norm(theta)
        if theta_norm < 1e-10:
            return np.eye(3)
        
        K = self._skew_symmetric(theta / theta_norm)
        return np.eye(3) + np.sin(theta_norm) * K + \
               (1 - np.cos(theta_norm)) * (K @ K)
    
    def reset(self):
        """重置预积分器"""
        self.delta_rot = np.eye(3)
        self.delta_vel = np.zeros(3)
        self.delta_pos = np.zeros(3)
        self.J_rot_bias = np.zeros((3, 3))
        self.J_vel_bias = np.zeros((3, 3))
        self.J_pos_bias = np.zeros((3, 3))
        self.covariance = np.zeros((9, 9))
        self.dt_total = 0
        self.measurements = []
```

### 2.2 VINS-Mono深度解析

**论文核心思想（Qin et al., 2018）**

VINS-Mono是港科大沈劭劼团队提出的单目视觉惯性SLAM系统，是视觉惯性SLAM领域的里程碑工作。

```python
class VINSMono:
    """
    VINS-Mono系统实现
    
    论文：Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A Robust and 
         Versatile Monocular Visual-Inertial State Estimator. IEEE T-RO.
    
    核心贡献：
    1. 基于优化的紧耦合VIO框架
    2. IMU预积分与视觉约束联合优化
    3. 鲁棒的初始化流程（视觉-惯性对齐）
    4. 基于4-DOF位姿图的回环检测
    5. 在线外参标定和时延估计
    
    系统架构：
    - 前端：KLT光流跟踪 + 关键帧选择
    - 初始化：纯视觉SfM + 视觉-惯性对齐
    - 后端：滑动窗口优化 + 边缘化
    - 回环：DBoW2 + 4-DOF位姿图优化
    
    优点：
    - 精度高，在EuRoC数据集上达到厘米级精度
    - 鲁棒性强，支持室内外各种场景
    - 实时性好，可在嵌入式平台运行
    - 功能完整，包含初始化、重定位、回环
    
    缺点：
    - 对IMU质量要求高
    - 初始化需要一定运动激励
    - 计算量相对较大
    - 单目尺度在初始化后固定
    """
    
    def __init__(self, camera_matrix, dist_coeffs, imu_params):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.imu_params = imu_params
        
        # 前端
        self.feature_tracker = FeatureTracker(camera_matrix, dist_coeffs)
        self.imu_preintegrator = IMUPreintegrator(
            acc_noise=imu_params['acc_noise'],
            gyro_noise=imu_params['gyro_noise']
        )
        
        # 状态
        self.initialized = False
        self.sliding_window = SlidingWindow(window_size=10)
        
        # 后端优化
        self.optimizer = VIOOptimizer()
        
        # 回环检测
        self.loop_detector = LoopDetector()
        self.pose_graph = PoseGraph4DOF()
        
        # 参数
        self.gravity = np.array([0, 0, -9.81])
        
    def process_imu(self, acc, gyro, timestamp):
        """
        处理IMU数据
        
        流程：
        1. 数据预处理（去偏置、单位转换）
        2. IMU预积分
        3. 中值积分用于状态预测
        """
        dt = timestamp - self.last_imu_time if hasattr(self, 'last_imu_time') else 0.01
        self.last_imu_time = timestamp
        
        # IMU预积分
        self.imu_preintegrator.integrate(acc, gyro, dt)
        
        # 如果已初始化，用IMU预测当前状态
        if self.initialized:
            self._predict_state(acc, gyro, dt)
    
    def process_image(self, img, timestamp):
        """
        处理图像
        
        流程：
        1. 特征跟踪（KLT光流）
        2. 关键帧判断
        3. 如果未初始化 -> 初始化流程
        4. 如果已初始化 -> VIO更新
        """
        # 特征跟踪
        tracked_features, new_features = self.feature_tracker.track(img)
        
        # 关键帧判断
        is_keyframe = self._is_keyframe(tracked_features)
        
        if not self.initialized:
            # 初始化流程
            self._initialization(img, tracked_features, timestamp)
        else:
            # VIO更新
            self._vio_update(tracked_features, is_keyframe, timestamp)
            
            # 回环检测
            if is_keyframe:
                self._loop_detection(img, timestamp)
    
    def _initialization(self, img, features, timestamp):
        """
        初始化流程
        
        步骤：
        1. 纯视觉SfM：估计相对位姿和特征点深度
        2. 视觉-惯性对齐：估计速度、重力方向、尺度、IMU偏置
        
        论文核心：通过求解线性方程组实现高效初始化
        """
        # 添加到滑动窗口
        self.sliding_window.add_frame(img, features, timestamp)
        
        # 检查是否有足够的视差
        if not self._check_parallax():
            return
        
        # 1. 纯视觉SfM
        success, poses, points_3d = self._visual_sfm()
        if not success:
            return
        
        # 2. 视觉-惯性对齐
        success = self._visual_inertial_alignment(poses, points_3d)
        if success:
            self.initialized = True
            print("初始化成功！")
    
    def _visual_sfm(self):
        """
        纯视觉SfM
        
        使用本质矩阵或基础矩阵估计初始位姿
        然后三角化特征点
        """
        # 选择参考帧和当前帧
        frames = self.sliding_window.get_frames()
        if len(frames) < 2:
            return False, None, None
        
        ref_frame = frames[0]
        curr_frame = frames[-1]
        
        # 找到共视点
        matches = self._find_common_features(ref_frame, curr_frame)
        
        if len(matches) < 8:
            return False, None, None
        
        # 计算本质矩阵
        E, mask = cv2.findEssentialMat(
            matches['pts1'], matches['pts2'],
            self.camera_matrix, method=cv2.RANSAC
        )
        
        # 分解本质矩阵得到R, t
        _, R, t, mask = cv2.recoverPose(
            E, matches['pts1'], matches['pts2'], self.camera_matrix
        )
        
        # 三角化
        points_3d = self._triangulate(R, t, matches)
        
        # 构建位姿序列
        poses = [np.eye(4)]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.flatten()
        poses.append(pose)
        
        return True, poses, points_3d
    
    def _visual_inertial_alignment(self, poses, points_3d):
        """
        视觉-惯性对齐
        
        估计：
        - 陀螺仪偏置
        - 速度
        - 重力方向
        - 尺度
        
        论文方法：通过求解线性最小二乘问题
        """
        # 获取IMU预积分数据
        preintegrations = self.sliding_window.get_preintegrations()
        
        # 1. 估计陀螺仪偏置
        gyro_bias = self._estimate_gyro_bias(poses, preintegrations)
        
        # 2. 修正预积分
        for preint in preintegrations:
            preint.correct_bias(np.zeros(3), gyro_bias)
        
        # 3. 估计速度、重力和尺度
        velocity, gravity, scale = self._estimate_velocity_gravity_scale(
            poses, points_3d, preintegrations
        )
        
        # 4. 优化所有参数
        success = self._refine_initialization(
            poses, points_3d, velocity, gravity, scale, gyro_bias
        )
        
        return success
    
    def _vio_update(self, features, is_keyframe, timestamp):
        """
        VIO更新
        
        滑动窗口优化：
        - 状态变量：位姿、速度、IMU偏置、特征点深度
        - 约束：IMU预积分约束、视觉重投影约束、先验约束
        """
        # 添加新帧到滑动窗口
        frame = self.sliding_window.add_keyframe(features, timestamp)
        
        # 构建优化问题
        problem = OptimizationProblem()
        
        # 添加IMU约束
        imu_factor = IMUFactor(self.imu_preintegrator)
        problem.add_factor(imu_factor)
        
        # 添加视觉约束
        for feat_id, feature in features.items():
            if feature.is_tracked:
                visual_factor = ReprojectionFactor(
                    feature.observations, self.camera_matrix
                )
                problem.add_factor(visual_factor)
        
        # 添加先验约束（边缘化产生的先验）
        if self.sliding_window.marginalized_prior is not None:
            problem.add_prior(self.sliding_window.marginalized_prior)
        
        # 优化
        self.optimizer.optimize(problem)
        
        # 边缘化
        if self.sliding_window.need_marginalization():
            self.sliding_window.marginalize_oldest()
        
        # 重置预积分器
        current_state = self.optimizer.get_current_state()
        self.imu_preintegrator.reset()
        self.imu_preintegrator.b_acc = current_state['b_acc']
        self.imu_preintegrator.b_gyro = current_state['b_gyro']
    
    def _loop_detection(self, img, timestamp):
        """
        回环检测
        
        使用DBoW2进行词袋匹配
        检测到回环后进行4-DOF位姿图优化
        """
        # 检测回环
        loop_result = self.loop_detector.detect(img)
        
        if loop_result.detected:
            # 计算相对位姿
            relative_pose = self._compute_loop_pose(loop_result)
            
            # 添加到位姿图
            self.pose_graph.add_loop_constraint(
                loop_result.query_id,
                loop_result.match_id,
                relative_pose
            )
            
            # 4-DOF位姿图优化
            self.pose_graph.optimize()
            
            # 更新VIO状态
            self._update_vio_from_pose_graph()


class FeatureTracker:
    """特征跟踪器（KLT光流）"""
    
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # 参数
        self.max_features = 150
        self.min_distance = 30
        self.window_size = (21, 21)
        self.max_level = 3
        
        # 状态
        self.prev_img = None
        self.prev_features = {}
        self.next_feature_id = 0
        
    def track(self, img):
        """
        跟踪特征点
        
        使用Lucas-Kanade光流法
        """
        if self.prev_img is None:
            # 第一帧，检测新特征
            features = self._detect_features(img)
            self.prev_img = img
            self.prev_features = features
            return features, {}
        
        # KLT光流跟踪
        prev_pts = np.array([f['pt'] for f in self.prev_features.values()])
        
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_img, img, prev_pts, None,
            winSize=self.window_size,
            maxLevel=self.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # 筛选好的跟踪点
        tracked_features = {}
        new_features = {}
        
        for i, (prev_id, prev_feat) in enumerate(self.prev_features.items()):
            if status[i]:
                tracked_features[prev_id] = {
                    'pt': curr_pts[i],
                    'id': prev_id,
                    'age': prev_feat.get('age', 0) + 1
                }
        
        # 检测新特征补充
        if len(tracked_features) < self.max_features:
            mask = self._create_mask(img.shape, tracked_features)
            new_features = self._detect_features(img, mask)
            
            # 合并
            for feat in new_features.values():
                tracked_features[self.next_feature_id] = feat
                self.next_feature_id += 1
        
        self.prev_img = img
        self.prev_features = tracked_features
        
        return tracked_features, new_features
    
    def _detect_features(self, img, mask=None):
        """检测Shi-Tomasi角点"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        corners = cv2.goodFeaturesToTrack(
            gray, self.max_features, 0.01, self.min_distance, mask=mask
        )
        
        features = {}
        if corners is not None:
            for i, corner in enumerate(corners):
                features[self.next_feature_id + i] = {
                    'pt': corner[0],
                    'id': self.next_feature_id + i,
                    'age': 0
                }
        
        return features
    
    def _create_mask(self, img_shape, existing_features):
        """创建掩码避免特征点过于集中"""
        mask = np.ones(img_shape[:2], dtype=np.uint8) * 255
        
        for feat in existing_features.values():
            pt = tuple(feat['pt'].astype(int))
            cv2.circle(mask, pt, self.min_distance, 0, -1)
        
        return mask


class SlidingWindow:
    """滑动窗口管理"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.frames = []
        self.marginalized_prior = None
    
    def add_frame(self, img, features, timestamp):
        """添加帧"""
        frame = {
            'img': img,
            'features': features,
            'timestamp': timestamp,
            'is_keyframe': False
        }
        self.frames.append(frame)
        return frame
    
    def add_keyframe(self, features, timestamp):
        """添加关键帧"""
        frame = {
            'features': features,
            'timestamp': timestamp,
            'is_keyframe': True
        }
        self.frames.append(frame)
        
        # 保持窗口大小
        if len(self.frames) > self.window_size:
            self.marginalize_oldest()
        
        return frame
    
    def marginalize_oldest(self):
        """边缘化最老的帧"""
        if len(self.frames) > 0:
            # 计算边缘化先验
            self.marginalized_prior = self._compute_marginalization()
            self.frames.pop(0)
    
    def _compute_marginalization(self):
        """计算边缘化先验（简化）"""
        # 实际实现需要计算舒尔补
        return None
    
    def get_frames(self):
        """获取所有帧"""
        return self.frames
    
    def need_marginalization(self):
        """检查是否需要边缘化"""
        return len(self.frames) > self.window_size


class LoopDetector:
    """回环检测器（基于DBoW2）"""
    
    def __init__(self):
        # 使用DBoW2或DBoW3
        self.vocabulary = None
        self.database = None
        self.keyframes = []
        
    def detect(self, img):
        """检测回环"""
        # 计算BoW向量
        bow_vector = self._compute_bow(img)
        
        # 查询数据库
        if self.database is not None:
            results = self.database.query(bow_vector, max_results=5)
            
            for result in results:
                if result.score > 0.05:  # 阈值
                    # 几何验证
                    if self._geometric_verification(result):
                        return LoopResult(detected=True, match_id=result.id)
        
        # 添加到数据库
        self._add_to_database(bow_vector)
        
        return LoopResult(detected=False)
    
    def _compute_bow(self, img):
        """计算BoW向量"""
        # 提取ORB特征
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img, None)
        
        # 转换为BoW向量
        # 实际使用DBoW2的transform方法
        return des
    
    def _geometric_verification(self, result):
        """几何验证"""
        # RANSAC验证几何一致性
        return True
    
    def _add_to_database(self, bow_vector):
        """添加到数据库"""
        if self.database is not None:
            self.database.add(bow_vector)


class LoopResult:
    """回环检测结果"""
    
    def __init__(self, detected=False, match_id=None, query_id=None):
        self.detected = detected
        self.match_id = match_id
        self.query_id = query_id


class PoseGraph4DOF:
    """4-DOF位姿图（x, y, z, yaw）"""
    
    def __init__(self):
        self.nodes = []
        self.edges = []
    
    def add_node(self, pose, timestamp):
        """添加节点"""
        node = {
            'id': len(self.nodes),
            'pose': pose,
            'timestamp': timestamp
        }
        self.nodes.append(node)
        return node['id']
    
    def add_loop_constraint(self, id1, id2, relative_pose):
        """添加回环约束"""
        edge = {
            'from': id1,
            'to': id2,
            'relative_pose': relative_pose
        }
        self.edges.append(edge)
    
    def optimize(self):
        """4-DOF位姿图优化"""
        # 使用g2o或GTSAM进行优化
        # 只优化x, y, z, yaw，保持roll和pitch不变
        pass


class VIOOptimizer:
    """VIO优化器"""
    
    def __init__(self):
        self.current_state = None
    
    def optimize(self, problem):
        """优化问题"""
        # 使用Ceres Solver或g2o
        # 构建目标函数并求解
        pass
    
    def get_current_state(self):
        """获取当前状态"""
        return self.current_state


class OptimizationProblem:
    """优化问题"""
    
    def __init__(self):
        self.factors = []
        self.priors = []
    
    def add_factor(self, factor):
        """添加因子"""
        self.factors.append(factor)
    
    def add_prior(self, prior):
        """添加先验"""
        self.priors.append(prior)


class IMUFactor:
    """IMU因子"""
    
    def __init__(self, preintegration):
        self.preintegration = preintegration


class ReprojectionFactor:
    """重投影误差因子"""
    
    def __init__(self, observations, camera_matrix):
        self.observations = observations
        self.camera_matrix = camera_matrix
```

### 2.3 MSCKF：多状态约束卡尔曼滤波

**论文核心思想（Mourikis & Roumeliotis, 2007）**

MSCKF是早期视觉惯性融合的代表性工作，使用EKF框架实现紧耦合融合。

```python
class MSCKF:
    """
    MSCKF (Multi-State Constraint Kalman Filter)
    
    论文：Mourikis, A. I., & Roumeliotis, S. I. (2007). A Multi-State 
         Constraint Kalman Filter for Vision-aided Inertial Navigation. ICRA.
    
    核心思想：
    1. 在状态向量中维护一个相机位姿的滑动窗口
    2. 特征点不加入状态向量，而是作为约束使用
    3. 当特征点跟踪丢失时，利用多帧观测构建几何约束
    
    优点：
    - 计算效率高（特征点不加入状态）
    - 适用于资源受限平台
    - 滤波方法理论成熟
    
    缺点：
    - 线性化误差累积
    - 无法回环（纯滤波方法）
    - 精度不如优化方法
    
    代表实现：S-MSCKF (Sun et al., 2018)
    """
    
    def __init__(self, imu_params, camera_params):
        # IMU状态（15维）：位置、速度、姿态、加速度偏置、陀螺仪偏置
        self.imu_state = IMUState()
        
        # 相机状态（滑动窗口）
        self.camera_states = []
        self.max_camera_states = 10
        
        # 协方差矩阵
        self.P = np.eye(15)
        
        # 特征点数据库
        self.feature_database = {}
        
        # 参数
        self.imu_params = imu_params
        self.camera_params = camera_params
        
    def predict(self, acc, gyro, dt):
        """
        IMU预测步骤
        
        传播IMU状态和协方差
        """
        # 状态传播
        self.imu_state.propagate(acc, gyro, dt)
        
        # 协方差传播
        F = self._compute_state_transition_matrix(acc, gyro, dt)
        Q = self._compute_process_noise(dt)
        
        self.P = F @ self.P @ F.T + Q
        
        # 相机状态传播（静态假设）
        for cam_state in self.camera_states:
            cam_state.propagate(self.imu_state, dt)
    
    def update(self, features):
        """
        更新步骤
        
        当特征点跟踪丢失时，利用多帧观测进行更新
        """
        # 更新特征点数据库
        lost_features = self._update_feature_database(features)
        
        # 对丢失的特征点进行更新
        for feature in lost_features:
            self._update_with_feature(feature)
        
        # 状态管理
        self._manage_states()
    
    def _update_with_feature(self, feature):
        """
        利用特征点进行MSCKF更新
        
        核心：利用特征点在多个相机状态下的观测，
        构建几何约束，但不将特征点加入状态向量
        """
        # 三角化特征点
        point_3d = self._triangulate_feature(feature)
        
        if point_3d is None:
            return
        
        # 计算残差和雅可比
        residuals = []
        H = []
        
        for obs in feature.observations:
            cam_id = obs['camera_id']
            cam_state = self._get_camera_state(cam_id)
            
            # 预测观测
            predicted = self._project(point_3d, cam_state)
            
            # 残差
            residual = obs['measurement'] - predicted
            residuals.append(residual)
            
            # 雅可比（关于相机状态）
            H_cam = self._compute_measurement_jacobian(point_3d, cam_state)
            H.append(H_cam)
        
        # 堆叠残差和雅可比
        r = np.concatenate(residuals)
        H_matrix = np.vstack(H)
        
        # 零空间投影（消除特征点位置的不确定性）
        H_proj = self._null_space_projection(H_matrix, point_3d)
        r_proj = self._null_space_projection_residual(r, H_matrix, point_3d)
        
        # EKF更新
        self._ekf_update(H_proj, r_proj)
    
    def _null_space_projection(self, H, point_3d):
        """
        零空间投影
        
        消除特征点位置的不确定性，只保留对相机状态的约束
        """
        # 计算H的左零空间
        # 实际实现使用QR分解
        return H
    
    def _ekf_update(self, H, r):
        """标准EKF更新"""
        # 卡尔曼增益
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 状态更新
        delta_x = K @ r
        self._apply_state_update(delta_x)
        
        # 协方差更新
        I_KH = np.eye(self.P.shape[0]) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
    
    def _manage_states(self):
        """状态管理"""
        # 添加新的相机状态
        if len(self.camera_states) < self.max_camera_states:
            new_cam_state = CameraState(self.imu_state)
            self.camera_states.append(new_cam_state)
        else:
            # 边缘化最老的相机状态
            self._marginalize_oldest_camera()
    
    def _marginalize_oldest_camera(self):
        """边缘化最老的相机状态"""
        # 移除最老的相机状态
        self.camera_states.pop(0)
        
        # 更新协方差矩阵
        # 移除对应的行和列
        pass


class IMUState:
    """IMU状态"""
    
    def __init__(self):
        self.p = np.zeros(3)  # 位置
        self.v = np.zeros(3)  # 速度
        self.q = np.array([1, 0, 0, 0])  # 四元数姿态
        self.b_a = np.zeros(3)  # 加速度偏置
        self.b_g = np.zeros(3)  # 陀螺仪偏置
    
    def propagate(self, acc, gyro, dt):
        """状态传播"""
        # 四元数积分
        omega = gyro - self.b_g
        q_dot = 0.5 * self._quaternion_multiply(self.q, np.concatenate([[0], omega]))
        self.q = self.q + q_dot * dt
        self.q = self.q / np.linalg.norm(self.q)
        
        # 速度积分
        R = self._quaternion_to_rotation(self.q)
        self.v += (R @ (acc - self.b_a) + np.array([0, 0, -9.81])) * dt
        
        # 位置积分
        self.p += self.v * dt
    
    def _quaternion_multiply(self, q1, q2):
        """四元数乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _quaternion_to_rotation(self, q):
        """四元数转旋转矩阵"""
        w, x, y, z = q
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)]
        ])


class CameraState:
    """相机状态"""
    
    def __init__(self, imu_state):
        self.timestamp = None
        self.pose = None
        self._compute_pose_from_imu(imu_state)
    
    def propagate(self, imu_state, dt):
        """传播（基于IMU）"""
        self._compute_pose_from_imu(imu_state)
    
    def _compute_pose_from_imu(self, imu_state):
        """从IMU状态计算相机位姿"""
        # 考虑IMU-相机外参
        pass
```

---

## 3. 激光视觉融合

### 3.1 相机-激光雷达外参标定

**问题的提出**

相机和激光雷达的融合需要精确的外参标定（旋转和平移），但两者数据形式差异大，标定困难。

```python
class CameraLidarCalibration:
    """
    相机-激光雷达外参标定
    
    问题：
    1. 相机获取2D图像（强度/颜色），激光获取3D点云
    2. 需要找到两者的对应关系
    3. 标定精度直接影响融合效果
    
    解决方案：
    1. 使用标定板（棋盘格/圆点板）
    2. 在图像中检测角点
    3. 在点云中检测标定板平面和边缘
    4. 求解PnP问题或ICP配准
    
    优点：
    - 精度高（可达像素级）
    - 可重复性好
    
    缺点：
    - 需要专用标定板
    - 标定过程繁琐
    - 外参可能随时间漂移
    
    自动标定方法：
    - 基于互信息的标定（无需标定板）
    - 基于边缘对齐的标定
    """
    
    def __init__(self):
        self.R_cl = np.eye(3)  # 激光到相机的旋转
        self.t_cl = np.zeros(3)  # 激光到相机的平移
        self.calibrated = False
    
    def calibrate_with_chessboard(self, images, point_clouds, chessboard_size):
        """
        使用棋盘格标定
        
        参数:
            images: 标定板图像列表
            point_clouds: 对应的点云列表
            chessboard_size: 棋盘格尺寸 (rows, cols)
        """
        camera_points = []  # 图像中的角点
        lidar_points = []   # 点云中的对应点
        
        for img, pc in zip(images, point_clouds):
            # 1. 检测图像中的棋盘格角点
            corners_img = self._detect_chessboard(img, chessboard_size)
            if corners_img is None:
                continue
            
            # 2. 检测点云中的标定板
            corners_lidar = self._detect_chessboard_in_lidar(pc, chessboard_size)
            if corners_lidar is None:
                continue
            
            camera_points.extend(corners_img)
            lidar_points.extend(corners_lidar)
        
        if len(camera_points) < 6:
            print("标定数据不足")
            return False
        
        # 3. 求解外参（PnP或ICP）
        success = self._solve_extrinsics(camera_points, lidar_points)
        
        return success
    
    def _detect_chessboard(self, img, pattern_size):
        """检测图像中的棋盘格角点"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # 亚像素精化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners.reshape(-1, 2)
        
        return None
    
    def _detect_chessboard_in_lidar(self, points, pattern_size):
        """在点云中检测标定板"""
        # 1. 平面分割
        plane_points = self._segment_plane(points)
        
        if len(plane_points) < 100:
            return None
        
        # 2. 提取标定板边缘
        edges = self._extract_edges(plane_points)
        
        # 3. 拟合角点
        corners = self._fit_corners(edges, pattern_size)
        
        return corners
    
    def _segment_plane(self, points, distance_threshold=0.02):
        """使用RANSAC分割平面"""
        # 简化的平面分割
        # 实际使用PCL或Open3D的RANSAC
        return points
    
    def _extract_edges(self, points):
        """提取边缘点"""
        # 基于法向量变化检测边缘
        return points
    
    def _fit_corners(self, edges, pattern_size):
        """拟合角点位置"""
        # 根据棋盘格几何结构拟合角点
        return edges[:pattern_size[0]*pattern_size[1]]
    
    def _solve_extrinsics(self, camera_points, lidar_points):
        """求解外参"""
        # 使用P3P或EPnP求解
        # 这里简化，实际使用OpenCV的solvePnP
        
        camera_points = np.array(camera_points)
        lidar_points = np.array(lidar_points)
        
        # 假设已知相机内参
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
        dist = np.zeros(4)
        
        # 使用ICP精化
        success, rvec, tvec = cv2.solvePnP(
            lidar_points, camera_points, K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            self.R_cl, _ = cv2.Rodrigues(rvec)
            self.t_cl = tvec.flatten()
            self.calibrated = True
        
        return success
    
    def project_lidar_to_image(self, points, camera_matrix, img_shape):
        """
        将激光点云投影到图像
        
        用于验证标定结果和可视化
        """
        if not self.calibrated:
            print("请先进行标定")
            return None, None
        
        # 变换到相机坐标系
        points_cam = (self.R_cl @ points.T).T + self.t_cl
        
        # 筛选相机前方的点
        mask_front = points_cam[:, 2] > 0
        points_cam = points_cam[mask_front]
        
        # 投影到图像平面
        points_2d = (camera_matrix @ points_cam.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        
        # 筛选图像范围内的点
        mask_img = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_shape[1]) & \
                   (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_shape[0])
        
        return points_2d[mask_img], points_cam[mask_img, 2]
    
    def colorize_point_cloud(self, points, img, camera_matrix):
        """给点云着色"""
        points_2d, depths = self.project_lidar_to_image(
            points, camera_matrix, img.shape[:2]
        )
        
        colors = []
        for pt in points_2d.astype(int):
            if 0 <= pt[1] < img.shape[0] and 0 <= pt[0] < img.shape[1]:
                colors.append(img[pt[1], pt[0]])
            else:
                colors.append([128, 128, 128])
        
        return np.array(colors)
```

### 3.2 激光视觉融合SLAM系统

```python
class LidarVisualSLAM:
    """
    激光视觉融合SLAM
    
    融合策略：
    1. 松耦合：各自独立估计，结果融合
    2. 紧耦合：原始测量联合优化
    
    典型系统：
    - LVI-SAM (Shan et al., 2021)
    - R2LIVE (Lin et al., 2021)
    - FAST-LIVO (Xu et al., 2023)
    """
    
    def __init__(self, camera_matrix, dist_coeffs, lidar_config):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        # 外参（需要预先标定）
        self.T_cl = np.eye(4)  # 激光到相机变换
        
        # 子系统
        self.visual_frontend = VisualFrontend(camera_matrix, dist_coeffs)
        self.lidar_frontend = LidarFrontend(lidar_config)
        self.imu_preintegrator = IMUPreintegrator()
        
        # 后端
        self.backend = TightlyCoupledBackend()
        
        # 地图
        self.sparse_map = SparseMap()  # 视觉特征地图
        self.dense_map = DenseMap()    # 激光点云地图
        
    def process_frame(self, img, points, imu_data=None):
        """
        处理一帧数据
        
        流程：
        1. 分别处理视觉和激光数据
        2. 提取互补特征
        3. 联合优化
        """
        # 视觉前端
        visual_features = self.visual_frontend.process(img)
        
        # 激光前端
        lidar_features = self.lidar_frontend.process(points)
        
        # IMU预积分
        if imu_data is not None:
            for acc, gyro, dt in imu_data:
                self.imu_preintegrator.integrate(acc, gyro, dt)
        
        # 特征关联
        associated_features = self._associate_features(
            visual_features, lidar_features
        )
        
        # 后端优化
        self.backend.add_frame(
            visual_features, lidar_features, 
            self.imu_preintegrator, associated_features
        )
        
        # 更新地图
        self._update_maps(visual_features, lidar_features)
        
        # 重置IMU预积分
        self.imu_preintegrator.reset()
    
    def _associate_features(self, visual_features, lidar_features):
        """
        关联视觉和激光特征
        
        方法：
        1. 将激光点投影到图像
        2. 找到视觉特征附近的深度信息
        3. 为视觉特征提供深度约束
        """
        associations = {}
        
        # 将激光点投影到图像
        lidar_points = lidar_features['points']
        points_2d, depths = self._project_lidar_to_image(lidar_points)
        
        # 为每个视觉特征找到最近的激光深度
        for feat_id, feat in visual_features.items():
            pt = feat['pt']
            
            # 找到最近的激光投影点
            distances = np.linalg.norm(points_2d - pt, axis=1)
            nearest_idx = np.argmin(distances)
            
            if distances[nearest_idx] < 5:  # 像素阈值
                associations[feat_id] = {
                    'visual': feat,
                    'depth': depths[nearest_idx],
                    'lidar_point': lidar_points[nearest_idx]
                }
        
        return associations
    
    def _project_lidar_to_image(self, points):
        """投影激光点到图像"""
        # 应用外参变换
        points_cam = (self.T_cl[:3, :3] @ points.T).T + self.T_cl[:3, 3]
        
        # 投影
        points_2d = (self.camera_matrix @ points_cam.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        
        return points_2d, points_cam[:, 2]
    
    def _update_maps(self, visual_features, lidar_features):
        """更新地图"""
        # 更新稀疏地图（视觉特征）
        self.sparse_map.add_features(visual_features)
        
        # 更新稠密地图（激光点云）
        self.dense_map.add_points(lidar_features['points'])


class VisualFrontend:
    """视觉前端"""
    
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tracker = FeatureTracker(camera_matrix, dist_coeffs)
    
    def process(self, img):
        """处理图像"""
        features, _ = self.tracker.track(img)
        return features


class LidarFrontend:
    """激光前端"""
    
    def __init__(self, config):
        self.config = config
        self.feature_extractor = LidarFeatureExtractor()
    
    def process(self, points):
        """处理点云"""
        # 去畸变
        undistorted = self._undistort(points)
        
        # 特征提取
        features = self.feature_extractor.extract(undistorted)
        
        return {
            'points': undistorted,
            'edge_features': features['edge'],
            'planar_features': features['planar']
        }
    
    def _undistort(self, points):
        """点云去畸变"""
        return points


class LidarFeatureExtractor:
    """激光特征提取器"""
    
    def extract(self, points):
        """提取边缘和平面特征"""
        # 计算曲率
        curvatures = self._compute_curvature(points)
        
        # 分类特征
        edge_features = points[curvatures > 0.1]
        planar_features = points[curvatures < 0.05]
        
        return {
            'edge': edge_features,
            'planar': planar_features
        }
    
    def _compute_curvature(self, points):
        """计算曲率"""
        # 简化的曲率计算
        return np.zeros(len(points))


class TightlyCoupledBackend:
    """紧耦合后端优化"""
    
    def __init__(self):
        self.window_size = 10
        self.frames = []
        self.optimizer = None
    
    def add_frame(self, visual_features, lidar_features, 
                  imu_preintegration, associations):
        """添加帧到后端"""
        frame = {
            'visual': visual_features,
            'lidar': lidar_features,
            'imu': imu_preintegration,
            'associations': associations
        }
        
        self.frames.append(frame)
        
        # 滑动窗口管理
        if len(self.frames) > self.window_size:
            self._marginalize_oldest()
        
        # 优化
        self._optimize()
    
    def _optimize(self):
        """联合优化"""
        # 构建优化问题
        # 包含视觉重投影误差、激光配准误差、IMU误差
        pass
    
    def _marginalize_oldest(self):
        """边缘化最老帧"""
        self.frames.pop(0)


class SparseMap:
    """稀疏地图（视觉特征）"""
    
    def __init__(self):
        self.landmarks = {}
        self.next_id = 0
    
    def add_features(self, features):
        """添加特征"""
        for feat_id, feat in features.items():
            if feat_id not in self.landmarks:
                self.landmarks[feat_id] = {
                    'observations': [],
                    '3d_position': None
                }
            
            self.landmarks[feat_id]['observations'].append(feat)


class DenseMap:
    """稠密地图（激光点云）"""
    
    def __init__(self):
        self.point_clouds = []
    
    def add_points(self, points):
        """添加点云"""
        self.point_clouds.append(points)
```

---

## 4. 更多传感器融合

### 4.1 GPS与SLAM融合

**问题的提出**

SLAM存在累积漂移，而GPS提供绝对位置但更新慢、室内失效。如何有效融合两者？

```python
class GPSFusion:
    """
    GPS与SLAM融合
    
    问题：
    1. GPS更新频率低（1-10Hz）vs SLAM高频（10-60Hz）
    2. GPS存在多路径效应和噪声
    3. 坐标系转换（WGS84到局部坐标）
    4. GPS室内/遮挡环境失效
    
    解决方案：
    1. 松耦合：GPS作为位姿先验或回环约束
    2. 紧耦合：GPS伪距与SLAM联合优化
    3. 自适应权重：根据GPS质量调整融合权重
    
    优点：
    - 消除长期漂移
    - 提供绝对尺度
    - 增强鲁棒性
    
    缺点：
    - GPS依赖外部环境
    - 坐标系对齐复杂
    - 时间同步要求高
    """
    
    def __init__(self):
        # 状态
        self.pose = np.eye(4)
        self.cov = np.eye(6)
        
        # GPS到局部坐标的转换
        self.gps_origin = None
        self.origin_set = False
        
        # GPS质量评估
        self.gps_quality_threshold = 5.0  # 米
        self.gps_history = []
        
    def set_gps_origin(self, gps_data):
        """
        设置GPS原点
        
        将第一帧GPS作为局部坐标系原点
        """
        self.gps_origin = {
            'lat': gps_data['lat'],
            'lon': gps_data['lon'],
            'alt': gps_data.get('alt', 0)
        }
        self.origin_set = True
    
    def gps_to_local(self, gps_data):
        """
        GPS坐标转局部坐标
        
        使用简化的平面投影（适用于小范围）
        大范围需要使用UTM或其他投影
        """
        if not self.origin_set:
            return None
        
        lat, lon = gps_data['lat'], gps_data['lon']
        alt = gps_data.get('alt', 0)
        
        lat0, lon0 = self.gps_origin['lat'], self.gps_origin['lon']
        alt0 = self.gps_origin.get('alt', 0)
        
        # 简化的墨卡托投影
        # 1度纬度约111.32km
        # 1度经度约111.32km * cos(lat)
        x = (lat - lat0) * 111320.0
        y = (lon - lon0) * 111320.0 * np.cos(np.radians(lat0))
        z = alt - alt0
        
        return np.array([x, y, z])
    
    def evaluate_gps_quality(self, gps_data):
        """
        评估GPS质量
        
        指标：
        - HDOP/VDOP（精度因子）
        - 卫星数量
        - 信号强度
        - 与历史数据的一致性
        """
        quality_score = 1.0
        
        # 检查HDOP
        if 'hdop' in gps_data and gps_data['hdop'] > 5.0:
            quality_score *= 0.5
        
        # 检查卫星数
        if 'num_satellites' in gps_data and gps_data['num_satellites'] < 6:
            quality_score *= 0.7
        
        # 检查与历史的一致性
        if len(self.gps_history) > 0:
            last_pos = self.gps_history[-1]
            curr_pos = self.gps_to_local(gps_data)
            
            if curr_pos is not None:
                distance = np.linalg.norm(curr_pos - last_pos)
                time_diff = gps_data['timestamp'] - self.gps_history_timestamp[-1]
                
                # 检查速度合理性
                if time_diff > 0:
                    velocity = distance / time_diff
                    if velocity > 50:  # 50m/s = 180km/h
                        quality_score *= 0.3
        
        return quality_score
    
    def update(self, slam_pose, slam_cov, gps_data=None):
        """
        融合更新
        
        使用卡尔曼滤波或因子图优化
        """
        if gps_data is None or not self.origin_set:
            # 纯SLAM模式
            self.pose = slam_pose
            self.cov = slam_cov
            return
        
        # 评估GPS质量
        gps_quality = self.evaluate_gps_quality(gps_data)
        
        if gps_quality < 0.3:
            # GPS质量差，降低权重
            gps_cov = np.eye(3) * 100.0
        else:
            gps_cov = np.eye(3) * gps_data.get('accuracy', 5.0)
        
        # 卡尔曼滤波融合
        gps_pos = self.gps_to_local(gps_data)
        if gps_pos is not None:
            # 位置更新
            H = np.hstack([np.eye(3), np.zeros((3, 3))])  # 只观测位置
            
            # 融合
            y = gps_pos - slam_pose[:3, 3]
            S = H @ slam_cov @ H.T + gps_cov
            K = slam_cov @ H.T @ np.linalg.inv(S)
            
            # 更新
            delta_x = K @ y
            self.pose = self._apply_delta_pose(slam_pose, delta_x)
            self.cov = (np.eye(6) - K @ H) @ slam_cov
        else:
            self.pose = slam_pose
            self.cov = slam_cov
    
    def _apply_delta_pose(self, pose, delta_x):
        """应用位姿增量"""
        new_pose = pose.copy()
        new_pose[:3, 3] += delta_x[:3]
        # 旋转更新（简化）
        return new_pose


class WheelOdometer:
    """
    轮速计里程计
    
    问题：
    1. 轮胎打滑导致误差
    2. 轮胎变形影响精度
    3. 地面类型影响（沙地、冰面）
    
    解决方案：
    1. 与IMU融合检测打滑
    2. 在线标定轮速计参数
    3. 多传感器融合校正
    
    优点：
    - 简单可靠
    - 成本低
    - 实时性好
    
    缺点：
    - 累积误差
    - 受地面条件影响
    """
    
    def __init__(self, wheel_base=1.0, wheel_radius=0.3):
        self.wheel_base = wheel_base  # 轮距
        self.wheel_radius = wheel_radius  # 轮半径
        
        # 状态
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # 协方差
        self.cov = np.eye(3) * 0.01
        
        # 参数
        self.left_ticks_prev = 0
        self.right_ticks_prev = 0
        self.ticks_per_rev = 1000  # 每转脉冲数
    
    def update_from_ticks(self, left_ticks, right_ticks, dt):
        """
        从编码器脉冲更新
        
        使用差速驱动模型
        """
        # 计算增量
        delta_left = (left_ticks - self.left_ticks_prev) / self.ticks_per_rev * 2 * np.pi * self.wheel_radius
        delta_right = (right_ticks - self.right_ticks_prev) / self.ticks_per_rev * 2 * np.pi * self.wheel_radius
        
        self.left_ticks_prev = left_ticks
        self.right_ticks_prev = right_ticks
        
        # 差速模型
        delta_s = (delta_left + delta_right) / 2
        delta_theta = (delta_right - delta_left) / self.wheel_base
        
        # 更新位姿
        if abs(delta_theta) < 1e-6:
            # 直线运动
            self.x += delta_s * np.cos(self.theta)
            self.y += delta_s * np.sin(self.theta)
        else:
            # 圆弧运动
            r = delta_s / delta_theta
            self.x += r * (np.sin(self.theta + delta_theta) - np.sin(self.theta))
            self.y += r * (np.cos(self.theta) - np.cos(self.theta + delta_theta))
            self.theta += delta_theta
        
        # 协方差传播
        self._propagate_covariance(delta_s, delta_theta)
    
    def update_from_velocity(self, v_left, v_right, dt):
        """
        从轮速更新
        
        参数:
            v_left: 左轮速度 (m/s)
            v_right: 右轮速度 (m/s)
            dt: 时间间隔
        """
        v = (v_left + v_right) / 2
        omega = (v_right - v_left) / self.wheel_base
        
        # 积分
        if abs(omega) < 1e-6:
            self.x += v * dt * np.cos(self.theta)
            self.y += v * dt * np.sin(self.theta)
        else:
            r = v / omega
            self.x += r * (np.sin(self.theta + omega * dt) - np.sin(self.theta))
            self.y += r * (np.cos(self.theta) - np.cos(self.theta + omega * dt))
            self.theta += omega * dt
    
    def _propagate_covariance(self, delta_s, delta_theta):
        """协方差传播"""
        # 简化的噪声模型
        noise_s = 0.01 * abs(delta_s)
        noise_theta = 0.01 * abs(delta_theta)
        
        # 雅可比
        J = np.array([
            [1, 0, -delta_s * np.sin(self.theta)],
            [0, 1, delta_s * np.cos(self.theta)],
            [0, 0, 1]
        ])
        
        Q = np.diag([noise_s**2, noise_s**2, noise_theta**2])
        
        self.cov = J @ self.cov @ J.T + Q
    
    def get_pose(self):
        """获取位姿（4x4矩阵）"""
        pose = np.eye(4)
        pose[0, 0] = np.cos(self.theta)
        pose[0, 1] = -np.sin(self.theta)
        pose[1, 0] = np.sin(self.theta)
        pose[1, 1] = np.cos(self.theta)
        pose[0, 3] = self.x
        pose[1, 3] = self.y
        return pose
    
    def detect_slip(self, imu_accel, imu_gyro, dt):
        """
        检测轮胎打滑
        
        通过比较轮速计和IMU的加速度差异
        """
        # 从轮速计计算加速度
        v = self._get_current_velocity()
        
        # 与IMU比较
        accel_diff = np.linalg.norm(imu_accel[:2]) - abs(v / dt)
        
        if accel_diff > 2.0:  # 阈值
            return True
        return False
    
    def _get_current_velocity(self):
        """获取当前速度"""
        # 简化实现
        return 0.0


class FactorGraphFusion:
    """
    因子图融合框架
    
    论文核心思想（Dellaert & Kaess, 2006; Grisetti et al., 2010）:
    将SLAM问题建模为因子图，每个传感器提供一种因子，
    通过图优化统一求解。
    
    因子类型：
    - 先验因子：初始位姿约束
    - 里程计因子：相对位姿约束
    - IMU因子：预积分约束
    - 视觉因子：重投影约束
    - 激光因子：点云配准约束
    - GPS因子：绝对位置约束
    - 回环因子：回环检测约束
    
    优点：
    - 模块化，易于扩展
    - 可以处理各种传感器组合
    - 精度高
    
    缺点：
    - 计算量大
    - 需要良好的初始值
    """
    
    def __init__(self):
        # 节点（位姿和路标点）
        self.pose_nodes = {}
        self.landmark_nodes = {}
        
        # 因子
        self.factors = []
        
        # 下一个索引
        self.next_pose_idx = 0
        self.next_landmark_idx = 0
        
        # 优化器
        self.optimizer = None
    
    def add_pose_node(self, pose, timestamp):
        """添加位姿节点"""
        idx = self.next_pose_idx
        self.pose_nodes[idx] = {
            'pose': pose,
            'timestamp': timestamp,
            'fixed': False
        }
        self.next_pose_idx += 1
        return idx
    
    def add_landmark_node(self, position):
        """添加路标节点"""
        idx = self.next_landmark_idx
        self.landmark_nodes[idx] = {
            'position': position
        }
        self.next_landmark_idx += 1
        return idx
    
    def add_prior_factor(self, pose_idx, prior_pose, information):
        """
        添加先验因子
        
        约束特定节点的位姿
        """
        factor = {
            'type': 'prior',
            'pose_idx': pose_idx,
            'prior': prior_pose,
            'information': information
        }
        self.factors.append(factor)
    
    def add_odometry_factor(self, from_idx, to_idx, relative_pose, information):
        """
        添加里程计因子
        
        约束两个位姿之间的相对变换
        """
        factor = {
            'type': 'odometry',
            'from': from_idx,
            'to': to_idx,
            'relative_pose': relative_pose,
            'information': information
        }
        self.factors.append(factor)
    
    def add_imu_factor(self, from_idx, to_idx, preintegration, information):
        """
        添加IMU因子
        
        使用IMU预积分约束
        """
        factor = {
            'type': 'imu',
            'from': from_idx,
            'to': to_idx,
            'preintegration': preintegration,
            'information': information
        }
        self.factors.append(factor)
    
    def add_visual_factor(self, pose_idx, landmark_idx, measurement, camera_matrix):
        """
        添加视觉因子
        
        重投影误差约束
        """
        factor = {
            'type': 'visual',
            'pose_idx': pose_idx,
            'landmark_idx': landmark_idx,
            'measurement': measurement,
            'camera_matrix': camera_matrix
        }
        self.factors.append(factor)
    
    def add_gps_factor(self, pose_idx, gps_position, covariance):
        """
        添加GPS因子
        
        绝对位置约束
        """
        factor = {
            'type': 'gps',
            'pose_idx': pose_idx,
            'position': gps_position,
            'covariance': covariance
        }
        self.factors.append(factor)
    
    def add_loop_factor(self, from_idx, to_idx, relative_pose, information):
        """
        添加回环因子
        
        消除累积误差
        """
        factor = {
            'type': 'loop',
            'from': from_idx,
            'to': to_idx,
            'relative_pose': relative_pose,
            'information': information
        }
        self.factors.append(factor)
    
    def optimize(self):
        """
        执行图优化
        
        使用Levenberg-Marquardt或Dogleg算法
        """
        # 构建优化问题
        # 使用g2o或GTSAM
        
        # 1. 定义变量
        # 2. 添加因子
        # 3. 求解
        
        print(f"优化: {len(self.pose_nodes)} 位姿节点, "
              f"{len(self.landmark_nodes)} 路标节点, "
              f"{len(self.factors)} 因子")
        
        # 实际实现调用优化库
        pass
    
    def marginalize(self, pose_idx):
        """
        边缘化位姿节点
        
        使用舒尔补将旧状态边缘化，产生先验约束
        """
        # 计算边缘化先验
        # 添加到因子图
        pass


---

## 5. 经典系统深度解析

### 5.1 LVI-SAM：激光-视觉-惯性融合

**论文核心思想（Shan et al., 2021）**

LVI-SAM是MIT团队提出的紧耦合激光-视觉-惯性SLAM系统，结合了LIO-SAM和VINS-Mono的优点。

```python
class LVISAM:
    """
    LVI-SAM系统
    
    论文：Shan, T., Englot, B., Meyers, D., et al. (2021). LVI-SAM: 
         Tightly-coupled Lidar-Visual-Inertial Odometry via Smoothing and Mapping. ICRA.
    
    核心贡献：
    1. 两个子系统松耦合：视觉-惯性系统(VIS)和激光-惯性系统(LIS)
    2. VIS为LIS提供初始值和回环检测
    3. LIS为VIS提供深度信息
    4. 因子图优化统一处理所有约束
    
    系统架构：
    - VIS子系统：基于VINS-Mono，处理相机和IMU
    - LIS子系统：基于LIO-SAM，处理激光和IMU
    - 因子图：统一优化框架
    
    优点：
    - 充分利用各传感器优势
    - VIS在几何退化环境提供约束
    - LIS提供精确深度
    - 鲁棒性强
    
    缺点：
    - 系统复杂度高
    - 计算量大
    - 需要精确标定
    """
    
    def __init__(self, config):
        self.config = config
        
        # 子系统
        self.vis = VisualInertialSystem(config['vis'])
        self.lis = LidarInertialSystem(config['lis'])
        
        # 因子图
        self.factor_graph = FactorGraphFusion()
        
        # 状态
        self.current_pose = np.eye(4)
        self.initialized = False
    
    def process_imu(self, acc, gyro, timestamp):
        """处理IMU数据"""
        # 同时发送给两个子系统
        self.vis.process_imu(acc, gyro, timestamp)
        self.lis.process_imu(acc, gyro, timestamp)
    
    def process_image(self, img, timestamp):
        """处理图像"""
        vis_result = self.vis.process_image(img, timestamp)
        
        # 如果VIS初始化成功，更新全局状态
        if vis_result['initialized']:
            self._update_from_vis(vis_result)
    
    def process_lidar(self, points, timestamp):
        """处理激光数据"""
        lis_result = self.lis.process_lidar(points, timestamp)
        
        # 更新全局状态
        self._update_from_lis(lis_result)
    
    def _update_from_vis(self, vis_result):
        """从VIS更新"""
        # VIS提供：
        # 1. 初始位姿估计
        # 2. 回环检测结果
        # 3. 视觉约束
        pass
    
    def _update_from_lis(self, lis_result):
        """从LIS更新"""
        # LIS提供：
        # 1. 精确位姿估计
        # 2. 深度信息（用于VIS）
        # 3. 激光约束
        pass
```

### 5.2 R2LIVE：鲁棒实时多传感器融合

**论文核心思想（Lin et al., 2021）**

R2LIVE是香港大学提出的鲁棒实时多传感器状态估计器。

```python
class R2LIVE:
    """
    R2LIVE系统
    
    论文：Lin, J., & Zhang, F. (2021). R2LIVE: A Robust, Real-time, 
         LiDAR-Inertial-Visual tightly-coupled state Estimator and mapping. ICRA.
    
    核心贡献：
    1. 基于误差状态迭代卡尔曼滤波(ESIKF)的紧耦合框架
    2. 激光、视觉、IMU测量统一在状态估计中
    3. 动态外点剔除
    4. 实时性能优异
    
    优点：
    - 实时性好（100Hz IMU更新）
    - 鲁棒性强
    - 适用于快速运动
    
    缺点：
    - 滤波方法存在线性化误差
    - 无法回环
    """
    
    def __init__(self):
        # 状态（误差状态形式）
        self.state = State()
        self.error_state = ErrorState()
        
        # 协方差
        self.covariance = np.eye(18)
        
        # 特征管理
        self.visual_features = []
        self.lidar_features = []
    
    def predict(self, acc, gyro, dt):
        """ESIKF预测步骤"""
        # 名义状态传播
        self.state.propagate(acc, gyro, dt)
        
        # 误差状态传播
        F = self._compute_error_state_transition()
        Q = self._compute_process_noise()
        
        self.covariance = F @ self.covariance @ F.T + Q
    
    def update(self, visual_measurements, lidar_measurements):
        """ESIKF更新步骤"""
        # 计算观测雅可比
        H_visual = self._compute_visual_jacobian(visual_measurements)
        H_lidar = self._compute_lidar_jacobian(lidar_measurements)
        
        H = np.vstack([H_visual, H_lidar])
        
        # 计算残差
        r_visual = self._compute_visual_residual(visual_measurements)
        r_lidar = self._compute_lidar_residual(lidar_measurements)
        
        r = np.concatenate([r_visual, r_lidar])
        
        # 迭代更新
        self._iterated_update(H, r)
    
    def _iterated_update(self, H, r):
        """迭代更新"""
        for iteration in range(3):
            # 卡尔曼增益
            S = H @ self.covariance @ H.T + self.R
            K = self.covariance @ H.T @ np.linalg.inv(S)
            
            # 更新误差状态
            delta_x = K @ r
            
            # 更新名义状态
            self.state.update(delta_x)
            
            # 重置误差状态
            self.error_state.reset()
            
            # 重新计算残差
            r = self._recompute_residual()
```

---

## 6. 挑战与前沿方向

### 6.1 当前挑战

1. **标定问题**：
   - 多传感器外参标定复杂
   - 在线标定方法不够鲁棒
   - 温度、振动导致参数漂移

2. **时间同步**：
   - 各传感器采样频率不同
   - 硬件触发 vs 软件同步
   - 传输延迟不确定

3. **计算资源**：
   - 紧耦合融合计算量大
   - 嵌入式平台资源受限
   - 实时性与精度的平衡

4. **鲁棒性**：
   - 传感器失效处理
   - 动态环境适应
   - 极端条件（雨雪雾）

### 6.2 前沿方向

1. **深度学习增强的多模态SLAM**：
   - 端到端特征学习
   - 神经网络数据关联
   - 不确定性估计

2. **事件相机融合**：
   - 高动态范围
   - 微秒级延迟
   - 低功耗

3. **多智能体协同SLAM**：
   - 分布式优化
   - 数据共享与隐私
   - 协作建图

4. **语义多模态SLAM**：
   - 语义信息辅助定位
   - 动态物体检测
   - 场景理解

---

## 7. 实践练习

### 练习1：IMU预积分实现

```python
def exercise_imu_preintegration():
    """IMU预积分练习"""
    print("=== IMU预积分练习 ===")
    
    # 参数
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
        omega_true = np.array([0.01 * np.sin(i * 0.01), 
                               0.01 * np.cos(i * 0.01), 0.1])
        acc_true = np.array([0.1 * np.cos(i * 0.02), 
                             0.1 * np.sin(i * 0.02), 0.0]) + g
        
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
            delta_R = np.eye(3) + np.sin(theta_norm) * K + \
                     (1 - np.cos(theta_norm)) * (K @ K)
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
    print(f"位置误差: {np.linalg.norm(p_j - true_p_j):.6f} m")
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
            "year": 2018,
            "sensors": "相机 + IMU",
            "method": "紧耦合优化",
            "frontend": "KLT光流",
            "backend": "滑动窗口优化",
            "pros": ["精度高", "尺度可观", "功能完整"],
            "cons": ["对IMU质量要求高", "初始化需要运动激励"],
            "dataset_performance": "EuRoC: 0.1-0.3m RMSE"
        },
        {
            "name": "LIO-SAM",
            "year": 2020,
            "sensors": "激光 + IMU",
            "method": "因子图",
            "frontend": "特征提取",
            "backend": "iSAM2",
            "pros": ["稳定", "精度高", "回环检测"],
            "cons": ["计算量大", "需要激光雷达"],
            "dataset_performance": "KITTI: 0.5-1.0m RMSE"
        },
        {
            "name": "LVI-SAM",
            "year": 2021,
            "sensors": "激光 + 视觉 + IMU",
            "method": "松耦合双系统",
            "frontend": "KLT + 特征提取",
            "backend": "联合因子图",
            "pros": ["完整", "鲁棒", "互补优势"],
            "cons": ["非常复杂", "标定要求高", "计算量大"],
            "dataset_performance": "多数据集验证"
        },
        {
            "name": "R2LIVE",
            "year": 2021,
            "sensors": "激光 + 视觉 + IMU",
            "method": "ESIKF紧耦合",
            "frontend": "直接法 + 特征提取",
            "backend": "误差状态滤波",
            "pros": ["实时性好", "鲁棒性强", "快速运动适应"],
            "cons": ["线性化误差", "无法回环"],
            "dataset_performance": "100Hz IMU更新"
        },
        {
            "name": "FAST-LIO2",
            "year": 2021,
            "sensors": "激光 + IMU",
            "method": "ESIKF + ikd-Tree",
            "frontend": "直接点云配准",
            "backend": "迭代卡尔曼滤波",
            "pros": ["极快", "高精度", "增量式k-d树"],
            "cons": ["需要激光雷达", "无视觉信息"],
            "dataset_performance": "50-100Hz更新"
        }
    ]
    
    for s in systems:
        print(f"\n{s['name']} ({s['year']})")
        print(f"  传感器: {s['sensors']}")
        print(f"  方法: {s['method']}")
        print(f"  前端: {s['frontend']}")
        print(f"  后端: {s['backend']}")
        print(f"  优点: {', '.join(s['pros'])}")
        print(f"  缺点: {', '.join(s['cons'])}")
        print(f"  性能: {s['dataset_performance']}")

# exercise_multimodal_comparison()
```

### 练习3：因子图构建

```python
def exercise_factor_graph():
    """因子图构建练习"""
    print("=== 因子图构建练习 ===")
    
    # 创建因子图
    graph = FactorGraphFusion()
    
    # 添加位姿节点
    poses = []
    for i in range(10):
        pose = np.eye(4)
        pose[0, 3] = i * 1.0  # 沿x轴移动
        idx = graph.add_pose_node(pose, timestamp=i)
        poses.append(idx)
    
    # 添加先验（固定第一个位姿）
    graph.add_prior_factor(poses[0], np.eye(4), np.eye(6) * 1000)
    
    # 添加里程计因子
    for i in range(len(poses) - 1):
        rel_pose = np.eye(4)
        rel_pose[0, 3] = 1.0
        info = np.eye(6) * 100
        graph.add_odometry_factor(poses[i], poses[i+1], rel_pose, info)
    
    # 添加回环因子（假设第9帧与第0帧形成回环）
    loop_pose = np.eye(4)
    loop_pose[0, 3] = -9.0
    graph.add_loop_factor(poses[9], poses[0], loop_pose, np.eye(6) * 50)
    
    print(f"位姿节点数: {len(graph.pose_nodes)}")
    print(f"因子数: {len(graph.factors)}")
    print(f"  - 先验因子: {sum(1 for f in graph.factors if f['type'] == 'prior')}")
    print(f"  - 里程计因子: {sum(1 for f in graph.factors if f['type'] == 'odometry')}")
    print(f"  - 回环因子: {sum(1 for f in graph.factors if f['type'] == 'loop')}")
    print("\n注: 实际优化请使用GTSAM或g2o库")

# exercise_factor_graph()
```

### 练习4：传感器融合权重自适应

```python
def exercise_adaptive_fusion():
    """自适应融合权重练习"""
    print("=== 自适应融合权重练习 ===")
    
    # 模拟不同场景下的传感器质量
    scenarios = [
        {"name": "室内走廊", "gps_quality": 0.0, "visual_quality": 0.8, "lidar_quality": 0.9},
        {"name": "开阔室外", "gps_quality": 0.9, "visual_quality": 0.9, "lidar_quality": 0.9},
        {"name": "隧道", "gps_quality": 0.0, "visual_quality": 0.3, "lidar_quality": 0.6},
        {"name": "动态环境", "gps_quality": 0.7, "visual_quality": 0.4, "lidar_quality": 0.8},
        {"name": "夜间", "gps_quality": 0.8, "visual_quality": 0.2, "lidar_quality": 0.9},
    ]
    
    for scenario in scenarios:
        print(f"\n场景: {scenario['name']}")
        
        # 计算归一化权重
        total_quality = (scenario['gps_quality'] + 
                        scenario['visual_quality'] + 
                        scenario['lidar_quality'])
        
        if total_quality > 0:
            gps_weight = scenario['gps_quality'] / total_quality
            visual_weight = scenario['visual_quality'] / total_quality
            lidar_weight = scenario['lidar_quality'] / total_quality
        else:
            gps_weight = visual_weight = lidar_weight = 0.33
        
        print(f"  GPS权重: {gps_weight:.2f}")
        print(f"  视觉权重: {visual_weight:.2f}")
        print(f"  激光权重: {lidar_weight:.2f}")
        
        # 推荐策略
        if scenario['visual_quality'] < 0.5 and scenario['lidar_quality'] > 0.7:
            print(f"  推荐: 主要依赖激光+IMU，视觉辅助")
        elif scenario['gps_quality'] > 0.8:
            print(f"  推荐: GPS提供绝对约束，消除漂移")
        elif scenario['visual_quality'] > 0.7 and scenario['lidar_quality'] > 0.7:
            print(f"  推荐: 视觉+激光紧耦合，精度最高")
        else:
            print(f"  推荐: 保守估计，增大不确定性")

# exercise_adaptive_fusion()
```

---

**下一节**：[地图构建](05-map-building.md)

---

## 参考文献

1. Qin, T., Li, P., & Shen, S. (2018). VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator. IEEE Transactions on Robotics, 34(4), 1004-1020.

2. Mourikis, A. I., & Roumeliotis, S. I. (2007). A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation. IEEE International Conference on Robotics and Automation (ICRA).

3. Forster, C., Carlone, L., Dellaert, F., & Scaramuzza, D. (2017). IMU Preintegration on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation. Robotics: Science and Systems (RSS).

4. Shan, T., Englot, B., Meyers, D., et al. (2020). LIO-SAM: Tightly-coupled LIDAR Inertial Odometry via Smoothing and Mapping. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

5. Shan, T., Englot, B., Ratti, C., & Rus, D. (2021). LVI-SAM: Tightly-coupled Lidar-Visual-Inertial Odometry via Smoothing and Mapping. IEEE International Conference on Robotics and Automation (ICRA).

6. Lin, J., & Zhang, F. (2021). R2LIVE: A Robust, Real-time, LiDAR-Inertial-Visual tightly-coupled state Estimator and mapping. IEEE International Conference on Robotics and Automation (ICRA).

7. Sun, K., Mohta, K., Pfrommer, B., et al. (2018). Robust Stereo Visual Inertial Odometry for Fast Autonomous Flight. IEEE Robotics and Automation Letters.

8. Dellaert, F., & Kaess, M. (2006). Square Root SAM: Simultaneous Localization and Mapping via Square Root Information Smoothing. International Journal of Robotics Research.

9. Grisetti, G., Kümmerle, R., Stachniss, C., & Burgard, W. (2010). A Tutorial on Graph-based SLAM. IEEE Intelligent Transportation Systems Magazine.

10. Xu, W., Cai, Y., He, D., et al. (2023). FAST-LIVO: Fast LiDAR-Inertial-Visual Odometry. IEEE Transactions on Robotics.
