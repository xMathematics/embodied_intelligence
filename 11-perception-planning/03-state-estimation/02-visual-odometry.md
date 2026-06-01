# 3.2 视觉里程计

## 目录

- [1. 引言](#1-引言)
- [2. 视觉里程计概述](#2-视觉里程计概述)
- [3. 特征点检测与匹配](#3-特征点检测与匹配)
- [4. 直接法视觉里程计](#4-直接法视觉里程计)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 视觉里程计的重要性

**视觉里程计 (Visual Odometry, VO)** 仅使用相机估计相机的运动轨迹，是SLAM和机器人自主导航的核心技术之一，无需额外传感器。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **无人机导航** | 无GPS环境定位 | 室内无人机飞行 |
| **AR/VR** | 设备姿态追踪 | 沉浸式体验 |
| **自主机器人** | 视觉定位 | 扫地机器人、服务机器人 |
| **自动驾驶** | 车辆位姿估计 | 多传感器融合定位 |

---

## 2. 视觉里程计概述

### 2.1 视觉里程计分类

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

class VisualOdometryBase:
    """视觉里程计基类"""
    
    def __init__(self, camera_matrix):
        """
        初始化VO
        
        参数:
            camera_matrix: 相机内参矩阵
        """
        self.camera_matrix = camera_matrix
        self.poses = []  # 位姿列表
        self.prev_frame = None
        self.prev_kp = None
        self.prev_des = None
    
    def process_frame(self, frame):
        """
        处理一帧图像
        
        参数:
            frame: 输入图像
        
        返回:
            pose: 当前估计的位姿
        """
        pass
    
    def get_trajectory(self):
        """获取轨迹"""
        return np.array(self.poses)
```

### 2.2 坐标变换

```python
class Pose:
    """位姿表示"""
    
    def __init__(self, R=None, t=None):
        """
        初始化位姿
        
        参数:
            R: 旋转矩阵 (3x3)
            t: 平移向量 (3)
        """
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)
        
        self.R = R
        self.t = t
    
    def to_matrix(self):
        """转换为4x4变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T
    
    def inverse(self):
        """求逆"""
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Pose(R_inv, t_inv)
    
    def compose(self, other):
        """位姿组合"""
        R_new = self.R @ other.R
        t_new = self.R @ other.t + self.t
        return Pose(R_new, t_new)
```

---

## 3. 特征点检测与匹配

### 3.1 特征点检测

```python
class FeatureDetector:
    """特征检测器"""
    
    def __init__(self, detector_type='orb'):
        """
        初始化特征检测器
        
        参数:
            detector_type: 'orb', 'sift', 'surf', 'akaze'
        """
        self.detector_type = detector_type
        
        if detector_type == 'orb':
            self.detector = cv2.ORB_create(nfeatures=1000)
        elif detector_type == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=1000)
        elif detector_type == 'akaze':
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"未知检测器类型: {detector_type}")
    
    def detect_and_compute(self, img):
        """
        检测特征点并计算描述符
        
        参数:
            img: 输入图像
        
        返回:
            kp: 特征点
            des: 描述符
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        kp, des = self.detector.detectAndCompute(gray, None)
        return kp, des
    
    def visualize_features(self, img, kp):
        """可视化特征点"""
        vis = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))
        return vis
```

### 3.2 特征点匹配

```python
class FeatureMatcher:
    """特征匹配器"""
    
    def __init__(self, matcher_type='bf', norm_type=None):
        """
        初始化匹配器
        
        参数:
            matcher_type: 'bf' (暴力匹配) 或 'flann' (快速最近邻)
            norm_type: 距离度量
        """
        self.matcher_type = matcher_type
        
        if matcher_type == 'bf':
            if norm_type is None:
                norm_type = cv2.NORM_HAMMING
            self.matcher = cv2.BFMatcher(norm_type, crossCheck=True)
        elif matcher_type == 'flann':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    def match(self, des1, des2):
        """
        匹配特征描述符
        
        参数:
            des1, des2: 描述符
        
        返回:
            matches: 匹配对
        """
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def match_knn(self, des1, des2, k=2):
        """KNN匹配"""
        matches = self.matcher.knnMatch(des1, des2, k=k)
        return matches
    
    def apply_ratio_test(self, matches, ratio=0.75):
        """应用比率测试"""
        good_matches = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        return good_matches
    
    def visualize_matches(self, img1, kp1, img2, kp2, matches):
        """可视化匹配"""
        vis = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return vis
```

### 3.3 基于特征的VO

```python
class FeatureBasedVO(VisualOdometryBase):
    """基于特征的视觉里程计"""
    
    def __init__(self, camera_matrix, detector_type='orb'):
        super().__init__(camera_matrix)
        
        self.feature_detector = FeatureDetector(detector_type)
        self.feature_matcher = FeatureMatcher()
        
        self.pts3d = None  # 3D点
        self.pose = Pose()  # 当前位姿
        self.poses.append(self.pose.to_matrix())
    
    def process_frame(self, frame):
        """处理一帧图像"""
        # 检测特征
        kp, des = self.feature_detector.detect_and_compute(frame)
        
        if self.prev_frame is None:
            # 第一帧
            self.prev_frame = frame
            self.prev_kp = kp
            self.prev_des = des
            return self.pose.to_matrix()
        
        # 匹配特征
        matches = self.feature_matcher.match(self.prev_des, des)
        
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
        
        # 组合位姿
        T_prev_to_curr = Pose(R, t.flatten())
        self.pose = self.pose.compose(T_prev_to_curr)
        self.poses.append(self.pose.to_matrix())
        
        # 更新
        self.prev_frame = frame
        self.prev_kp = kp
        self.prev_des = des
        
        return self.pose.to_matrix()
    
    def triangulate(self, pts1, pts2, P1, P2):
        """三角化恢复3D点"""
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = pts4d[:3] / pts4d[3]
        return pts3d.T
```

---

## 4. 直接法视觉里程计

### 4.1 光流跟踪

```python
class OpticalFlowTracker:
    """光流跟踪器"""
    
    def __init__(self, win_size=(15, 15), max_level=3):
        """
        初始化光流跟踪器
        
        参数:
            win_size: 窗口大小
            max_level: 金字塔层数
        """
        self.win_size = win_size
        self.max_level = max_level
        
        # 角点检测参数
        self.feature_params = dict(
            maxCorners=1000,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # LK光流参数
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
    
    def detect(self, img):
        """检测特征点"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        return pts
    
    def track(self, img1, img2, pts1):
        """
        跟踪特征点
        
        参数:
            img1: 上一帧图像
            img2: 当前帧图像
            pts1: 上一帧特征点
        
        返回:
            pts2: 当前帧特征点
            status: 跟踪状态
            err: 跟踪误差
        """
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1
            gray2 = img2
        
        pts2, status, err = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, pts1, None, **self.lk_params
        )
        
        return pts2, status, err
    
    def visualize_track(self, img1, img2, pts1, pts2, status):
        """可视化光流跟踪"""
        vis = np.hstack((img1, img2))
        
        good_pts1 = pts1[status == 1]
        good_pts2 = pts2[status == 1]
        
        for i, (p1, p2) in enumerate(zip(good_pts1, good_pts2)):
            x1, y1 = p1.ravel()
            x2, y2 = p2.ravel()
            x2_shifted = x2 + img1.shape[1]
            
            cv2.circle(vis, (int(x1), int(y1)), 3, (0, 255, 0), -1)
            cv2.circle(vis, (int(x2_shifted), int(y2)), 3, (0, 0, 255), -1)
            cv2.line(vis, (int(x1), int(y1)), (int(x2_shifted), int(y2)), (0, 255, 255), 1)
        
        return vis
```

### 4.2 直接法VO

```python
class DirectVO(VisualOdometryBase):
    """直接法视觉里程计"""
    
    def __init__(self, camera_matrix):
        super().__init__(camera_matrix)
        
        self.tracker = OpticalFlowTracker()
        self.prev_pts = None
        self.pose = Pose()
        self.poses.append(self.pose.to_matrix())
    
    def process_frame(self, frame):
        """处理一帧图像"""
        if self.prev_frame is None:
            self.prev_frame = frame
            self.prev_pts = self.tracker.detect(frame)
            return self.pose.to_matrix()
        
        # 光流跟踪
        pts, status, err = self.tracker.track(self.prev_frame, frame, self.prev_pts)
        
        # 筛选好的跟踪点
        good_pts1 = self.prev_pts[status == 1].reshape(-1, 1, 2)
        good_pts2 = pts[status == 1].reshape(-1, 1, 2)
        
        if len(good_pts1) > 20:
            # 计算本质矩阵
            E, mask = cv2.findEssentialMat(
                good_pts1, good_pts2, self.camera_matrix,
                cv2.RANSAC, 0.999, 1.0
            )
            
            if E is not None:
                # 恢复位姿
                _, R, t, mask_pose = cv2.recoverPose(E, good_pts1, good_pts2, self.camera_matrix)
                
                # 组合位姿
                T_prev_to_curr = Pose(R, t.flatten())
                self.pose = self.pose.compose(T_prev_to_curr)
        
        self.poses.append(self.pose.to_matrix())
        
        # 更新
        self.prev_frame = frame
        self.prev_pts = good_pts2
        
        return self.pose.to_matrix()
```

---

## 5. 实践练习

### 练习1：特征检测与匹配

```python
def exercise_feature_detection():
    """特征检测与匹配练习"""
    print("=== 特征检测与匹配练习 ===")
    
    # 创建模拟图像（实际应用中使用真实图像）
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 检测特征
    detector = FeatureDetector('orb')
    kp1, des1 = detector.detect_and_compute(img1)
    kp2, des2 = detector.detect_and_compute(img2)
    
    print(f"图像1检测到 {len(kp1)} 个特征点")
    print(f"图像2检测到 {len(kp2)} 个特征点")
    
    # 匹配
    matcher = FeatureMatcher('bf')
    matches = matcher.match(des1, des2)
    
    print(f"匹配到 {len(matches)} 对特征")
    
    # 可视化
    if des1 is not None and des2 is not None and len(matches) > 0:
        vis = matcher.visualize_matches(img1, kp1, img2, kp2, matches[:50])
        cv2.imwrite('feature_matches.png', vis)
        print("匹配可视化已保存")

# exercise_feature_detection()
```

### 练习2：光流跟踪

```python
def exercise_optical_flow():
    """光流跟踪练习"""
    print("=== 光流跟踪练习 ===")
    
    # 创建模拟图像序列
    num_frames = 10
    frames = []
    for i in range(num_frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # 添加平移
        M = np.float32([[1, 0, i*2], [0, 1, 0]])
        frame_shifted = cv2.warpAffine(frame, M, (640, 480))
        frames.append(frame_shifted)
    
    # 光流跟踪
    tracker = OpticalFlowTracker()
    prev_frame = frames[0]
    prev_pts = tracker.detect(prev_frame)
    
    print(f"初始检测 {len(prev_pts)} 个特征点")
    
    for i in range(1, num_frames):
        curr_frame = frames[i]
        curr_pts, status, err = tracker.track(prev_frame, curr_frame, prev_pts)
        
        num_good = np.sum(status)
        print(f"帧 {i}: 跟踪到 {num_good} 个点")
        
        if num_good > 0:
            vis = tracker.visualize_track(prev_frame, curr_frame, prev_pts, curr_pts, status)
            cv2.imwrite(f'flow_track_{i}.png', vis)
        
        prev_frame = curr_frame
        prev_pts = curr_pts[status == 1].reshape(-1, 1, 2)
        
        # 补充新特征
        if len(prev_pts) < 50:
            new_pts = tracker.detect(prev_frame)
            if new_pts is not None:
                prev_pts = np.vstack((prev_pts, new_pts))[:50]

# exercise_optical_flow()
```

### 练习3：轨迹可视化

```python
def exercise_trajectory_visualization():
    """轨迹可视化练习"""
    print("=== 轨迹可视化练习 ===")
    
    # 模拟相机内参
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])
    
    # 创建模拟图像序列
    num_frames = 50
    frames = []
    for i in range(num_frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        theta = i * 0.05
        dx = 5 * np.sin(theta)
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        frame_shifted = cv2.warpAffine(frame, M, (640, 480))
        frames.append(frame_shifted)
    
    # 运行直接法VO
    vo = DirectVO(K)
    
    for frame in frames:
        vo.process_frame(frame)
    
    # 获取轨迹
    trajectory = vo.get_trajectory()
    positions = trajectory[:, :3, 3]
    
    print(f"估计了 {len(positions)} 个位置")
    
    # 绘制轨迹
    plt.figure(figsize=(8, 8))
    plt.plot(positions[:, 0], positions[:, 1], 'b-', label='估计轨迹')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('视觉里程计轨迹')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('vo_trajectory.png')
    print("轨迹图已保存")

# exercise_trajectory_visualization()
```

---

**下一节**：[位姿估计](03-pose-estimation.md)

---

## 参考文献

1. Nistér, D., et al. (2004). Visual Odometry.
2. Hartley, R., & Zisserman, A. (2004). Multiple View Geometry in Computer Vision.
3. Forster, C., et al. (2014). SVO: Fast Semi-Direct Monocular Visual Odometry.
4. Engel, J., et al. (2014). LSD-SLAM: Large-Scale Direct Monocular SLAM.
