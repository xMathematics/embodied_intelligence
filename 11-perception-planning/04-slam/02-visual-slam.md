# 4.2 视觉SLAM

## 目录

- [1. 视觉SLAM简介](#1-视觉slam简介)
- [2. 特征法视觉SLAM](#2-特征法视觉slam)
- [3. 直接法视觉SLAM](#3-直接法视觉slam)
- [4. 经典视觉SLAM系统](#4-经典视觉slam系统)
- [5. 实践练习](#5-实践练习)

---

## 1. 视觉SLAM简介

### 1.1 什么是视觉SLAM

**视觉SLAM (Visual SLAM)** 使用相机作为主要传感器，从图像序列中同时估计相机运动和重建环境。

### 1.2 视觉SLAM分类

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

class VisualSLAMTypes:
    """视觉SLAM类型"""
    
    @staticmethod
    def feature_based():
        """特征法"""
        return {
            "name": "特征法",
            "methods": ["PTAM", "ORB-SLAM"],
            "pros": ["鲁棒性好", "计算效率高"],
            "cons": ["信息损失", "依赖特征"]
        }
    
    @staticmethod
    def direct():
        """直接法"""
        return {
            "name": "直接法",
            "methods": ["DTAM", "LSD-SLAM"],
            "pros": ["稠密信息", "不依赖特征"],
            "cons": ["光照敏感", "计算量大"]
        }
    
    @staticmethod
    def semi_direct():
        """半直接法"""
        return {
            "name": "半直接法",
            "methods": ["SVO"],
            "pros": ["两者优点"],
            "cons": ["复杂"]
        }
    
    @staticmethod
    def dense():
        """稠密法"""
        return {
            "name": "稠密法",
            "methods": ["ElasticFusion"],
            "pros": ["完整地图"],
            "cons": ["计算量巨大"]
        }
```

---

## 2. 特征法视觉SLAM

### 2.1 ORB特征提取和匹配

```python
class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, method='orb', num_features=2000):
        self.method = method
        
        if method == 'orb':
            self.detector = cv2.ORB_create(nfeatures=num_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif method == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=num_features)
            self.matcher = cv2.FlannBasedMatcher()
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def detect_and_compute(self, img):
        """检测特征并计算描述符"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        kp, des = self.detector.detectAndCompute(gray, None)
        return kp, des
    
    def match(self, des1, des2):
        """匹配描述符"""
        if self.method == 'sift':
            matches = self.matcher.knnMatch(des1, des2, k=2)
            # Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            return good_matches
        else:
            matches = self.matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches
```

### 2.2 本质矩阵和基础矩阵

```python
class EpipolarGeometry:
    """极线几何"""
    
    def __init__(self, K):
        self.K = K
    
    def find_essential_matrix(self, pts1, pts2):
        """计算本质矩阵"""
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, cv2.RANSAC, 0.999, 1.0
        )
        return E, mask
    
    def find_fundamental_matrix(self, pts1, pts2):
        """计算基础矩阵"""
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        return F, mask
    
    def recover_pose(self, E, pts1, pts2):
        """恢复位姿"""
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        return R, t, mask
    
    def triangulate(self, P1, P2, pts1, pts2):
        """三角化"""
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T
```

### 2.3 回环检测

```python
class LoopDetector:
    """回环检测器"""
    
    def __init__(self, method='dbow'):
        self.method = method
        self.keyframes = []
        self.database = None
    
    def add_keyframe(self, frame, descriptors):
        """添加关键帧到数据库"""
        self.keyframes.append({
            "frame": frame,
            "descriptors": descriptors,
            "timestamp": len(self.keyframes)
        })
    
    def detect_loop(self, descriptors, min_distance=10):
        """
        检测回环
        
        返回: 回环帧ID或None
        """
        if len(self.keyframes) < min_distance + 1:
            return None
        
        # 简化实现
        # 真实实现用DBoW等词袋模型
        return None
    
    def verify_loop(self, frame1, frame2):
        """验证回环"""
        return False
```

### 2.4 简化的ORB-SLAM流程

```python
class SimpleORBSLAM:
    """简化的ORB-SLAM"""
    
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
        self.epipolar = EpipolarGeometry(camera_matrix)
        self.feature_extractor = FeatureExtractor(method='orb')
        
        # 状态
        self.last_frame = None
        self.last_kp = None
        self.last_des = None
        self.current_pose = np.eye(4)
        self.trajectory = [self.current_pose.copy()]
        
        # 地图
        self.map_points = {}
        self.keyframes = []
    
    def process_frame(self, img):
        """处理一帧图像"""
        kp, des = self.feature_extractor.detect_and_compute(img)
        
        if self.last_frame is None:
            # 第一帧
            self.last_frame = img
            self.last_kp = kp
            self.last_des = des
            return self.current_pose
        
        # 匹配特征
        matches = self.feature_extractor.match(self.last_des, des)
        
        # 获取点
        pts1 = np.float32([self.last_kp[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
        
        # 计算本质矩阵
        E, mask_e = self.epipolar.find_essential_matrix(pts1, pts2)
        
        # 恢复位姿
        R, t, mask_p = self.epipolar.recover_pose(E, pts1, pts2)
        
        # 更新位姿
        delta_pose = np.eye(4)
        delta_pose[:3, :3] = R
        delta_pose[:3, 3] = t.flatten()
        
        self.current_pose = self.current_pose @ delta_pose
        self.trajectory.append(self.current_pose.copy())
        
        # 如果是关键帧
        if self._check_keyframe():
            self._add_keyframe(img, kp, des)
            # 三角化
            # 添加地图点
        
        # 更新
        self.last_frame = img
        self.last_kp = kp
        self.last_des = des
        
        return self.current_pose
    
    def _check_keyframe(self):
        """是否是关键帧"""
        # 简化的判断
        return len(self.keyframes) == 0 or len(self.trajectory) % 5 == 0
    
    def _add_keyframe(self, img, kp, des):
        """添加关键帧"""
        keyframe = {
            "pose": self.current_pose.copy(),
            "keypoints": kp,
            "descriptors": des,
            "frame": img
        }
        self.keyframes.append(keyframe)
```

---

## 3. 直接法视觉SLAM

### 3.1 直接法原理

```python
class DirectMethod:
    """直接法"""
    
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix
    
    def compute_photometric_error(self, img1, img2, point1, point2):
        """计算光度误差"""
        # 简化实现
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        x1, y1 = point1
        x2, y2 = point2
        
        if 0 <= int(x1) < img1.shape[1] and 0 <= int(y1) < img1.shape[0]:
            if 0 <= int(x2) < img2.shape[1] and 0 <= int(y2) < img2.shape[0]:
                error = float(gray1[int(y1), int(x1)]) - float(gray2[int(y2), int(x2)])
                return error
        
        return 0.0
    
    def optimize_pose_direct(self, img_ref, img_cur, points_3d, initial_pose):
        """
        直接法优化位姿
        
        这里简化，真实实现需要优化
        """
        # 直接法计算位姿
        # 这里只是占位
        return initial_pose
```

### 3.2 直接法分类

```python
class DirectMethodTypes:
    """直接法分类"""
    
    @staticmethod
    def sparse_direct():
        """稀疏直接法"""
        return {
            "name": "稀疏直接法",
            "examples": ["SVO"],
            "points": "稀疏特征点",
            "speed": "快"
        }
    
    @staticmethod
    def semi_dense_direct():
        """半稠密直接法"""
        return {
            "name": "半稠密直接法",
            "examples": ["LSD-SLAM"],
            "points": "梯度大的像素",
            "speed": "中"
        }
    
    @staticmethod
    def dense_direct():
        """稠密直接法"""
        return {
            "name": "稠密直接法",
            "examples": ["DTAM", "ElasticFusion"],
            "points": "所有像素",
            "speed": "慢"
        }
```

---

## 4. 经典视觉SLAM系统

### 4.1 PTAM

```python
class PTAM:
    """PTAM (Parallel Tracking and Mapping)"""
    
    def __init__(self):
        self.tracking_thread = None
        self.mapping_thread = None
        self.camera = None
        self.map = None
    
    def track(self, img):
        """跟踪线程"""
        # 1. 估计相机位姿
        # 2. 决策是否加关键帧
        pass
    
    def map(self):
        """建图线程"""
        # 1. 管理地图
        # 2. 优化地图
        pass
```

### 4.2 ORB-SLAM系统

```python
class ORBSLAM:
    """ORB-SLAM"""
    
    def __init__(self):
        # 三个线程
        self.tracking = None
        self.local_mapping = None
        self.loop_closing = None
        
        # 地图
        self.map = None
        
        # 关键帧数据库
        self.keyframe_database = None
    
    def initialize(self):
        """初始化"""
        pass
    
    def track(self, img):
        """跟踪"""
        pass
    
    def create_new_map(self, points_3d, poses):
        """创建新地图"""
        pass
    
    def local_bundle_adjustment(self):
        """局部BA"""
        pass
    
    def detect_loop_closure(self):
        """回环检测"""
        pass
    
    def optimize_pose_graph(self):
        """位姿图优化"""
        pass
```

### 4.3 LSD-SLAM

```python
class LSDSLAM:
    """LSD-SLAM (Large-Scale Direct Monocular SLAM)"""
    
    def __init__(self):
        self.current_frame = None
        self.last_frame = None
        self.keyframe_graph = None
        self.sim3_pose = None
    
    def track_direct(self, img):
        """直接法跟踪"""
        pass
    
    def estimate_depth(self):
        """深度估计"""
        pass
    
    def update_keyframe(self):
        """更新关键帧"""
        pass
    
    def pose_graph_optimization(self):
        """位姿图优化"""
        pass
```

---

## 5. 实践练习

### 练习1：特征匹配

```python
def exercise_feature_matching():
    """特征匹配练习"""
    print("=== 特征匹配练习 ===")
    
    # 读取两张示例图像
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 添加简单的移动
    M = np.float32([[1, 0, 50], [0, 1, 20]])
    img2_shifted = cv2.warpAffine(img2, M, (640, 480))
    
    # 提取和匹配
    extractor = FeatureExtractor('orb')
    kp1, des1 = extractor.detect_and_compute(img1)
    kp2, des2 = extractor.detect_and_compute(img2_shifted)
    
    matches = extractor.match(des1, des2)
    
    print(f"检测到 {len(kp1)} 和 {len(kp2)} 个特征")
    print(f"匹配到 {len(matches)} 对")
    
    # 绘制
    if len(matches) > 0:
        match_img = cv2.drawMatches(img1, kp1, img2_shifted, kp2, matches[:50], None)
        cv2.imwrite('feature_matches.png', match_img)
        print("匹配图已保存")

# exercise_feature_matching()
```

### 练习2：简单的VO

```python
def exercise_simple_vo():
    """简单的视觉里程计"""
    print("=== 简单视觉里程计练习 ===")
    
    # 相机参数
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])
    
    # 创建SLAM
    slam = SimpleORBSLAM(K)
    
    # 模拟图像序列
    num_frames = 20
    for i in range(num_frames):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 平移
        M = np.float32([[1, 0, i * 10], [0, 1, 0]])
        img_shifted = cv2.warpAffine(img, M, (640, 480))
        
        pose = slam.process_frame(img_shifted)
        print(f"帧 {i}, 位置: {pose[:3, 3]}")
    
    # 绘制轨迹
    trajectory = np.array(slam.trajectory)
    positions = trajectory[:, :3, 3]
    
    plt.figure(figsize=(10, 10))
    plt.plot(positions[:, 0], positions[:, 1], 'o-')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('VO轨迹')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('vo_trajectory.png')
    print("轨迹已保存")

# exercise_simple_vo()
```

### 练习3：视觉SLAM对比

```python
def exercise_visual_slam_comparison():
    """视觉SLAM系统对比"""
    print("=== 视觉SLAM系统对比 ===\n")
    
    systems = [
        {
            "name": "PTAM",
            "year": 2007,
            "type": "特征法",
            "features": "FAST",
            "threads": 2,
            "pros": ["第一个实时", "并行跟踪建图"],
            "cons": ["小场景", "无回环检测"]
        },
        {
            "name": "ORB-SLAM",
            "year": 2015,
            "type": "特征法",
            "features": "ORB",
            "threads": 3,
            "pros": ["完整系统", "回环检测"],
            "cons": ["大场景慢"]
        },
        {
            "name": "LSD-SLAM",
            "year": 2014,
            "type": "直接法",
            "features": "半稠密",
            "threads": 2,
            "pros": ["半稠密", "直接法"],
            "cons": ["尺度飘移"]
        },
        {
            "name": "SVO",
            "year": 2014,
            "type": "半直接",
            "features": "FAST",
            "threads": 2,
            "pros": ["速度快", "精度高"],
            "cons": ["无回环检测"]
        }
    ]
    
    for s in systems:
        print(f"{s['name']} ({s['year']})")
        print(f"  类型: {s['type']}")
        print(f"  特征: {s['features']}")
        print(f"  线程: {s['threads']}")
        print(f"  优点: {', '.join(s['pros'])}")
        print(f"  缺点: {', '.join(s['cons'])}")
        print()

# exercise_visual_slam_comparison()
```

---

**下一节**：[激光SLAM](03-lidar-slam.md)

---

## 参考文献

1. Klein, G., & Murray, D. (2007). Parallel Tracking and Mapping for Small AR Workspaces.
2. Mur-Artal, R., et al. (2015). ORB-SLAM: A Versatile and Accurate Monocular SLAM System.
3. Engel, J., et al. (2014). LSD-SLAM: Large-Scale Direct Monocular SLAM.
4. Forster, C., et al. (2014). SVO: Fast Semi-Direct Monocular Visual Odometry.
5. Newcombe, R. A., et al. (2011). DTAM: Dense Tracking and Mapping in Real-Time.
