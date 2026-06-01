# 4.1 SLAM概述

## 目录

- [1. SLAM简介](#1-slam简介)
- [2. SLAM问题建模](#2-slam问题建模)
- [3. SLAM系统架构](#3-slam系统架构)
- [4. 主要SLAM方法分类](#4-主要slam方法分类)
- [5. 实践练习](#5-实践练习)

---

## 1. SLAM简介

### 1.1 SLAM的定义

**SLAM (Simultaneous Localization and Mapping)**，即同时定位与地图构建，是指机器人在未知环境中，同时：

1. **定位 (Localization)**：确定自身在环境中的位姿
2. **建图 (Mapping)**：构建环境的地图

### 1.2 SLAM的发展历程

```python
import matplotlib.pyplot as plt
import numpy as np

class SLAMHistory:
    """SLAM发展历程"""
    
    def __init__(self):
        self.milestones = [
            (1986, "Kalman Filter SLAM", "使用EKF的早期SLAM"),
            (1995, "FastSLAM", "粒子滤波SLAM"),
            (2002, "EKF-SLAM", "扩展卡尔曼滤波"),
            (2006, "PTAM", "实时单目SLAM"),
            (2011, "RGBD-SLAM", "RGB-D相机SLAM"),
            (2015, "ORB-SLAM", "特征法SLAM"),
            (2016, "LSD-SLAM", "直接法SLAM"),
            (2017, "LOAM", "激光SLAM"),
            (2018, "VINS-Mono", "视觉惯性SLAM"),
            (2019, "LIO-SAM", "激光惯性SLAM"),
        ]
    
    def plot_timeline(self):
        """绘制时间线"""
        years = [y for y, _, _ in self.milestones]
        labels = [name for _, name, _ in self.milestones]
        
        plt.figure(figsize=(12, 6))
        for i, (year, name, desc) in enumerate(self.milestones):
            plt.plot(year, i, 'o', markersize=10)
            plt.text(year + 0.5, i, name, va='center')
        
        plt.yticks([])
        plt.xlabel("年份")
        plt.title("SLAM发展历程")
        plt.grid(True, alpha=0.3)
        plt.savefig("slam_timeline.png")
```

---

## 2. SLAM问题建模

### 2.1 概率SLAM

```python
class ProbabilisticSLAM:
    """概率SLAM"""
    
    def __init__(self):
        pass
    
    def motion_model(self, x_prev, u):
        """
        运动模型
        
        x_prev: 上一时刻状态
        u: 控制输入
        """
        pass
    
    def measurement_model(self, x, z):
        """
        观测模型
        
        x: 当前状态
        z: 观测
        """
        pass
    
    def bayes_filter(self, bel_prev, u, z):
        """
        贝叶斯滤波
        
        bel_prev: 上一时刻置信度
        u: 控制
        z: 观测
        """
        # 预测
        bel_pred = self.predict(bel_prev, u)
        
        # 更新
        bel = self.update(bel_pred, z)
        
        return bel
    
    def predict(self, bel, u):
        """预测步"""
        pass
    
    def update(self, bel, z):
        """更新步"""
        pass
```

### 2.2 图优化SLAM

```python
import numpy as np

class PoseGraphNode:
    """位姿图节点"""
    
    def __init__(self, idx, pose):
        self.idx = idx
        self.pose = pose.copy()


class PoseGraphEdge:
    """位姿图边"""
    
    def __init__(self, id1, id2, relative_pose, information):
        self.id1 = id1
        self.id2 = id2
        self.relative_pose = relative_pose.copy()
        self.information = information.copy()


class PoseGraphSLAM:
    """图优化SLAM"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.next_idx = 0
    
    def add_node(self, pose):
        """添加节点"""
        node = PoseGraphNode(self.next_idx, pose)
        self.nodes[self.next_idx] = node
        self.next_idx += 1
        return node
    
    def add_edge(self, id1, id2, relative_pose, information=None):
        """添加边"""
        if information is None:
            information = np.eye(6)
        
        edge = PoseGraphEdge(id1, id2, relative_pose, information)
        self.edges.append(edge)
        return edge
    
    def optimize(self, num_iterations=10):
        """
        优化位姿图
        
        这里简化，实际用g2o等库
        """
        print(f"位姿图优化 {num_iterations} 次")
        # 真实的实现使用g2o或gtsam
        pass
    
    def compute_error(self):
        """计算总误差"""
        total_error = 0
        
        for edge in self.edges:
            pose1 = self.nodes[edge.id1].pose
            pose2 = self.nodes[edge.id2].pose
            
            # 计算相对变换误差
            # 这里简化
            pass
        
        return total_error
```

---

## 3. SLAM系统架构

### 3.1 前端-后端架构

```python
class SLAMFrontend:
    """SLAM前端"""
    
    def __init__(self):
        self.current_frame = None
        self.last_frame = None
        self.keyframes = []
    
    def process_frame(self, data):
        """
        处理一帧数据
        
        data: 图像/点云等
        """
        pass
    
    def track(self, current_data, last_data):
        """跟踪"""
        pass
    
    def create_keyframe(self, frame):
        """创建关键帧"""
        pass
    
    def detect_loop_closure(self, frame):
        """检测回环"""
        pass


class SLAMBackend:
    """SLAM后端"""
    
    def __init__(self):
        self.pose_graph = PoseGraphSLAM()
        self.keyframe_database = []
    
    def add_keyframe(self, keyframe):
        """添加关键帧到后端"""
        pass
    
    def optimize(self):
        """执行优化"""
        pass
    
    def update_from_loop_closure(self, loop_edge):
        """从回环更新"""
        pass
```

### 3.2 经典SLAM系统架构

```python
class ClassicSLAMSystem:
    """经典SLAM系统"""
    
    def __init__(self):
        self.frontend = SLAMFrontend()
        self.backend = SLAMBackend()
        self.map = None
        
        # 传感器
        self.camera = None
        self.lidar = None
        self.imu = None
    
    def initialize(self, config):
        """初始化"""
        pass
    
    def process_data(self, sensor_data):
        """处理传感器数据"""
        # 1. 前端处理
        keyframe = self.frontend.process_frame(sensor_data)
        
        # 2. 如果是关键帧
        if keyframe:
            # 添加到后端
            self.backend.add_keyframe(keyframe)
            
            # 检测回环
            loop_edge = self.frontend.detect_loop_closure(keyframe)
            
            if loop_edge:
                self.backend.update_from_loop_closure(loop_edge)
            
            # 后端优化
            self.backend.optimize()
        
        # 3. 返回估计的位姿
        return self.get_current_pose()
    
    def get_current_pose(self):
        """获取当前位姿"""
        pass
    
    def get_map(self):
        """获取地图"""
        return self.map
```

---

## 4. 主要SLAM方法分类

### 4.1 按传感器类型分类

```python
class SLAMType:
    """SLAM类型"""
    
    MONOCULAR = "monocular"
    STEREO = "stereo"
    RGBD = "rgbd"
    LIDAR = "lidar"
    IMU = "imu"
    FUSED = "fused"


class SLAMBySensor:
    """按传感器的SLAM分类"""
    
    @staticmethod
    def monocular_slam():
        """单目SLAM"""
        return {
            "name": "单目SLAM",
            "pros": ["成本低", "硬件简单"],
            "cons": ["尺度不确定性", "纯旋转问题"],
            "examples": ["PTAM", "ORB-SLAM", "LSD-SLAM"]
        }
    
    @staticmethod
    def stereo_slam():
        """双目SLAM"""
        return {
            "name": "双目SLAM",
            "pros": ["有尺度", "可稀疏"],
            "cons": ["硬件复杂", "基线约束"],
            "examples": ["ORB-SLAM2 (stereo)", "SVO"]
        }
    
    @staticmethod
    def rgbd_slam():
        """RGB-D SLAM"""
        return {
            "name": "RGB-D SLAM",
            "pros": ["直接深度", "稠密地图"],
            "cons": ["范围有限", "室外差"],
            "examples": ["RGBD-SLAM", "ElasticFusion"]
        }
    
    @staticmethod
    def lidar_slam():
        """激光SLAM"""
        return {
            "name": "激光SLAM",
            "pros": ["精度高", "不受光照"],
            "cons": ["成本高", "无纹理信息"],
            "examples": ["LOAM", "Cartographer", "LIO-SAM"]
        }
    
    @staticmethod
    def visual_inertial_slam():
        """视觉惯性SLAM"""
        return {
            "name": "视觉惯性SLAM",
            "pros": ["互补", "鲁棒"],
            "cons": ["复杂", "标定难"],
            "examples": ["VINS-Mono", "MSCKF", "ROVIO"]
        }
```

### 4.2 按方法分类

```python
class SLAMByMethod:
    """按方法的SLAM分类"""
    
    @staticmethod
    def ekf_slam():
        """EKF-SLAM"""
        return {
            "name": "EKF-SLAM",
            "type": "滤波",
            "description": "扩展卡尔曼滤波",
            "pros": ["简单", "经典"],
            "cons": ["计算量大", "线性化误差"]
        }
    
    @staticmethod
    def graph_slam():
        """图优化SLAM"""
        return {
            "name": "图优化SLAM",
            "type": "优化",
            "description": "基于图的优化",
            "pros": ["精度高", "可回环"],
            "cons": ["计算量大", "需要初始解"]
        }
    
    @staticmethod
    def particle_slam():
        """FastSLAM"""
        return {
            "name": "FastSLAM",
            "type": "粒子滤波",
            "description": "粒子滤波",
            "pros": ["非线性好"],
            "cons": ["粒子退化", "计算量大"]
        }
    
    @staticmethod
    def feature_based_slam():
        """特征法SLAM"""
        return {
            "name": "特征法SLAM",
            "type": "特征",
            "description": "基于特征点",
            "pros": ["高效", "鲁棒"],
            "cons": ["依赖特征", "信息少"]
        }
    
    @staticmethod
    def direct_slam():
        """直接法SLAM"""
        return {
            "name": "直接法SLAM",
            "type": "直接",
            "description": "直接用像素",
            "pros": ["信息多", "稠密"],
            "cons": ["对光照敏感"]
        }
```

---

## 5. 实践练习

### 练习1：简单的位姿图

```python
def exercise_pose_graph_slam():
    """位姿图SLAM练习"""
    print("=== 位姿图SLAM练习 ===")
    
    # 创建位姿图
    slam = PoseGraphSLAM()
    
    # 添加节点（模拟轨迹
    num_nodes = 10
    for i in range(num_nodes):
        # 圆形轨迹
        theta = i * 0.2
        x = np.cos(theta) * 5
        y = np.sin(theta) * 5
        
        pose = np.eye(4)
        pose[0, 3] = x
        pose[1, 3] = y
        
        # 添加噪声
        pose[:3, 3] += np.random.randn(3) * 0.1
        
        slam.add_node(pose)
    
    # 添加顺序边
    for i in range(num_nodes - 1):
        # 相对变换
        pose_i = slam.nodes[i].pose
        pose_j = slam.nodes[i + 1].pose
        
        # T_i^j = (T_i)^{-1} @ T_j
        T_i_inv = np.eye(4)
        T_i_inv[:3, :3] = pose_i[:3, :3].T
        T_i_inv[:3, 3] = -pose_i[:3, :3].T @ pose_i[:3, 3]
        
        relative_pose = T_i_inv @ pose_j
        
        # 添加边
        info = np.eye(6) * 10
        slam.add_edge(i, i + 1, relative_pose, info)
    
    # 添加回环边
    if num_nodes > 3:
        pose_first = slam.nodes[0].pose
        pose_last = slam.nodes[num_nodes - 1].pose
        
        T_first_inv = np.eye(4)
        T_first_inv[:3, :3] = pose_first[:3, :3].T
        T_first_inv[:3, 3] = -pose_first[:3, :3].T @ pose_first[:3, 3]
        
        loop_pose = T_first_inv @ pose_last
        
        info = np.eye(6) * 100
        slam.add_edge(0, num_nodes - 1, loop_pose, info)
    
    print(f"节点数: {len(slam.nodes)}")
    print(f"边数: {len(slam.edges)}")
    
    # 获取节点位置
    positions = np.array([node.pose[:3, 3] for node in slam.nodes.values()])
    
    # 绘制
    plt.figure(figsize=(10, 10))
    plt.plot(positions[:, 0], positions[:, 1], 'o-', label='轨迹')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('位姿图SLAM示例')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('pose_graph_slam.png')
    print("位姿图已保存")

# exercise_pose_graph_slam()
```

### 练习2：SLAM分类对比

```python
def exercise_slam_comparison():
    """SLAM方法对比练习"""
    print("=== SLAM方法对比 ===")
    
    # 定义各种方法
    methods = [
        SLAMByMethod.ekf_slam(),
        SLAMByMethod.graph_slam(),
        SLAMByMethod.particle_slam(),
        SLAMByMethod.feature_based_slam(),
        SLAMByMethod.direct_slam(),
    ]
    
    # 打印对比
    print(f"{'方法':<20} {'类型':<10} {'主要特点'}")
    print("-" * 70)
    
    for m in methods:
        print(f"{m['name']:<20} {m['type']:<10} {m['description']}")
    
    print("\n传感器类型:")
    sensors = [
        SLAMBySensor.monocular_slam(),
        SLAMBySensor.stereo_slam(),
        SLAMBySensor.rgbd_slam(),
        SLAMBySensor.lidar_slam(),
        SLAMBySensor.visual_inertial_slam(),
    ]
    
    for s in sensors:
        print(f"\n{s['name']}:")
        print(f"  优点: {', '.join(s['pros'])}")
        print(f"  缺点: {', '.join(s['cons'])}")
        print(f"  例子: {', '.join(s['examples'])}")

# exercise_slam_comparison()
```

---

**下一节**：[视觉SLAM](02-visual-slam.md)

---

## 参考文献

1. Durrant-Whyte, H., & Bailey, T. (2006). Simultaneous Localization and Mapping: Part I.
2. Bailey, T., & Durrant-Whyte, H. (2006). Simultaneous Localization and Mapping (SLAM): Part II.
3. Grisetti, G., et al. (2010). A Tutorial on Graph-Based SLAM.
4. Cadena, C., et al. (2016). Past, Present, and Future of Simultaneous Localization and Mapping: Toward the Robust-Perceptual Age.
