# 6.1 运动规划基础

## 目录

- [1. 引言](#1-引言)
- [2. 构型空间](#2-构型空间)
- [3. 位形空间表示](#3-位形空间表示)
- [4. 运动规划问题定义](#4-运动规划问题定义)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 运动规划与路径规划的区别

| 特性 | 路径规划 | 运动规划 |
|------|---------|---------|
| 考虑因素 | 几何约束 | 几何 + 动力学约束 |
| 输出 | 几何路径 | 时间参数化轨迹 |
| 环境维度 | 通常2D/3D | 高维构型空间 |
| 机器人类型 | 移动机器人 | 机械臂、无人机等 |

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MotionPlanning:
    """运动规划基础"""
    
    def __init__(self, dim=3):
        self.dim = dim
```

---

## 2. 构型空间

### 2.1 构型空间 (C-Space)

```python
class CSpace:
    """构型空间"""
    
    def __init__(self, bounds):
        """
        bounds: [(min1, max1), (min2, max2), ...]
        """
        self.bounds = bounds
        self.dim = len(bounds)
    
    def in_bounds(self, q):
        """检查是否在边界内"""
        for i in range(self.dim):
            if not (self.bounds[i][0] <= q[i] <= self.bounds[i][1]):
                return False
        return True
    
    def sample(self):
        """随机采样"""
        q = []
        for (min_val, max_val) in self.bounds:
            q.append(np.random.uniform(min_val, max_val))
        return np.array(q)
```

### 2.2 障碍物空间 (C-Obstacle)

```python
class SphereObstacle:
    """球形障碍物"""
    
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
    
    def is_in_collision(self, point, robot_radius=0.1):
        """检查碰撞"""
        dist = np.linalg.norm(point - self.center)
        return dist < (self.radius + robot_radius)
    
    def distance(self, point):
        """计算距离"""
        dist = np.linalg.norm(point - self.center)
        return max(0, dist - self.radius)


class BoxObstacle:
    """长方体障碍物"""
    
    def __init__(self, center, size):
        self.center = np.array(center)
        self.size = np.array(size)
    
    def is_in_collision(self, point, robot_radius=0.1):
        """简化的碰撞检查"""
        dx = abs(point[0] - self.center[0])
        dy = abs(point[1] - self.center[1])
        dz = abs(point[2] - self.center[2]) if len(point) > 2 else 0
        
        if len(point) > 2:
            return (dx < self.size[0]/2 + robot_radius and 
                    dy < self.size[1]/2 + robot_radius and
                    dz < self.size[2]/2 + robot_radius)
        else:
            return (dx < self.size[0]/2 + robot_radius and 
                    dy < self.size[1]/2 + robot_radius)
```

---

## 3. 位形空间表示

### 3.1 刚体变换

```python
class RigidBody:
    """刚体"""
    
    @staticmethod
    def rotation_matrix_2d(theta):
        """2D旋转矩阵"""
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
    
    @staticmethod
    def rotation_matrix_3d(roll, pitch, yaw):
        """3D旋转矩阵 (ZYX欧拉角)"""
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx
    
    @staticmethod
    def transform_2d(point, pose):
        """2D变换"""
        x, y, theta = pose
        R = RigidBody.rotation_matrix_2d(theta)
        return R @ point + np.array([x, y])
    
    @staticmethod
    def transform_3d(point, pose):
        """3D变换"""
        x, y, z, roll, pitch, yaw = pose
        R = RigidBody.rotation_matrix_3d(roll, pitch, yaw)
        return R @ point + np.array([x, y, z])
```

### 3.2 机器人模型

```python
class SimpleRobot:
    """简单机器人模型"""
    
    def __init__(self, size=0.3):
        self.size = size
    
    def get_footprint(self, q):
        """获取在构型q下的足迹"""
        if len(q) == 2:  # 平面点机器人
            return [q.copy()]
        elif len(q) == 3:  # 平面圆形机器人 (x, y, theta)
            # 返回机器人边界
            x, y, theta = q
            points = []
            for a in np.linspace(0, 2*np.pi, 8):
                px = x + self.size * np.cos(a + theta)
                py = y + self.size * np.sin(a + theta)
                points.append(np.array([px, py]))
            return points
        return [q.copy()]
    
    def check_collision(self, q, obstacles):
        """检查在构型q下是否碰撞"""
        footprint = self.get_footprint(q)
        for point in footprint:
            for obs in obstacles:
                if obs.is_in_collision(point, 0):
                    return True
        return False
```

---

## 4. 运动规划问题定义

### 4.1 运动规划问题

```python
class MotionPlanningProblem:
    """运动规划问题定义"""
    
    def __init__(self, cspace, robot, obstacles):
        self.cspace = cspace
        self.robot = robot
        self.obstacles = obstacles
    
    def is_valid(self, q):
        """检查构型是否有效"""
        if not self.cspace.in_bounds(q):
            return False
        if self.robot.check_collision(q, self.obstacles):
            return False
        return True
    
    def is_edge_valid(self, q1, q2, num_checks=10):
        """检查边是否有效"""
        for alpha in np.linspace(0, 1, num_checks):
            q = q1 + alpha * (q2 - q1)
            if not self.is_valid(q):
                return False
        return True
    
    def distance(self, q1, q2):
        """构型空间距离"""
        return np.linalg.norm(q1 - q2)
```

---

## 5. 实践练习

### 练习1：C-Space可视化

```python
def exercise_cspace_visualization():
    """构型空间可视化"""
    print("=== C-Space 可视化 ===")
    
    # 定义C-Space
    cspace = CSpace(bounds=[(-2, 6), (-2, 6)])
    
    # 障碍物
    obstacles = [
        BoxObstacle(center=(1, 1), size=(2, 2)),
        BoxObstacle(center=(4, 3), size=(2, 2)),
    ]
    
    # 可视化C-Space
    xx, yy = np.meshgrid(np.linspace(-2, 6, 100), np.linspace(-2, 6, 100))
    collision_grid = np.zeros_like(xx)
    
    robot = SimpleRobot(size=0.2)
    
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            q = np.array([xx[i, j], yy[i, j], 0.0])
            if robot.check_collision(q, obstacles):
                collision_grid[i, j] = 1
    
    plt.figure(figsize=(10, 10))
    plt.contourf(xx, yy, collision_grid, levels=[0, 0.5, 1], colors=['white', 'gray'], alpha=0.5)
    
    # 画障碍物
    for obs in obstacles:
        rect = plt.Rectangle(
            (obs.center[0] - obs.size[0]/2, obs.center[1] - obs.size[1]/2),
            obs.size[0], obs.size[1], facecolor='red', alpha=0.5
        )
        plt.gca().add_patch(rect)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('C-Space (Free Space: White, C-Obstacle: Gray)')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('cspace_visualization.png')
    print("C-Space图已保存")

# exercise_cspace_visualization()
```

### 练习2：刚体变换演示

```python
def exercise_rigid_transform():
    """刚体变换演示"""
    print("=== 刚体变换演示 ===")
    
    # 机器人原始形状（三角）
    points = np.array([
        [0.5, 0.0],
        [-0.3, 0.3],
        [-0.3, -0.3]
    ])
    
    # 变换
    poses = [
        (0, 0, 0),
        (1, 1, np.pi/4),
        (2, 2, np.pi/2),
        (3, 1, np.pi)
    ]
    
    plt.figure(figsize=(10, 10))
    
    for pose in poses:
        transformed = []
        for p in points:
            transformed.append(RigidBody.transform_2d(p, pose))
        transformed = np.array(transformed)
        
        # 绘制
        plt.fill(transformed[:, 0], transformed[:, 1], alpha=0.3)
        plt.plot(np.append(transformed[:, 0], transformed[0, 0]),
                np.append(transformed[:, 1], transformed[0, 1]), '-', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rigid Body Transforms')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('rigid_transform.png')
    print("变换演示图已保存")

# exercise_rigid_transform()
```

---

**下一节**：[采样规划](02-sampling-based-planning.md)

---

## 参考文献

1. Latombe, J. C. (1991). Robot Motion Planning.
2. LaValle, S. M. (2006). Planning Algorithms.
3. Choset, H., et al. (2005). Principles of Robot Motion: Theory, Algorithms, and Implementations.
4. Lozano-Pérez, T. (1983). Spatial Planning: A Configuration Space Approach.
