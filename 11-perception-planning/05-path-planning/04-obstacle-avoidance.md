# 5.4 避障算法

## 目录

- [1. 引言](#1-引言)
- [2. 人工势场法](#2-人工势场法)
- [3. 向量场直方图](#3-向量场直方图)
- [4. 动态避障](#4-动态避障)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 避障的重要性

避障是机器人安全导航的基本能力，机器人需要实时响应环境变化，避开静态和动态障碍物。

### 1.2 避障方法分类

```python
import numpy as np
import matplotlib.pyplot as plt

class ObstacleAvoidance:
    """避障基类"""
    
    def __init__(self, robot_radius=0.3):
        self.robot_radius = robot_radius
    
    def compute_command(self, robot_state, goal, obstacles):
        """计算控制命令"""
        raise NotImplementedError
```

---

## 2. 人工势场法

### 2.1 APF实现

```python
class ArtificialPotentialField(ObstacleAvoidance):
    """人工势场法"""
    
    def __init__(self, robot_radius=0.3):
        super().__init__(robot_radius)
        
        # 势场参数
        self.k_att = 1.0
        self.k_rep = 10.0
        self.d0 = 1.0  # 排斥势场作用范围
    
    def attractive_force(self, position, goal):
        """引力"""
        direction = goal - position
        distance = np.linalg.norm(direction)
        force = self.k_att * direction
        return force
    
    def repulsive_force(self, position, obstacle):
        """单个障碍物的斥力"""
        direction = position - obstacle
        distance = np.linalg.norm(direction)
        
        if distance <= self.d0:
            force_mag = self.k_rep * (1/distance - 1/self.d0) * (1/distance**2)
            force = force_mag * direction / (distance + 1e-6)
            return force
        return np.zeros(2)
    
    def total_force(self, position, goal, obstacles):
        """总力"""
        force = self.attractive_force(position, goal)
        
        for obs in obstacles:
            force += self.repulsive_force(position, obs)
        
        return force
    
    def compute_command(self, robot_state, goal, obstacles):
        """计算控制命令"""
        x, y, theta, v, w = robot_state
        position = np.array([x, y])
        
        # 计算合力
        force = self.total_force(position, goal, obstacles)
        
        # 转换为速度
        force_norm = np.linalg.norm(force)
        if force_norm > 1e-3:
            # 速度方向
            target_dir = np.arctan2(force[1], force[0])
            angle_diff = target_dir - theta
            
            # 归一化到[-pi, pi]
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # 线速度
            v_cmd = min(1.0, force_norm) * np.cos(angle_diff)
            v_cmd = max(0, v_cmd)
            
            # 角速度
            w_cmd = angle_diff * 2.0
            w_cmd = np.clip(w_cmd, -1.0, 1.0)
        else:
            v_cmd, w_cmd = 0, 0
        
        return v_cmd, w_cmd
```

---

## 3. 向量场直方图

### 3.1 VFH实现

```python
class VectorFieldHistogram(ObstacleAvoidance):
    """向量场直方图法"""
    
    def __init__(self, robot_radius=0.3, num_bins=36):
        super().__init__(robot_radius)
        self.num_bins = num_bins
        self.safety_dist = 0.5
        self.max_speed = 1.0
    
    def compute_histogram(self, position, obstacles):
        """计算直方图"""
        bins = np.zeros(self.num_bins)
        bin_angle = 2 * np.pi / self.num_bins
        
        for obs in obstacles:
            rel_pos = obs - position
            distance = np.linalg.norm(rel_pos)
            
            if distance < self.safety_dist:
                angle = np.arctan2(rel_pos[1], rel_pos[0])
                bin_idx = int((angle + np.pi) / bin_angle)
                bin_idx = bin_idx % self.num_bins
                
                # 障碍权重
                weight = (self.safety_dist - distance) / self.safety_dist
                bins[bin_idx] += weight
        
        return bins
    
    def find_best_direction(self, bins, target_dir):
        """寻找最佳方向"""
        bin_angle = 2 * np.pi / self.num_bins
        best_bin = -1
        best_score = -float('inf')
        
        for i in range(self.num_bins):
            # 检查这个方向是否可行
            if bins[i] < 0.1:  # 阈值
                # 计算与目标方向的角度差
                bin_center = i * bin_angle - np.pi
                angle_diff = abs((bin_center - target_dir + np.pi) % (2 * np.pi) - np.pi)
                
                score = -angle_diff  # 越小越好
                
                if score > best_score:
                    best_score = score
                    best_bin = i
        
        if best_bin >= 0:
            return best_bin * bin_angle - np.pi
        
        # 如果没有可行方向，返回距离障碍最远的方向
        return np.argmin(bins) * bin_angle - np.pi
    
    def compute_command(self, robot_state, goal, obstacles):
        """计算控制命令"""
        x, y, theta, v, w = robot_state
        position = np.array([x, y])
        
        # 目标方向
        target_dir = np.arctan2(goal[1] - y, goal[0] - x)
        
        # 计算直方图
        bins = self.compute_histogram(position, obstacles)
        
        # 寻找最佳方向
        desired_dir = self.find_best_direction(bins, target_dir)
        
        # 转向
        angle_diff = desired_dir - theta
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # 速度
        dist_to_goal = np.linalg.norm(goal - position)
        v_cmd = min(self.max_speed, dist_to_goal) * (1 - abs(angle_diff) / np.pi)
        w_cmd = angle_diff * 2.0
        w_cmd = np.clip(w_cmd, -1.0, 1.0)
        
        return v_cmd, w_cmd
```

---

## 4. 动态避障

### 4.1 速度障碍法

```python
class VelocityObstacle(ObstacleAvoidance):
    """速度障碍法"""
    
    def __init__(self, robot_radius=0.3):
        super().__init__(robot_radius)
        self.time_horizon = 3.0
    
    def compute_vo(self, robot_pos, robot_radius, obs_pos, obs_radius, obs_vel):
        """计算速度障碍"""
        rel_pos = obs_pos - robot_pos
        dist = np.linalg.norm(rel_pos)
        combined_radius = robot_radius + obs_radius
        
        if dist <= combined_radius:
            # 已经碰撞
            return None
        
        # 速度障碍锥
        alpha = np.arcsin(combined_radius / dist)
        dir_vec = rel_pos / dist
        rot_matrix = np.array([
            [np.cos(alpha), -np.sin(alpha)],
            [np.sin(alpha), np.cos(alpha)]
        ])
        left_dir = rot_matrix @ dir_vec
        rot_matrix = np.array([
            [np.cos(-alpha), -np.sin(-alpha)],
            [np.sin(-alpha), np.cos(-alpha)]
        ])
        right_dir = rot_matrix @ dir_vec
        
        return obs_vel, left_dir, right_dir, combined_radius / dist
    
    def is_velocity_safe(self, v_robot, robot_pos, obstacles, obstacle_velocities):
        """检查速度是否安全"""
        for i, obs in enumerate(obstacles):
            obs_vel = obstacle_velocities[i] if obstacle_velocities else np.zeros(2)
            
            vo = self.compute_vo(robot_pos, self.robot_radius, obs, 0.3, obs_vel)
            
            if vo:
                obs_vel_v, left_dir, right_dir, time_factor = vo
                
                # 相对速度
                v_rel = v_robot - obs_vel_v
                
                # 检查是否在VO锥内
                v_rel_dir = v_rel / (np.linalg.norm(v_rel) + 1e-6)
                
                # 简化检查
                left_cross = np.cross(left_dir, v_rel_dir)
                right_cross = np.cross(right_dir, v_rel_dir)
                
                if left_cross > 0 and right_cross < 0:
                    # 在VO内，会发生碰撞
                    return False
        
        return True
    
    def compute_command(self, robot_state, goal, obstacles, obstacle_velocities=None):
        """计算命令"""
        x, y, theta, v_current, w_current = robot_state
        position = np.array([x, y])
        
        # 优先前往目标的速度
        to_goal = goal - position
        target_dir = np.arctan2(to_goal[1], to_goal[0])
        v_goal = min(1.0, np.linalg.norm(to_goal))
        v_desired = np.array([v_goal * np.cos(target_dir), v_goal * np.sin(target_dir)])
        
        # 搜索最优安全速度
        best_v, best_w = 0, 0
        best_score = -float('inf')
        
        for v_candidate in np.linspace(0, 1.0, 11):
            for w_candidate in np.linspace(-1.0, 1.0, 21):
                # 转换为速度向量
                dx = v_candidate * np.cos(theta)
                dy = v_candidate * np.sin(theta)
                v_vec = np.array([dx, dy])
                
                if self.is_velocity_safe(v_vec, position, obstacles, obstacle_velocities):
                    # 评分：接近目标速度优先
                    score = -np.linalg.norm(v_vec - v_desired)
                    if score > best_score:
                        best_score = score
                        best_v, best_w = v_candidate, w_candidate
        
        return best_v, best_w
```

---

## 5. 实践练习

### 练习1：人工势场演示

```python
def exercise_apf():
    """人工势场法演示"""
    print("=== 人工势场法 ===")
    
    apf = ArtificialPotentialField()
    
    # 环境
    goal = np.array([5.0, 5.0])
    obstacles = np.array([
        [2.5, 2.5],
        [3.0, 3.5],
        [1.5, 3.0]
    ])
    
    # 模拟
    robot_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    trajectory = []
    
    for i in range(200):
        trajectory.append((robot_state[0], robot_state[1]))
        
        # 计算命令
        v, w = apf.compute_command(robot_state, goal, obstacles)
        
        # 更新
        x, y, theta, _, _ = robot_state
        x += v * np.cos(theta) * 0.1
        y += v * np.sin(theta) * 0.1
        theta += w * 0.1
        robot_state = np.array([x, y, theta, v, w])
        
        # 停止
        if np.linalg.norm((x, y) - goal) < 0.3:
            print("到达目标!")
            break
    
    # 可视化
    trajectory = np.array(trajectory)
    
    plt.figure(figsize=(10, 10))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Trajectory')
    plt.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')
    plt.plot(obstacles[:, 0], obstacles[:, 1], 'ro', markersize=10, label='Obstacles')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Artificial Potential Field')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('apf_demo.png')
    print("APF演示已保存")

# exercise_apf()
```

### 练习2：势场可视化

```python
def exercise_potential_field_visualization():
    """势场可视化"""
    print("=== 势场可视化 ===")
    
    apf = ArtificialPotentialField()
    
    goal = np.array([5.0, 5.0])
    obstacles = np.array([[2.5, 2.5], [3.5, 3.5]])
    
    # 计算势场
    xx, yy = np.meshgrid(np.linspace(0, 6, 100), np.linspace(0, 6, 100))
    U = np.zeros_like(xx)
    
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            pos = np.array([xx[i, j], yy[i, j]])
            # 引力势
            U_att = 0.5 * apf.k_att * np.linalg.norm(pos - goal)**2
            # 斥力势
            U_rep = 0
            for obs in obstacles:
                d = np.linalg.norm(pos - obs)
                if d <= apf.d0:
                    U_rep += 0.5 * apf.k_rep * (1/d - 1/apf.d0)**2
            U[i, j] = U_att + U_rep
    
    # 可视化
    plt.figure(figsize=(10, 8))
    im = plt.contourf(xx, yy, U, 50, cmap='viridis')
    plt.colorbar(im, label='Potential U')
    plt.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')
    plt.plot(obstacles[:, 0], obstacles[:, 1], 'ro', markersize=10, label='Obstacles')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Potential Field')
    plt.legend()
    plt.savefig('potential_field.png')
    print("势场图已保存")

# exercise_potential_field_visualization()
```

---

**下一节**：[动态规划](05-dynamic-planning.md)

---

## 参考文献

1. Khatib, O. (1986). Real-Time Obstacle Avoidance for Manipulators and Mobile Robots.
2. Ulrich, I., & Borenstein, J. (1998). VFH+: Reliable Obstacle Avoidance for Fast Mobile Robots.
3. Fiorini, P., & Shiller, Z. (1998). Motion Planning in Dynamic Environments Using Velocity Obstacles.
4. van den Berg, J., et al. (2011). Reciprocal Velocity Obstacles for Real-Time Multi-Agent Navigation.
5. Fox, D., et al. (1997). The Dynamic Window Approach to Collision Avoidance.
