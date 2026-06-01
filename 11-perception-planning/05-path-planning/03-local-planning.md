# 5.3 局部规划

## 目录

- [1. 引言](#1-引言)
- [2. 动态窗口法](#2-动态窗口法)
- [3. 弹性带算法](#3-弹性带算法)
- [4. 模型预测控制](#4-模型预测控制)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 局部规划的作用

局部规划在全局路径的指导下，考虑机器人动力学约束和动态障碍物，实时生成可行的运动轨迹。

```python
import numpy as np
import matplotlib.pyplot as plt

class LocalPlanner:
    """局部规划器基类"""
    
    def __init__(self, max_vel, max_acc):
        self.max_vel = max_vel
        self.max_acc = max_acc
    
    def compute_velocity(self, robot_state, goal, obstacles):
        """计算速度"""
        raise NotImplementedError
```

---

## 2. 动态窗口法

### 2.1 DWA实现

```python
class DynamicWindowApproach(LocalPlanner):
    """动态窗口法"""
    
    def __init__(self, max_vel=1.0, max_vel_w=1.0, max_acc=0.5, 
                 max_acc_w=0.5, dt=0.1, predict_time=1.0):
        super().__init__(max_vel, max_acc)
        self.max_vel_w = max_vel_w
        self.max_acc_w = max_acc_w
        self.dt = dt
        self.predict_time = predict_time
        
        # 权重
        self.alpha = 0.1  # 速度
        self.beta = 0.6   # 方向
        self.gamma = 0.3  # 距离
    
    def compute_dynamic_window(self, v, w):
        """计算动态窗口"""
        # 速度边界
        v_min = max(0, v - self.max_acc * self.dt)
        v_max = min(self.max_vel, v + self.max_acc * self.dt)
        
        w_min = max(-self.max_vel_w, w - self.max_acc_w * self.dt)
        w_max = min(self.max_vel_w, w + self.max_acc_w * self.dt)
        
        return (v_min, v_max), (w_min, w_max)
    
    def simulate_trajectory(self, v, w, robot_state):
        """模拟轨迹"""
        x, y, theta = robot_state
        trajectory = [(x, y)]
        
        t = 0
        while t < self.predict_time:
            x = x + v * np.cos(theta) * self.dt
            y = y + v * np.sin(theta) * self.dt
            theta = theta + w * self.dt
            trajectory.append((x, y))
            t += self.dt
        
        return np.array(trajectory)
    
    def evaluate_trajectory(self, trajectory, goal, obstacles):
        """评估轨迹"""
        if len(trajectory) < 2:
            return 0
        
        # 1. 速度得分
        vel_score = np.linalg.norm(trajectory[-1] - trajectory[-2]) / self.dt
        
        # 2. 方向得分
        dx, dy = goal - trajectory[-1]
        angle_diff = np.arctan2(dy, dx) - np.arctan2(
            trajectory[-1][1] - trajectory[-2][1],
            trajectory[-1][0] - trajectory[-2][0]
        )
        angle_diff = np.abs(angle_diff)
        heading_score = 1.0 - (angle_diff / np.pi)
        
        # 3. 距离得分
        min_dist = float('inf')
        for obs in obstacles:
            dists = np.linalg.norm(trajectory - obs[:2], axis=1)
            min_dist = min(min_dist, np.min(dists))
        dist_score = min(1.0, min_dist / 1.0)
        
        # 总得分
        total_score = self.alpha * vel_score + self.beta * heading_score + self.gamma * dist_score
        return total_score
    
    def compute_velocity(self, robot_state, goal, obstacles):
        """计算最优速度"""
        x, y, theta, v, w = robot_state
        
        (v_min, v_max), (w_min, w_max) = self.compute_dynamic_window(v, w)
        
        # 搜索速度组合
        best_score = -float('inf')
        best_v, best_w = 0, 0
        
        n_v = int((v_max - v_min) / 0.1) + 1
        n_w = int((w_max - w_min) / 0.1) + 1
        
        for i in range(n_v):
            v_candidate = v_min + i * 0.1
            for j in range(n_w):
                w_candidate = w_min + j * 0.1
                
                # 模拟
                trajectory = self.simulate_trajectory(v_candidate, w_candidate, (x, y, theta))
                
                # 评估
                score = self.evaluate_trajectory(trajectory, goal, obstacles)
                
                if score > best_score:
                    best_score = score
                    best_v, best_w = v_candidate, w_candidate
        
        return best_v, best_w
```

---

## 3. 弹性带算法

### 3.1 TEB实现

```python
class ElasticBand:
    """弹性带"""
    
    def __init__(self, waypoints, robot_radius=0.3):
        self.waypoints = np.array(waypoints)
        self.robot_radius = robot_radius
        
        # 弹簧参数
        self.k_obstacle = 10.0
        self.k_internal = 1.0
        self.k_waypoint = 5.0
    
    def internal_forces(self):
        """内力"""
        n = len(self.waypoints)
        forces = np.zeros_like(self.waypoints)
        
        for i in range(1, n-1):
            prev = self.waypoints[i-1]
            curr = self.waypoints[i]
            next_pt = self.waypoints[i+1]
            
            # 弹性力
            force_prev = self.k_internal * (prev - curr)
            force_next = self.k_internal * (next_pt - curr)
            forces[i] = force_prev + force_next
        
        return forces
    
    def obstacle_forces(self, obstacles):
        """障碍物斥力"""
        n = len(self.waypoints)
        forces = np.zeros_like(self.waypoints)
        
        for i in range(n):
            for obs in obstacles:
                dist = np.linalg.norm(self.waypoints[i] - obs)
                if dist < self.robot_radius * 3:
                    # 斥力
                    direction = self.waypoints[i] - obs
                    force = self.k_obstacle * (1.0 / dist - 1.0 / (3 * self.robot_radius))
                    forces[i] += direction / (dist + 1e-6) * force
        
        return forces
    
    def optimize(self, obstacles, iterations=100, dt=0.01):
        """优化弹性带"""
        for _ in range(iterations):
            forces_int = self.internal_forces()
            forces_obs = self.obstacle_forces(obstacles)
            total_forces = forces_int + forces_obs
            
            # 更新
            self.waypoints += total_forces * dt
    
    def get_trajectory(self):
        """获取轨迹"""
        return self.waypoints.copy()
```

---

## 4. 模型预测控制

### 4.1 MPC实现

```python
class ModelPredictiveControl(LocalPlanner):
    """模型预测控制"""
    
    def __init__(self, max_vel=1.0, horizon=10, dt=0.1):
        super().__init__(max_vel, max_acc=0.5)
        self.horizon = horizon
        self.dt = dt
        
        # 权重
        self.q_goal = 10.0
        self.q_velocity = 1.0
        self.r_control = 0.1
    
    def kinematic_model(self, state, v, w):
        """运动学模型"""
        x, y, theta = state
        x_new = x + v * np.cos(theta) * self.dt
        y_new = y + v * np.sin(theta) * self.dt
        theta_new = theta + w * self.dt
        return np.array([x_new, y_new, theta_new])
    
    def predict(self, initial_state, velocity_seq):
        """预测轨迹"""
        states = [initial_state]
        
        x, y, theta = initial_state
        for v, w in velocity_seq:
            x, y, theta = self.kinematic_model((x, y, theta), v, w)
            states.append([x, y, theta])
        
        return np.array(states)
    
    def compute_cost(self, states, velocity_seq, goal, obstacles):
        """计算代价"""
        cost = 0
        
        # 终点代价
        end_pos = states[-1, :2]
        cost += self.q_goal * np.linalg.norm(end_pos - goal)**2
        
        # 速度代价
        for v, w in velocity_seq:
            cost += self.q_velocity * (v**2 + w**2)
        
        # 控制平滑代价
        for i in range(1, len(velocity_seq)):
            dv = velocity_seq[i][0] - velocity_seq[i-1][0]
            dw = velocity_seq[i][1] - velocity_seq[i-1][1]
            cost += self.r_control * (dv**2 + dw**2)
        
        # 障碍物代价
        for state in states:
            for obs in obstacles:
                dist = np.linalg.norm(state[:2] - obs)
                if dist < 0.5:
                    cost += 1000.0 * (0.5 - dist)**2
        
        return cost
    
    def compute_velocity(self, robot_state, goal, obstacles):
        """计算最优速度（简化版）"""
        x, y, theta, v_current, w_current = robot_state
        initial_state = np.array([x, y, theta])
        
        best_cost = float('inf')
        best_v, best_w = 0, 0
        
        # 简单搜索
        for dv in np.linspace(-0.5, 0.5, 5):
            for dw in np.linspace(-0.5, 0.5, 5):
                v_seq = [(v_current + dv, w_current + dw)] * self.horizon
                
                # 预测
                states = self.predict(initial_state, v_seq)
                
                # 计算代价
                cost = self.compute_cost(states, v_seq, goal, obstacles)
                
                if cost < best_cost:
                    best_cost = cost
                    best_v, best_w = v_seq[0]
        
        return best_v, best_w
```

---

## 5. 实践练习

### 练习1：DWA可视化

```python
def exercise_dwa():
    """DWA练习"""
    print("=== 动态窗口法 ===")
    
    # 创建DWA
    dwa = DynamicWindowApproach(max_vel=1.0, max_vel_w=1.0)
    
    # 环境
    goal = np.array([5.0, 5.0])
    obstacles = np.array([
        [3.0, 3.0],
        [2.0, 4.0],
        [4.0, 2.0]
    ])
    
    # 初始状态
    robot_state = (0.0, 0.0, 0.0, 0.0, 0.0)  # x, y, theta, v, w
    
    # 模拟
    trajectory = []
    for i in range(100):
        trajectory.append((robot_state[0], robot_state[1]))
        
        # 计算速度
        v, w = dwa.compute_velocity(robot_state, goal, obstacles)
        
        # 更新
        x, y, theta, v_old, w_old = robot_state
        x_new = x + v * np.cos(theta) * 0.1
        y_new = y + v * np.sin(theta) * 0.1
        theta_new = theta + w * 0.1
        robot_state = (x_new, y_new, theta_new, v, w)
        
        # 检查是否到达
        if np.linalg.norm((x_new, y_new) - goal) < 0.3:
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
    plt.title('Dynamic Window Approach')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('dwa_demo.png')
    print("DWA演示已保存")

# exercise_dwa()
```

### 练习2：弹性带演示

```python
def exercise_elastic_band():
    """弹性带练习"""
    print("=== 弹性带算法 ===")
    
    # 初始路点
    waypoints = np.linspace((0, 0), (5, 5), 20)
    
    # 创建弹性带
    band = ElasticBand(waypoints)
    
    # 障碍物
    obstacles = np.array([
        [2.0, 2.0],
        [3.0, 3.0],
        [1.5, 3.5]
    ])
    
    # 优化前
    traj_before = band.get_trajectory().copy()
    
    # 优化
    band.optimize(obstacles, iterations=200)
    
    # 优化后
    traj_after = band.get_trajectory()
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(121)
    plt.plot(traj_before[:, 0], traj_before[:, 1], 'b-o', label='Initial')
    plt.plot(obstacles[:, 0], obstacles[:, 1], 'ro', label='Obstacles')
    plt.title('Before Optimization')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(traj_before[:, 0], traj_before[:, 1], 'b-o', alpha=0.3, label='Initial')
    plt.plot(traj_after[:, 0], traj_after[:, 1], 'g-o', label='Optimized')
    plt.plot(obstacles[:, 0], obstacles[:, 1], 'ro', label='Obstacles')
    plt.title('After Optimization')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    
    plt.savefig('elastic_band.png')
    print("弹性带演示已保存")

# exercise_elastic_band()
```

---

**下一节**：[避障算法](04-obstacle-avoidance.md)

---

## 参考文献

1. Fox, D., et al. (1997). The Dynamic Window Approach to Collision Avoidance.
2. Quinlan, S., & Khatib, O. (1993). Elastic Bands: Connecting Path Planning and Control.
3. Werling, M., et al. (2012). Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame.
4. Diehl, M., et al. (2009). Fast Direct Multiple Shooting Algorithms for Optimal Robot Control.
5. Rosmann, C., et al. (2017). Kinodynamic Trajectory Optimization and Control for Mobile Robots.
