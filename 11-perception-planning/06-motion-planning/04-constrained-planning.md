# 6.4 约束规划

## 目录

- [1. 引言](#1-引言)
- [2. 约束类型](#2-约束类型)
- [3. 惩罚函数法](#3-惩罚函数法)
- [4. 时间参数化](#4-时间参数化)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 机器人约束

真实机器人有许多约束：
- 关节角度限制
- 速度/加速度限制
- 扭矩限制
- 避障约束

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## 2. 约束类型

### 2.1 运动学约束

```python
class KinematicConstraints:
    """运动学约束"""
    
    def __init__(self, q_min, q_max, q_dot_max, q_ddot_max):
        self.q_min = np.array(q_min)
        self.q_max = np.array(q_max)
        self.q_dot_max = np.array(q_dot_max)
        self.q_ddot_max = np.array(q_ddot_max)
    
    def check_position(self, q):
        """检查位置约束"""
        return np.all(q >= self.q_min) and np.all(q <= self.q_max)
    
    def check_velocity(self, q_dot):
        """检查速度约束"""
        return np.all(np.abs(q_dot) <= self.q_dot_max)
    
    def check_acceleration(self, q_ddot):
        """检查加速度约束"""
        return np.all(np.abs(q_ddot) <= self.q_ddot_max)
    
    def clamp_position(self, q):
        """夹紧位置"""
        return np.clip(q, self.q_min, self.q_max)
    
    def clamp_velocity(self, q_dot):
        """夹紧速度"""
        return np.clip(q_dot, -self.q_dot_max, self.q_dot_max)
    
    def clamp_acceleration(self, q_ddot):
        """夹紧加速度"""
        return np.clip(q_ddot, -self.q_ddot_max, self.q_ddot_max)
```

---

## 3. 惩罚函数法

### 3.1 带约束的优化

```python
class ConstrainedOptimizer:
    """带约束的优化器"""
    
    def __init__(self, waypoints, constraints):
        self.waypoints = np.array(waypoints)
        self.constraints = constraints
        self.n, self.dim = self.waypoints.shape
    
    def position_penalty(self):
        """位置约束惩罚"""
        penalty = 0
        for q in self.waypoints:
            over_max = np.maximum(q - self.constraints.q_max, 0)
            under_min = np.maximum(self.constraints.q_min - q, 0)
            penalty += np.sum(over_max**2 + under_min**2)
        return penalty
    
    def position_penalty_gradient(self):
        """位置约束梯度"""
        grad = np.zeros_like(self.waypoints)
        for i, q in enumerate(self.waypoints):
            over_max = np.maximum(q - self.constraints.q_max, 0)
            under_min = np.maximum(self.constraints.q_min - q, 0)
            grad[i] = 2 * (over_max - under_min)
        return grad
    
    def smoothness_cost(self):
        """光滑代价"""
        cost = 0
        for i in range(1, self.n - 1):
            acc = self.waypoints[i-1] - 2*self.waypoints[i] + self.waypoints[i+1]
            cost += np.sum(acc**2)
        return cost
    
    def smoothness_gradient(self):
        """光滑代价梯度"""
        grad = np.zeros_like(self.waypoints)
        for i in range(1, self.n - 1):
            term = self.waypoints[i-1] - 2*self.waypoints[i] + self.waypoints[i+1]
            grad[i-1] += 2 * term
            grad[i] -= 4 * term
            grad[i+1] += 2 * term
        return grad
    
    def optimize(self, fixed_start, fixed_goal, 
                max_iter=500, step_size=0.01,
                w_smooth=1.0, w_constraint=100.0):
        """优化"""
        for _ in range(max_iter):
            grad = w_smooth * self.smoothness_gradient() + \
                   w_constraint * self.position_penalty_gradient()
            
            self.waypoints -= step_size * grad
            
            # 固定
            self.waypoints[0] = fixed_start
            self.waypoints[-1] = fixed_goal
            
            # 显式夹紧（辅助）
            for i in range(self.n):
                self.waypoints[i] = self.constraints.clamp_position(self.waypoints[i])
        
        return self.waypoints.copy()
```

---

## 4. 时间参数化

### 4.1 时间最优参数化

```python
class TimeParameterizer:
    """时间参数化"""
    
    def __init__(self, constraints, dt=0.01):
        self.constraints = constraints
        self.dt = dt
    
    def parameterize(self, path, max_v_scale=1.0):
        """参数化"""
        n = len(path)
        if n < 2:
            return path, [0]
        
        # 1. 计算距离
        dists = [0]
        for i in range(1, n):
            dist = np.linalg.norm(path[i] - path[i-1])
            dists.append(dists[-1] + dist)
        
        total_dist = dists[-1]
        if total_dist < 1e-6:
            return path, [0]
        
        # 2. 简单时间缩放（真实需要梯形速度剖面）
        v_max = np.max(self.constraints.q_dot_max) * max_v_scale
        times = []
        
        t = 0
        for d in dists:
            # 简单的匀速
            ideal_t = d / v_max
            times.append(ideal_t)
        
        return times
```

### 4.2 梯形速度剖面

```python
class TrapezoidalProfile:
    """梯形速度剖面"""
    
    def __init__(self, v_max, a_max):
        self.v_max = v_max
        self.a_max = a_max
    
    def plan(self, start, goal):
        """规划梯形速度"""
        q0 = np.array(start)
        q1 = np.array(goal)
        delta = q1 - q0
        dist = np.linalg.norm(delta)
        
        if dist < 1e-6:
            return [q0], [0], [0]
        
        dir_vec = delta / dist
        
        # 加速/减速距离
        accel_dist = self.v_max**2 / (2 * self.a_max)
        
        if 2 * accel_dist < dist:
            # 梯形：加速 -> 匀速 -> 减速
            t_accel = self.v_max / self.a_max
            const_dist = dist - 2 * accel_dist
            t_const = const_dist / self.v_max
            total_time = 2 * t_accel + t_const
        else:
            # 三角形：加速 -> 减速
            t_accel = np.sqrt(dist / self.a_max)
            v_peak = self.a_max * t_accel
            total_time = 2 * t_accel
        
        # 采样
        t_samples = np.linspace(0, total_time, 50)
        positions = []
        velocities = []
        
        for t in t_samples:
            if 2 * accel_dist < dist:
                # 梯形
                if t < t_accel:
                    v = self.a_max * t
                    d = 0.5 * self.a_max * t**2
                elif t < t_accel + t_const:
                    v = self.v_max
                    d = accel_dist + self.v_max * (t - t_accel)
                else:
                    t_decel = t - t_accel - t_const
                    v = self.v_max - self.a_max * t_decel
                    d = accel_dist + const_dist + self.v_max * t_decel - \
                        0.5 * self.a_max * t_decel**2
            else:
                # 三角形
                if t < t_accel:
                    v = self.a_max * t
                    d = 0.5 * self.a_max * t**2
                else:
                    t_decel = t - t_accel
                    v = v_peak - self.a_max * t_decel
                    d = accel_dist + v_peak * t_decel - \
                        0.5 * self.a_max * t_decel**2
            
            pos = q0 + dir_vec * min(d, dist)
            vel = dir_vec * v
            
            positions.append(pos)
            velocities.append(vel)
        
        return positions, velocities, t_samples
```

---

## 5. 实践练习

### 练习1：约束优化演示

```python
def exercise_constrained_optimization():
    """约束优化演示"""
    print("=== 约束优化 ===")
    
    # 约束
    constraints = KinematicConstraints(
        q_min=[-1, -1],
        q_max=[1, 1],
        q_dot_max=[0.5, 0.5],
        q_ddot_max=[0.5, 0.5]
    )
    
    # 初始路径（超出边界）
    waypoints = np.linspace([-1.5, -1.5], [1.5, 1.5], 20)
    
    optimizer = ConstrainedOptimizer(waypoints, constraints)
    
    optimized = optimizer.optimize(waypoints[0], waypoints[-1],
                                   w_smooth=1.0, w_constraint=50.0)
    
    plt.figure(figsize=(10, 10))
    
    # 边界
    rect = plt.Rectangle((-1, -1), 2, 2, linewidth=2, 
                        edgecolor='red', facecolor='none', 
                        linestyle='--', label='Constraints')
    plt.gca().add_patch(rect)
    
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'bo-', 
            label='Initial (Violation)', linewidth=2, alpha=0.5)
    plt.plot(optimized[:, 0], optimized[:, 1], 'g-', 
            label='Optimized', linewidth=3)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Constrained Optimization')
    plt.legend()
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.savefig('constrained_opt.png')
    print("约束优化图已保存")

# exercise_constrained_optimization()
```

### 练习2：梯形速度剖面

```python
def exercise_trapezoidal_profile():
    """梯形速度演示"""
    print("=== 梯形速度剖面 ===")
    
    profile = TrapezoidalProfile(v_max=1.0, a_max=0.5)
    
    start = [0, 0]
    goal = [5, 3]
    
    positions, velocities, times = profile.plan(start, goal)
    
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(times, positions[:, 0], 'b-', label='X')
    ax1.plot(times, positions[:, 1], 'r-', label='Y')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Position')
    ax1.set_title('Position Profile')
    ax1.legend()
    ax1.grid(True)
    
    speed = np.linalg.norm(velocities, axis=1)
    ax2.plot(times, speed, 'g-', linewidth=2, label='Speed')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Speed')
    ax2.set_title('Velocity Profile')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('trapezoidal_profile.png')
    print("梯形剖面演示图已保存")

# exercise_trapezoidal_profile()
```

---

**下一节**：[机械臂规划](05-arm-planning.md)

---

## 参考文献

1. Nocedal, J., & Wright, S. J. (2006). Numerical Optimization.
2. Bobrow, J. E., et al. (1985). Time-Optimal Control of Robotic Manipulators Along Specified Paths.
3. Shin, K. G., & McKay, N. D. (1985). A New Approach to Dynamic Time Scaling for Manipulator Trajectory Generation.
4. Pham, Q. C., & Nakamura, Y. (2014). A Fast Time-Optimal Parameterization of Paths with Bounded Derivatives.
5. Hauser, K., & Ng-Thow-Hing, V. (2010). Fast Smoothing of Manipulator Trajectories.
