# 6.3 轨迹优化

## 目录

- [1. 引言](#1-引言)
- [2. 曲线参数化](#2-曲线参数化)
- [3. 数值优化](#3-数值优化)
- [4. CHOMP](#4-chomp)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 为什么需要轨迹优化？

采样规划给出的路径可能不够光滑、不够短，无法满足动力学约束。轨迹优化让路径变得更好！

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## 2. 曲线参数化

### 2.1 贝塞尔曲线

```python
class BezierCurve:
    """贝塞尔曲线"""
    
    def __init__(self, control_points):
        self.points = np.array(control_points)
        self.n = len(self.points) - 1
    
    def evaluate(self, t):
        """评估t∈[0,1]处的值"""
        b = 0
        for i in range(self.n + 1):
            # 二项式系数
            binom = np.math.comb(self.n, i)
            b += binom * (1 - t)**(self.n - i) * t**i * self.points[i]
        return b
    
    def derivative(self, t):
        """导数（速度）"""
        if self.n == 0:
            return 0 * self.points[0]
        
        d = 0
        for i in range(self.n):
            binom = np.math.comb(self.n - 1, i)
            d += self.n * binom * (1 - t)**(self.n - 1 - i) * t**i * \
                (self.points[i + 1] - self.points[i])
        return d


class CubicSpline:
    """三次样条"""
    
    def __init__(self, waypoints):
        self.waypoints = np.array(waypoints)
        self.n = len(waypoints)
        self.coeffs = self._compute_coeffs()
    
    def _compute_coeffs(self):
        """计算样条系数（简化版）"""
        # 简单实现：返回多项式系数
        # 真实样条需要三对角矩阵求解
        n = self.n - 1
        coeffs = []
        for i in range(n):
            p0 = self.waypoints[i]
            p1 = self.waypoints[i + 1]
            coeffs.append((p0, 3*(p1 - p0), 3*(p0 - p1), p1 - p0))
        return coeffs
    
    def evaluate(self, t):
        """评估"""
        segment = int(t)
        local_t = t - segment
        if segment >= len(self.coeffs):
            segment = len(self.coeffs) - 1
            local_t = 1
        
        a, b, c, d = self.coeffs[segment]
        return a + b * local_t + c * local_t**2 + d * local_t**3
```

---

## 3. 数值优化

### 3.1 梯度下降

```python
class TrajectoryOptimizer:
    """轨迹优化器"""
    
    def __init__(self, num_waypoints, dim=2):
        self.num_waypoints = num_waypoints
        self.dim = dim
        self.waypoints = np.zeros((num_waypoints, dim))
    
    def set_initial(self, waypoints):
        """设置初始路径"""
        self.waypoints = np.array(waypoints)
    
    def smoothness_cost(self):
        """光滑代价（加速度）"""
        cost = 0
        for i in range(1, self.num_waypoints - 1):
            acc = self.waypoints[i-1] - 2*self.waypoints[i] + self.waypoints[i+1]
            cost += np.sum(acc**2)
        return cost
    
    def smoothness_gradient(self):
        """光滑代价梯度"""
        grad = np.zeros_like(self.waypoints)
        for i in range(1, self.num_waypoints - 1):
            # dCost/dq_{i-1}
            grad[i-1] += 2 * (self.waypoints[i-1] - 2*self.waypoints[i] + self.waypoints[i+1])
            # dCost/dq_i
            grad[i] -= 4 * (self.waypoints[i-1] - 2*self.waypoints[i] + self.waypoints[i+1])
            # dCost/dq_{i+1}
            grad[i+1] += 2 * (self.waypoints[i-1] - 2*self.waypoints[i] + self.waypoints[i+1])
        return grad
    
    def length_cost(self):
        """长度代价"""
        cost = 0
        for i in range(self.num_waypoints - 1):
            diff = self.waypoints[i+1] - self.waypoints[i]
            cost += np.sum(diff**2)
        return cost
    
    def length_gradient(self):
        """长度代价梯度"""
        grad = np.zeros_like(self.waypoints)
        for i in range(self.num_waypoints - 1):
            diff = self.waypoints[i+1] - self.waypoints[i]
            grad[i] -= 2 * diff
            grad[i+1] += 2 * diff
        return grad
    
    def optimize(self, fixed_start, fixed_goal, 
                max_iter=100, step_size=0.01, 
                w_smooth=1.0, w_length=0.1):
        """优化"""
        for _ in range(max_iter):
            grad = w_smooth * self.smoothness_gradient() + \
                   w_length * self.length_gradient()
            
            # 更新
            self.waypoints -= step_size * grad
            
            # 固定起点和终点
            self.waypoints[0] = fixed_start
            self.waypoints[-1] = fixed_goal
        
        return self.waypoints.copy()
```

---

## 4. CHOMP

### 4.1 协变哈密顿优化

```python
class CHOMP:
    """Covariant Hamiltonian Optimization for Motion Planning"""
    
    def __init__(self, waypoints, obstacles, robot_radius=0.2):
        self.waypoints = np.array(waypoints)
        self.obstacles = obstacles
        self.robot_radius = robot_radius
        self.n = len(waypoints)
    
    def obstacle_cost(self, q):
        """障碍物代价"""
        cost = 0
        for obs in self.obstacles:
            dist = obs.distance(q)
            if dist < 2 * self.robot_radius:
                cost += max(0, 2 * self.robot_radius - dist)**2
        return cost
    
    def obstacle_gradient(self, q):
        """障碍物梯度"""
        grad = np.zeros_like(q)
        for obs in self.obstacles:
            dist = obs.distance(q)
            if dist < 2 * self.robot_radius:
                dir_to_obs = (q - obs.center) / max(dist, 1e-6)
                grad += -2 * max(0, 2 * self.robot_radius - dist) * dir_to_obs
        return grad
    
    def smoothness_matrix(self):
        """光滑矩阵（二阶差分）"""
        n = self.n
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 2
            if i > 0:
                K[i, i-1] = -1
            if i < n - 1:
                K[i, i+1] = -1
        return K
    
    def optimize(self, fixed_start, fixed_goal, 
                max_iter=200, step_size=0.01, 
                w_smooth=10.0, w_obstacle=1.0):
        """CHOMP优化"""
        K = self.smoothness_matrix()
        
        for _ in range(max_iter):
            # 光滑梯度
            smooth_grad = w_smooth * (K @ self.waypoints)
            
            # 障碍物梯度
            obs_grad = np.zeros_like(self.waypoints)
            for i in range(self.n):
                obs_grad[i] = w_obstacle * self.obstacle_gradient(self.waypoints[i])
            
            # 总梯度
            grad = smooth_grad + obs_grad
            
            # 更新
            self.waypoints -= step_size * grad
            
            # 固定边界
            self.waypoints[0] = fixed_start
            self.waypoints[-1] = fixed_goal
        
        return self.waypoints.copy()
```

---

## 5. 实践练习

### 练习1：贝塞尔曲线演示

```python
def exercise_bezier():
    """贝塞尔曲线练习"""
    print("=== 贝塞尔曲线 ===")
    
    # 控制点
    control_points = [
        (0, 0),
        (1, 2),
        (3, 2),
        (4, 0)
    ]
    
    bezier = BezierCurve(control_points)
    
    # 采样
    ts = np.linspace(0, 1, 100)
    curve = np.array([bezier.evaluate(t) for t in ts])
    
    plt.figure(figsize=(10, 10))
    
    # 画多边形
    control_points = np.array(control_points)
    plt.plot(control_points[:, 0], control_points[:, 1], 'bo-', 
            label='Control Points', linewidth=2, alpha=0.5)
    
    # 画曲线
    plt.plot(curve[:, 0], curve[:, 1], 'r-', label='Bezier Curve', linewidth=3)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bezier Curve')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('bezier_curve.png')
    print("贝塞尔曲线图已保存")

# exercise_bezier()
```

### 练习2：轨迹优化演示

```python
def exercise_trajectory_optimization():
    """轨迹优化演示"""
    print("=== 轨迹优化 ===")
    
    from .01-motion-planning-basics import BoxObstacle
    
    # 初始路径（有抖动）
    np.random.seed(42)
    num_wp = 20
    initial = np.zeros((num_wp, 2))
    for i in range(num_wp):
        initial[i] = [
            5 * i / (num_wp - 1),
            5 * i / (num_wp - 1) + 0.3 * np.random.randn()
        ]
    
    optimizer = TrajectoryOptimizer(num_wp)
    optimizer.set_initial(initial)
    
    # 优化
    optimized = optimizer.optimize(initial[0], initial[-1], 
                                  w_smooth=2.0, w_length=0.5)
    
    plt.figure(figsize=(10, 10))
    
    plt.plot(initial[:, 0], initial[:, 1], 'bo-', 
            label='Initial', linewidth=2, alpha=0.5)
    
    plt.plot(optimized[:, 0], optimized[:, 1], 'g-', 
            label='Optimized', linewidth=3)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory Optimization')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('trajectory_opt.png')
    print("轨迹优化图已保存")

# exercise_trajectory_optimization()
```

---

**下一节**：[约束规划](04-constrained-planning.md)

---

## 参考文献

1. Ratliff, N. D., et al. (2009). CHOMP: Gradient Optimization Techniques for Efficient Motion Planning.
2. Schulman, J., et al. (2013). Finding Locally Optimal, Collision-Free Trajectories with Sequential Convex Optimization.
3. Park, J., et al. (2018). Trajectory Optimization Using Sequential Quadratic Programming and Covariant Matrix Adaptation.
4. Zucker, M., et al. (2013). CHOMP: Covariant Hamiltonian Optimization for Motion Planning.
5. Nocedal, J., & Wright, S. J. (2006). Numerical Optimization (2nd ed.).
