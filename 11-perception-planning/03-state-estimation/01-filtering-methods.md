# 3.1 滤波方法

## 目录

- [1. 引言](#1-引言)
- [2. 状态估计概述](#2-状态估计概述)
- [3. 卡尔曼滤波](#3-卡尔曼滤波)
- [4. 扩展卡尔曼滤波](#4-扩展卡尔曼滤波)
- [5. 无迹卡尔曼滤波](#5-无迹卡尔曼滤波)
- [6. 实践练习](#6-实践练习)

---

## 1. 引言

### 1.1 滤波方法的重要性

**滤波方法**是状态估计的核心技术，用于从含噪声的传感器观测中递归估计动态系统的状态，是机器人、导航、控制等领域的基础。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **机器人导航** | 位置和姿态估计 | 移动机器人定位 |
| **自动驾驶** | 车辆状态追踪 | IMU+GPS融合定位 |
| **飞行器控制** | 姿态估计 | 无人机飞控系统 |
| **经济预测** | 时间序列预测 | 股票价格分析 |

---

## 2. 状态估计概述

### 2.1 状态空间模型

```python
import numpy as np
import matplotlib.pyplot as plt

class StateSpaceModel:
    """状态空间模型"""
    
    def __init__(self, F, H, Q, R, B=None):
        """
        初始化状态空间模型
        
        参数:
            F: 状态转移矩阵
            H: 观测矩阵
            Q: 过程噪声协方差
            R: 观测噪声协方差
            B: 控制矩阵
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B if B is not None else np.zeros((F.shape[0], 1))
    
    def predict_state(self, x, u=None):
        """状态预测"""
        if u is None:
            u = np.zeros((self.B.shape[1], 1))
        return self.F @ x + self.B @ u
    
    def predict_observation(self, x):
        """观测预测"""
        return self.H @ x
    
    def simulate(self, x0, num_steps, u=None):
        """
        仿真系统
        
        参数:
            x0: 初始状态
            num_steps: 步数
            u: 控制序列
        
        返回:
            x_true: 真实状态序列
            z: 观测序列
        """
        dim_x = self.F.shape[0]
        dim_z = self.H.shape[0]
        
        x_true = np.zeros((num_steps, dim_x))
        z = np.zeros((num_steps, dim_z))
        
        x = x0.copy()
        x_true[0] = x
        
        for t in range(1, num_steps):
            # 状态转移（带过程噪声）
            u_t = u[t] if u is not None else np.zeros(self.B.shape[1])
            process_noise = np.random.multivariate_normal(
                np.zeros(dim_x), self.Q
            )
            x = self.F @ x + self.B @ u_t + process_noise
            x_true[t] = x
            
            # 观测（带观测噪声）
            observation_noise = np.random.multivariate_normal(
                np.zeros(dim_z), self.R
            )
            z[t] = self.H @ x + observation_noise
        
        return x_true, z
```

### 2.2 贝叶斯滤波

```python
class BayesianFilter:
    """贝叶斯滤波器基类"""
    
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
    
    def predict(self, u=None):
        """预测步骤"""
        pass
    
    def update(self, z):
        """更新步骤"""
        pass
    
    def get_state(self):
        """获取当前状态"""
        return self.x.copy()
```

---

## 3. 卡尔曼滤波

### 3.1 基本卡尔曼滤波

```python
class KalmanFilter(BayesianFilter):
    """卡尔曼滤波器"""
    
    def __init__(self, F, H, Q, R, B=None):
        """
        初始化卡尔曼滤波器
        
        参数:
            F: 状态转移矩阵
            H: 观测矩阵
            Q: 过程噪声协方差
            R: 观测噪声协方差
            B: 控制矩阵
        """
        dim_x = F.shape[0]
        dim_z = H.shape[0]
        super().__init__(dim_x, dim_z)
        
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.B = B if B is not None else np.zeros((dim_x, 1))
        
        # 后验状态和协方差
        self.x_post = np.zeros(dim_x)
        self.P_post = np.eye(dim_x)
    
    def predict(self, u=None):
        """
        预测步骤
        
        参数:
            u: 控制量
        """
        if u is None:
            u = np.zeros(self.B.shape[1])
        
        # 先验状态预测
        self.x = self.F @ self.x_post + self.B @ u
        
        # 先验协方差预测
        self.P = self.F @ self.P_post @ self.F.T + self.Q
    
    def update(self, z):
        """
        更新步骤
        
        参数:
            z: 观测值
        """
        # 卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 后验状态更新
        y = z - self.H @ self.x
        self.x_post = self.x + K @ y
        
        # 后验协方差更新
        I = np.eye(self.dim_x)
        self.P_post = (I - K @ self.H) @ self.P
    
    def filter(self, z_list, u_list=None):
        """
        完整滤波过程
        
        参数:
            z_list: 观测序列
            u_list: 控制序列
        
        返回:
            estimates: 估计状态序列
        """
        num_steps = len(z_list)
        estimates = np.zeros((num_steps, self.dim_x))
        
        for t in range(num_steps):
            # 预测
            u = u_list[t] if u_list is not None else None
            self.predict(u)
            
            # 更新
            self.update(z_list[t])
            
            # 保存
            estimates[t] = self.x_post.copy()
        
        return estimates
```

### 3.2 一维运动跟踪示例

```python
def kalman_filter_1d_demo():
    """一维运动跟踪演示"""
    # 系统定义：位置和速度
    dt = 0.1  # 时间步长
    
    F = np.array([
        [1, dt],
        [0, 1]
    ])
    H = np.array([[1, 0]])
    
    Q = np.array([
        [0.1, 0.05],
        [0.05, 0.1]
    ])
    R = np.array([[1.0]])
    
    # 初始状态
    x0 = np.array([0, 1])
    
    # 创建模型
    model = StateSpaceModel(F, H, Q, R)
    
    # 仿真
    num_steps = 50
    x_true, z = model.simulate(x0, num_steps)
    
    # 卡尔曼滤波
    kf = KalmanFilter(F, H, Q, R)
    kf.x_post = x0.copy()
    kf.P_post = np.eye(2) * 0.1
    estimates = kf.filter(z)
    
    # 绘图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_true[:, 0], label='真实位置')
    plt.plot(z[:, 0], 'x', label='观测值')
    plt.plot(estimates[:, 0], label='滤波估计')
    plt.xlabel('时间步')
    plt.ylabel('位置')
    plt.legend()
    plt.title('位置估计')
    
    plt.subplot(1, 2, 2)
    plt.plot(x_true[:, 1], label='真实速度')
    plt.plot(estimates[:, 1], label='滤波估计')
    plt.xlabel('时间步')
    plt.ylabel('速度')
    plt.legend()
    plt.title('速度估计')
    
    plt.tight_layout()
    plt.savefig('kalman_1d_demo.png')
    print("卡尔曼滤波一维演示完成，结果已保存")

# kalman_filter_1d_demo()
```

---

## 4. 扩展卡尔曼滤波

### 4.1 EKF实现

```python
class ExtendedKalmanFilter(BayesianFilter):
    """扩展卡尔曼滤波器 (EKF)"""
    
    def __init__(self, dim_x, dim_z):
        super().__init__(dim_x, dim_z)
        
        self.x_post = np.zeros(dim_x)
        self.P_post = np.eye(dim_x)
        
        # 函数对象
        self.f = None
        self.h = None
        self.F_jacobian = None
        self.H_jacobian = None
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
    
    def set_functions(self, f, h, F_jacobian, H_jacobian, Q, R):
        """设置非线性函数和雅可比矩阵"""
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
    
    def predict(self, u=None):
        """预测步骤"""
        if u is None:
            u = np.zeros(1)
        
        # 先验状态预测（非线性）
        self.x = self.f(self.x_post, u)
        
        # 雅可比矩阵
        F = self.F_jacobian(self.x_post, u)
        
        # 先验协方差预测
        self.P = F @ self.P_post @ F.T + self.Q
    
    def update(self, z):
        """更新步骤"""
        # 雅可比矩阵
        H = self.H_jacobian(self.x)
        
        # 卡尔曼增益
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 观测预测（非线性）
        z_pred = self.h(self.x)
        
        # 后验状态更新
        y = z - z_pred
        self.x_post = self.x + K @ y
        
        # 后验协方差更新
        I = np.eye(self.dim_x)
        self.P_post = (I - K @ H) @ self.P
```

### 4.2 非线运动示例

```python
def ekf_nonlinear_demo():
    """EKF非线性演示"""
    # 单摆模型
    dt = 0.01
    g = 9.81
    l = 1.0
    
    # 状态：[角度, 角速度]
    def f(x, u):
        theta, omega = x
        theta_new = theta + omega * dt
        omega_new = omega - (g / l) * np.sin(theta) * dt
        return np.array([theta_new, omega_new])
    
    def h(x, u=None):
        return np.array([x[0]])
    
    def F_jacobian(x, u):
        theta, omega = x
        return np.array([
            [1, dt],
            [-(g / l) * np.cos(theta), 1]
        ])
    
    def H_jacobian(x):
        return np.array([[1, 0]])
    
    # 噪声
    Q = np.diag([0.0001, 0.001])
    R = np.array([[0.01]])
    
    # 创建EKF
    ekf = ExtendedKalmanFilter(dim_x=2, dim_z=1)
    ekf.set_functions(f, h, F_jacobian, H_jacobian, Q, R)
    
    # 初始状态
    x0 = np.array([np.pi / 4, 0])
    ekf.x_post = x0.copy()
    ekf.P_post = np.eye(2) * 0.1
    
    # 仿真
    num_steps = 500
    x_true = np.zeros((num_steps, 2))
    z = np.zeros((num_steps, 1))
    estimates = np.zeros((num_steps, 2))
    
    x = x0.copy()
    x_true[0] = x
    
    for t in range(1, num_steps):
        # 真实状态
        x = f(x, None)
        x_true[t] = x
        
        # 观测
        z[t] = x[0] + np.random.normal(0, np.sqrt(R[0, 0]))
        
        # EKF
        ekf.predict()
        ekf.update(z[t])
        estimates[t] = ekf.x_post
    
    # 绘图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_true[:, 0], label='真实角度')
    plt.plot(z[:, 0], '.', label='观测', alpha=0.5)
    plt.plot(estimates[:, 0], label='EKF估计')
    plt.xlabel('时间步')
    plt.ylabel('角度 (rad)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x_true[:, 1], label='真实角速度')
    plt.plot(estimates[:, 1], label='EKF估计')
    plt.xlabel('时间步')
    plt.ylabel('角速度 (rad/s)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ekf_demo.png')
    print("EKF非线性演示完成")

# ekf_nonlinear_demo()
```

---

## 5. 无迹卡尔曼滤波

### 5.1 UKF实现

```python
class UnscentedKalmanFilter(BayesianFilter):
    """无迹卡尔曼滤波器 (UKF)"""
    
    def __init__(self, dim_x, dim_z, alpha=1e-3, beta=2, kappa=0):
        """
        初始化UKF
        
        参数:
            dim_x: 状态维度
            dim_z: 观测维度
            alpha: 缩放参数
            beta: 先验知识参数
            kappa: 缩放参数
        """
        super().__init__(dim_x, dim_z)
        
        self.x_post = np.zeros(dim_x)
        self.P_post = np.eye(dim_x)
        
        # UKF参数
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha**2 * (dim_x + kappa) - dim_x
        
        # 函数对象
        self.f = None
        self.h = None
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
    
    def set_functions(self, f, h, Q, R):
        """设置非线性函数"""
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
    
    def generate_sigma_points(self, x, P):
        """生成sigma点"""
        n = self.dim_x
        sigma_points = np.zeros((2 * n + 1, n))
        
        sqrt_matrix = np.linalg.cholesky((n + self.lam) * P)
        
        sigma_points[0] = x
        for k in range(n):
            sigma_points[k + 1] = x + sqrt_matrix[:, k]
            sigma_points[k + 1 + n] = x - sqrt_matrix[:, k]
        
        return sigma_points
    
    def compute_weights(self):
        """计算权重"""
        n = self.dim_x
        
        Wm = np.zeros(2 * n + 1)
        Wc = np.zeros(2 * n + 1)
        
        Wm[0] = self.lam / (n + self.lam)
        Wc[0] = self.lam / (n + self.lam) + (1 - self.alpha**2 + self.beta)
        
        for k in range(1, 2 * n + 1):
            Wm[k] = 1 / (2 * (n + self.lam))
            Wc[k] = 1 / (2 * (n + self.lam))
        
        return Wm, Wc
    
    def predict(self, u=None):
        """预测步骤"""
        if u is None:
            u = np.zeros(1)
        
        # 生成sigma点
        sigma_points = self.generate_sigma_points(self.x_post, self.P_post)
        
        # 通过状态方程传播
        sigma_points_pred = np.zeros_like(sigma_points)
        for k in range(2 * self.dim_x + 1):
            sigma_points_pred[k] = self.f(sigma_points[k], u)
        
        # 计算权重
        Wm, Wc = self.compute_weights()
        
        # 先验均值
        self.x = np.sum(Wm.reshape(-1, 1) * sigma_points_pred, axis=0)
        
        # 先验协方差
        self.P = np.zeros((self.dim_x, self.dim_x))
        for k in range(2 * self.dim_x + 1):
            diff = (sigma_points_pred[k] - self.x).reshape(-1, 1)
            self.P += Wc[k] * (diff @ diff.T)
        self.P += self.Q
    
    def update(self, z):
        """更新步骤"""
        # 生成sigma点
        sigma_points = self.generate_sigma_points(self.x, self.P)
        
        # 通过观测方程传播
        sigma_points_z = np.zeros((2 * self.dim_x + 1, self.dim_z))
        for k in range(2 * self.dim_x + 1):
            sigma_points_z[k] = self.h(sigma_points[k])
        
        # 计算权重
        Wm, Wc = self.compute_weights()
        
        # 观测均值
        z_pred = np.sum(Wm.reshape(-1, 1) * sigma_points_z, axis=0)
        
        # 观测协方差
        Pzz = np.zeros((self.dim_z, self.dim_z))
        for k in range(2 * self.dim_x + 1):
            diff = (sigma_points_z[k] - z_pred).reshape(-1, 1)
            Pzz += Wc[k] * (diff @ diff.T)
        Pzz += self.R
        
        # 交叉协方差
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for k in range(2 * self.dim_x + 1):
            diff_x = (sigma_points[k] - self.x).reshape(-1, 1)
            diff_z = (sigma_points_z[k] - z_pred).reshape(-1, 1)
            Pxz += Wc[k] * (diff_x @ diff_z.T)
        
        # 卡尔曼增益
        K = Pxz @ np.linalg.inv(Pzz)
        
        # 后验状态更新
        self.x_post = self.x + K @ (z - z_pred)
        
        # 后验协方差更新
        self.P_post = self.P - K @ Pzz @ K.T
```

### 5.2 UKF示例

```python
def ukf_demo():
    """UKF演示"""
    # 使用与EKF相同的单摆模型
    dt = 0.01
    g = 9.81
    l = 1.0
    
    def f(x, u):
        theta, omega = x
        theta_new = theta + omega * dt
        omega_new = omega - (g / l) * np.sin(theta) * dt
        return np.array([theta_new, omega_new])
    
    def h(x):
        return np.array([x[0]])
    
    Q = np.diag([0.0001, 0.001])
    R = np.array([[0.01]])
    
    # 创建UKF
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1)
    ukf.set_functions(f, h, Q, R)
    
    # 初始状态
    x0 = np.array([np.pi / 4, 0])
    ukf.x_post = x0.copy()
    ukf.P_post = np.eye(2) * 0.1
    
    # 仿真
    num_steps = 500
    x_true = np.zeros((num_steps, 2))
    z = np.zeros((num_steps, 1))
    estimates = np.zeros((num_steps, 2))
    
    x = x0.copy()
    x_true[0] = x
    
    for t in range(1, num_steps):
        # 真实状态
        x = f(x, None)
        x_true[t] = x
        
        # 观测
        z[t] = x[0] + np.random.normal(0, np.sqrt(R[0, 0]))
        
        # UKF
        ukf.predict()
        ukf.update(z[t])
        estimates[t] = ukf.x_post
    
    # 绘图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_true[:, 0], label='真实角度')
    plt.plot(z[:, 0], '.', label='观测', alpha=0.5)
    plt.plot(estimates[:, 0], label='UKF估计')
    plt.xlabel('时间步')
    plt.ylabel('角度 (rad)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x_true[:, 1], label='真实角速度')
    plt.plot(estimates[:, 1], label='UKF估计')
    plt.xlabel('时间步')
    plt.ylabel('角速度 (rad/s)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ukf_demo.png')
    print("UKF演示完成")

# ukf_demo()
```

---

## 6. 实践练习

### 练习1：卡尔曼滤波基础

```python
def exercise_kalman_basics():
    """卡尔曼滤波基础练习"""
    print("=== 卡尔曼滤波基础练习 ===")
    
    # 简单的一维系统
    F = np.array([[1]])
    H = np.array([[1]])
    Q = np.array([[0.1]])
    R = np.array([[0.5]])
    
    # 初始状态
    x0 = np.array([10])
    
    # 仿真
    model = StateSpaceModel(F, H, Q, R)
    x_true, z = model.simulate(x0, num_steps=100)
    
    # 滤波
    kf = KalmanFilter(F, H, Q, R)
    kf.x_post = np.array([0])
    kf.P_post = np.array([[5]])
    estimates = kf.filter(z)
    
    # 计算误差
    mse_kf = np.mean((estimates[:, 0] - x_true[:, 0])**2)
    mse_obs = np.mean((z[:, 0] - x_true[:, 0])**2)
    
    print(f"观测MSE: {mse_obs:.4f}")
    print(f"滤波MSE: {mse_kf:.4f}")
    print(f"误差减少: {(1 - mse_kf/mse_obs)*100:.2f}%")

# exercise_kalman_basics()
```

### 练习2：比较KF、EKF、UKF

```python
def exercise_compare_filters():
    """比较不同滤波器"""
    print("=== 滤波器比较练习 ===")
    
    # 非线性模型
    dt = 0.01
    g = 9.81
    l = 1.0
    
    def f(x, u):
        theta, omega = x
        theta_new = theta + omega * dt
        omega_new = omega - (g / l) * np.sin(theta) * dt
        return np.array([theta_new, omega_new])
    
    def h(x):
        return np.array([x[0]])
    
    def F_jacobian(x, u):
        theta, omega = x
        return np.array([
            [1, dt],
            [-(g / l) * np.cos(theta), 1]
        ])
    
    def H_jacobian(x):
        return np.array([[1, 0]])
    
    Q = np.diag([0.0001, 0.001])
    R = np.array([[0.01]])
    
    # 初始化滤波器
    x0 = np.array([np.pi / 4, 0])
    
    ekf = ExtendedKalmanFilter(2, 1)
    ekf.set_functions(f, h, F_jacobian, H_jacobian, Q, R)
    ekf.x_post = x0.copy()
    ekf.P_post = np.eye(2) * 0.1
    
    ukf = UnscentedKalmanFilter(2, 1)
    ukf.set_functions(f, h, Q, R)
    ukf.x_post = x0.copy()
    ukf.P_post = np.eye(2) * 0.1
    
    # 仿真和滤波
    num_steps = 200
    x_true = np.zeros((num_steps, 2))
    z = np.zeros((num_steps, 1))
    est_ekf = np.zeros((num_steps, 2))
    est_ukf = np.zeros((num_steps, 2))
    
    x = x0.copy()
    x_true[0] = x
    
    for t in range(1, num_steps):
        x = f(x, None)
        x_true[t] = x
        z[t] = x[0] + np.random.normal(0, np.sqrt(R[0, 0]))
        
        ekf.predict()
        ekf.update(z[t])
        est_ekf[t] = ekf.x_post
        
        ukf.predict()
        ukf.update(z[t])
        est_ukf[t] = ukf.x_post
    
    # 计算MSE
    mse_ekf_theta = np.mean((est_ekf[:, 0] - x_true[:, 0])**2)
    mse_ukf_theta = np.mean((est_ukf[:, 0] - x_true[:, 0])**2)
    
    print(f"EKF 角度MSE: {mse_ekf_theta:.6f}")
    print(f"UKF 角度MSE: {mse_ukf_theta:.6f}")

# exercise_compare_filters()
```

---

**下一节**：[视觉里程计](02-visual-odometry.md)

---

## 参考文献

1. Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems.
2. Welch, G., & Bishop, G. (2006). An Introduction to the Kalman Filter.
3. Julier, S. J., et al. (1995). A New Extension of the Kalman Filter to Nonlinear Systems.
4. Thrun, S., et al. (2005). Probabilistic Robotics.
