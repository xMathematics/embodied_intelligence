# 3.1 拉格朗日力学

## 目录

- [1. 拉格朗日方程](#1-拉格朗日方程)
- [2. 动能与势能](#2-动能与势能)
- [3. 广义坐标](#3-广义坐标)
- [4. 拉格朗日函数](#4-拉格朗日函数)
- [5. 约束与拉格朗日乘子](#5-约束与拉格朗日乘子)
- [6. 在具身智能中的应用](#6-在具身智能中的应用)
- [7. 相关论文](#7-相关论文)
- [8. 实践练习](#8-实践练习)

---

## 1. 拉格朗日方程

### 1.1 定义

**拉格朗日方程**是分析力学的核心方程，描述系统的运动。

**基本形式**：

$$\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}_i} \right) - \frac{\partial L}{\partial q_i} = Q_i$$

其中：
- $L = T - V$ 是拉格朗日函数
- $T$ 是动能
- $V$ 是势能
- $q_i$ 是广义坐标
- $Q_i$ 是广义力

### 1.2 无阻尼保守系统

**当系统只有保守力时**：

$$\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}_i} \right) - \frac{\partial L}{\partial q_i} = 0$$

### 1.3 与牛顿力学的关系

**拉格朗日力学 vs 牛顿力学**：

| 方面 | 牛顿力学 | 拉格朗日力学 |
|------|---------|-------------|
| 描述方式 | 力和加速度 | 能量和广义坐标 |
| 坐标系 | 笛卡尔坐标 | 任意广义坐标 |
| 约束处理 | 需要约束力 | 自动处理约束 |
| 适用范围 | 质点系 | 复杂系统 |

---

## 2. 动能与势能

### 2.1 动能

**质点的动能**：

$$T = \frac{1}{2}mv^2$$

**质点系的动能**：

$$T = \sum_{i=1}^{n} \frac{1}{2}m_i v_i^2$$

**刚体的动能**：

$$T = \frac{1}{2}Mv_c^2 + \frac{1}{2} \boldsymbol{\omega}^T \mathbf{I} \boldsymbol{\omega}$$

其中 $M$ 是总质量，$v_c$ 是质心速度，$\boldsymbol{\omega}$ 是角速度，$\mathbf{I}$ 是惯性张量。

### 2.2 势能

**保守力场中的势能**：

$$V = -\int \mathbf{F} \cdot d\mathbf{r}$$

**重力势能**：

$$V = mgh$$

**弹性势能**：

$$V = \frac{1}{2}kx^2$$

**引力势能**：

$$V = -\frac{GMm}{r}$$

---

## 3. 广义坐标

### 3.1 定义

**广义坐标**：描述系统自由度的独立参数。

**自由度**：系统可以独立变化的坐标数。

**示例**：

| 系统 | 自由度 | 广义坐标 |
|------|--------|---------|
| 单摆 | 1 | 摆角 $\theta$ |
| 双摆 | 2 | 两个摆角 $\theta_1, \theta_2$ |
| 平面机械臂（2DOF） | 2 | 关节角度 $\theta_1, \theta_2$ |

### 3.2 广义速度

**广义速度**：广义坐标对时间的导数。

$$\dot{q}_i = \frac{dq_i}{dt}$$

### 3.3 虚功原理

**虚位移**：在约束允许的情况下，系统的微小位移。

**虚功**：

$$\delta W = \sum_{i=1}^{n} \mathbf{F}_i \cdot \delta \mathbf{r}_i$$

**虚功原理**：

$$\delta W = 0$$

---

## 4. 拉格朗日函数

### 4.1 定义

**拉格朗日函数**：

$$L = T - V$$

### 4.2 拉格朗日方程的推导

**从虚功原理出发**：

$$\sum_{i=1}^{n} \left( \frac{d}{dt} \left( \frac{\partial T}{\partial \dot{q}_i} \right) - \frac{\partial T}{\partial q_i} - Q_i \right) \delta q_i = 0$$

**对于保守系统**：

$$Q_i = -\frac{\partial V}{\partial q_i}$$

**最终得到**：

$$\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}_i} \right) - \frac{\partial L}{\partial q_i} = 0$$

### 4.3 示例：单摆

**广义坐标**：$\theta$

**动能**：

$$T = \frac{1}{2}ml^2 \dot{\theta}^2$$

**势能**：

$$V = mgl(1 - \cos\theta)$$

**拉格朗日函数**：

$$L = \frac{1}{2}ml^2 \dot{\theta}^2 - mgl(1 - \cos\theta)$$

**拉格朗日方程**：

$$\frac{d}{dt} \left( ml^2 \dot{\theta} \right) - (-mgl \sin\theta) = 0$$

$$ml^2 \ddot{\theta} + mgl \sin\theta = 0$$

$$\ddot{\theta} + \frac{g}{l} \sin\theta = 0$$

---

## 5. 约束与拉格朗日乘子

### 5.1 约束分类

**完整约束**：可以表示为坐标的函数。

$$f(q_1, q_2, \ldots, q_n, t) = 0$$

**非完整约束**：包含速度项。

$$f(q_1, \ldots, q_n, \dot{q}_1, \ldots, \dot{q}_n, t) = 0$$

**定常约束**：不随时间变化。

**非定常约束**：随时间变化。

### 5.2 拉格朗日乘子法

**处理约束的方法**：

$$\frac{d}{dt} \left( \frac{\partial L}{\partial \dot{q}_i} \right) - \frac{\partial L}{\partial q_i} = \sum_{j=1}^{m} \lambda_j \frac{\partial f_j}{\partial q_i}$$

其中 $\lambda_j$ 是拉格朗日乘子，$f_j$ 是约束方程。

### 5.3 约束力

**约束力可以通过拉格朗日乘子求得**：

$$F_j = \lambda_j \frac{\partial f_j}{\partial q_i}$$

---

## 6. 在具身智能中的应用

### 6.1 机器人动力学建模

**机械臂动力学**：使用拉格朗日方法建立机器人的动力学方程。

**多体系统**：分析由多个刚体组成的系统。

### 6.2 物理仿真

**系统建模**：为物理仿真提供准确的动力学模型。

**实时仿真**：高效计算系统的运动。

### 6.3 最优控制

**最优轨迹规划**：基于拉格朗日力学的最优控制。

**能量最优控制**：最小化能量消耗。

### 6.4 状态估计

**动力学模型**：用于状态估计和滤波。

**传感器融合**：结合动力学模型进行状态估计。

---

## 7. 相关论文

### 经典文献

1. **"Lagrangian Mechanics"** - Hand & Finch (1998)
   - 拉格朗日力学教材

2. **"Classical Mechanics"** - Goldstein (2002)
   - 经典力学权威教材

### 应用相关

3. **"Robot Dynamics and Control"** - Spong et al. (2006)
   - 机器人动力学与控制

4. **"Planning Algorithms"** - LaValle (2006)
   - 运动规划算法

---

## 8. 实践练习

### 练习1：单摆的拉格朗日方程

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def pendulum_lagrangian(t, state, m, l, g):
    """
    单摆的拉格朗日方程
    state = [theta, theta_dot]
    """
    theta, theta_dot = state
    
    # 拉格朗日方程: d/dt(dL/dtheta_dot) - dL/dtheta = 0
    # L = 0.5*m*l^2*theta_dot^2 - m*g*l*(1 - cos(theta))
    # dL/dtheta_dot = m*l^2*theta_dot
    # d/dt(dL/dtheta_dot) = m*l^2*theta_ddot
    # dL/dtheta = -m*g*l*sin(theta)
    # 方程: m*l^2*theta_ddot + m*g*l*sin(theta) = 0
    # theta_ddot = -(g/l)*sin(theta)
    
    theta_ddot = -(g / l) * np.sin(theta)
    
    return [theta_dot, theta_ddot]

# 参数
m = 1.0  # 质量 (kg)
l = 1.0  # 摆长 (m)
g = 9.8  # 重力加速度 (m/s^2)

# 初始条件
theta0 = np.pi / 4  # 初始角度（45度）
theta_dot0 = 0.0  # 初始角速度

# 时间范围
t_span = [0, 10]

# 求解
sol = solve_ivp(pendulum_lagrangian, t_span, [theta0, theta_dot0], args=(m, l, g), dense_output=True)

# 计算解
t = np.linspace(0, 10, 100)
state = sol.sol(t)
theta = state[0]
theta_dot = state[1]

# 计算能量
T = 0.5 * m * l**2 * theta_dot**2
V = m * g * l * (1 - np.cos(theta))
E = T + V

print("单摆运动模拟:")
print(f"周期（小角度近似）: {2*np.pi*np.sqrt(l/g):.4f} s")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 角度和角速度
axes[0].plot(t, np.degrees(theta), 'b-', label='角度')
axes[0].plot(t, theta_dot, 'r--', label='角速度')
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('角度(°)/角速度(rad/s)')
axes[0].set_title('单摆运动')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 能量
axes[1].plot(t, T, 'b-', label='动能')
axes[1].plot(t, V, 'r--', label='势能')
axes[1].plot(t, E, 'k:', label='总能量')
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('能量 (J)')
axes[1].set_title('能量变化')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 练习2：双摆的拉格朗日方程

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def double_pendulum_lagrangian(t, state, m1, m2, l1, l2, g):
    """
    双摆的拉格朗日方程
    state = [theta1, theta1_dot, theta2, theta2_dot]
    """
    theta1, theta1_dot, theta2, theta2_dot = state
    
    # 动能
    T1 = 0.5 * m1 * l1**2 * theta1_dot**2
    T2 = 0.5 * m2 * (l1**2 * theta1_dot**2 + l2**2 * theta2_dot**2 + 
                      2 * l1 * l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2))
    T = T1 + T2
    
    # 势能
    V1 = m1 * g * l1 * (1 - np.cos(theta1))
    V2 = m2 * g * (l1 * (1 - np.cos(theta1)) + l2 * (1 - np.cos(theta2)))
    V = V1 + V2
    
    # 拉格朗日函数
    L = T - V
    
    # 计算导数
    dL_dtheta1 = m1*g*l1*np.sin(theta1) + m2*g*l1*np.sin(theta1) + \
                 m2*l1*l2*theta1_dot*theta2_dot*np.sin(theta1-theta2)
    dL_dtheta1_dot = m1*l1**2*theta1_dot + m2*l1**2*theta1_dot + \
                     m2*l1*l2*theta2_dot*np.cos(theta1-theta2)
    
    dL_dtheta2 = m2*g*l2*np.sin(theta2) - m2*l1*l2*theta1_dot*theta2_dot*np.sin(theta1-theta2)
    dL_dtheta2_dot = m2*l2**2*theta2_dot + m2*l1*l2*theta1_dot*np.cos(theta1-theta2)
    
    # 拉格朗日方程
    theta1_ddot = (dL_dtheta1 - np.cos(theta1-theta2)*dL_dtheta2/l2) / \
                  (m1*l1**2 + m2*l1**2 - m2*l1**2*np.cos(theta1-theta2)**2) * l1
    theta2_ddot = (dL_dtheta2 - np.cos(theta1-theta2)*(m1*l1**2*theta1_dot + m2*l1**2*theta1_dot)/l1) / \
                  (m2*l2**2)
    
    # 简化的数值解法
    delta = theta1 - theta2
    den = (m1 + m2) * l1 - m2 * l1 * np.cos(delta)**2
    
    theta1_ddot = (m2 * l1 * theta1_dot**2 * np.sin(delta) * np.cos(delta) +
                   m2 * g * np.sin(theta2) * np.cos(delta) +
                   m2 * l2 * theta2_dot**2 * np.sin(delta) -
                   (m1 + m2) * g * np.sin(theta1)) / den
    
    theta2_ddot = (-m2 * l2 * theta2_dot**2 * np.sin(delta) * np.cos(delta) +
                   (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                   (m1 + m2) * l1 * theta1_dot**2 * np.sin(delta) -
                   (m1 + m2) * g * np.sin(theta2)) / ((l2 / l1) * den)
    
    return [theta1_dot, theta1_ddot, theta2_dot, theta2_ddot]

# 参数
m1 = 1.0  # 质量1
m2 = 1.0  # 质量2
l1 = 1.0  # 摆长1
l2 = 1.0  # 摆长2
g = 9.8   # 重力加速度

# 初始条件
theta1_0 = np.pi / 2
theta2_0 = np.pi / 2
theta1_dot_0 = 0.0
theta2_dot_0 = 0.0

# 时间范围
t_span = [0, 20]

# 求解
sol = solve_ivp(double_pendulum_lagrangian, t_span, 
                [theta1_0, theta1_dot_0, theta2_0, theta2_dot_0], 
                args=(m1, m2, l1, l2, g), dense_output=True)

# 计算解
t = np.linspace(0, 20, 200)
state = sol.sol(t)
theta1 = state[0]
theta2 = state[2]

# 计算末端位置
x1 = l1 * np.sin(theta1)
y1 = -l1 * np.cos(theta1)
x2 = x1 + l2 * np.sin(theta2)
y2 = y1 - l2 * np.cos(theta2)

# 可视化轨迹
plt.figure(figsize=(8, 6))
plt.plot(x1, y1, 'b-', label='第一个摆球')
plt.plot(x2, y2, 'r-', label='第二个摆球')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('双摆运动轨迹')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

### 练习3：弹簧-质量系统的拉格朗日方程

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def spring_mass_lagrangian(t, state, m, k):
    """
    弹簧-质量系统的拉格朗日方程
    state = [x, x_dot]
    """
    x, x_dot = state
    
    # 动能
    T = 0.5 * m * x_dot**2
    
    # 势能
    V = 0.5 * k * x**2
    
    # 拉格朗日函数
    L = T - V
    
    # 拉格朗日方程
    # d/dt(dL/dx_dot) - dL/dx = 0
    # dL/dx_dot = m*x_dot
    # d/dt(dL/dx_dot) = m*x_ddot
    # dL/dx = -k*x
    # m*x_ddot + k*x = 0
    
    x_ddot = -k * x / m
    
    return [x_dot, x_ddot]

# 参数
m = 1.0  # 质量 (kg)
k = 1.0  # 弹簧常数 (N/m)

# 初始条件
x0 = 1.0  # 初始位移
x_dot0 = 0.0  # 初始速度

# 时间范围
t_span = [0, 10]

# 求解
sol = solve_ivp(spring_mass_lagrangian, t_span, [x0, x_dot0], args=(m, k), dense_output=True)

# 计算解
t = np.linspace(0, 10, 100)
state = sol.sol(t)
x = state[0]
x_dot = state[1]

# 计算能量
T = 0.5 * m * x_dot**2
V = 0.5 * k * x**2
E = T + V

print("弹簧-质量系统:")
print(f"角频率: {np.sqrt(k/m):.4f} rad/s")
print(f"周期: {2*np.pi*np.sqrt(m/k):.4f} s")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 位移和速度
axes[0].plot(t, x, 'b-', label='位移')
axes[0].plot(t, x_dot, 'r--', label='速度')
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('位移/速度')
axes[0].set_title('弹簧-质量系统运动')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 能量
axes[1].plot(t, T, 'b-', label='动能')
axes[1].plot(t, V, 'r--', label='势能')
axes[1].plot(t, E, 'k:', label='总能量')
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('能量 (J)')
axes[1].set_title('能量变化')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 练习4：刚体平面运动的拉格朗日方程

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def rigid_body_planar(t, state, m, I, Fx, Fy, tau):
    """
    刚体平面运动的拉格朗日方程
    state = [x, y, theta, vx, vy, omega]
    """
    x, y, theta, vx, vy, omega = state
    
    # 动能：平动 + 转动
    T = 0.5 * m * (vx**2 + vy**2) + 0.5 * I * omega**2
    
    # 势能（假设在水平面上）
    V = 0
    
    # 拉格朗日函数
    L = T - V
    
    # 广义坐标：x, y, theta
    # 广义力：Fx, Fy, tau
    
    # 拉格朗日方程
    x_ddot = Fx / m
    y_ddot = Fy / m
    theta_ddot = tau / I
    
    return [vx, vy, omega, x_ddot, y_ddot, theta_ddot]

# 参数
m = 2.0  # 质量 (kg)
I = 0.1  # 转动惯量 (kg·m²)
Fx = 1.0  # x方向力 (N)
Fy = 0.5  # y方向力 (N)
tau = 0.1  # 力矩 (N·m)

# 初始条件
x0 = 0.0
y0 = 0.0
theta0 = 0.0
vx0 = 0.0
vy0 = 0.0
omega0 = 0.0

# 时间范围
t_span = [0, 5]

# 求解
sol = solve_ivp(rigid_body_planar, t_span, 
                [x0, y0, theta0, vx0, vy0, omega0], 
                args=(m, I, Fx, Fy, tau), dense_output=True)

# 计算解
t = np.linspace(0, 5, 100)
state = sol.sol(t)
x = state[0]
y = state[1]
theta = state[2]

print("刚体平面运动:")
print(f"最终位置: ({x[-1]:.2f}, {y[-1]:.2f})")
print(f"最终角度: {np.degrees(theta[-1]):.2f}°")

# 可视化
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='轨迹')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('刚体平面运动轨迹')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
```

### 练习5：带阻尼的拉格朗日方程

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def damped_pendulum_lagrangian(t, state, m, l, g, b):
    """
    带阻尼的单摆拉格朗日方程
    state = [theta, theta_dot]
    """
    theta, theta_dot = state
    
    # 动能
    T = 0.5 * m * l**2 * theta_dot**2
    
    # 势能
    V = m * g * l * (1 - np.cos(theta))
    
    # 拉格朗日函数
    L = T - V
    
    # 广义力（阻尼力）
    Q = -b * theta_dot  # 阻尼力矩
    
    # 拉格朗日方程：d/dt(dL/dtheta_dot) - dL/dtheta = Q
    theta_ddot = -(g / l) * np.sin(theta) - (b / (m * l**2)) * theta_dot
    
    return [theta_dot, theta_ddot]

# 参数
m = 1.0  # 质量 (kg)
l = 1.0  # 摆长 (m)
g = 9.8  # 重力加速度 (m/s^2)
b = 0.5  # 阻尼系数 (N·m·s/rad)

# 初始条件
theta0 = np.pi / 4
theta_dot0 = 0.0

# 时间范围
t_span = [0, 10]

# 求解
sol = solve_ivp(damped_pendulum_lagrangian, t_span, [theta0, theta_dot0], 
                args=(m, l, g, b), dense_output=True)

# 计算解
t = np.linspace(0, 10, 100)
state = sol.sol(t)
theta = state[0]
theta_dot = state[1]

# 计算能量
T = 0.5 * m * l**2 * theta_dot**2
V = m * g * l * (1 - np.cos(theta))
E = T + V

print("带阻尼的单摆:")
print(f"阻尼比: {b/(2*np.sqrt(m*l**2 * m*g/l)):.4f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 角度和角速度
axes[0].plot(t, np.degrees(theta), 'b-', label='角度')
axes[0].plot(t, theta_dot, 'r--', label='角速度')
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('角度(°)/角速度(rad/s)')
axes[0].set_title('带阻尼单摆运动')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 能量
axes[1].plot(t, T, 'b-', label='动能')
axes[1].plot(t, V, 'r--', label='势能')
axes[1].plot(t, E, 'k:', label='总能量')
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('能量 (J)')
axes[1].set_title('能量衰减')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

**下一步**：[机器人动力学](02-robot-dynamics.md)