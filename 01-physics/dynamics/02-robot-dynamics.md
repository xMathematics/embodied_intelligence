# 3.2 机器人动力学

## 目录

- [1. 关节空间动力学](#1-关节空间动力学)
- [2. 操作空间动力学](#2-操作空间动力学)
- [3. 递归牛顿-欧拉算法](#3-递归牛顿-欧拉算法)
- [4. 动力学方程的数值求解](#4-动力学方程的数值求解)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 关节空间动力学

### 1.1 动力学方程

**机器人动力学方程**：

$$\mathbf{M}(\mathbf{q}) \ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) \dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) = \boldsymbol{\tau}$$

其中：
- $\mathbf{M}(\mathbf{q})$：质量矩阵（惯性矩阵）
- $\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})$：科里奥利/离心力矩阵
- $\mathbf{G}(\mathbf{q})$：重力向量
- $\boldsymbol{\tau}$：关节力矩

### 1.2 质量矩阵

**定义**：

$$M_{ij}(\mathbf{q}) = \sum_{k=1}^{n} m_k \frac{\partial \mathbf{r}_k}{\partial q_i} \cdot \frac{\partial \mathbf{r}_k}{\partial q_j}$$

**性质**：
- 对称正定矩阵
- 维度为 $n \times n$（$n$ 为自由度）

### 1.3 科里奥利/离心力

**定义**：

$$\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) \dot{\mathbf{q}} = \sum_{i=1}^{n} \sum_{j=1}^{n} h_{ijk}(\mathbf{q}) \dot{q}_i \dot{q}_j$$

其中 $h_{ijk}$ 是 Christoffel 符号。

### 1.4 重力项

**定义**：

$$G_i(\mathbf{q}) = \frac{\partial V(\mathbf{q})}{\partial q_i}$$

其中 $V(\mathbf{q})$ 是系统的势能。

---

## 2. 操作空间动力学

### 2.1 定义

**操作空间**：末端执行器所在的笛卡尔空间。

**操作空间动力学方程**：

$$\Lambda(\mathbf{q}) \dot{\mathbf{v}} + \mu(\mathbf{q}, \dot{\mathbf{q}}) + \mathbf{p}(\mathbf{q}) = \mathbf{F}$$

其中：
- $\Lambda(\mathbf{q}) = \mathbf{J}(\mathbf{q}) \mathbf{M}^{-1}(\mathbf{q}) \mathbf{J}^T(\mathbf{q})$：操作空间惯性矩阵
- $\mu(\mathbf{q}, \dot{\mathbf{q}}) = \mathbf{J}(\mathbf{q}) \mathbf{M}^{-1}(\mathbf{q}) (\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) \dot{\mathbf{q}} + \mathbf{G}(\mathbf{q})) - \dot{\mathbf{J}}(\mathbf{q}) \dot{\mathbf{q}}$：科里奥利/离心力和重力项
- $\mathbf{F}$：末端执行器的力/力矩

### 2.2 关节空间与操作空间的关系

**速度关系**：

$$\mathbf{v} = \mathbf{J}(\mathbf{q}) \dot{\mathbf{q}}$$

**力关系**：

$$\boldsymbol{\tau} = \mathbf{J}^T(\mathbf{q}) \mathbf{F}$$

---

## 3. 递归牛顿-欧拉算法

### 3.1 算法概述

**递归牛顿-欧拉算法**（RNEA）是高效计算机器人动力学的方法。

**正向递归**：从基座到末端，计算各连杆的速度和加速度。

**反向递归**：从末端到基座，计算各关节的力矩。

### 3.2 正向递归

**连杆i的速度**：

$$\boldsymbol{\omega}_i = \boldsymbol{\omega}_{i-1} + \dot{q}_i \mathbf{z}_{i-1}$$

$$\mathbf{v}_i = \mathbf{v}_{i-1} + \boldsymbol{\omega}_{i-1} \times \mathbf{r}_{i-1,i}$$

**连杆i的加速度**：

$$\dot{\boldsymbol{\omega}}_i = \dot{\boldsymbol{\omega}}_{i-1} + \ddot{q}_i \mathbf{z}_{i-1} + \boldsymbol{\omega}_{i-1} \times (\dot{q}_i \mathbf{z}_{i-1})$$

$$\dot{\mathbf{v}}_i = \dot{\mathbf{v}}_{i-1} + \dot{\boldsymbol{\omega}}_{i-1} \times \mathbf{r}_{i-1,i} + \boldsymbol{\omega}_{i-1} \times (\boldsymbol{\omega}_{i-1} \times \mathbf{r}_{i-1,i})$$

### 3.3 反向递归

**连杆i的惯性力**：

$$\mathbf{f}_i = m_i \dot{\mathbf{v}}_i$$

$$\boldsymbol{\tau}_i = \mathbf{I}_i \dot{\boldsymbol{\omega}}_i + \boldsymbol{\omega}_i \times (\mathbf{I}_i \boldsymbol{\omega}_i)$$

**关节i的力矩**：

$$\tau_i = \mathbf{f}_i \cdot \mathbf{z}_{i-1} + \boldsymbol{\tau}_i \cdot \mathbf{z}_{i-1}$$

---

## 4. 动力学方程的数值求解

### 4.1 显式欧拉法

**公式**：

$$\dot{\mathbf{q}}_{k+1} = \dot{\mathbf{q}}_k + \Delta t \cdot \mathbf{M}^{-1}(\mathbf{q}_k) (\boldsymbol{\tau}_k - \mathbf{C}(\mathbf{q}_k, \dot{\mathbf{q}}_k) \dot{\mathbf{q}}_k - \mathbf{G}(\mathbf{q}_k))$$

$$\mathbf{q}_{k+1} = \mathbf{q}_k + \Delta t \cdot \dot{\mathbf{q}}_{k+1}$$

### 4.2 Runge-Kutta法

**RK4**：

$$k_1 = \mathbf{f}(\mathbf{q}_k, \dot{\mathbf{q}}_k, t_k)$$
$$k_2 = \mathbf{f}(\mathbf{q}_k + \frac{\Delta t}{2} \cdot \dot{\mathbf{q}}_k, \dot{\mathbf{q}}_k + \frac{\Delta t}{2} \cdot k_1, t_k + \frac{\Delta t}{2})$$
$$k_3 = \mathbf{f}(\mathbf{q}_k + \frac{\Delta t}{2} \cdot \dot{\mathbf{q}}_k + \frac{\Delta t^2}{4} \cdot k_1, \dot{\mathbf{q}}_k + \frac{\Delta t}{2} \cdot k_2, t_k + \frac{\Delta t}{2})$$
$$k_4 = \mathbf{f}(\mathbf{q}_k + \Delta t \cdot \dot{\mathbf{q}}_k + \frac{\Delta t^2}{2} \cdot k_2, \dot{\mathbf{q}}_k + \Delta t \cdot k_3, t_k + \Delta t)$$

$$\dot{\mathbf{q}}_{k+1} = \dot{\mathbf{q}}_k + \frac{\Delta t}{6} (k_1 + 2k_2 + 2k_3 + k_4)$$
$$\mathbf{q}_{k+1} = \mathbf{q}_k + \Delta t \cdot \dot{\mathbf{q}}_k + \frac{\Delta t^2}{6} (k_1 + k_2 + k_3)$$

### 4.3 隐式方法

**适用于刚性系统**：

$$\mathbf{q}_{k+1} = \mathbf{q}_k + \Delta t \cdot \dot{\mathbf{q}}_{k+1}$$

$$\dot{\mathbf{q}}_{k+1} = \dot{\mathbf{q}}_k + \Delta t \cdot \mathbf{M}^{-1}(\mathbf{q}_{k+1}) (\boldsymbol{\tau}_{k+1} - \mathbf{C}(\mathbf{q}_{k+1}, \dot{\mathbf{q}}_{k+1}) \dot{\mathbf{q}}_{k+1} - \mathbf{G}(\mathbf{q}_{k+1}))$$

---

## 5. 在具身智能中的应用

### 5.1 机器人控制

**计算力矩控制**：基于动力学模型的控制。

**自适应控制**：在线估计动力学参数。

### 5.2 物理仿真

**实时仿真**：高效模拟机器人运动。

**碰撞检测与响应**：处理机器人与环境的交互。

### 5.3 轨迹优化

**动力学约束**：考虑动力学约束的轨迹优化。

**能量优化**：最小化能量消耗。

### 5.4 状态估计

**扩展卡尔曼滤波**：结合动力学模型的状态估计。

**无迹卡尔曼滤波**：处理非线性系统。

---

## 6. 相关论文

### 经典文献

1. **"Robot Dynamics and Control"** - Spong et al. (2006)
   - 机器人动力学与控制

2. **"Rigid Body Dynamics Algorithms"** - Featherstone (2008)
   - 刚体动力学算法

### 应用相关

3. **"Planning Algorithms"** - LaValle (2006)
   - 运动规划算法

4. **"Applied Robotics"** - Craig (2005)
   - 机器人学导论

---

## 7. 实践练习

### 练习1：2自由度机械臂动力学

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def robot_dynamics(t, state, m1, m2, l1, l2, g):
    """
    2自由度机械臂动力学方程
    state = [q1, q2, q1_dot, q2_dot]
    """
    q1, q2, q1_dot, q2_dot = state
    
    # 质量矩阵
    M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * np.cos(q2)
    M12 = m2 * l2**2 + m2 * l1 * l2 * np.cos(q2)
    M21 = M12
    M22 = m2 * l2**2
    
    M = np.array([[M11, M12], [M21, M22]])
    
    # 科里奥利/离心力
    C1 = -m2 * l1 * l2 * np.sin(q2) * (2 * q1_dot * q2_dot + q2_dot**2)
    C2 = m2 * l1 * l2 * np.sin(q2) * q1_dot**2
    
    C = np.array([C1, C2])
    
    # 重力项
    G1 = (m1 + m2) * g * l1 * np.cos(q1) + m2 * g * l2 * np.cos(q1 + q2)
    G2 = m2 * g * l2 * np.cos(q1 + q2)
    
    G = np.array([G1, G2])
    
    # 动力学方程：M(q) * q_ddot + C(q, q_dot) + G(q) = tau
    # 假设无外力矩：tau = 0
    q_ddot = np.linalg.inv(M) @ (-C - G)
    
    return [q1_dot, q2_dot, q_ddot[0], q_ddot[1]]

# 参数
m1 = 1.0  # 连杆1质量 (kg)
m2 = 1.0  # 连杆2质量 (kg)
l1 = 1.0  # 连杆1长度 (m)
l2 = 1.0  # 连杆2长度 (m)
g = 9.8   # 重力加速度 (m/s^2)

# 初始条件
q1_0 = np.pi / 4
q2_0 = np.pi / 4
q1_dot_0 = 0.0
q2_dot_0 = 0.0

# 时间范围
t_span = [0, 5]

# 求解
sol = solve_ivp(robot_dynamics, t_span, 
                [q1_0, q2_0, q1_dot_0, q2_dot_0], 
                args=(m1, m2, l1, l2, g), dense_output=True)

# 计算解
t = np.linspace(0, 5, 100)
state = sol.sol(t)
q1 = state[0]
q2 = state[1]

# 计算末端位置
x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)

print("2自由度机械臂动力学模拟:")
print(f"初始关节角度: q1={np.degrees(q1_0):.1f}°, q2={np.degrees(q2_0):.1f}°")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 关节角度
axes[0].plot(t, np.degrees(q1), 'b-', label='q1')
axes[0].plot(t, np.degrees(q2), 'r--', label='q2')
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('角度 (°)')
axes[0].set_title('关节角度变化')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 末端轨迹
axes[1].plot(x, y, 'b-')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title('末端执行器轨迹')
axes[1].grid(True, alpha=0.3)
axes[1].axis('equal')

plt.tight_layout()
plt.show()
```

### 练习2：质量矩阵计算

```python
import numpy as np

def compute_mass_matrix_2dof(q, m1, m2, l1, l2):
    """
    计算2自由度机械臂的质量矩阵
    """
    q1, q2 = q
    
    M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * np.cos(q2)
    M12 = m2 * l2**2 + m2 * l1 * l2 * np.cos(q2)
    M21 = M12
    M22 = m2 * l2**2
    
    return np.array([[M11, M12], [M21, M22]])

# 参数
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0

# 不同关节角度下的质量矩阵
print("不同关节角度下的质量矩阵:")

# 情况1：q1=0, q2=0（伸直）
q = [0, 0]
M = compute_mass_matrix_2dof(q, m1, m2, l1, l2)
print(f"\n情况1: q1=0°, q2=0°")
print(M)

# 情况2：q1=0, q2=90°（弯曲）
q = [0, np.pi/2]
M = compute_mass_matrix_2dof(q, m1, m2, l1, l2)
print(f"\n情况2: q1=0°, q2=90°")
print(M)

# 情况3：q1=45°, q2=45°
q = [np.pi/4, np.pi/4]
M = compute_mass_matrix_2dof(q, m1, m2, l1, l2)
print(f"\n情况3: q1=45°, q2=45°")
print(M)

# 验证质量矩阵的对称性
print("\n验证质量矩阵的对称性:")
print(f"对称: {np.allclose(M, M.T)}")
print(f"正定: {np.linalg.eigvalsh(M).min() > 0}")
```

### 练习3：操作空间动力学

```python
import numpy as np

def jacobian_2dof(q, l1, l2):
    """2自由度机械臂Jacobian矩阵"""
    q1, q2 = q
    J = np.array([
        [-l1*np.sin(q1) - l2*np.sin(q1+q2), -l2*np.sin(q1+q2)],
        [l1*np.cos(q1) + l2*np.cos(q1+q2), l2*np.cos(q1+q2)],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1]
    ])
    return J[:2, :]  # 简化为2D位置

def operational_space_inertia(q, m1, m2, l1, l2):
    """计算操作空间惯性矩阵"""
    M = compute_mass_matrix_2dof(q, m1, m2, l1, l2)
    J = jacobian_2dof(q, l1, l2)
    
    # 操作空间惯性矩阵：Lambda = J * M^{-1} * J^T
    Lambda = J @ np.linalg.inv(M) @ J.T
    
    return Lambda

# 参数
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0

# 计算操作空间惯性矩阵
q = [np.pi/4, np.pi/4]
Lambda = operational_space_inertia(q, m1, m2, l1, l2)

print("操作空间惯性矩阵:")
print(Lambda)

# 计算条件数
cond_num = np.linalg.cond(Lambda)
print(f"\n条件数: {cond_num:.2f}")

# 奇异值
svd = np.linalg.svd(Lambda, compute_uv=False)
print(f"奇异值: {svd}")
```

### 练习4：递归牛顿-欧拉算法

```python
import numpy as np

def rnea_2dof(q, q_dot, q_ddot, m1, m2, l1, l2, g):
    """
    递归牛顿-欧拉算法计算关节力矩
    """
    q1, q2 = q
    q1_dot, q2_dot = q_dot
    q1_ddot, q2_ddot = q_ddot
    
    # 正向递归：计算速度和加速度
    # 连杆1
    v1 = np.array([-l1 * q1_dot * np.sin(q1), 
                   l1 * q1_dot * np.cos(q1), 
                   0])
    a1 = np.array([-l1 * (q1_ddot * np.sin(q1) + q1_dot**2 * np.cos(q1)),
                   l1 * (q1_ddot * np.cos(q1) - q1_dot**2 * np.sin(q1)),
                   0]) + np.array([0, -g, 0])
    
    # 连杆2
    v2 = v1 + np.array([-l2 * (q1_dot + q2_dot) * np.sin(q1 + q2),
                        l2 * (q1_dot + q2_dot) * np.cos(q1 + q2),
                        0])
    a2 = a1 + np.array([-l2 * ((q1_ddot + q2_ddot) * np.sin(q1 + q2) + (q1_dot + q2_dot)**2 * np.cos(q1 + q2)),
                        l2 * ((q1_ddot + q2_ddot) * np.cos(q1 + q2) - (q1_dot + q2_dot)**2 * np.sin(q1 + q2)),
                        0])
    
    # 反向递归：计算力和力矩
    # 末端力（假设为零）
    f2 = m2 * a2
    tau2 = np.cross(np.array([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2), 0]), f2)[2]
    
    f1 = m1 * a1 + f2
    tau1 = np.cross(np.array([l1 * np.cos(q1), l1 * np.sin(q1), 0]), f1)[2] + tau2
    
    return np.array([tau1, tau2])

# 参数
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
g = 9.8

# 状态
q = [np.pi/4, np.pi/4]
q_dot = [0.5, 0.3]
q_ddot = [0.1, 0.2]

# 计算关节力矩
tau = rnea_2dof(q, q_dot, q_ddot, m1, m2, l1, l2, g)

print("递归牛顿-欧拉算法:")
print(f"关节力矩: tau1={tau[0]:.4f} N·m, tau2={tau[1]:.4f} N·m")
```

### 练习5：动力学仿真

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def robot_dynamics_with_torque(t, state, m1, m2, l1, l2, g, tau):
    """
    带有力矩输入的机器人动力学
    state = [q1, q2, q1_dot, q2_dot]
    """
    q1, q2, q1_dot, q2_dot = state
    
    # 质量矩阵
    M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * np.cos(q2)
    M12 = m2 * l2**2 + m2 * l1 * l2 * np.cos(q2)
    M21 = M12
    M22 = m2 * l2**2
    
    M = np.array([[M11, M12], [M21, M22]])
    
    # 科里奥利/离心力
    C1 = -m2 * l1 * l2 * np.sin(q2) * (2 * q1_dot * q2_dot + q2_dot**2)
    C2 = m2 * l1 * l2 * np.sin(q2) * q1_dot**2
    
    C = np.array([C1, C2])
    
    # 重力项
    G1 = (m1 + m2) * g * l1 * np.cos(q1) + m2 * g * l2 * np.cos(q1 + q2)
    G2 = m2 * g * l2 * np.cos(q1 + q2)
    
    G = np.array([G1, G2])
    
    # 关节力矩（简单的PD控制）
    q_des = np.array([np.pi/2, np.pi/3])
    Kp = 10
    Kd = 2
    torque = Kp * (q_des - np.array([q1, q2])) - Kd * np.array([q1_dot, q2_dot])
    
    # 动力学方程
    q_ddot = np.linalg.inv(M) @ (torque - C - G)
    
    return [q1_dot, q2_dot, q_ddot[0], q_ddot[1]]

# 参数
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
g = 9.8

# 初始条件（远离目标位置）
q1_0 = 0.0
q2_0 = 0.0
q1_dot_0 = 0.0
q2_dot_0 = 0.0

# 时间范围
t_span = [0, 10]

# 求解
sol = solve_ivp(robot_dynamics_with_torque, t_span, 
                [q1_0, q2_0, q1_dot_0, q2_dot_0], 
                args=(m1, m2, l1, l2, g, 0), dense_output=True)

# 计算解
t = np.linspace(0, 10, 100)
state = sol.sol(t)
q1 = state[0]
q2 = state[1]

print("机器人动力学控制仿真:")
print(f"目标关节角度: q1={np.degrees(np.pi/2):.1f}°, q2={np.degrees(np.pi/3):.1f}°")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 关节角度
axes[0].plot(t, np.degrees(q1), 'b-', label='q1')
axes[0].plot(t, np.degrees(q2), 'r--', label='q2')
axes[0].axhline(np.degrees(np.pi/2), color='b', linestyle=':', label='q1目标')
axes[0].axhline(np.degrees(np.pi/3), color='r', linestyle=':', label='q2目标')
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('角度 (°)')
axes[0].set_title('关节角度跟踪')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 末端轨迹
x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
axes[1].plot(x, y, 'b-')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title('末端执行器轨迹')
axes[1].grid(True, alpha=0.3)
axes[1].axis('equal')

plt.tight_layout()
plt.show()
```

---

**动力学部分完成！** 🎉

**物理基础模块全部完成！**