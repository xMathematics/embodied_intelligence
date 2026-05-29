# 1.2 刚体运动

## 目录

- [1. 刚体的平动与转动](#1-刚体的平动与转动)
- [2. 角动量](#2-角动量)
- [3. 惯性张量](#3-惯性张量)
- [4. 刚体动力学方程](#4-刚体动力学方程)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 刚体的平动与转动

### 1.1 刚体定义

**刚体**：物体内任意两点之间的距离保持不变。

**运动类型**：
- **平动**：刚体上所有点的运动轨迹平行
- **转动**：刚体绕某一固定轴旋转
- **平面运动**：平动与转动的组合

### 1.2 平动

**质心运动定理**：

$$M\ddot{\mathbf{r}}_c = \sum \mathbf{F}_{\text{外}}$$

其中 $M$ 是总质量，$\mathbf{r}_c$ 是质心位置。

**质心坐标**：

$$\mathbf{r}_c = \frac{1}{M} \sum_{i=1}^{n} m_i \mathbf{r}_i$$

### 1.3 转动

**角位移**：

$$\Delta\theta = \theta_f - \theta_i$$

**角速度**：

$$\omega = \frac{d\theta}{dt}$$

**角加速度**：

$$\alpha = \frac{d\omega}{dt} = \frac{d^2\theta}{dt^2}$$

### 1.4 线速度与角速度的关系

**刚体上一点的速度**：

$$\mathbf{v} = \mathbf{v}_c + \boldsymbol{\omega} \times (\mathbf{r} - \mathbf{r}_c)$$

**刚体上一点的加速度**：

$$\mathbf{a} = \mathbf{a}_c + \boldsymbol{\alpha} \times (\mathbf{r} - \mathbf{r}_c) + \boldsymbol{\omega} \times (\boldsymbol{\omega} \times (\mathbf{r} - \mathbf{r}_c))$$

---

## 2. 角动量

### 2.1 质点的角动量

**定义**：

$$\mathbf{L} = \mathbf{r} \times \mathbf{p} = \mathbf{r} \times (m\mathbf{v})$$

**分量形式**：

$$L_x = yp_z - zp_y$$
$$L_y = zp_x - xp_z$$
$$L_z = xp_y - yp_x$$

### 2.2 刚体的角动量

**绕固定轴的角动量**：

$$L = I\omega$$

其中 $I$ 是转动惯量。

**一般情况**：

$$\mathbf{L} = \mathbf{I} \cdot \boldsymbol{\omega}$$

其中 $\mathbf{I}$ 是惯性张量。

### 2.3 角动量守恒

**定律**：

$$\text{若 } \sum \boldsymbol{\tau}_{\text{外}} = 0, \quad \text{则 } \mathbf{L} = \text{常数}$$

**应用**：
- 陀螺效应
- 花样滑冰运动员旋转

### 2.4 力矩

**定义**：

$$\boldsymbol{\tau} = \mathbf{r} \times \mathbf{F}$$

**转动定律**：

$$\boldsymbol{\tau} = \frac{d\mathbf{L}}{dt}$$

---

## 3. 惯性张量

### 3.1 定义

**惯性张量**：

$$\mathbf{I} = \begin{bmatrix} I_{xx} & -I_{xy} & -I_{xz} \\ -I_{yx} & I_{yy} & -I_{yz} \\ -I_{zx} & -I_{zy} & I_{zz} \end{bmatrix}$$

**分量**：

$$I_{xx} = \int (y^2 + z^2) dm$$
$$I_{yy} = \int (x^2 + z^2) dm$$
$$I_{zz} = \int (x^2 + y^2) dm$$
$$I_{xy} = I_{yx} = \int xy\ dm$$
$$I_{xz} = I_{zx} = \int xz\ dm$$
$$I_{yz} = I_{zy} = \int yz\ dm$$

### 3.2 转动惯量

**绕某轴的转动惯量**：

$$I = \mathbf{n}^T \mathbf{I} \mathbf{n}$$

其中 $\mathbf{n}$ 是轴的单位向量。

### 3.3 常见刚体的转动惯量

| 刚体类型 | 转轴位置 | 转动惯量 |
|---------|---------|---------|
| 细杆（长度L） | 过一端 | $I = \frac{1}{3}ML^2$ |
| 细杆（长度L） | 过质心 | $I = \frac{1}{12}ML^2$ |
| 圆盘（半径R） | 过中心垂直盘面 | $I = \frac{1}{2}MR^2$ |
| 球壳（半径R） | 过球心 | $I = \frac{2}{3}MR^2$ |
| 实心球（半径R） | 过球心 | $I = \frac{2}{5}MR^2$ |

### 3.4 平行轴定理

**定理**：

$$I = I_c + Md^2$$

其中 $I_c$ 是绕质心轴的转动惯量，$d$ 是两轴之间的距离。

---

## 4. 刚体动力学方程

### 4.1 平动方程

$$M\ddot{\mathbf{r}}_c = \sum \mathbf{F}_{\text{外}}$$

### 4.2 转动方程

**在惯性系中**：

$$\boldsymbol{\tau} = \frac{d}{dt}(\mathbf{I} \cdot \boldsymbol{\omega}) = \mathbf{I} \cdot \boldsymbol{\alpha} + \boldsymbol{\omega} \times (\mathbf{I} \cdot \boldsymbol{\omega})$$

**在体坐标系中**：

$$\boldsymbol{\tau} = \mathbf{I} \cdot \dot{\boldsymbol{\omega}} + \boldsymbol{\omega} \times (\mathbf{I} \cdot \boldsymbol{\omega})$$

这就是 **Euler动力学方程**。

### 4.3 Euler方程的分量形式

设 $\mathbf{I} = \text{diag}(I_1, I_2, I_3)$（主轴坐标系），则：

$$\tau_1 = I_1 \dot{\omega}_1 + (I_3 - I_2)\omega_2\omega_3$$
$$\tau_2 = I_2 \dot{\omega}_2 + (I_1 - I_3)\omega_1\omega_3$$
$$\tau_3 = I_3 \dot{\omega}_3 + (I_2 - I_1)\omega_1\omega_2$$

### 4.4 定点转动

**陀螺运动**：

$$\dot{\boldsymbol{\omega}} = \mathbf{I}^{-1} (\boldsymbol{\tau} - \boldsymbol{\omega} \times (\mathbf{I} \cdot \boldsymbol{\omega}))$$

**章动**：自转轴的摆动

**进动**：自转轴绕固定轴的旋转

---

## 5. 在具身智能中的应用

### 5.1 机器人动力学

**多体动力学**：分析由多个刚体组成的机器人系统。

**递归牛顿-欧拉算法**：高效计算机器人动力学。

### 5.2 物理仿真

**刚体仿真**：模拟机器人的运动。

**碰撞检测**：检测刚体之间的碰撞。

### 5.3 姿态控制

**姿态估计**：使用陀螺仪和加速度计估计刚体姿态。

**姿态控制**：控制刚体的旋转。

### 5.4 机械臂设计

**惯性匹配**：设计机械臂时考虑惯性张量。

**动力学优化**：优化机械臂的动态性能。

---

## 6. 相关论文

### 刚体力学

1. **"Classical Mechanics"** - Goldstein (2002)
   - 经典力学权威教材

2. **"Rigid Body Dynamics for Beginners"** - Murray et al. (1994)
   - 刚体动力学入门

### 应用相关

3. **"Robot Dynamics and Control"** - Spong et al. (2006)
   - 机器人动力学与控制

4. **"Featherstone's Rigid Body Dynamics Algorithms"** - Featherstone (2008)
   - 刚体动力学算法

---

## 7. 实践练习

### 练习1：刚体的平动与转动

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rigid_body_motion(t, r0, v0, omega, points):
    """
    计算刚体上各点的位置
    """
    # 质心位置
    r_c = r0 + v0 * t
    
    # 旋转矩阵（绕z轴）
    theta = omega[2] * t
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    
    # 刚体上各点的位置
    positions = []
    for p in points:
        # 相对于质心的位置
        p_rel = p - np.mean(points, axis=0)
        # 旋转后的位置
        p_rot = R @ p_rel
        # 加上质心位置
        p_final = r_c + p_rot
        positions.append(p_final)
    
    return np.array(positions)

# 初始条件
r0 = np.array([0, 0, 0])
v0 = np.array([1, 0.5, 0])
omega = np.array([0, 0, 2])  # 绕z轴旋转

# 刚体上的点（一个正方形）
points = np.array([[-1, -1, 0],
                   [1, -1, 0],
                   [1, 1, 0],
                   [-1, 1, 0]])

# 计算不同时刻的位置
times = [0, 0.5, 1.0, 1.5, 2.0]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for t in times:
    pos = rigid_body_motion(t, r0, v0, omega, points)
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'o-', label=f't={t}s')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('刚体的平动与转动')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### 练习2：转动惯量计算

```python
import numpy as np

def moment_of_inertia_cylinder(mass, radius, length):
    """
    计算圆柱体的惯性张量
    """
    # 绕中心轴（z轴）的转动惯量
    I_zz = 0.5 * mass * radius**2
    
    # 绕垂直于中心轴的轴（x或y轴）的转动惯量
    I_xx = I_yy = (1/12) * mass * length**2 + (1/4) * mass * radius**2
    
    # 惯性张量（主轴坐标系）
    I = np.diag([I_xx, I_yy, I_zz])
    
    return I

def moment_of_inertia_sphere(mass, radius):
    """
    计算球体的惯性张量
    """
    I = (2/5) * mass * radius**2
    return np.diag([I, I, I])

# 示例：圆柱体
m_cyl = 2.0  # kg
r_cyl = 0.1  # m
l_cyl = 0.5  # m

I_cyl = moment_of_inertia_cylinder(m_cyl, r_cyl, l_cyl)
print("圆柱体的惯性张量:")
print(I_cyl)

# 示例：球体
m_sphere = 1.0  # kg
r_sphere = 0.1  # m

I_sphere = moment_of_inertia_sphere(m_sphere, r_sphere)
print("\n球体的惯性张量:")
print(I_sphere)

# 平行轴定理示例
# 圆柱体绕一端的转动惯量
d = l_cyl / 2  # 质心到端点的距离
I_xx_end = I_cyl[0, 0] + m_cyl * d**2
print(f"\n圆柱体绕一端的转动惯量: {I_xx_end:.6f} kg·m²")

# 验证：直接计算
I_xx_end_direct = (1/3) * m_cyl * l_cyl**2 + (1/4) * m_cyl * r_cyl**2
print(f"直接计算: {I_xx_end_direct:.6f} kg·m²")
```

### 练习3：角动量守恒

```python
import numpy as np
import matplotlib.pyplot as plt

def angular_momentum_conservation(I_initial, omega_initial, I_final):
    """
    角动量守恒：I_initial * omega_initial = I_final * omega_final
    """
    L_initial = I_initial * omega_initial
    omega_final = L_initial / I_final
    return omega_final, L_initial

# 示例：花样滑冰运动员
I_initial = 2.0  # kg·m²（伸开手臂）
omega_initial = 1.0  # rad/s
I_final = 1.0  # kg·m²（收拢手臂）

omega_final, L = angular_momentum_conservation(I_initial, omega_initial, I_final)

print("角动量守恒示例:")
print(f"初始转动惯量: {I_initial} kg·m²")
print(f"初始角速度: {omega_initial} rad/s")
print(f"初始角动量: {L} kg·m²/s")
print(f"\n最终转动惯量: {I_final} kg·m²")
print(f"最终角速度: {omega_final} rad/s")
print(f"最终角动量: {L} kg·m²/s")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 转动惯量和角速度
axes[0].bar(['初始', '最终'], [I_initial, I_final], label='转动惯量', alpha=0.7)
axes[0].bar(['初始', '最终'], [omega_initial, omega_final], label='角速度', alpha=0.7)
axes[0].set_ylabel('数值')
axes[0].set_title('转动惯量与角速度')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 角动量
axes[1].bar(['初始', '最终'], [L, L], color='green')
axes[1].set_ylabel('角动量 (kg·m²/s)')
axes[1].set_title('角动量守恒')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 练习4：Euler动力学方程

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def euler_equations(t, state, I):
    """
    Euler动力学方程
    state = [omega_x, omega_y, omega_z]
    """
    omega = state
    I1, I2, I3 = I
    
    # Euler方程
    domega1 = ((I2 - I3) / I1) * omega[1] * omega[2]
    domega2 = ((I3 - I1) / I2) * omega[0] * omega[2]
    domega3 = ((I1 - I2) / I3) * omega[0] * omega[1]
    
    return [domega1, domega2, domega3]

# 参数：不对称陀螺
I = [1.0, 2.0, 3.0]  # 三个主轴的转动惯量

# 初始角速度
omega0 = [1.0, 0.5, 0.3]

# 时间范围
t_span = [0, 10]

# 求解
sol = solve_ivp(euler_equations, t_span, omega0, args=(I,), dense_output=True)

# 计算解
t = np.linspace(0, 10, 1000)
omega = sol.sol(t)

print("Euler动力学方程解:")
print(f"初始角速度: {omega0}")
print(f"最终角速度: {omega[:, -1]}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 角速度分量
axes[0].plot(t, omega[0], 'r-', label='ωx')
axes[0].plot(t, omega[1], 'g-', label='ωy')
axes[0].plot(t, omega[2], 'b-', label='ωz')
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('角速度 (rad/s)')
axes[0].set_title('角速度变化')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 角动量
L = I[0] * omega[0]**2 + I[1] * omega[1]**2 + I[2] * omega[2]**2
axes[1].plot(t, L, 'k-')
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('角动量大小 (kg·m²/s)')
axes[1].set_title('角动量守恒')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 练习5：陀螺进动

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gyroscope(t, state, I, L, g, l):
    """
    陀螺进动方程
    state = [theta, phi, psi]
    """
    theta, phi, psi = state
    
    # 简化模型：快速旋转的陀螺
    # 角动量 L = I3 * omega3
    dtheta = 0  # 假设章动角恒定
    dphi = L / (I * np.sin(theta))  # 进动角速度
    dpsi = (L * np.cos(theta) - m * g * l * np.sin(theta)) / I  # 自转角速度
    
    return [dtheta, dphi, dpsi]

# 参数
I = 0.01  # kg·m²（绕自转轴的转动惯量）
L = 1.0   # kg·m²/s（角动量）
g = 9.8   # m/s²
l = 0.5   # m（质心到支点的距离）
m = 1.0   # kg

# 初始条件
theta0 = np.pi / 4  # 章动角
phi0 = 0  # 进动角
psi0 = 0  # 自转角

# 时间范围
t_span = [0, 10]

# 求解
sol = solve_ivp(gyroscope, t_span, [theta0, phi0, psi0], args=(I, L, g, l), dense_output=True)

# 计算解
t = np.linspace(0, 10, 100)
state = sol.sol(t)
theta = state[0]
phi = state[1]
psi = state[2]

# 计算自转轴的位置
x = l * np.sin(theta) * np.cos(phi)
y = l * np.sin(theta) * np.sin(phi)
z = l * np.cos(theta)

print("陀螺进动示例:")
print(f"章动角: {np.degrees(theta0):.1f}°")
print(f"进动角速度: {L/(I*np.sin(theta0)):.2f} rad/s")

# 可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 支点
ax.scatter(0, 0, 0, c='red', s=100, label='支点')

# 陀螺轨迹
ax.plot(x, y, z, 'b-', label='自转轴轨迹')

# 初始位置
ax.quiver(0, 0, 0, x[0], y[0], z[0], color='green', label='初始轴')

# 最终位置
ax.quiver(0, 0, 0, x[-1], y[-1], z[-1], color='blue', label='最终轴')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('陀螺进动')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

---

**经典力学部分完成！** 🎉

**下一步**：[运动学](../kinematics/README.md)