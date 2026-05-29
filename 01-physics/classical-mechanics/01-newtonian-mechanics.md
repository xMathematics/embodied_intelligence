# 1.1 牛顿力学

## 目录

- [1. 牛顿三大定律](#1-牛顿三大定律)
- [2. 力与加速度](#2-力与加速度)
- [3. 动量与冲量](#3-动量与冲量)
- [4. 能量守恒](#4-能量守恒)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 牛顿三大定律

### 1.1 第一定律（惯性定律）

**陈述**：物体在没有外力作用时，将保持静止或匀速直线运动状态。

**数学表达**：

$$\text{若 } \sum \mathbf{F} = 0, \quad \text{则 } \mathbf{v} = \text{常数}$$

**物理意义**：物体具有惯性，质量是惯性的量度。

### 1.2 第二定律（运动定律）

**陈述**：物体的加速度与所受合外力成正比，与质量成反比。

**数学表达**：

$$\mathbf{F} = m\mathbf{a} = m\frac{d\mathbf{v}}{dt} = \frac{d\mathbf{p}}{dt}$$

其中 $\mathbf{p} = m\mathbf{v}$ 是动量。

**分量形式**：

$$F_x = m\frac{d^2x}{dt^2}, \quad F_y = m\frac{d^2y}{dt^2}, \quad F_z = m\frac{d^2z}{dt^2}$$

### 1.3 第三定律（作用反作用定律）

**陈述**：作用力与反作用力大小相等、方向相反、作用在同一直线上。

**数学表达**：

$$\mathbf{F}_{AB} = -\mathbf{F}_{BA}$$

**重要性质**：作用力和反作用力作用在不同物体上，不能相互抵消。

---

## 2. 力与加速度

### 2.1 常见力

**重力**：

$$\mathbf{F}_g = m\mathbf{g}$$

其中 $\mathbf{g} = (0, 0, -9.8)\ \text{m/s}^2$（地球表面）。

**摩擦力**：

$$f_k = \mu_k N \quad \text{(滑动摩擦)}$$
$$f_s \leq \mu_s N \quad \text{(静摩擦)}$$

其中 $\mu_k$ 和 $\mu_s$ 分别是动摩擦系数和静摩擦系数，$N$ 是法向力。

**弹簧力（胡克定律）**：

$$\mathbf{F} = -k\mathbf{x}$$

其中 $k$ 是弹簧常数，$\mathbf{x}$ 是位移。

**万有引力**：

$$\mathbf{F} = -G\frac{m_1 m_2}{r^2} \hat{\mathbf{r}}$$

其中 $G$ 是万有引力常数。

### 2.2 力的合成

**矢量合成**：

$$\mathbf{F}_{\text{合}} = \sum_{i=1}^{n} \mathbf{F}_i$$

**分量合成**：

$$F_{x,\text{合}} = \sum_{i=1}^{n} F_{i,x}$$
$$F_{y,\text{合}} = \sum_{i=1}^{n} F_{i,y}$$
$$F_{z,\text{合}} = \sum_{i=1}^{n} F_{i,z}$$

### 2.3 运动方程

**一维运动**：

$$m\frac{d^2x}{dt^2} = F(x, \dot{x}, t)$$

**二维运动**：

$$\begin{cases} m\frac{d^2x}{dt^2} = F_x(x, y, \dot{x}, \dot{y}, t) \\ m\frac{d^2y}{dt^2} = F_y(x, y, \dot{x}, \dot{y}, t) \end{cases}$$

---

## 3. 动量与冲量

### 3.1 动量

**定义**：

$$\mathbf{p} = m\mathbf{v}$$

**动量守恒定律**：

$$\text{若 } \sum \mathbf{F}_{\text{外}} = 0, \quad \text{则 } \sum \mathbf{p}_i = \text{常数}$$

### 3.2 冲量

**定义**：

$$\mathbf{J} = \int_{t_1}^{t_2} \mathbf{F}(t) dt$$

**冲量-动量定理**：

$$\mathbf{J} = \Delta \mathbf{p} = \mathbf{p}_2 - \mathbf{p}_1$$

### 3.3 碰撞

**弹性碰撞**：动量和动能都守恒。

**非弹性碰撞**：动量守恒，动能不守恒。

**完全非弹性碰撞**：碰撞后物体粘在一起。

**恢复系数**：

$$e = \frac{v_{2f} - v_{1f}}{v_{1i} - v_{2i}}$$

其中 $e = 1$ 为完全弹性碰撞，$e = 0$ 为完全非弹性碰撞。

---

## 4. 能量守恒

### 4.1 功

**定义**：

$$W = \int_{a}^{b} \mathbf{F} \cdot d\mathbf{r}$$

**恒力做功**：

$$W = \mathbf{F} \cdot \Delta\mathbf{r} = Fd\cos\theta$$

### 4.2 动能

**定义**：

$$K = \frac{1}{2}mv^2$$

**动能定理**：

$$W_{\text{合}} = \Delta K = K_f - K_i$$

### 4.3 势能

**重力势能**：

$$U_g = mgh$$

**弹性势能**：

$$U_s = \frac{1}{2}kx^2$$

**引力势能**：

$$U_g = -\frac{GMm}{r}$$

### 4.4 机械能守恒

**条件**：只有保守力做功。

**守恒定律**：

$$E = K + U = \text{常数}$$

**能量转换**：

$$\Delta K = -\Delta U$$

---

## 5. 在具身智能中的应用

### 5.1 机器人运动学

**关节力分析**：使用牛顿定律分析关节受力。

**动态响应**：分析机器人在力作用下的运动响应。

### 5.2 物理仿真

**刚体动力学**：模拟物体的运动和碰撞。

**接触力**：计算机器人与环境的接触力。

### 5.3 控制理论

**力控制**：基于力反馈的控制策略。

**阻抗控制**：调节机器人的刚度和阻尼。

### 5.4 传感器融合

**惯性测量**：使用加速度计和陀螺仪测量运动。

**状态估计**：结合牛顿力学模型进行状态估计。

---

## 6. 相关论文

### 经典力学

1. **"Classical Mechanics"** - Goldstein (2002)
   - 经典力学权威教材

2. **"Physics for Scientists and Engineers"** - Serway (2018)
   - 物理学入门教材

### 应用相关

3. **"Robot Dynamics and Control"** - Spong et al. (2006)
   - 机器人动力学与控制

4. **"Applied Mechanics Reviews"** - Various authors
   - 应用力学综述期刊

---

## 7. 实践练习

### 练习1：自由落体运动

```python
import numpy as np
import matplotlib.pyplot as plt

def free_fall(h0, g=9.8, dt=0.01, t_max=5):
    """
    模拟自由落体运动
    """
    t = np.arange(0, t_max, dt)
    y = h0 - 0.5 * g * t**2
    
    # 找到落地时间
    t_landing = np.sqrt(2 * h0 / g)
    mask = t <= t_landing
    
    return t[mask], y[mask]

# 初始高度
h0 = 100  # 米

# 模拟
t, y = free_fall(h0)

print(f"初始高度: {h0} 米")
print(f"落地时间: {t[-1]:.2f} 秒")
print(f"落地速度: {np.sqrt(2 * 9.8 * h0):.2f} m/s")

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(t, y, 'b-', linewidth=2)
plt.xlabel('时间 (s)')
plt.ylabel('高度 (m)')
plt.title('自由落体运动')
plt.grid(True, alpha=0.3)
plt.show()

# 不同初始高度
heights = [50, 100, 150, 200]
plt.figure(figsize=(10, 5))
for h in heights:
    t, y = free_fall(h)
    plt.plot(t, y, label=f'h0={h}m')

plt.xlabel('时间 (s)')
plt.ylabel('高度 (m)')
plt.title('不同初始高度的自由落体')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 练习2：弹簧振子

```python
import numpy as np
import matplotlib.pyplot as plt

def spring_oscillator(m, k, x0, v0=0, dt=0.01, t_max=10):
    """
    模拟弹簧振子运动（使用欧拉方法）
    """
    t = np.arange(0, t_max, dt)
    x = np.zeros(len(t))
    v = np.zeros(len(t))
    
    x[0] = x0
    v[0] = v0
    
    for i in range(len(t) - 1):
        a = -k * x[i] / m
        v[i+1] = v[i] + a * dt
        x[i+1] = x[i] + v[i+1] * dt
    
    # 计算能量
    KE = 0.5 * m * v**2
    PE = 0.5 * k * x**2
    E = KE + PE
    
    return t, x, v, KE, PE, E

# 参数
m = 1.0  # 质量 (kg)
k = 1.0  # 弹簧常数 (N/m)
x0 = 1.0  # 初始位移 (m)

# 模拟
t, x, v, KE, PE, E = spring_oscillator(m, k, x0)

print(f"弹簧振子参数:")
print(f"  质量 m = {m} kg")
print(f"  弹簧常数 k = {k} N/m")
print(f"  角频率 ω = {np.sqrt(k/m):.4f} rad/s")
print(f"  周期 T = {2*np.pi/np.sqrt(k/m):.4f} s")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 位移和速度
axes[0].plot(t, x, 'b-', label='位移')
axes[0].plot(t, v, 'r--', label='速度')
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('位移/速度')
axes[0].set_title('弹簧振子运动')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 能量
axes[1].plot(t, KE, 'b-', label='动能')
axes[1].plot(t, PE, 'r--', label='势能')
axes[1].plot(t, E, 'k:', label='总能量')
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('能量 (J)')
axes[1].set_title('能量变化')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 练习3：动量守恒与碰撞

```python
import numpy as np

def elastic_collision(m1, m2, v1i, v2i):
    """
    弹性碰撞后的速度
    """
    v1f = ((m1 - m2) * v1i + 2 * m2 * v2i) / (m1 + m2)
    v2f = ((m2 - m1) * v2i + 2 * m1 * v1i) / (m1 + m2)
    return v1f, v2f

def inelastic_collision(m1, m2, v1i, v2i):
    """
    完全非弹性碰撞后的速度
    """
    vf = (m1 * v1i + m2 * v2i) / (m1 + m2)
    return vf, vf

# 示例：两个小球碰撞
m1 = 2.0  # kg
m2 = 3.0  # kg
v1i = 5.0  # m/s
v2i = -2.0  # m/s

# 弹性碰撞
v1f_elastic, v2f_elastic = elastic_collision(m1, m2, v1i, v2i)

# 完全非弹性碰撞
v1f_inelastic, v2f_inelastic = inelastic_collision(m1, m2, v1i, v2i)

print("碰撞前:")
print(f"  球1: m={m1}kg, v={v1i}m/s")
print(f"  球2: m={m2}kg, v={v2i}m/s")
print(f"  总动量: {m1*v1i + m2*v2i:.2f} kg·m/s")
print(f"  总动能: {0.5*m1*v1i**2 + 0.5*m2*v2i**2:.2f} J")

print("\n弹性碰撞后:")
print(f"  球1: v={v1f_elastic:.2f}m/s")
print(f"  球2: v={v2f_elastic:.2f}m/s")
print(f"  总动量: {m1*v1f_elastic + m2*v2f_elastic:.2f} kg·m/s")
print(f"  总动能: {0.5*m1*v1f_elastic**2 + 0.5*m2*v2f_elastic**2:.2f} J")

print("\n完全非弹性碰撞后:")
print(f"  球1: v={v1f_inelastic:.2f}m/s")
print(f"  球2: v={v2f_inelastic:.2f}m/s")
print(f"  总动量: {m1*v1f_inelastic + m2*v2f_inelastic:.2f} kg·m/s")
print(f"  总动能: {0.5*m1*v1f_inelastic**2 + 0.5*m2*v2f_inelastic**2:.2f} J")

# 可视化碰撞过程
t = np.linspace(0, 2, 100)
x1 = np.where(t < 1, v1i * t, v1f_elastic * (t - 1))
x2 = np.where(t < 1, v2i * t + 5, v2f_elastic * (t - 1) + 3)

plt.figure(figsize=(10, 4))
plt.plot(t, x1, 'b-', label='球1')
plt.plot(t, x2, 'r--', label='球2')
plt.axvline(1, color='k', linestyle=':', label='碰撞时刻')
plt.xlabel('时间 (s)')
plt.ylabel('位置 (m)')
plt.title('弹性碰撞过程')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 练习4：抛体运动

```python
import numpy as np
import matplotlib.pyplot as plt

def projectile_motion(v0, theta, g=9.8, dt=0.01):
    """
    模拟抛体运动
    """
    # 初始速度分量
    v0x = v0 * np.cos(theta)
    v0y = v0 * np.sin(theta)
    
    # 计算飞行时间
    t_flight = 2 * v0y / g
    
    t = np.arange(0, t_flight, dt)
    
    # 位置
    x = v0x * t
    y = v0y * t - 0.5 * g * t**2
    
    # 速度
    vx = np.ones(len(t)) * v0x
    vy = v0y - g * t
    
    return t, x, y, vx, vy

# 参数
v0 = 20  # m/s
theta = np.pi / 4  # 45度

# 模拟
t, x, y, vx, vy = projectile_motion(v0, theta)

print(f"抛体运动参数:")
print(f"  初速度: {v0} m/s")
print(f"  抛射角: {np.degrees(theta):.1f}°")
print(f"  飞行时间: {t[-1]:.2f} s")
print(f"  水平射程: {x[-1]:.2f} m")
print(f"  最大高度: {np.max(y):.2f} m")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 轨迹
axes[0].plot(x, y, 'b-', linewidth=2)
axes[0].set_xlabel('水平距离 (m)')
axes[0].set_ylabel('高度 (m)')
axes[0].set_title('抛体运动轨迹')
axes[0].grid(True, alpha=0.3)
axes[0].set_aspect('equal')

# 速度分量
axes[1].plot(t, vx, 'b-', label='水平速度')
axes[1].plot(t, vy, 'r--', label='竖直速度')
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('速度 (m/s)')
axes[1].set_title('速度变化')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 不同角度的轨迹
angles = [np.pi/6, np.pi/4, np.pi/3]
plt.figure(figsize=(10, 5))
for theta in angles:
    t, x, y, _, _ = projectile_motion(v0, theta)
    plt.plot(x, y, label=f'{np.degrees(theta):.0f}°')

plt.xlabel('水平距离 (m)')
plt.ylabel('高度 (m)')
plt.title('不同抛射角的轨迹')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 练习5：牛顿第二定律的数值求解

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def newton_law(t, state, m, k, b):
    """
    牛顿第二定律：m*d²x/dt² = -kx - b*dx/dt
    """
    x, v = state
    dxdt = v
    dvdt = (-k * x - b * v) / m
    return [dxdt, dvdt]

# 参数
m = 1.0  # kg
k = 1.0  # N/m
b = 0.1  # N·s/m (阻尼系数)

# 初始条件
x0 = 1.0
v0 = 0.0
initial_state = [x0, v0]

# 时间范围
t_span = [0, 20]

# 求解
sol = solve_ivp(newton_law, t_span, initial_state, args=(m, k, b), dense_output=True)

# 计算解
t = np.linspace(0, 20, 100)
y = sol.sol(t)
x = y[0]
v = y[1]

# 计算能量
KE = 0.5 * m * v**2
PE = 0.5 * k * x**2
E = KE + PE

print(f"阻尼振子参数:")
print(f"  质量 m = {m} kg")
print(f"  弹簧常数 k = {k} N/m")
print(f"  阻尼系数 b = {b} N·s/m")
print(f"  阻尼比 ζ = {b/(2*np.sqrt(m*k)):.4f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 位移和速度
axes[0].plot(t, x, 'b-', label='位移')
axes[0].plot(t, v, 'r--', label='速度')
axes[0].set_xlabel('时间 (s)')
axes[0].set_ylabel('位移/速度')
axes[0].set_title('阻尼振子运动')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 能量
axes[1].plot(t, KE, 'b-', label='动能')
axes[1].plot(t, PE, 'r--', label='势能')
axes[1].plot(t, E, 'k:', label='总能量')
axes[1].set_xlabel('时间 (s)')
axes[1].set_ylabel('能量 (J)')
axes[1].set_title('能量变化')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

**下一步**：[刚体运动](1.2-rigid-body-motion.md)