# 4.3 微分方程求解

## 目录

- [1. 常微分方程](#1-常微分方程)
- [2. 偏微分方程](#2-偏微分方程)
- [3. 刚性问题](#3-刚性问题)
- [4. 在具身智能中的应用](#4-在具身智能中的应用)
- [5. 相关论文](#5-相关论文)
- [6. 实践练习](#6-实践练习)

---

## 1. 常微分方程

### 1.1 问题定义

**一阶常微分方程**：

$$\frac{dy}{dt} = f(t, y)$$
$$y(t_0) = y_0$$

**高阶常微分方程**：可以转换为一阶方程组。

### 1.2 欧拉方法

**显式欧拉**：

$$y_{n+1} = y_n + h f(t_n, y_n)$$

**局部截断误差**：$O(h^2)$

**全局截断误差**：$O(h)$

**隐式欧拉**：

$$y_{n+1} = y_n + h f(t_{n+1}, y_{n+1})$$

**稳定性**：无条件稳定。

### 1.3 Runge-Kutta方法

#### 二阶Runge-Kutta（RK2）

**中点法**：

$$k_1 = f(t_n, y_n)$$
$$k_2 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_1)$$
$$y_{n+1} = y_n + h k_2$$

**局部截断误差**：$O(h^3)$

#### 四阶Runge-Kutta（RK4）

$$k_1 = f(t_n, y_n)$$
$$k_2 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_1)$$
$$k_3 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2} k_2)$$
$$k_4 = f(t_n + h, y_n + h k_3)$$
$$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

**局部截断误差**：$O(h^5)$

**全局截断误差**：$O(h^4)$

### 1.4 线性多步法

#### Adams-Bashforth方法

**两步法**：

$$y_{n+1} = y_n + \frac{h}{2}(3f(t_n, y_n) - f(t_{n-1}, y_{n-1}))$$

**三步法**：

$$y_{n+1} = y_n + \frac{h}{12}(23f(t_n, y_n) - 16f(t_{n-1}, y_{n-1}) + 5f(t_{n-2}, y_{n-2}))$$

#### Adams-Moulton方法

**隐式两步法**：

$$y_{n+1} = y_n + \frac{h}{12}(5f(t_{n+1}, y_{n+1}) + 8f(t_n, y_n) - f(t_{n-1}, y_{n-1}))$$

### 1.5 自适应步长控制

**思想**：根据局部误差调整步长。

**步骤**：
1. 用步长 $h$ 计算一次
2. 用步长 $h/2$ 计算两次
3. 比较结果，估计误差
4. 根据误差调整步长

---

## 2. 偏微分方程

### 2.1 分类

**椭圆型方程**：$\nabla^2 u = f$（拉普拉斯方程）

**抛物型方程**：$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$（热传导方程）

**双曲型方程**：$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$（波动方程）

### 2.2 有限差分法

#### 一阶导数

**向前差分**：

$$\frac{\partial u}{\partial x} \approx \frac{u_{i+1,j} - u_{i,j}}{h}$$

**向后差分**：

$$\frac{\partial u}{\partial x} \approx \frac{u_{i,j} - u_{i-1,j}}{h}$$

**中心差分**：

$$\frac{\partial u}{\partial x} \approx \frac{u_{i+1,j} - u_{i-1,j}}{2h}$$

#### 二阶导数

**中心差分**：

$$\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2}$$

### 2.3 热传导方程

**显式格式**：

$$u_{i,j+1} = u_{i,j} + \alpha \frac{\Delta t}{\Delta x^2}(u_{i+1,j} - 2u_{i,j} + u_{i-1,j})$$

**稳定性条件**：$\alpha \frac{\Delta t}{\Delta x^2} \leq \frac{1}{2}$

**隐式格式**：

$$u_{i,j+1} - \alpha \frac{\Delta t}{\Delta x^2}(u_{i+1,j+1} - 2u_{i,j+1} + u_{i-1,j+1}) = u_{i,j}$$

**无条件稳定**。

### 2.4 波动方程

**显式格式**：

$$u_{i,j+1} = 2u_{i,j} - u_{i,j-1} + c^2 \frac{\Delta t^2}{\Delta x^2}(u_{i+1,j} - 2u_{i,j} + u_{i-1,j})$$

**稳定性条件**：$c \frac{\Delta t}{\Delta x} \leq 1$

### 2.5 有限元法

**思想**：将区域划分为单元，在每个单元上用基函数近似解。

**步骤**：
1. 区域离散化
2. 构造基函数
3. 建立刚度矩阵和载荷向量
4. 求解线性方程组

---

## 3. 刚性问题

### 3.1 定义

**刚性问题**：方程中存在相差很大的时间尺度。

**例子**：

$$\frac{dy}{dt} = -1000y + 999e^{-t}$$

**解析解**：$y(t) = e^{-1000t} + e^{-t}$

### 3.2 隐式方法

**优点**：无条件稳定。

**缺点**：每步需要求解方程组。

### 3.3 Rosenbrock方法

**半隐式Runge-Kutta方法**：

$$k_1 = f(t_n, y_n)$$
$$k_2 = f(t_n + \gamma h, y_n + \gamma h k_1)$$
$$y_{n+1} = y_n + h(b_1 k_1 + b_2 k_2)$$

**稳定性**：适用于刚性问题。

### 3.4 向后差分公式（BDF）

**k阶BDF**：

$$\sum_{j=0}^{k} \alpha_j y_{n+j} = h \beta_k f(t_{n+k}, y_{n+k})$$

**稳定性**：k=1,2,...,6时稳定。

---

## 4. 在具身智能中的应用

### 4.1 机器人动力学

**牛顿-欧拉方程**：

$$M(q) \ddot{q} + C(q, \dot{q}) \dot{q} + G(q) = \tau$$

**数值积分**：使用RK4或BDF方法。

### 4.2 物理仿真

**刚体动力学**：

$$\frac{d}{dt} \begin{bmatrix} r \\ v \\ R \\ \omega \end{bmatrix} = \begin{bmatrix} v \\ M^{-1}(f - \omega \times M v) \\ R \Omega(\omega) \\ I^{-1}(\tau - \omega \times I \omega) \end{bmatrix}$$

**碰撞检测**：使用显式或隐式方法处理碰撞。

### 4.3 控制理论

**状态方程**：

$$\dot{x} = Ax + Bu$$
$$y = Cx + Du$$

**数值求解**：用于仿真和控制器设计。

### 4.4 流体仿真

**Navier-Stokes方程**：

$$\frac{\partial u}{\partial t} + u \cdot \nabla u = -\frac{1}{\rho} \nabla p + \nu \nabla^2 u$$
$$\nabla \cdot u = 0$$

**数值方法**：有限差分法、有限体积法、有限元法。

---

## 5. 相关论文

### 微分方程数值解

1. **"Numerical Solution of Ordinary Differential Equations"** - Hairer et al. (2008)
   - 常微分方程数值解权威教材

2. **"Finite Difference Methods for Ordinary and Partial Differential Equations"** - LeVeque (2007)
   - 有限差分法教材

### 应用相关

3. **"Numerical Methods for Robotics"** - Lynch & Park (2017)
   - 机器人学数值方法

4. **"Computational Fluid Dynamics"** - Anderson (2010)
   - 计算流体力学

---

## 6. 实践练习

### 练习1：欧拉方法

```python
import numpy as np
import matplotlib.pyplot as plt

def euler_method(f, t0, y0, tf, h):
    """
    显式欧拉方法
    """
    t = np.arange(t0, tf + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    
    for i in range(len(t) - 1):
        y[i+1] = y[i] + h * f(t[i], y[i])
    
    return t, y

# 测试方程: dy/dt = -y, y(0) = 1
f = lambda t, y: -y
t0, y0 = 0, np.array([1.0])
tf = 5
h = 0.1

# 欧拉方法
t, y_euler = euler_method(f, t0, y0, tf, h)

# 解析解
y_exact = np.exp(-t)

print("欧拉方法结果:")
print(f"t = {tf:.1f}, y = {y_euler[-1][0]:.6f}")
print(f"解析解: y = {y_exact[-1]:.6f}")
print(f"误差: {abs(y_euler[-1][0] - y_exact[-1]):.6f}")

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(t, y_euler, 'b--', label='Euler')
plt.plot(t, y_exact, 'r-', label='Exact')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Euler Method')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 不同步长比较
for h in [0.5, 0.1, 0.01]:
    t, y = euler_method(f, t0, y0, tf, h)
    plt.plot(t, y, label=f'h={h}')

plt.plot(t, np.exp(-t), 'k-', label='Exact')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 练习2：四阶Runge-Kutta方法

```python
import numpy as np

def rk4_method(f, t0, y0, tf, h):
    """
    四阶Runge-Kutta方法
    """
    t = np.arange(t0, tf + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    
    for i in range(len(t) - 1):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y

# 测试方程: dy/dt = y, y(0) = 1
f = lambda t, y: y
t0, y0 = 0, np.array([1.0])
tf = 5
h = 0.5

# RK4方法
t, y_rk4 = rk4_method(f, t0, y0, tf, h)

# 解析解
y_exact = np.exp(t)

print("RK4方法结果:")
print(f"t = {tf:.1f}, y = {y_rk4[-1][0]:.6f}")
print(f"解析解: y = {y_exact[-1]:.6f}")
print(f"误差: {abs(y_rk4[-1][0] - y_exact[-1]):.6f}")

# 与欧拉方法比较
t_euler, y_euler = euler_method(f, t0, y0, tf, h)

plt.figure(figsize=(10, 5))
plt.plot(t, y_euler, 'b--', label='Euler')
plt.plot(t, y_rk4, 'g--', label='RK4')
plt.plot(t, y_exact, 'r-', label='Exact')
plt.xlabel('t')
plt.ylabel('y')
plt.title('RK4 vs Euler')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 练习3：二阶常微分方程

```python
import numpy as np
import matplotlib.pyplot as plt

def solve_second_order(f, t0, y0, dy0, tf, h):
    """
    求解二阶常微分方程: d^2y/dt^2 = f(t, y, dy/dt)
    """
    # 转换为一阶方程组
    def system(t, z):
        y, dy = z
        return np.array([dy, f(t, y, dy)])
    
    return rk4_method(system, t0, np.array([y0, dy0]), tf, h)

# 单摆方程: d^2θ/dt^2 = -(g/L) sin(θ)
g = 9.8
L = 1.0
f = lambda t, theta, dtheta: -(g/L) * np.sin(theta)

# 初始条件
t0, theta0, dtheta0 = 0, np.pi/4, 0
tf = 10
h = 0.01

# 求解
t, z = solve_second_order(f, t0, theta0, dtheta0, tf, h)
theta = z[:, 0]
dtheta = z[:, 1]

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, theta, 'b-')
plt.xlabel('t')
plt.ylabel('θ')
plt.title('Pendulum Angle')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(theta, dtheta, 'r-')
plt.xlabel('θ')
plt.ylabel('dθ/dt')
plt.title('Phase Portrait')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 能量守恒验证
energy = 0.5 * L**2 * dtheta**2 + g * L * (1 - np.cos(theta))
print(f"能量变化: {np.max(energy) - np.min(energy):.6f}")
```

### 练习4：热传导方程

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def heat_equation_1d(nx, nt, alpha, L, T):
    """
    求解一维热传导方程
    """
    dx = L / (nx - 1)
    dt = T / nt
    
    # 稳定性检查
    r = alpha * dt / dx**2
    if r > 0.5:
        print(f"警告: r = {r} > 0.5，显式格式不稳定")
    
    u = np.zeros((nt + 1, nx))
    
    # 初始条件: u(0, x) = sin(πx/L)
    x = np.linspace(0, L, nx)
    u[0] = np.sin(np.pi * x / L)
    
    # 边界条件: u(t, 0) = u(t, L) = 0
    u[:, 0] = 0
    u[:, -1] = 0
    
    # 显式差分
    for n in range(nt):
        for i in range(1, nx - 1):
            u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
    
    return x, u

# 参数设置
nx = 50
nt = 1000
alpha = 1.0
L = 1.0
T = 0.5

# 求解
x, u = heat_equation_1d(nx, nt, alpha, L, T)

# 可视化
fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot(x, u[0], 'b-')
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Heat Equation')
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3)

def update(frame):
    line.set_ydata(u[frame])
    return line,

ani = FuncAnimation(fig, update, frames=range(0, nt, 10), interval=50)
plt.show()

# 解析解
def exact_solution(x, t, alpha, L):
    return np.sin(np.pi * x / L) * np.exp(-alpha * (np.pi/L)**2 * t)

# 比较数值解和解析解
t_compare = 0.5
idx = int(t_compare * nt / T)
u_exact = exact_solution(x, t_compare, alpha, L)

plt.figure(figsize=(8, 4))
plt.plot(x, u[idx], 'b--', label='Numerical')
plt.plot(x, u_exact, 'r-', label='Exact')
plt.xlabel('x')
plt.ylabel('u')
plt.title(f'T = {t_compare}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 练习5：刚性问题

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 刚性问题: dy/dt = -1000y + 999e^{-t}
def stiff_equation(t, y):
    return -1000 * y + 999 * np.exp(-t)

# 解析解
def exact_solution(t):
    return np.exp(-1000 * t) + np.exp(-t)

# 初始条件
t0, y0 = 0, np.array([0.0])
tf = 0.01

# 使用solve_ivp（自动选择方法）
result = solve_ivp(stiff_equation, [t0, tf], y0, method='RK45')

print("求解结果:")
print(f"t = {result.t[-1]:.6f}, y = {result.y[0, -1]:.6f}")
print(f"解析解: y = {exact_solution(result.t[-1]):.6f}")

# 比较不同方法
methods = ['RK45', 'RK23', 'BDF', 'Radau']
plt.figure(figsize=(10, 6))

for method in methods:
    result = solve_ivp(stiff_equation, [t0, tf], y0, method=method)
    plt.plot(result.t, result.y[0], label=method)

plt.plot(result.t, exact_solution(result.t), 'k--', label='Exact')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Stiff Problem')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 长时间积分
tf_long = 5.0
result = solve_ivp(stiff_equation, [t0, tf_long], y0, method='BDF')

plt.figure(figsize=(10, 6))
plt.plot(result.t, result.y[0], 'b-', label='BDF')
plt.plot(result.t, exact_solution(result.t), 'k--', label='Exact')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Long Time Integration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

**数值计算部分完成！** 🎉

**下一步**：[几何基础](../geometry/README.md)