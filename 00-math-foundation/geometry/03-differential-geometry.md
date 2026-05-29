# 5.3 微分几何

## 目录

- [1. 曲线论](#1-曲线论)
- [2. 曲面论](#2-曲面论)
- [3. 流形](#3-流形)
- [4. 在具身智能中的应用](#4-在具身智能中的应用)
- [5. 相关论文](#5-相关论文)
- [6. 实践练习](#6-实践练习)

---

## 1. 曲线论

### 1.1 参数曲线

**定义**：

$$\mathbf{r}(t) = (x(t), y(t), z(t))$$

**切线向量**：

$$\mathbf{r}'(t) = \left(\frac{dx}{dt}, \frac{dy}{dt}, \frac{dz}{dt}\right)$$

**弧长参数**：

$$s(t) = \int_{t_0}^t \|\mathbf{r}'(\tau)\| d\tau$$

**单位切向量**：

$$\mathbf{T}(s) = \frac{d\mathbf{r}}{ds} = \frac{\mathbf{r}'(t)}{\|\mathbf{r}'(t)\|}$$

### 1.2 Frenet标架

**单位法向量**：

$$\mathbf{N}(s) = \frac{\mathbf{T}'(s)}{\|\mathbf{T}'(s)\|}$$

**副法向量**：

$$\mathbf{B}(s) = \mathbf{T}(s) \times \mathbf{N}(s)$$

**Frenet-Serret公式**：

$$\frac{d}{ds} \begin{bmatrix} \mathbf{T} \\ \mathbf{N} \\ \mathbf{B} \end{bmatrix} = \begin{bmatrix} 0 & \kappa & 0 \\ -\kappa & 0 & \tau \\ 0 & -\tau & 0 \end{bmatrix} \begin{bmatrix} \mathbf{T} \\ \mathbf{N} \\ \mathbf{B} \end{bmatrix}$$

其中 $\kappa$ 是曲率，$\tau$ 是挠率。

### 1.3 曲率与挠率

**曲率**：

$$\kappa = \|\mathbf{T}'(s)\| = \frac{\|\mathbf{r}'(t) \times \mathbf{r}''(t)\|}{\|\mathbf{r}'(t)\|^3}$$

**挠率**：

$$\tau = \frac{(\mathbf{r}'(t) \times \mathbf{r}''(t)) \cdot \mathbf{r}'''(t)}{\|\mathbf{r}'(t) \times \mathbf{r}''(t)\|^2}$$

### 1.4 曲线的局部近似

**密切平面**：由 $\mathbf{T}$ 和 $\mathbf{N}$ 张成的平面。

**曲率圆**：在密切平面内，以 $\mathbf{r}(s) + \frac{1}{\kappa} \mathbf{N}(s)$ 为圆心，$\frac{1}{\kappa}$ 为半径的圆。

---

## 2. 曲面论

### 2.1 参数曲面

**定义**：

$$\mathbf{r}(u, v) = (x(u, v), y(u, v), z(u, v))$$

**切向量**：

$$\mathbf{r}_u = \frac{\partial \mathbf{r}}{\partial u}, \quad \mathbf{r}_v = \frac{\partial \mathbf{r}}{\partial v}$$

**法向量**：

$$\mathbf{n}(u, v) = \frac{\mathbf{r}_u \times \mathbf{r}_v}{\|\mathbf{r}_u \times \mathbf{r}_v\|}$$

### 2.2 第一基本形式

**度量张量**：

$$\mathbf{I} = \begin{bmatrix} E & F \\ F & G \end{bmatrix}$$

其中：
- $E = \mathbf{r}_u \cdot \mathbf{r}_u$
- $F = \mathbf{r}_u \cdot \mathbf{r}_v$
- $G = \mathbf{r}_v \cdot \mathbf{r}_v$

**弧长元素**：

$$ds^2 = E du^2 + 2F du dv + G dv^2$$

**面积元素**：

$$dA = \sqrt{EG - F^2} du dv$$

### 2.3 第二基本形式

**曲率张量**：

$$\mathbf{II} = \begin{bmatrix} L & M \\ M & N \end{bmatrix}$$

其中：
- $L = \mathbf{r}_{uu} \cdot \mathbf{n}$
- $M = \mathbf{r}_{uv} \cdot \mathbf{n}$
- $N = \mathbf{r}_{vv} \cdot \mathbf{n}$

### 2.4 主曲率与高斯曲率

**主曲率**：二次型 $\mathbf{II}$ 的特征值。

**高斯曲率**：

$$K = \frac{LN - M^2}{EG - F^2}$$

**平均曲率**：

$$H = \frac{EN + GL - 2FM}{2(EG - F^2)}$$

### 2.5 曲面分类

**椭圆点**：$K > 0$

**双曲点**：$K < 0$

**抛物点**：$K = 0$

**极小曲面**：$H = 0$

---

## 3. 流形

### 3.1 定义

**d维流形**：一个豪斯多夫空间，每个点都有一个邻域同胚于 $\mathbb{R}^d$。

**坐标图**：$(U, \phi)$，其中 $U$ 是流形上的开集，$\phi: U \to \mathbb{R}^d$ 是同胚。

**坐标变换**：

$$\psi \circ \phi^{-1}: \phi(U \cap V) \to \psi(U \cap V)$$

### 3.2 切空间

**切向量**：在点 $p$ 处的切向量是曲线 $\gamma(t)$ 在 $t=0$ 时的速度向量，其中 $\gamma(0) = p$。

**切空间**：所有切向量构成的向量空间，记为 $T_p M$。

**切丛**：$TM = \bigcup_{p \in M} T_p M$

### 3.3 黎曼流形

**黎曼度量**：在每个切空间上定义内积 $\langle \cdot, \cdot \rangle_p$。

**测地线**：局部最短路径。

**测地方程**：

$$\frac{d^2 x^i}{dt^2} + \Gamma^i_{jk} \frac{dx^j}{dt} \frac{dx^k}{dt} = 0$$

其中 $\Gamma^i_{jk}$ 是克里斯托费尔符号。

### 3.4 联络

**仿射联络**：定义向量场的协变导数。

**克里斯托费尔符号**：

$$\Gamma^i_{jk} = \frac{1}{2} g^{il} \left(\frac{\partial g_{lj}}{\partial x^k} + \frac{\partial g_{lk}}{\partial x^j} - \frac{\partial g_{jk}}{\partial x^l}\right)$$

---

## 4. 在具身智能中的应用

### 4.1 机器人运动学

**关节空间**：机器人关节角度构成的流形。

**配置空间**：机器人末端执行器位姿构成的流形（SE(3)）。

**运动规划**：在配置空间中寻找路径。

### 4.2 状态估计

**流形上的滤波**：在非欧几里得空间上进行状态估计。

**李群滤波**：使用李群结构进行滤波。

### 4.3 三维重建

**曲面重建**：从点云重建曲面。

**网格参数化**：将曲面参数化到平面区域。

### 4.4 计算机视觉

**姿态估计**：在SO(3)上进行优化。

**光束平差法**：在流形上进行优化。

---

## 5. 相关论文

### 微分几何

1. **"Differential Geometry of Curves and Surfaces"** - Do Carmo (1976)
   - 微分几何经典教材

2. **"Riemannian Geometry"** - Petersen (2016)
   - 黎曼几何教材

### 应用相关

3. **"Geometric Control of Mechanical Systems"** - Bullo & Lewis (2004)
   - 几何控制理论

4. **"Manifold Optimization for Machine Learning"** - Absil et al. (2008)
   - 流形优化

---

## 6. 实践练习

### 练习1：曲线的切线和法线

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 参数曲线
def curve(t):
    return np.array([np.cos(t), np.sin(t), t])

# 切线向量
def tangent(t):
    return np.array([-np.sin(t), np.cos(t), 1])

# 计算Frenet标架
def frenet_frame(t):
    r = curve(t)
    r_prime = tangent(t)
    
    # 单位切向量
    T = r_prime / np.linalg.norm(r_prime)
    
    # 二阶导数
    r_double_prime = np.array([-np.cos(t), -np.sin(t), 0])
    
    # 曲率向量
    kappa_vector = r_double_prime - np.dot(r_double_prime, T) * T
    
    # 单位法向量
    N = kappa_vector / np.linalg.norm(kappa_vector)
    
    # 副法向量
    B = np.cross(T, N)
    
    return T, N, B

# 示例点
t0 = np.pi/4
T, N, B = frenet_frame(t0)
r0 = curve(t0)

print("Frenet标架:")
print(f"切向量 T: {T}")
print(f"法向量 N: {N}")
print(f"副法向量 B: {B}")

# 验证正交性
print("\n验证正交性:")
print(f"T · N = {np.dot(T, N):.6f}")
print(f"T · B = {np.dot(T, B):.6f}")
print(f"N · B = {np.dot(N, B):.6f}")

# 可视化
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲线
t = np.linspace(0, 4*np.pi, 100)
x, y, z = curve(t)
ax.plot(x, y, z, 'b-', label='Curve')

# 绘制Frenet标架
scale = 0.5
ax.quiver(r0[0], r0[1], r0[2], T[0], T[1], T[2], color='red', label='T')
ax.quiver(r0[0], r0[1], r0[2], N[0], N[1], N[2], color='green', label='N')
ax.quiver(r0[0], r0[1], r0[2], B[0], B[1], B[2], color='blue', label='B')

ax.scatter(r0[0], r0[1], r0[2], c='black', s=50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Frenet Frame')
ax.legend()
plt.show()
```

### 练习2：曲率和挠率

```python
import numpy as np

# 螺旋线
def helix(t):
    return np.array([np.cos(t), np.sin(t), t])

def helix_derivative(t, order=1):
    """计算螺旋线的各阶导数"""
    if order == 1:
        return np.array([-np.sin(t), np.cos(t), 1])
    elif order == 2:
        return np.array([-np.cos(t), -np.sin(t), 0])
    elif order == 3:
        return np.array([np.sin(t), -np.cos(t), 0])

def curvature(t):
    """计算曲率"""
    r_prime = helix_derivative(t, 1)
    r_double_prime = helix_derivative(t, 2)
    
    cross = np.cross(r_prime, r_double_prime)
    numerator = np.linalg.norm(cross)
    denominator = np.linalg.norm(r_prime)**3
    
    return numerator / denominator

def torsion(t):
    """计算挠率"""
    r_prime = helix_derivative(t, 1)
    r_double_prime = helix_derivative(t, 2)
    r_triple_prime = helix_derivative(t, 3)
    
    cross = np.cross(r_prime, r_double_prime)
    numerator = np.dot(cross, r_triple_prime)
    denominator = np.linalg.norm(cross)**2
    
    return numerator / denominator

# 计算螺旋线的曲率和挠率
t = np.linspace(0, 4*np.pi, 100)
kappa = [curvature(ti) for ti in t]
tau = [torsion(ti) for ti in t]

print("螺旋线的曲率和挠率:")
print(f"平均曲率: {np.mean(kappa):.6f}")
print(f"平均挠率: {np.mean(tau):.6f}")

# 解析解：对于螺旋线 r(t) = (cos t, sin t, t)
# 曲率 kappa = 1/2
# 挠率 tau = 1/2
print(f"\n解析解:")
print(f"曲率 kappa = 1/2 = 0.5")
print(f"挠率 tau = 1/2 = 0.5")

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, kappa, 'r-')
plt.axhline(0.5, color='k', linestyle='--', label='Analytic')
plt.xlabel('t')
plt.ylabel('Curvature')
plt.title('Curvature')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(t, tau, 'b-')
plt.axhline(0.5, color='k', linestyle='--', label='Analytic')
plt.xlabel('t')
plt.ylabel('Torsion')
plt.title('Torsion')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 练习3：曲面的第一基本形式

```python
import numpy as np

# 球面参数化
def sphere(u, v):
    return np.array([np.sin(u)*np.cos(v), 
                     np.sin(u)*np.sin(v), 
                     np.cos(u)])

def sphere_derivatives(u, v):
    """计算球面的偏导数"""
    # 一阶偏导
    r_u = np.array([np.cos(u)*np.cos(v), 
                    np.cos(u)*np.sin(v), 
                    -np.sin(u)])
    
    r_v = np.array([-np.sin(u)*np.sin(v), 
                    np.sin(u)*np.cos(v), 
                    0])
    
    # 二阶偏导
    r_uu = np.array([-np.sin(u)*np.cos(v), 
                     -np.sin(u)*np.sin(v), 
                     -np.cos(u)])
    
    r_uv = np.array([-np.cos(u)*np.sin(v), 
                     np.cos(u)*np.cos(v), 
                     0])
    
    r_vv = np.array([-np.sin(u)*np.cos(v), 
                     -np.sin(u)*np.sin(v), 
                     0])
    
    return r_u, r_v, r_uu, r_uv, r_vv

def first_fundamental_form(u, v):
    """计算第一基本形式"""
    r_u, r_v, _, _, _ = sphere_derivatives(u, v)
    
    E = np.dot(r_u, r_u)
    F = np.dot(r_u, r_v)
    G = np.dot(r_v, r_v)
    
    return E, F, G

def second_fundamental_form(u, v):
    """计算第二基本形式"""
    r_u, r_v, r_uu, r_uv, r_vv = sphere_derivatives(u, v)
    
    # 法向量
    n = np.cross(r_u, r_v)
    n = n / np.linalg.norm(n)
    
    L = np.dot(r_uu, n)
    M = np.dot(r_uv, n)
    N = np.dot(r_vv, n)
    
    return L, M, N

# 计算球面的基本形式
u0, v0 = np.pi/4, np.pi/4

E, F, G = first_fundamental_form(u0, v0)
L, M, N = second_fundamental_form(u0, v0)

print("第一基本形式系数:")
print(f"E = {E:.6f}")
print(f"F = {F:.6f}")
print(f"G = {G:.6f}")

print("\n第二基本形式系数:")
print(f"L = {L:.6f}")
print(f"M = {M:.6f}")
print(f"N = {N:.6f}")

# 计算高斯曲率和平均曲率
K = (L*N - M**2) / (E*G - F**2)
H = (E*N + G*L - 2*F*M) / (2*(E*G - F**2))

print(f"\n高斯曲率 K = {K:.6f}")
print(f"平均曲率 H = {H:.6f}")

# 解析解：单位球面的高斯曲率 K = 1，平均曲率 H = 1
print(f"\n解析解:")
print(f"高斯曲率 K = 1")
print(f"平均曲率 H = 1")
```

### 练习4：测地线

```python
import numpy as np
import matplotlib.pyplot as plt

# 球面测地线（大圆）
def spherical_geodesic(t, theta0=0, phi0=0, theta1=np.pi/2, phi1=np.pi/2):
    """
    球面上两点之间的测地线
    """
    # 将球坐标转换为笛卡尔坐标
    def spherical_to_cartesian(theta, phi):
        return np.array([np.sin(theta)*np.cos(phi),
                         np.sin(theta)*np.sin(phi),
                         np.cos(theta)])
    
    # 起点和终点
    p0 = spherical_to_cartesian(theta0, phi0)
    p1 = spherical_to_cartesian(theta1, phi1)
    
    # 计算旋转轴（垂直于p0和p1）
    axis = np.cross(p0, p1)
    axis = axis / np.linalg.norm(axis)
    
    # 计算旋转角度
    angle = np.arccos(np.dot(p0, p1))
    
    # 旋转矩阵
    def rotation_matrix(axis, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        x, y, z = axis
        
        return np.array([[c + x**2*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
                         [y*x*(1-c) + z*s, c + y**2*(1-c), y*z*(1-c) - x*s],
                         [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z**2*(1-c)]])
    
    # 参数化测地线
    R = rotation_matrix(axis, angle * t)
    point = R @ p0
    
    return point

# 绘制球面上的测地线
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制球面
u = np.linspace(0, np.pi, 50)
v = np.linspace(0, 2*np.pi, 50)
U, V = np.meshgrid(u, v)
X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)
ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.3)

# 绘制测地线
t = np.linspace(0, 1, 100)
geodesic = np.array([spherical_geodesic(ti) for ti in t])
ax.plot(geodesic[:, 0], geodesic[:, 1], geodesic[:, 2], 'r-', linewidth=2, label='Geodesic')

# 标记起点和终点
ax.scatter(1, 0, 0, c='green', s=100, label='Start')
ax.scatter(0, 1, 0, c='red', s=100, label='End')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Spherical Geodesic')
ax.legend()
plt.show()
```

### 练习5：李群SO(3)

```python
import numpy as np

# SO(3)旋转矩阵
def rotation_matrix(angle, axis):
    """
    绕任意轴旋转的旋转矩阵
    """
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    
    return np.array([[c + x**2*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
                     [y*x*(1-c) + z*s, c + y**2*(1-c), y*z*(1-c) - x*s],
                     [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z**2*(1-c)]])

# 李代数so(3)
def skew_symmetric(v):
    """
    向量到反对称矩阵的映射
    """
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]])

def exp_rotation(v):
    """
    指数映射：so(3) -> SO(3)
    """
    theta = np.linalg.norm(v)
    
    if theta < 1e-6:
        return np.eye(3) + skew_symmetric(v)
    
    axis = v / theta
    c = np.cos(theta)
    s = np.sin(theta)
    K = skew_symmetric(axis)
    
    return np.eye(3) + s*K + (1 - c)*K@K

def log_rotation(R):
    """
    对数映射：SO(3) -> so(3)
    """
    trace = np.trace(R)
    theta = np.arccos((trace - 1) / 2)
    
    if theta < 1e-6:
        # 接近单位矩阵，使用一阶近似
        return (R - R.T) / 2
    
    K = (R - R.T) / (2 * np.sin(theta))
    return theta * K

# 示例：旋转矩阵的指数和对数映射
angle = np.pi/4
axis = np.array([1, 0, 0])

# 使用Rodrigues公式
R1 = rotation_matrix(angle, axis)

# 使用指数映射
R2 = exp_rotation(angle * axis)

print("旋转矩阵:")
print(f"Rodrigues公式:\n{R1}")
print(f"指数映射:\n{R2}")
print(f"误差: {np.linalg.norm(R1 - R2):.6e}")

# 对数映射
v = log_rotation(R1)
print(f"\n对数映射结果: {v}")
print(f"原始向量: {angle * axis}")
print(f"误差: {np.linalg.norm(v - angle * axis):.6e}")

# 验证群运算
R3 = rotation_matrix(np.pi/6, np.array([0, 1, 0]))
R4 = rotation_matrix(np.pi/3, np.array([0, 0, 1]))

# 群乘法
R_product = R3 @ R4

# 验证正交性
print(f"\n验证正交性:")
print(f"R^T R = I? {np.allclose(R_product.T @ R_product, np.eye(3))}")

# 验证行列式
print(f"det(R) = {np.linalg.det(R_product):.6f}")
```

---

**几何基础部分完成！** 🎉

**数学基础模块全部完成！** 🎊