
# 2.3 向量微积分

## 目录

- [1. 向量场](#1-向量场)
- [2. 梯度、散度、旋度](#2-梯度散度旋度)
- [3. 线积分](#3-线积分)
- [4. 面积分](#4-面积分)
- [5. 重要定理](#5-重要定理)
- [6. 在具身智能中的应用](#6-在具身智能中的应用)
- [7. 相关论文](#7-相关论文)
- [8. 实践练习](#8-实践练习)

---

## 1. 向量场

### 1.1 定义

**向量场**是在空间每一点赋予一个向量的函数：

$$\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$$

**2D向量场**：

$$\mathbf{F}(x, y) = P(x, y)\mathbf{i} + Q(x, y)\mathbf{j}$$

**3D向量场**：

$$\mathbf{F}(x, y, z) = P(x, y, z)\mathbf{i} + Q(x, y, z)\mathbf{j} + R(x, y, z)\mathbf{k}$$

### 1.2 物理实例

- **速度场**：流体中每一点的速度
- **力场**：重力场、电场、磁场
- **梯度场**：标量场的梯度

### 1.3 向量场的可视化

**箭头图**：用箭头表示向量的方向和大小

**流线**：曲线每点的切线方向与向量场方向一致

---

## 2. 梯度、散度、旋度

### 2.1 梯度（Gradient）

对于标量场 $f(x, y, z)$：

$$\nabla f = \frac{\partial f}{\partial x}\mathbf{i} + \frac{\partial f}{\partial y}\mathbf{j} + \frac{\partial f}{\partial z}\mathbf{k}$$

**性质**：
- 指向函数增长最快的方向
- 垂直于等值面
- 模表示最大变化率

### 2.2 散度（Divergence）

对于向量场 $\mathbf{F} = P\mathbf{i} + Q\mathbf{j} + R\mathbf{k}$：

$$\nabla \cdot \mathbf{F} = \frac{\partial P}{\partial x} + \frac{\partial Q}{\partial y} + \frac{\partial R}{\partial z}$$

**物理意义**：
- $\nabla \cdot \mathbf{F} > 0$：源（发散）
- $\nabla \cdot \mathbf{F} < 0$：汇（汇聚）
- $\nabla \cdot \mathbf{F} = 0$：无源场

### 2.3 旋度（Curl）

$$\nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ P & Q & R \end{vmatrix}$$

展开：

$$\nabla \times \mathbf{F} = \left(\frac{\partial R}{\partial y} - \frac{\partial Q}{\partial z}\right)\mathbf{i} + \left(\frac{\partial P}{\partial z} - \frac{\partial R}{\partial x}\right)\mathbf{j} + \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right)\mathbf{k}$$

**物理意义**：
- 表示向量场的旋转程度
- 方向是旋转轴方向（右手定则）
- 模是旋转强度

### 2.4 拉普拉斯算子

**标量场的拉普拉斯**：

$$\nabla^2 f = \nabla \cdot (\nabla f) = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}$$

**向量场的拉普拉斯**：

$$\nabla^2 \mathbf{F} = \nabla(\nabla \cdot \mathbf{F}) - \nabla \times (\nabla \times \mathbf{F})$$

### 2.5 重要恒等式

$$\nabla \times (\nabla f) = \mathbf{0}$$

$$\nabla \cdot (\nabla \times \mathbf{F}) = 0$$

$$\nabla \cdot (f\mathbf{F}) = f(\nabla \cdot \mathbf{F}) + (\nabla f) \cdot \mathbf{F}$$

$$\nabla \times (f\mathbf{F}) = f(\nabla \times \mathbf{F}) + (\nabla f) \times \mathbf{F}$$

---

## 3. 线积分

### 3.1 定义

**第一类线积分**（对弧长）：

$$\int_C f(x, y) ds = \int_a^b f(x(t), y(t)) \sqrt{\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2} dt$$

**第二类线积分**（对坐标）：

$$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_C P dx + Q dy + R dz$$

### 3.2 参数化计算

若曲线 $C$ 参数化为 $\mathbf{r}(t)$，$t \in [a, b]$：

$$\int_C \mathbf{F} \cdot d\mathbf{r} = \int_a^b \mathbf{F}(\mathbf{r}(t)) \cdot \mathbf{r}'(t) dt$$

### 3.3 物理应用

**功的计算**：

$$W = \int_C \mathbf{F} \cdot d\mathbf{r}$$

**环流量**：

$$\Gamma = \oint_C \mathbf{F} \cdot d\mathbf{r}$$

### 3.4 保守场

**定义**：若 $\mathbf{F} = \nabla f$，则 $\mathbf{F}$ 是保守场。

**性质**：
- 线积分与路径无关
- 沿闭合曲线的积分为零
- 旋度为零：$\nabla \times \mathbf{F} = \mathbf{0}$

---

## 4. 面积分

### 4.1 定义

**第一类面积分**（对面积）：

$$\iint_S f(x, y, z) dS$$

**第二类面积分**（通量）：

$$\iint_S \mathbf{F} \cdot \mathbf{n} dS$$

其中 $\mathbf{n}$ 是单位法向量。

### 4.2 参数化计算

若曲面 $S$ 参数化为 $\mathbf{r}(u, v)$：

$$\iint_S \mathbf{F} \cdot d\mathbf{S} = \iint_D \mathbf{F}(\mathbf{r}(u, v)) \cdot (\mathbf{r}_u \times \mathbf{r}_v) du dv$$

### 4.3 物理应用

**通量**：

$$\Phi = \iint_S \mathbf{F} \cdot \mathbf{n} dS$$

表示通过曲面的流量。

---

## 5. 重要定理

### 5.1 格林定理（Green's Theorem）

**连接线积分与二重积分**：

$$\oint_C P dx + Q dy = \iint_D \left(\frac{\partial Q}{\partial x} - \frac{\partial P}{\partial y}\right) dA$$

其中 $C$ 是区域 $D$ 的边界曲线（逆时针方向）。

### 5.2 斯托克斯定理（Stokes' Theorem）

**连接线积分与面积分**：

$$\oint_C \mathbf{F} \cdot d\mathbf{r} = \iint_S (\nabla \times \mathbf{F}) \cdot \mathbf{n} dS$$

其中 $C$ 是曲面 $S$ 的边界曲线。

**物理意义**：环流量等于旋度的通量。

### 5.3 高斯定理（散度定理）

**连接面积分与体积分**：

$$\iint_S \mathbf{F} \cdot \mathbf{n} dS = \iiint_V (\nabla \cdot \mathbf{F}) dV$$

其中 $S$ 是体积 $V$ 的边界曲面。

**物理意义**：通过闭合曲面的通量等于内部源的总量。

### 5.4 定理之间的关系

```
格林定理（2D）
    ↓ 推广
斯托克斯定理（3D曲面）
    ↓ 推广
高斯定理（3D体积）
```

---

## 6. 在具身智能中的应用

### 6.1 流体力学仿真

**纳维-斯托克斯方程**：

$$\rho\left(\frac{\partial \mathbf{v}}{\partial t} + \mathbf{v} \cdot \nabla \mathbf{v}\right) = -\nabla p + \mu\nabla^2\mathbf{v} + \mathbf{f}$$

### 6.2 电磁场计算

**麦克斯韦方程组**：

$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}$$

$$\nabla \cdot \mathbf{B} = 0$$

$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}$$

$$\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t}$$

### 6.3 势能场方法

**机器人导航**：

$$\mathbf{F}_{total} = -\nabla U_{attract} - \nabla U_{repulse}$$

### 6.4 热传导

**热方程**：

$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u$$

---

## 7. 相关论文

### 数学物理

1. **"Vector Calculus"** - Marsden & Tromba
   - 向量微积分经典教材

2. **"Mathematical Methods for Physicists"** - Arfken & Weber
   - 数学物理方法

### 应用相关

3. **"Computational Fluid Dynamics"** - Ferziger & Perić
   - 计算流体力学

4. **"Potential Fields for Robot Navigation"** - Ge & Lewis (1994)
   - 势能场在机器人导航中的应用

---

## 8. 实践练习

### 练习1：梯度、散度、旋度计算

```python
import numpy as np
import sympy as sp

# 定义符号
x, y, z = sp.symbols('x y z')

# 定义标量场
f = x**2 * y + y**2 * z + z**2 * x

# 定义向量场
P = x**2 + y*z
Q = y**2 + x*z
R = z**2 + x*y

# 计算梯度
grad_f = [sp.diff(f, var) for var in [x, y, z]]
print(f"梯度 ∇f = {grad_f}")

# 计算散度
div_F = sp.diff(P, x) + sp.diff(Q, y) + sp.diff(R, z)
print(f"散度 ∇·F = {div_F}")

# 计算旋度
curl_F_x = sp.diff(R, y) - sp.diff(Q, z)
curl_F_y = sp.diff(P, z) - sp.diff(R, x)
curl_F_z = sp.diff(Q, x) - sp.diff(P, y)
print(f"旋度 ∇×F = [{curl_F_x}, {curl_F_y}, {curl_F_z}]")

# 验证恒等式
print(f"\n验证 ∇·(∇×F) = 0: {sp.simplify(sp.diff(curl_F_x, x) + sp.diff(curl_F_y, y) + sp.diff(curl_F_z, z)) == 0}")
```

### 练习2：线积分计算

```python
import sympy as sp

# 定义参数
t = sp.Symbol('t')

# 定义向量场
def F(x, y, z):
    return [y*z, x*z, x*y]

# 定义曲线（螺旋线）
x_t = sp.cos(t)
y_t = sp.sin(t)
z_t = t

# 计算 dr/dt
dx_dt = sp.diff(x_t, t)
dy_dt = sp.diff(y_t, t)
dz_dt = sp.diff(z_t, t)

# 计算 F(r(t))
F_x = F(x_t, y_t, z_t)[0]
F_y = F(x_t, y_t, z_t)[1]
F_z = F(x_t, y_t, z_t)[2]

# 计算 F · dr/dt
integrand = F_x * dx_dt + F_y * dy_dt + F_z * dz_dt

# 计算线积分（t 从 0 到 2π）
line_integral = sp.integrate(integrand, (t, 0, 2*sp.pi))

print(f"线积分结果: {line_integral}")
print(f"数值结果: {line_integral.evalf()}")
```

### 练习3：可视化向量场

```python
import numpy as np
import matplotlib.pyplot as plt

# 创建网格
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# 定义向量场
P = -Y  # x方向分量
Q = X   # y方向分量

# 计算散度和旋度
div = np.gradient(P, x, axis=0) + np.gradient(Q, y, axis=1)
curl = np.gradient(Q, x, axis=0) - np.gradient(P, y, axis=1)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 向量场
axes[0].quiver(X, Y, P, Q, color='blue', alpha=0.7)
axes[0].set_title('Vector Field F = (-y, x)')
axes[0].set_aspect('equal')
axes[0].grid(True)

# 散度
im1 = axes[1].contourf(X, Y, div, levels=20, cmap='RdBu')
axes[1].set_title('Divergence')
axes[1].set_aspect('equal')
plt.colorbar(im1, ax=axes[1])

# 旋度
im2 = axes[2].contourf(X, Y, curl, levels=20, cmap='RdBu')
axes[2].set_title('Curl (z-component)')
axes[2].set_aspect('equal')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.show()
```

### 练习4：验证格林定理

```python
import sympy as sp

# 定义符号
x, y = sp.symbols('x y')

# 定义向量场
P = x**2 - y**2
Q = 2*x*y

# 方法1：计算线积分（沿单位圆）
t = sp.Symbol('t')
x_t = sp.cos(t)
y_t = sp.sin(t)

dx_dt = sp.diff(x_t, t)
dy_dt = sp.diff(y_t, t)

P_t = P.subs({x: x_t, y: y_t})
Q_t = Q.subs({x: x_t, y: y_t})

line_integral = sp.integrate(P_t * dx_dt + Q_t * dy_dt, (t, 0, 2*sp.pi))

# 方法2：计算二重积分（格林定理）
integrand = sp.diff(Q, x) - sp.diff(P, y)

# 转换为极坐标
r, theta = sp.symbols('r theta')
integrand_polar = integrand.subs({x: r*sp.cos(theta), y: r*sp.sin(theta)}) * r

double_integral = sp.integrate(integrand_polar, (r, 0, 1), (theta, 0, 2*sp.pi))

print(f"线积分结果: {line_integral}")
print(f"二重积分结果: {double_integral}")
print(f"验证格林定理: {sp.simplify(line_integral - double_integral) == 0}")
```

### 练习5：保守场验证

```python
import sympy as sp

x, y, z = sp.symbols('x y z')

# 定义向量场
P = 2*x*y + z**2
Q = x**2 + 2*y*z
R = y**2 + 2*x*z

# 计算旋度
curl_x = sp.diff(R, y) - sp.diff(Q, z)
curl_y = sp.diff(P, z) - sp.diff(R, x)
curl_z = sp.diff(Q, x) - sp.diff(P, y)

print(f"旋度: [{sp.simplify(curl_x)}, {sp.simplify(curl_y)}, {sp.simplify(curl_z)}]")

# 验证是否为保守场
is_conservative = (sp.simplify(curl_x) == 0 and 
                   sp.simplify(curl_y) == 0 and 
                   sp.simplify(curl_z) == 0)

print(f"是否为保守场: {is_conservative}")

# 如果是保守场，求势函数
if is_conservative:
    # 积分求势函数
    f = sp.integrate(P, x)
    print(f"\n势函数（初步）: f = {f}")
    
    # 验证并完善
    f_y = sp.diff(f, y)
    f_z = sp.diff(f, z)
    
    print(f"∂f/∂y = {f_y}, 应该等于 Q = {Q}")
    print(f"∂f/∂z = {f_z}, 应该等于 R = {R}")
```

---

**下一步**：[优化理论](04-optimization.md)
