
# 2.2 多元微积分

## 目录

- [1. 多元函数](#1-多元函数)
- [2. 偏导数](#2-偏导数)
- [3. 全微分与链式法则](#3-全微分与链式法则)
- [4. 方向导数与梯度](#4-方向导数与梯度)
- [5. 多元函数的极值](#5-多元函数的极值)
- [6. 在具身智能中的应用](#6-在具身智能中的应用)
- [7. 相关论文](#7-相关论文)
- [8. 实践练习](#8-实践练习)

---

## 1. 多元函数

### 1.1 定义

**多元函数**是从 $\mathbb{R}^n$ 到 $\mathbb{R}^m$ 的映射：

$$f: \mathbb{R}^n \to \mathbb{R}^m$$

**标量场**：$f: \mathbb{R}^n \to \mathbb{R}$，如温度分布、高度函数

**向量场**：$\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^n$，如速度场、力场

### 1.2 二元函数

$$z = f(x, y)$$

**定义域**：$(x, y)$ 的取值范围

**等值线**：$f(x, y) = c$ 的曲线

**示例**：
- 平面：$z = ax + by + c$
- 抛物面：$z = x^2 + y^2$
- 双曲面：$z = x^2 - y^2$

---

## 2. 偏导数

### 2.1 定义

**偏导数**是多元函数对其中一个变量的导数，其他变量保持不变。

$$\frac{\partial f}{\partial x}(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0+h, y_0) - f(x_0, y_0)}{h}$$

$$\frac{\partial f}{\partial y}(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0, y_0+h) - f(x_0, y_0)}{h}$$

### 2.2 记号

$$\frac{\partial f}{\partial x} = f_x = \partial_x f$$

### 2.3 计算规则

计算偏导数时，将其他变量视为常数。

**示例**：

$$f(x, y) = x^2y + 3xy^2$$

$$\frac{\partial f}{\partial x} = 2xy + 3y^2$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy$$

### 2.4 高阶偏导数

**二阶偏导数**：

$$\frac{\partial^2 f}{\partial x^2} = f_{xx}$$

$$\frac{\partial^2 f}{\partial y^2} = f_{yy}$$

$$\frac{\partial^2 f}{\partial x \partial y} = f_{xy}$$

$$\frac{\partial^2 f}{\partial y \partial x} = f_{yx}$$

**Clairaut定理**：若混合偏导数连续，则：

$$f_{xy} = f_{yx}$$

### 2.5 常见函数的偏导数

| 函数 $f(x, y)$ | $\frac{\partial f}{\partial x}$ | $\frac{\partial f}{\partial y}$ |
|---------------|--------------------------------|--------------------------------|
| $x^2 + y^2$ | $2x$ | $2y$ |
| $e^{xy}$ | $ye^{xy}$ | $xe^{xy}$ |
| $\ln(x^2 + y^2)$ | $\frac{2x}{x^2+y^2}$ | $\frac{2y}{x^2+y^2}$ |
| $\sin(xy)$ | $y\cos(xy)$ | $x\cos(xy)$ |

---

## 3. 全微分与链式法则

### 3.1 全微分

若 $z = f(x, y)$ 可微，则全微分为：

$$dz = \frac{\partial f}{\partial x}dx + \frac{\partial f}{\partial y}dy$$

**几何意义**：切平面的变化量

### 3.2 可微性

**定义**：$f$ 在 $(x_0, y_0)$ 可微，如果：

$$\Delta z = f(x_0+\Delta x, y_0+\Delta y) - f(x_0, y_0) = A\Delta x + B\Delta y + o(\sqrt{\Delta x^2 + \Delta y^2})$$

其中 $A = f_x(x_0, y_0)$，$B = f_y(x_0, y_0)$。

**充分条件**：若偏导数连续，则函数可微。

### 3.3 链式法则

#### 情形1：单变量中间变量

若 $z = f(x, y)$，$x = x(t)$，$y = y(t)$，则：

$$\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$

#### 情形2：多变量中间变量

若 $z = f(x, y)$，$x = x(s, t)$，$y = y(s, t)$，则：

$$\frac{\partial z}{\partial s} = \frac{\partial f}{\partial x}\frac{\partial x}{\partial s} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial s}$$

$$\frac{\partial z}{\partial t} = \frac{\partial f}{\partial x}\frac{\partial x}{\partial t} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial t}$$

#### 情形3：一般情形

$$\frac{\partial}{\partial x_i} f(u_1(\mathbf{x}), \ldots, u_n(\mathbf{x})) = \sum_{j=1}^{n} \frac{\partial f}{\partial u_j} \frac{\partial u_j}{\partial x_i}$$

### 3.4 隐函数定理

若 $F(x, y) = 0$ 定义了隐函数 $y = y(x)$，则：

$$\frac{dy}{dx} = -\frac{F_x}{F_y}$$

**条件**：$F_y \neq 0$

---

## 4. 方向导数与梯度

### 4.1 方向导数

**定义**：函数 $f$ 在点 $P$ 沿方向 $\mathbf{u}$（单位向量）的方向导数：

$$D_{\mathbf{u}}f = \lim_{h \to 0} \frac{f(P + h\mathbf{u}) - f(P)}{h}$$

**计算**：

$$D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u} = \frac{\partial f}{\partial x}u_1 + \frac{\partial f}{\partial y}u_2$$

### 4.2 梯度

**定义**：

$$\nabla f = \left[\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right]^T$$

**性质**：
- 梯度方向是函数增长最快的方向
- 梯度的模是最大增长率
- 梯度垂直于等值线

### 4.3 梯度的应用

**最速下降方向**：$-\nabla f$

**最速上升方向**：$\nabla f$

**等高线法线方向**：$\nabla f$

### 4.4 散度与旋度

#### 散度（Divergence）

对于向量场 $\mathbf{F} = [F_x, F_y, F_z]$：

$$\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}$$

**物理意义**：源或汇的强度

#### 旋度（Curl）

$$\nabla \times \mathbf{F} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ F_x & F_y & F_z \end{vmatrix}$$

**物理意义**：旋转强度

---

## 5. 多元函数的极值

### 5.1 无条件极值

**必要条件**：若 $f$ 在 $(x_0, y_0)$ 可微且取极值，则：

$$\nabla f(x_0, y_0) = \mathbf{0}$$

即 $f_x = 0$ 且 $f_y = 0$。

**充分条件**：使用Hessian矩阵

$$H = \begin{bmatrix} f_{xx} & f_{xy} \\ f_{yx} & f_{yy} \end{bmatrix}$$

**判别法**：
- 若 $H$ 正定（$f_{xx} > 0$ 且 $\det(H) > 0$）：极小值
- 若 $H$ 负定（$f_{xx} < 0$ 且 $\det(H) > 0$）：极大值
- 若 $\det(H) < 0$：鞍点
- 若 $\det(H) = 0$：无法判断

### 5.2 条件极值（拉格朗日乘数法）

**问题**：在约束 $g(x, y) = 0$ 下求 $f(x, y)$ 的极值。

**方法**：构造拉格朗日函数

$$\mathcal{L}(x, y, \lambda) = f(x, y) + \lambda g(x, y)$$

求解方程组：

$$\frac{\partial \mathcal{L}}{\partial x} = 0, \quad \frac{\partial \mathcal{L}}{\partial y} = 0, \quad \frac{\partial \mathcal{L}}{\partial \lambda} = 0$$

### 5.3 多个约束

对于多个约束 $g_1 = 0, g_2 = 0, \ldots$：

$$\mathcal{L} = f + \lambda_1 g_1 + \lambda_2 g_2 + \cdots$$

---

## 6. 在具身智能中的应用

### 6.1 梯度下降优化

**损失函数最小化**：

$$\theta_{n+1} = \theta_n - \eta \nabla_{\theta} L(\theta)$$

**应用**：神经网络训练

### 6.2 机器人运动规划

**势能场方法**：

$$\mathbf{F} = -\nabla U$$

其中 $U$ 是势能函数。

### 6.3 图像处理

**边缘检测**：

$$\|\nabla I\| = \sqrt{\left(\frac{\partial I}{\partial x}\right)^2 + \left(\frac{\partial I}{\partial y}\right)^2}$$

### 6.4 约束优化

**逆运动学**：

在关节角度约束下最小化末端误差。

---

## 7. 相关论文

### 优化理论

1. **"Convex Optimization"** - Boyd & Vandenberghe (2004)
   - 凸优化经典教材

2. **"Numerical Optimization"** - Nocedal & Wright (2006)
   - 数值优化权威参考书

### 应用相关

3. **"Gradient-Based Learning Applied to Document Recognition"** - LeCun et al. (1998)
   - 梯度下降在神经网络中的应用

4. **"Optimization for Machine Learning"** - Bottou et al. (2018)
   - 机器学习优化综述

---

## 8. 实践练习

### 练习1：偏导数计算

```python
import numpy as np
import sympy as sp

# 定义符号
x, y = sp.symbols('x y')

# 定义函数
f = x**2 * y + 3*x*y**2

# 计算偏导数
f_x = sp.diff(f, x)
f_y = sp.diff(f, y)

# 二阶偏导数
f_xx = sp.diff(f, x, 2)
f_yy = sp.diff(f, y, 2)
f_xy = sp.diff(f, x, y)

print(f"f(x, y) = {f}")
print(f"∂f/∂x = {f_x}")
print(f"∂f/∂y = {f_y}")
print(f"∂²f/∂x² = {f_xx}")
print(f"∂²f/∂y² = {f_yy}")
print(f"∂²f/∂x∂y = {f_xy}")

# 验证混合偏导数相等
print(f"\n验证 f_xy = f_yx: {sp.simplify(f_xy - sp.diff(f, y, x)) == 0}")
```

### 练习2：梯度计算与可视化

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    """目标函数"""
    return x**2 + y**2

def gradient(x, y):
    """梯度计算"""
    return np.array([2*x, 2*y])

# 创建网格
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 计算梯度
U, V = gradient(X, Y)

# 可视化
fig, ax = plt.subplots(figsize=(10, 8))

# 等高线
contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# 梯度场（箭头）
ax.quiver(X, Y, U, V, color='red', alpha=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gradient Field of f(x,y) = x² + y²')
ax.set_aspect('equal')
plt.colorbar(contour)
plt.show()
```

### 练习3：梯度下降优化

```python
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    """Rosenbrock函数"""
    return (1 - x)**2 + 100*(y - x**2)**2

def gradient(x, y):
    """梯度"""
    df_dx = -2*(1 - x) - 400*x*(y - x**2)
    df_dy = 200*(y - x**2)
    return np.array([df_dx, df_dy])

def gradient_descent(start_point, learning_rate=0.0002, num_iterations=1000):
    """梯度下降"""
    x, y = start_point
    trajectory = [(x, y)]
    
    for i in range(num_iterations):
        grad = gradient(x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        trajectory.append((x, y))
    
    return np.array(trajectory)

# 运行优化
start = np.array([-1.5, 1.5])
trajectory = gradient_descent(start)

print(f"初始点: {start}")
print(f"最终点: {trajectory[-1]}")
print(f"最终函数值: {f(trajectory[-1, 0], trajectory[-1, 1]):.6f}")

# 可视化
x = np.linspace(-2, 2, 100)
y = np.linspace(-0.5, 2.5, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.figure(figsize=(12, 5))

# 等高线图
plt.subplot(1, 2, 1)
contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-o', markersize=2, label='Trajectory')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent on Rosenbrock Function')
plt.legend()
plt.colorbar(contour)

# 3D图
plt.subplot(1, 2, 2, projection='3d')
plt.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis')
plt.plot(trajectory[:, 0], trajectory[:, 1], 
         f(trajectory[:, 0], trajectory[:, 1]), 
         'r-o', markersize=2, label='Trajectory')
plt.title('3D Visualization')
plt.tight_layout()
plt.show()
```

### 练习4：拉格朗日乘数法

```python
import sympy as sp

# 定义符号
x, y, lam = sp.symbols('x y lambda')

# 目标函数
f = x**2 + y**2

# 约束条件：x + y = 1
g = x + y - 1

# 拉格朗日函数
L = f + lam * g

# 求解方程组
eq1 = sp.diff(L, x)
eq2 = sp.diff(L, y)
eq3 = sp.diff(L, lam)

solution = sp.solve([eq1, eq2, eq3], [x, y, lam])

print(f"目标函数: f(x,y) = {f}")
print(f"约束条件: g(x,y) = {g} = 0")
print(f"\n解: {solution}")

# 验证
for sol in solution:
    x_val, y_val, lam_val = sol
    print(f"\n验证:")
    print(f"  x = {x_val}, y = {y_val}")
    print(f"  f(x,y) = {f.subs({x: x_val, y: y_val})}")
    print(f"  约束满足: {g.subs({x: x_val, y: y_val}) == 0}")
```

### 练习5：方向导数

```python
import numpy as np

def f(x, y):
    """目标函数"""
    return x**2 + 2*y**2

def gradient(x, y):
    """梯度"""
    return np.array([2*x, 4*y])

def directional_derivative(x, y, direction):
    """
    计算方向导数
    direction: 方向向量（不需要是单位向量）
    """
    # 归一化方向向量
    u = direction / np.linalg.norm(direction)
    
    # 计算梯度
    grad = gradient(x, y)
    
    # 方向导数 = 梯度 · 方向
    return np.dot(grad, u)

# 测试
point = (1, 2)
directions = [
    np.array([1, 0]),  # x方向
    np.array([0, 1]),  # y方向
    np.array([1, 1]),  # 45度方向
    np.array([1, -1]), # -45度方向
]

print(f"点: {point}")
print(f"梯度: {gradient(*point)}")
print(f"\n方向导数:")

for d in directions:
    dd = directional_derivative(*point, d)
    print(f"  方向 {d}: {dd:.4f}")

# 验证：最大方向导数在梯度方向
grad = gradient(*point)
max_dd = np.linalg.norm(grad)
print(f"\n最大方向导数（梯度方向）: {max_dd:.4f}")
```

---

**下一步**：[向量微积分](03-vector-calculus.md)
