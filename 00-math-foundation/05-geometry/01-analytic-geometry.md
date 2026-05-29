# 5.1 解析几何

## 目录

- [1. 点与向量](#1-点与向量)
- [2. 直线与平面](#2-直线与平面)
- [3. 曲线与曲面](#3-曲线与曲面)
- [4. 变换](#4-变换)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 点与向量

### 1.1 点的表示

**笛卡尔坐标系**：

$$P = (x, y, z)$$

**齐次坐标**：

$$P = (x, y, z, w)$$

其中 $w \neq 0$，对应的笛卡尔坐标为 $(x/w, y/w, z/w)$。

### 1.2 向量运算

**向量表示**：

$$\mathbf{v} = \begin{bmatrix} v_x \\ v_y \\ v_z \end{bmatrix}$$

**向量加法**：

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_x + v_x \\ u_y + v_y \\ u_z + v_z \end{bmatrix}$$

**数乘**：

$$k\mathbf{v} = \begin{bmatrix} kv_x \\ kv_y \\ kv_z \end{bmatrix}$$

**点积**：

$$\mathbf{u} \cdot \mathbf{v} = u_x v_x + u_y v_y + u_z v_z = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

**叉积**：

$$\mathbf{u} \times \mathbf{v} = \begin{bmatrix} u_y v_z - u_z v_y \\ u_z v_x - u_x v_z \\ u_x v_y - u_y v_x \end{bmatrix}$$

**混合积**：

$$[\mathbf{u}, \mathbf{v}, \mathbf{w}] = \mathbf{u} \cdot (\mathbf{v} \times \mathbf{w})$$

### 1.3 距离与角度

**两点距离**：

$$d(P_1, P_2) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}$$

**向量夹角**：

$$\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

---

## 2. 直线与平面

### 2.1 直线方程

**参数式**：

$$\mathbf{r}(t) = \mathbf{r}_0 + t\mathbf{v}$$

其中 $\mathbf{r}_0$ 是直线上一点，$\mathbf{v}$ 是方向向量。

**对称式**：

$$\frac{x - x_0}{a} = \frac{y - y_0}{b} = \frac{z - z_0}{c}$$

**两点式**：

$$\mathbf{r}(t) = (1-t)\mathbf{P}_1 + t\mathbf{P}_2$$

### 2.2 平面方程

**点法式**：

$$a(x - x_0) + b(y - y_0) + c(z - z_0) = 0$$

其中 $(a, b, c)$ 是法向量。

**一般式**：

$$ax + by + cz + d = 0$$

**截距式**：

$$\frac{x}{a} + \frac{y}{b} + \frac{z}{c} = 1$$

### 2.3 交点与距离

**直线与平面交点**：

$$t = \frac{ax_0 + by_0 + cz_0 + d}{-(a v_x + b v_y + c v_z)}$$

**点到平面距离**：

$$d = \frac{|ax_0 + by_0 + cz_0 + d|}{\sqrt{a^2 + b^2 + c^2}}$$

**两直线夹角**：

$$\cos\theta = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|}$$

---

## 3. 曲线与曲面

### 3.1 曲线

**参数方程**：

$$\mathbf{r}(t) = (x(t), y(t), z(t))$$

**切线向量**：

$$\mathbf{r}'(t) = \left(\frac{dx}{dt}, \frac{dy}{dt}, \frac{dz}{dt}\right)$$

**弧长**：

$$s(t) = \int_{t_0}^t \|\mathbf{r}'(\tau)\| d\tau$$

### 3.2 曲面

**参数方程**：

$$\mathbf{r}(u, v) = (x(u, v), y(u, v), z(u, v))$$

**切向量**：

$$\mathbf{r}_u = \left(\frac{\partial x}{\partial u}, \frac{\partial y}{\partial u}, \frac{\partial z}{\partial u}\right)$$

$$\mathbf{r}_v = \left(\frac{\partial x}{\partial v}, \frac{\partial y}{\partial v}, \frac{\partial z}{\partial v}\right)$$

**法向量**：

$$\mathbf{n} = \mathbf{r}_u \times \mathbf{r}_v$$

### 3.3 常见曲面

**平面**：$ax + by + cz + d = 0$

**球面**：$(x - x_0)^2 + (y - y_0)^2 + (z - z_0)^2 = r^2$

**圆柱面**：$(x - x_0)^2 + (y - y_0)^2 = r^2$

**圆锥面**：$z^2 = k^2(x^2 + y^2)$

**椭球面**：$\frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{c^2} = 1$

---

## 4. 变换

### 4.1 线性变换

**矩阵表示**：

$$\mathbf{v}' = \mathbf{A}\mathbf{v}$$

**常见变换**：
- 旋转
- 缩放
- 剪切

### 4.2 仿射变换

**定义**：

$$\mathbf{x}' = \mathbf{A}\mathbf{x} + \mathbf{b}$$

**齐次坐标表示**：

$$\begin{bmatrix} \mathbf{x}' \\ 1 \end{bmatrix} = \begin{bmatrix} \mathbf{A} & \mathbf{b} \\ \mathbf{0} & 1 \end{bmatrix} \begin{bmatrix} \mathbf{x} \\ 1 \end{bmatrix}$$

### 4.3 欧氏变换

**定义**：保持距离和角度不变的变换。

$$\mathbf{x}' = \mathbf{R}\mathbf{x} + \mathbf{t}$$

其中 $\mathbf{R}$ 是正交矩阵（$\mathbf{R}^T \mathbf{R} = \mathbf{I}$）。

**性质**：
- 保持长度
- 保持角度
- 保持定向

### 4.4 投影变换

**透视投影**：

$$x' = \frac{f x}{z}$$
$$y' = \frac{f y}{z}$$

其中 $f$ 是焦距。

**齐次坐标表示**：

$$\begin{bmatrix} x' \\ y' \\ z' \\ w' \end{bmatrix} = \begin{bmatrix} f & 0 & 0 & 0 \\ 0 & f & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 1/f & 0 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

---

## 5. 在具身智能中的应用

### 5.1 机器人运动学

**正运动学**：根据关节角度计算末端执行器位置。

**逆运动学**：根据末端执行器位置计算关节角度。

**雅可比矩阵**：

$$\mathbf{v}_e = \mathbf{J}(\mathbf{q}) \dot{\mathbf{q}}$$

### 5.2 SLAM

**相机位姿估计**：使用几何约束估计相机位置和方向。

**三角测量**：从多视图重建3D点。

**ICP**：迭代最近点算法。

### 5.3 计算机视觉

**特征匹配**：使用几何变换匹配特征点。

**姿态估计**：从2D-3D对应估计相机位姿。

**立体视觉**：从双目图像重建深度。

### 5.4 路径规划

**碰撞检测**：检测路径是否与障碍物相交。

**最短路径**：在自由空间中寻找最短路径。

---

## 6. 相关论文

### 几何基础

1. **"Analytic Geometry"** - Hartshorne (2000)
   - 解析几何经典教材

2. **"Elementary Differential Geometry"** - O'Neill (2006)
   - 微分几何入门教材

### 应用相关

3. **"Multiple View Geometry in Computer Vision"** - Hartley & Zisserman (2003)
   - 计算机视觉中的多视图几何

4. **"Robot Modeling and Control"** - Spong et al. (2006)
   - 机器人建模与控制

---

## 7. 实践练习

### 练习1：向量运算

```python
import numpy as np

# 向量定义
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

# 向量加法
print(f"向量加法: u + v = {u + v}")

# 数乘
print(f"数乘: 2 * u = {2 * u}")

# 点积
dot_product = np.dot(u, v)
print(f"点积: u · v = {dot_product}")

# 叉积
cross_product = np.cross(u, v)
print(f"叉积: u × v = {cross_product}")

# 混合积
w = np.array([7, 8, 9])
scalar_triple = np.dot(u, np.cross(v, w))
print(f"混合积: [u, v, w] = {scalar_triple}")

# 向量长度
norm_u = np.linalg.norm(u)
print(f"向量长度: ||u|| = {norm_u}")

# 单位向量
unit_u = u / norm_u
print(f"单位向量: u/||u|| = {unit_u}")

# 夹角
cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
theta = np.arccos(cos_theta)
print(f"夹角: θ = {np.degrees(theta):.2f}°")
```

### 练习2：直线与平面

```python
import numpy as np

# 直线参数方程
def line_point(t, r0, v):
    return r0 + t * v

# 平面方程
def plane_distance(point, normal, d):
    return np.abs(np.dot(normal, point) + d) / np.linalg.norm(normal)

# 示例：点到平面距离
plane_normal = np.array([1, 1, 1])
plane_d = -3  # x + y + z = 3
point = np.array([1, 2, 3])

distance = plane_distance(point, plane_normal, plane_d)
print(f"点 {point} 到平面 x+y+z=3 的距离: {distance:.4f}")

# 示例：直线与平面交点
line_r0 = np.array([0, 0, 0])
line_v = np.array([1, 1, 1])

# 求交点
denominator = np.dot(plane_normal, line_v)
if denominator == 0:
    print("直线与平面平行或重合")
else:
    t = -(np.dot(plane_normal, line_r0) + plane_d) / denominator
    intersection = line_point(t, line_r0, line_v)
    print(f"交点: {intersection}")
    print(f"参数 t = {t}")

# 示例：两直线夹角
line_v1 = np.array([1, 2, 3])
line_v2 = np.array([4, 5, 6])

cos_theta = np.dot(line_v1, line_v2) / (np.linalg.norm(line_v1) * np.linalg.norm(line_v2))
theta = np.arccos(cos_theta)
print(f"两直线夹角: {np.degrees(theta):.2f}°")
```

### 练习3：曲线与曲面

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 螺旋线
def helix(t):
    return np.array([np.cos(t), np.sin(t), t])

# 绘制螺旋线
t = np.linspace(0, 4*np.pi, 100)
x, y, z = helix(t)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, 'b-', linewidth=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Helix Curve')
plt.show()

# 球面
def sphere(u, v):
    return np.array([np.sin(u)*np.cos(v), 
                     np.sin(u)*np.sin(v), 
                     np.cos(u)])

# 绘制球面
u = np.linspace(0, np.pi, 50)
v = np.linspace(0, 2*np.pi, 50)
U, V = np.meshgrid(u, v)
X, Y, Z = sphere(U, V)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Sphere')
plt.show()

# 圆柱面
def cylinder(u, v, r=1):
    return np.array([r*np.cos(v), 
                     r*np.sin(v), 
                     u])

# 绘制圆柱面
u = np.linspace(-2, 2, 50)
v = np.linspace(0, 2*np.pi, 50)
U, V = np.meshgrid(u, v)
X, Y, Z = cylinder(U, V)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Cylinder')
plt.show()
```

### 练习4：变换

```python
import numpy as np

# 旋转矩阵
def rotation_matrix(angle, axis='z'):
    """
    绕坐标轴旋转的旋转矩阵
    """
    c = np.cos(angle)
    s = np.sin(angle)
    
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

# 平移向量
translation = np.array([1, 2, 3])

# 点
point = np.array([4, 5, 6])

# 旋转变换
rotated_point = rotation_matrix(np.pi/4, 'z') @ point
print(f"绕Z轴旋转45°: {rotated_point}")

# 仿射变换
def affine_transform(point, rotation, translation):
    """
    仿射变换：先旋转后平移
    """
    rotated = rotation @ point
    transformed = rotated + translation
    return transformed

transformed_point = affine_transform(point, rotation_matrix(np.pi/4, 'z'), translation)
print(f"仿射变换后: {transformed_point}")

# 齐次坐标变换
def homogeneous_transform(point, rotation, translation):
    """
    使用齐次坐标进行仿射变换
    """
    # 构造4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    
    # 转换为齐次坐标
    point_hom = np.append(point, 1)
    
    # 变换
    transformed_hom = T @ point_hom
    
    # 转换回笛卡尔坐标
    return transformed_hom[:3]

transformed_point_hom = homogeneous_transform(point, rotation_matrix(np.pi/4, 'z'), translation)
print(f"齐次坐标变换后: {transformed_point_hom}")
```

### 练习5：投影变换

```python
import numpy as np

# 透视投影
def perspective_projection(point, f=1):
    """
    透视投影（简单模型）
    """
    x, y, z = point
    if z == 0:
        return np.array([0, 0])
    return np.array([f * x / z, f * y / z])

# 示例点
points_3d = np.array([[1, 1, 1],
                      [1, -1, 2],
                      [-1, -1, 3],
                      [-1, 1, 4]])

# 投影
points_2d = np.array([perspective_projection(p) for p in points_3d])

print("3D点:")
print(points_3d)
print("\n投影后的2D点:")
print(points_2d)

# 使用齐次坐标的投影矩阵
def perspective_matrix(f=1):
    """
    透视投影矩阵
    """
    return np.array([[f, 0, 0, 0],
                     [0, f, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 1/f, 0]])

# 示例
P = perspective_matrix(f=1)
point_hom = np.array([1, 1, 1, 1])
projected_hom = P @ point_hom
projected = projected_hom[:2] / projected_hom[3]
print(f"\n使用投影矩阵: {projected}")

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 3D点
ax3d = fig.add_subplot(121, projection='3d')
ax3d.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='red')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.set_title('3D Points')

# 2D投影
axes[1].scatter(points_2d[:, 0], points_2d[:, 1], c='blue')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title('2D Projection')
axes[1].grid(True, alpha=0.3)
axes[1].axis('equal')

plt.tight_layout()
plt.show()
```

---

**下一步**：[射影几何](02-projective-geometry.md)