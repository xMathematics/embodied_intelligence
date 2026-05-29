# 5.2 射影几何

## 目录

- [1. 射影空间](#1-射影空间)
- [2. 齐次坐标](#2-齐次坐标)
- [3. 射影变换](#3-射影变换)
- [4. 对偶性](#4-对偶性)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 射影空间

### 1.1 定义

**射影平面**：在欧氏平面基础上添加无穷远线。

**射影空间 $\mathbb{P}^n$**：由所有通过原点的直线组成的空间。

**点的表示**：

$$[x_0 : x_1 : \ldots : x_n]$$

其中 $(x_0, x_1, \ldots, x_n) \neq (0, 0, \ldots, 0)$，且 $[k x_0 : k x_1 : \ldots : k x_n] = [x_0 : x_1 : \ldots : x_n]$ 对任意非零 $k$。

### 1.2 射影平面

**二维射影平面 $\mathbb{P}^2$**：

- **普通点**：$[x : y : 1]$，对应欧氏平面点 $(x, y)$
- **无穷远点**：$[x : y : 0]$，对应方向 $(x, y)$
- **无穷远线**：所有无穷远点的集合，方程为 $z = 0$

### 1.3 几何元素

**点**：$[x : y : z]$

**线**：$ax + by + cz = 0$，表示为 $[a : b : c]$

**交点**：两条线 $l_1$ 和 $l_2$ 的交点为 $l_1 \times l_2$

**连线**：两个点 $p_1$ 和 $p_2$ 的连线为 $p_1 \times p_2$

---

## 2. 齐次坐标

### 2.1 二维齐次坐标

**点**：

$$(x, y) \leftrightarrow [x : y : 1]$$

**线**：

$$ax + by + c = 0 \leftrightarrow [a : b : c]$$

**点在线上**：$ax + by + cz = 0$，即 $p \cdot l = 0$

### 2.2 三维齐次坐标

**点**：

$$(x, y, z) \leftrightarrow [x : y : z : 1]$$

**平面**：

$$ax + by + cz + d = 0 \leftrightarrow [a : b : c : d]$$

### 2.3 无穷远元素

**无穷远线**（二维）：$[0 : 0 : 1]$

**无穷远平面**（三维）：$[0 : 0 : 0 : 1]$

---

## 3. 射影变换

### 3.1 定义

**射影变换**：由可逆矩阵 $\mathbf{H}$ 定义的变换：

$$\mathbf{x}' = \mathbf{H} \mathbf{x}$$

其中 $\mathbf{x}$ 和 $\mathbf{x}'$ 是齐次坐标向量。

**性质**：
- 保持共线性
- 保持交比
- 保持接触关系

### 3.2 二维射影变换（单应性）

**3x3矩阵**：

$$\mathbf{H} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix}$$

**自由度**：8个（齐次性）

**应用**：图像配准、全景拼接。

### 3.3 三维射影变换

**4x4矩阵**：

$$\mathbf{H} = \begin{bmatrix} h_{11} & h_{12} & h_{13} & h_{14} \\ h_{21} & h_{22} & h_{23} & h_{24} \\ h_{31} & h_{32} & h_{33} & h_{34} \\ h_{41} & h_{42} & h_{43} & h_{44} \end{bmatrix}$$

**自由度**：15个。

### 3.4 特殊射影变换

**仿射变换**：

$$\mathbf{H} = \begin{bmatrix} \mathbf{A} & \mathbf{b} \\ \mathbf{0} & 1 \end{bmatrix}$$

其中 $\mathbf{A}$ 是2x2或3x3矩阵，$\mathbf{b}$ 是平移向量。

**欧氏变换**：

$$\mathbf{H} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix}$$

其中 $\mathbf{R}$ 是正交矩阵。

---

## 4. 对偶性

### 4.1 对偶原理

**对偶命题**：将命题中的"点"和"线"互换得到的命题。

**示例**：
- 原命题："两点确定一条线"
- 对偶命题："两线确定一个点"

### 4.2 对偶表示

**点与线的对偶**：

- 点 $[x : y : z]$ 的对偶是线 $[x : y : z]$
- 线 $[a : b : c]$ 的对偶是点 $[a : b : c]$

**性质**：
- 点 $p$ 在线 $l$ 上 $\Leftrightarrow$ 线 $p^*$ 通过点 $l^*$
- 点 $p_1, p_2$ 的连线是 $p_1 \times p_2$ $\Leftrightarrow$ 线 $l_1, l_2$ 的交点是 $l_1 \times l_2$

### 4.3 对偶变换

**配极变换**：

$$\mathbf{x} \mapsto \mathbf{C} \mathbf{x}$$

其中 $\mathbf{C}$ 是对称矩阵（配极二次曲线）。

---

## 5. 在具身智能中的应用

### 5.1 相机模型

**针孔相机模型**：

$$\mathbf{x} = \mathbf{K} \mathbf{P} \mathbf{X}$$

其中 $\mathbf{K}$ 是内参矩阵，$\mathbf{P} = [\mathbf{R} \mid \mathbf{t}]$ 是外参矩阵。

**投影矩阵**：

$$\mathbf{P} = \mathbf{K} [\mathbf{R} \mid \mathbf{t}]$$

### 5.2 多视图几何

**基本矩阵**：

$$\mathbf{x}_2^T \mathbf{F} \mathbf{x}_1 = 0$$

**本质矩阵**：

$$\mathbf{E} = \mathbf{R}_{21} [\mathbf{t}_{21}]_\times$$

**单应性矩阵**：

$$\mathbf{x}_2 = \mathbf{H} \mathbf{x}_1$$

### 5.3 三维重建

**三角测量**：从多视图重建3D点。

**光束平差法**：最小化重投影误差。

### 5.4 SLAM

**视觉SLAM**：使用射影几何进行相机位姿估计和地图构建。

**Bundle Adjustment**：全局优化相机位姿和地图点。

---

## 6. 相关论文

### 射影几何

1. **"Projective Geometry"** - Coxeter (1994)
   - 射影几何经典教材

2. **"Foundations of Projective Geometry"** - Seidenberg (1962)
   - 射影几何基础

### 应用相关

3. **"Multiple View Geometry in Computer Vision"** - Hartley & Zisserman (2003)
   - 多视图几何权威教材

4. **"An Invitation to 3D Vision"** - Ma et al. (2004)
   - 三维视觉入门

---

## 7. 实践练习

### 练习1：齐次坐标

```python
import numpy as np

# 齐次坐标转换
def cartesian_to_homogeneous(points):
    """
    笛卡尔坐标转齐次坐标
    """
    if points.ndim == 1:
        return np.append(points, 1)
    else:
        return np.column_stack([points, np.ones(points.shape[0])])

def homogeneous_to_cartesian(points):
    """
    齐次坐标转笛卡尔坐标
    """
    if points.ndim == 1:
        return points[:-1] / points[-1]
    else:
        return points[:, :-1] / points[:, -1, np.newaxis]

# 示例
points_cart = np.array([[1, 2], [3, 4], [5, 6]])
points_hom = cartesian_to_homogeneous(points_cart)

print("笛卡尔坐标:")
print(points_cart)
print("\n齐次坐标:")
print(points_hom)
print("\n转换回笛卡尔坐标:")
print(homogeneous_to_cartesian(points_hom))

# 无穷远点
infinite_point = np.array([1, 0, 0])  # x轴方向的无穷远点
print(f"\n无穷远点: {infinite_point}")
```

### 练习2：射影变换

```python
import numpy as np

# 单应性矩阵
def homography_matrix(src_points, dst_points):
    """
    计算单应性矩阵（DLT算法）
    """
    assert len(src_points) == len(dst_points) >= 4
    
    n = len(src_points)
    A = []
    
    for i in range(n):
        x, y = src_points[i]
        u, v = dst_points[i]
        
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    A = np.array(A)
    
    # SVD分解
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    
    # 归一化
    H = H / H[2, 2]
    
    return H

# 示例：正方形到四边形的变换
src_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
dst_points = np.array([[0, 0], [2, 0.5], [2.5, 2], [0.5, 1.5]])

H = homography_matrix(src_points, dst_points)
print("单应性矩阵 H:")
print(H)

# 验证变换
test_point = np.array([0.5, 0.5, 1])
transformed = H @ test_point
transformed = transformed / transformed[2]
print(f"\n点 (0.5, 0.5) 变换后: {transformed[:2]}")

# 应用变换到网格
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
X, Y = np.meshgrid(x, y)
grid_points = np.column_stack([X.ravel(), Y.ravel()])

# 转换为齐次坐标
grid_hom = cartesian_to_homogeneous(grid_points)

# 应用变换
transformed_hom = (H @ grid_hom.T).T
transformed_cart = homogeneous_to_cartesian(transformed_hom)

# 可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.subplot(121)
plt.scatter(src_points[:, 0], src_points[:, 1], c='red')
plt.scatter(grid_points[:, 0], grid_points[:, 1], c='blue', s=10)
plt.title('Original')
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.subplot(122)
plt.scatter(dst_points[:, 0], dst_points[:, 1], c='red')
plt.scatter(transformed_cart[:, 0], transformed_cart[:, 1], c='blue', s=10)
plt.title('Transformed')
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.show()
```

### 练习3：对偶性

```python
import numpy as np

# 点与线的对偶
def point_to_line(point):
    """点的对偶是线"""
    return point

def line_to_point(line):
    """线的对偶是点"""
    return line

# 示例：三点共线
points = np.array([[1, 2, 1], [2, 4, 1], [3, 6, 1]])

# 计算连线
line12 = np.cross(points[0], points[1])
print(f"点1和点2的连线: {line12}")

# 检查点3是否在这条线上
dot_product = np.dot(points[2], line12)
print(f"点3在线12上吗? {np.isclose(dot_product, 0)}")

# 对偶：三线共点
lines = np.array([[1, 2, -5], [2, 4, -10], [3, 6, -15]])

# 计算交点
point12 = np.cross(lines[0], lines[1])
print(f"\n线1和线2的交点: {point12}")

# 检查线3是否通过这个点
dot_product = np.dot(point12, lines[2])
print(f"线3通过点12吗? {np.isclose(dot_product, 0)}")

# 对偶变换
def duality_transform(points, lines):
    """演示对偶性"""
    print("\n原命题: 三点共线")
    print(f"点: {points}")
    line = np.cross(points[0], points[1])
    print(f"连线: {line}")
    print(f"验证: 点3 · 连线 = {np.dot(points[2], line)}")
    
    print("\n对偶命题: 三线共点")
    print(f"线: {lines}")
    point = np.cross(lines[0], lines[1])
    print(f"交点: {point}")
    print(f"验证: 线3 · 交点 = {np.dot(lines[2], point)}")

duality_transform(points, lines)
```

### 练习4：相机投影

```python
import numpy as np

# 相机内参矩阵
def intrinsic_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])

# 相机外参矩阵
def extrinsic_matrix(R, t):
    return np.hstack([R, t.reshape(-1, 1)])

# 投影矩阵
def projection_matrix(K, R, t):
    P = K @ extrinsic_matrix(R, t)
    return P

# 示例
fx, fy, cx, cy = 500, 500, 320, 240  # 相机内参
K = intrinsic_matrix(fx, fy, cx, cy)

# 相机位姿
R = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])  # 旋转矩阵（单位矩阵）
t = np.array([0, 0, 5])   # 平移向量（相机在z=5处）

# 投影矩阵
P = projection_matrix(K, R, t)
print("投影矩阵 P:")
print(P)

# 3D点
X = np.array([1, 2, 0, 1])  # 齐次坐标

# 投影到2D
x = P @ X
x = x / x[2]  # 归一化
print(f"\n3D点 {X[:3]} 投影到2D: {x[:2]}")

# 多个点
points_3d = np.array([[0, 0, 0, 1],
                      [1, 0, 0, 1],
                      [0, 1, 0, 1],
                      [1, 1, 0, 1],
                      [0, 0, 1, 1]])

# 投影
points_2d = (P @ points_3d.T).T
points_2d = points_2d[:, :2] / points_2d[:, 2, np.newaxis]

print("\n3D点:")
print(points_3d[:, :3])
print("\n投影后的2D点:")
print(points_2d)
```

### 练习5：基本矩阵

```python
import numpy as np

# 计算基本矩阵
def fundamental_matrix(points1, points2):
    """
    使用8点算法计算基本矩阵
    """
    n = len(points1)
    A = []
    
    for i in range(n):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        
        A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
    
    A = np.array(A)
    
    # SVD分解
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    
    # 强制秩2约束
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V
    
    return F

# 示例：模拟两个视图的对应点
np.random.seed(42)
n_points = 8
points1 = np.random.rand(n_points, 2)

# 模拟第二个视图（简单平移）
points2 = points1 + np.array([0.1, 0.05])

# 添加齐次坐标
points1_hom = cartesian_to_homogeneous(points1)
points2_hom = cartesian_to_homogeneous(points2)

# 计算基本矩阵
F = fundamental_matrix(points1, points2)
print("基本矩阵 F:")
print(F)

# 验证极线约束
print("\n验证极线约束 x2^T F x1 = 0:")
for i in range(n_points):
    constraint = points2_hom[i] @ F @ points1_hom[i]
    print(f"点{i}: {constraint:.6e}")

# 绘制极线
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 图像1中的点和对应的极线
axes[0].scatter(points1[:, 0], points1[:, 1], c='red')
for i in range(n_points):
    # 极线 l1 = F^T x2
    l1 = F.T @ points2_hom[i]
    # 绘制极线
    x_vals = np.linspace(0, 1, 100)
    y_vals = -(l1[0] * x_vals + l1[2]) / l1[1]
    axes[0].plot(x_vals, y_vals, 'b--', alpha=0.5)
axes[0].set_title('Image 1')
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

# 图像2中的点和对应的极线
axes[1].scatter(points2[:, 0], points2[:, 1], c='blue')
for i in range(n_points):
    # 极线 l2 = F x1
    l2 = F @ points1_hom[i]
    # 绘制极线
    x_vals = np.linspace(0, 1, 100)
    y_vals = -(l2[0] * x_vals + l2[2]) / l2[1]
    axes[1].plot(x_vals, y_vals, 'r--', alpha=0.5)
axes[1].set_title('Image 2')
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

plt.show()
```

---

**下一步**：[微分几何](03-differential-geometry.md)