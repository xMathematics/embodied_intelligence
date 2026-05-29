
# 1.4 向量空间与变换

## 目录

- [1. 向量空间](#1-向量空间)
- [2. 线性变换](#2-线性变换)
- [3. 仿射变换](#3-仿射变换)
- [4. 坐标变换](#4-坐标变换)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 向量空间

### 1.1 定义

**向量空间**（线性空间）是一个集合 $V$ 配合加法和数乘运算，满足以下公理：

**加法公理**：
- 封闭性：$\mathbf{u} + \mathbf{v} \in V$
- 交换律：$\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
- 结合律：$(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
- 零向量：存在 $\mathbf{0}$ 使得 $\mathbf{v} + \mathbf{0} = \mathbf{v}$
- 负向量：对每个 $\mathbf{v}$ 存在 $-\mathbf{v}$

**数乘公理**：
- 封闭性：$c\mathbf{v} \in V$
- 分配律：$c(\mathbf{u} + \mathbf{v}) = c\mathbf{u} + c\mathbf{v}$
- 分配律：$(c + d)\mathbf{v} = c\mathbf{v} + d\mathbf{v}$
- 结合律：$c(d\mathbf{v}) = (cd)\mathbf{v}$
- 单位元：$1\mathbf{v} = \mathbf{v}$

### 1.2 子空间

**定义**：$W \subseteq V$ 是子空间，如果：
- $\mathbf{0} \in W$
- 对加法封闭：$\mathbf{u}, \mathbf{v} \in W \Rightarrow \mathbf{u} + \mathbf{v} \in W$
- 对数乘封闭：$\mathbf{v} \in W, c \in \mathbb{R} \Rightarrow c\mathbf{v} \in W$

**重要子空间**：
- **列空间**（Column Space）：矩阵列向量的张成空间
- **零空间**（Null Space）：$\{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{0}\}$
- **行空间**（Row Space）：矩阵行向量的张成空间
- **左零空间**（Left Null Space）：$\{\mathbf{y} : \mathbf{y}^T\mathbf{A} = \mathbf{0}\}$

### 1.3 基与维度

**基**（Basis）：向量空间 $V$ 的基是一组线性无关的向量 $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\}$，使得 $V$ 中任何向量都可表示为它们的线性组合。

**维度**：基中向量的个数，记为 $\dim(V)$。

**标准基**：

$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, \ldots, \mathbf{e}_n = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$

### 1.4 线性无关与张成

**线性无关**：向量组 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ 线性无关，如果：

$$c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \Rightarrow c_1 = \cdots = c_k = 0$$

**张成空间**（Span）：

$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \{c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k : c_i \in \mathbb{R}\}$$

### 1.5 内积空间

**内积**定义了向量之间的"角度"概念：

$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T\mathbf{v} = \sum_{i=1}^{n} u_i v_i$$

**性质**：
- 对称性：$\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$
- 线性性：$\langle c\mathbf{u} + d\mathbf{v}, \mathbf{w} \rangle = c\langle \mathbf{u}, \mathbf{w} \rangle + d\langle \mathbf{v}, \mathbf{w} \rangle$
- 正定性：$\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$，等号成立当且仅当 $\mathbf{v} = \mathbf{0}$

**正交**：$\langle \mathbf{u}, \mathbf{v} \rangle = 0$

**正交归一基**：基向量两两正交且为单位长度。

---

## 2. 线性变换

### 2.1 定义

**线性变换** $T: V \to W$ 满足：
- $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
- $T(c\mathbf{v}) = cT(\mathbf{v})$

### 2.2 矩阵表示

给定基 $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ 和 $\{\mathbf{w}_1, \ldots, \mathbf{w}_m\}$，线性变换 $T$ 可用矩阵 $\mathbf{A}$ 表示：

$$T(\mathbf{v}_j) = \sum_{i=1}^{m} a_{ij}\mathbf{w}_i$$

$$\mathbf{A} = \begin{bmatrix} | & | & & | \\ T(\mathbf{v}_1) & T(\mathbf{v}_2) & \cdots & T(\mathbf{v}_n) \\ | & | & & | \end{bmatrix}$$

### 2.3 常见线性变换

#### 缩放

$$\mathbf{S} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

#### 旋转

2D旋转：

$$\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

3D旋转（绕各轴）：

$$\mathbf{R}_x(\alpha) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\alpha & -\sin\alpha \\ 0 & \sin\alpha & \cos\alpha \end{bmatrix}$$

$$\mathbf{R}_y(\beta) = \begin{bmatrix} \cos\beta & 0 & \sin\beta \\ 0 & 1 & 0 \\ -\sin\beta & 0 & \cos\beta \end{bmatrix}$$

$$\mathbf{R}_z(\gamma) = \begin{bmatrix} \cos\gamma & -\sin\gamma & 0 \\ \sin\gamma & \cos\gamma & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

#### 反射

关于x轴反射：

$$\mathbf{F}_x = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$

#### 剪切

$$\mathbf{H} = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$$

### 2.4 复合变换

变换的组合对应矩阵的乘法：

$$T_2(T_1(\mathbf{v})) = (\mathbf{A}_2\mathbf{A}_1)\mathbf{v}$$

**注意**：矩阵乘法不满足交换律，变换顺序很重要！

### 2.5 核与像

**核**（Kernel/Null Space）：

$$\ker(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}$$

**像**（Image/Range）：

$$\text{im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}$$

**秩-零化度定理**：

$$\dim(V) = \dim(\ker(T)) + \dim(\text{im}(T))$$

---

## 3. 仿射变换

### 3.1 定义

**仿射变换**是线性变换加上平移：

$$T(\mathbf{v}) = \mathbf{A}\mathbf{v} + \mathbf{b}$$

### 3.2 齐次坐标

使用齐次坐标将仿射变换表示为线性变换：

$$\begin{bmatrix} \mathbf{v}' \\ 1 \end{bmatrix} = \begin{bmatrix} \mathbf{A} & \mathbf{b} \\ \mathbf{0}^T & 1 \end{bmatrix} \begin{bmatrix} \mathbf{v} \\ 1 \end{bmatrix}$$

### 3.3 2D仿射变换

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} & t_x \\ a_{21} & a_{22} & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

### 3.4 3D刚体变换

**齐次变换矩阵**：

$$\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

其中：
- $\mathbf{R} \in SO(3)$：旋转矩阵（3×3正交矩阵，行列式为1）
- $\mathbf{t} \in \mathbb{R}^3$：平移向量

**逆变换**：

$$\mathbf{T}^{-1} = \begin{bmatrix} \mathbf{R}^T & -\mathbf{R}^T\mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

**变换组合**：

$$\mathbf{T}_{12} = \mathbf{T}_1 \mathbf{T}_2$$

---

## 4. 坐标变换

### 4.1 基变换

设 $V$ 有两组基 $\mathcal{B} = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ 和 $\mathcal{B}' = \{\mathbf{v}'_1, \ldots, \mathbf{v}'_n\}$。

**变换矩阵**：

$$\mathbf{P} = \begin{bmatrix} | & | & & | \\ [\mathbf{v}'_1]_\mathcal{B} & [\mathbf{v}'_2]_\mathcal{B} & \cdots & [\mathbf{v}'_n]_\mathcal{B} \\ | & | & & | \end{bmatrix}$$

坐标变换：

$$[\mathbf{v}]_\mathcal{B} = \mathbf{P}[\mathbf{v}]_{\mathcal{B}'}$$

### 4.2 坐标系变换

**世界坐标系** $\to$ **相机坐标系**：

$$\mathbf{X}_c = \mathbf{R}_{cw}\mathbf{X}_w + \mathbf{t}_{cw}$$

或齐次形式：

$$\mathbf{X}_c = \mathbf{T}_{cw}\mathbf{X}_w$$

### 4.3 欧拉角与旋转矩阵

**ZYX欧拉角**（偏航-俯仰-翻滚）：

$$\mathbf{R} = \mathbf{R}_z(\psi)\mathbf{R}_y(\theta)\mathbf{R}_x(\phi)$$

**万向节锁问题**：当俯仰角为±90°时，丢失一个自由度。

### 4.4 四元数

**定义**：$q = w + xi + yj + zk = [w, \mathbf{v}]$

**与旋转矩阵的转换**：

$$\mathbf{R} = \begin{bmatrix} 1-2y^2-2z^2 & 2xy-2zw & 2xz+2yw \\ 2xy+2zw & 1-2x^2-2z^2 & 2yz-2xw \\ 2xz-2yw & 2yz+2xw & 1-2x^2-2y^2 \end{bmatrix}$$

**优点**：
- 避免万向节锁
- 插值更平滑（球面线性插值SLERP）
- 存储更高效（4个数 vs 9个数）

---

## 5. 在具身智能中的应用

### 5.1 机器人运动学

**正运动学**：关节角度 $\to$ 末端位姿

$$\mathbf{T}_{ee} = \mathbf{T}_1(\theta_1)\mathbf{T}_2(\theta_2)\cdots\mathbf{T}_n(\theta_n)$$

**DH参数法**：用4个参数描述相邻连杆的变换

### 5.2 相机模型

**世界坐标 $\to$ 图像坐标**：

$$\lambda \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K}[\mathbf{R}|\mathbf{t}]\begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

其中 $\mathbf{K}$ 是内参矩阵。

### 5.3 SLAM中的位姿估计

**位姿图优化**：

$$\min_{\mathbf{T}_1, \ldots, \mathbf{T}_n} \sum_{(i,j) \in \mathcal{E}} \|\log(\mathbf{T}_{ij}^{-1}\mathbf{T}_i^{-1}\mathbf{T}_j)\|^2$$

### 5.4 点云配准

**ICP算法**：

寻找最优刚性变换：

$$\min_{\mathbf{R}, \mathbf{t}} \sum_{i} \|\mathbf{p}_i - (\mathbf{R}\mathbf{q}_i + \mathbf{t})\|^2$$

### 5.5 神经网络中的变换

**注意力机制**：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

---

## 6. 相关论文

### 刚体变换

1. **"A Mathematical Introduction to Robotic Manipulation"** - Murray, Li & Sastry
   - 机器人操作数学基础

2. **"Quaternions and Rotation Sequences"** - Kuipers
   - 四元数理论

### 坐标变换

3. **"Least-Squares Rigid Motion Using SVD"** - Umeyama (1991)
   - SVD在刚体配准中的应用

4. **"A Method for Registration of 3-D Shapes"** - Besl & McKay (1992)
   - ICP算法

### 李群李代数

5. **"State Estimation for Robotics"** - Barfoot
   - 李群李代数在状态估计中的应用

6. **"A Micro Lie Theory for State Estimation in Robotics"** - Sola et al.
   - 机器人学中的李理论

---

## 7. 实践练习

### 练习1：基变换

```python
import numpy as np

def change_of_basis_matrix(B_old, B_new):
    """
    计算基变换矩阵
    B_old: 旧基（列向量）
    B_new: 新基（列向量）
    """
    # 新基向量在旧基下的坐标
    P = np.linalg.solve(B_old, B_new)
    return P

# 定义两组基
B_standard = np.eye(2)
B_new = np.array([[1, 1], [0, 1]])  # 新基

# 计算变换矩阵
P = change_of_basis_matrix(B_standard, B_new)
print("基变换矩阵 P:")
print(P)

# 验证：向量在新基下的坐标
v = np.array([3, 2])  # 在标准基下的坐标
v_new = np.linalg.solve(P, v)  # 在新基下的坐标
print(f"\n向量在标准基下: {v}")
print(f"向量在新基下: {v_new}")
print(f"验证: {B_new @ v_new}")
```

### 练习2：2D变换组合

```python
import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix_2d(theta):
    """2D旋转矩阵"""
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def scaling_matrix(sx, sy):
    """缩放矩阵"""
    return np.array([
        [sx, 0],
        [0, sy]
    ])

def apply_transform(points, transform):
    """应用变换到点集"""
    return (transform @ points.T).T

# 创建正方形
square = np.array([
    [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]
])

# 组合变换：先旋转45度，再缩放
R = rotation_matrix_2d(np.pi/4)
S = scaling_matrix(1.5, 0.5)
T = S @ R  # 注意顺序：先R后S

transformed_square = apply_transform(square, T)

# 可视化
plt.figure(figsize=(8, 8))
plt.plot(square[:, 0], square[:, 1], 'b-', label='Original', linewidth=2)
plt.plot(transformed_square[:, 0], transformed_square[:, 1], 'r-', label='Transformed', linewidth=2)
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('2D Transformation')
plt.show()
```

### 练习3：3D刚体变换

```python
import numpy as np

def rotation_matrix_x(angle):
    """绕x轴旋转"""
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])

def rotation_matrix_y(angle):
    """绕y轴旋转"""
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rotation_matrix_z(angle):
    """绕z轴旋转"""
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])

def homogeneous_transform(R, t):
    """创建齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def apply_homogeneous_transform(T, points):
    """应用齐次变换"""
    # 转换为齐次坐标
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = (T @ points_h.T).T
    return transformed[:, :3]

# 创建3D点（立方体顶点）
cube = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
])

# 定义变换：旋转+平移
R = rotation_matrix_z(np.pi/4) @ rotation_matrix_y(np.pi/6)
t = np.array([2, 1, 0.5])
T = homogeneous_transform(R, t)

# 应用变换
cube_transformed = apply_homogeneous_transform(T, cube)

print("原始立方体:")
print(cube)
print("\n变换后立方体:")
print(cube_transformed)

# 验证逆变换
T_inv = np.linalg.inv(T)
cube_recovered = apply_homogeneous_transform(T_inv, cube_transformed)
print("\n恢复后的立方体:")
print(cube_recovered)
```

### 练习4：四元数实现

```python
import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self.q = np.array([w, x, y, z])
    
    def __mul__(self, other):
        """四元数乘法"""
        w1, x1, y1, z1 = self.q
        w2, x2, y2, z2 = other.q
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return Quaternion(w, x, y, z)
    
    def conjugate(self):
        """共轭"""
        return Quaternion(self.q[0], -self.q[1], -self.q[2], -self.q[3])
    
    def norm(self):
        """模长"""
        return np.linalg.norm(self.q)
    
    def normalize(self):
        """归一化"""
        norm = self.norm()
        self.q = self.q / norm
        return self
    
    def to_rotation_matrix(self):
        """转换为旋转矩阵"""
        w, x, y, z = self.q
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])
    
    @staticmethod
    def from_axis_angle(axis, angle):
        """从轴角表示创建四元数"""
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        w = np.cos(half_angle)
        x, y, z = np.sin(half_angle) * axis
        return Quaternion(w, x, y, z)

# 测试
q1 = Quaternion.from_axis_angle([0, 0, 1], np.pi/4)  # 绕z轴旋转45度
print("四元数:", q1.q)
print("旋转矩阵:")
print(q1.to_rotation_matrix())

# 验证与直接旋转矩阵一致
R_z = np.array([
    [np.cos(np.pi/4), -np.sin(np.pi/4), 0],
    [np.sin(np.pi/4), np.cos(np.pi/4), 0],
    [0, 0, 1]
])
print("\n直接旋转矩阵:")
print(R_z)
```

---

**下一步**：[张量基础](05-tensors.md)
