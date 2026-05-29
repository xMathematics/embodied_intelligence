# 2.1 位姿表示

## 目录

- [1. 位置表示](#1-位置表示)
- [2. 姿态表示](#2-姿态表示)
- [3. 旋转矩阵](#3-旋转矩阵)
- [4. 四元数](#4-四元数)
- [5. 欧拉角](#5-欧拉角)
- [6. 坐标变换](#6-坐标变换)
- [7. 在具身智能中的应用](#7-在具身智能中的应用)
- [8. 相关论文](#8-相关论文)
- [9. 实践练习](#9-实践练习)

---

## 1. 位置表示

### 1.1 笛卡尔坐标

**三维空间中的位置**：

$$\mathbf{p} = \begin{bmatrix} x \\ y \\ z \end{bmatrix}$$

**齐次坐标**：

$$\tilde{\mathbf{p}} = \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

### 1.2 向量运算

**向量加法**：

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_x + b_x \\ a_y + b_y \\ a_z + b_z \end{bmatrix}$$

**标量乘法**：

$$k\mathbf{a} = \begin{bmatrix} ka_x \\ ka_y \\ ka_z \end{bmatrix}$$

**点积**：

$$\mathbf{a} \cdot \mathbf{b} = a_xb_x + a_yb_y + a_zb_z = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$$

**叉积**：

$$\mathbf{a} \times \mathbf{b} = \begin{bmatrix} a_yb_z - a_zb_y \\ a_zb_x - a_xb_z \\ a_xb_y - a_yb_x \end{bmatrix}$$

---

## 2. 姿态表示

### 2.1 姿态的描述方法

| 表示方法 | 维度 | 优点 | 缺点 |
|---------|------|------|------|
| 旋转矩阵 | 9参数 | 直观，易于组合 | 冗余，需满足正交约束 |
| 四元数 | 4参数 | 无万向锁，插值平滑 | 稍复杂 |
| 欧拉角 | 3参数 | 直观，易理解 | 存在万向锁问题 |
| 轴角表示 | 4参数 | 几何意义明确 | 插值不便 |

### 2.2 旋转群 SO(3)

**特殊正交群**：所有3x3正交矩阵，行列式为+1。

$$SO(3) = \{ \mathbf{R} \in \mathbb{R}^{3 \times 3} \mid \mathbf{R}^T\mathbf{R} = \mathbf{I}, \det(\mathbf{R}) = 1 \}$$

---

## 3. 旋转矩阵

### 3.1 基本旋转矩阵

**绕x轴旋转**：

$$\mathbf{R}_x(\theta) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix}$$

**绕y轴旋转**：

$$\mathbf{R}_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}$$

**绕z轴旋转**：

$$\mathbf{R}_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

### 3.2 旋转矩阵的性质

1. **正交性**：$\mathbf{R}^T = \mathbf{R}^{-1}$
2. **行列式**：$\det(\mathbf{R}) = 1$
3. **复合旋转**：$\mathbf{R} = \mathbf{R}_1 \mathbf{R}_2$

### 3.3 从旋转矩阵提取角度

**迹**：$\text{tr}(\mathbf{R}) = 1 + 2\cos\theta$

**旋转角**：$\theta = \arccos\left(\frac{\text{tr}(\mathbf{R}) - 1}{2}\right)$

**旋转轴**：当 $\theta \neq 0$ 时，$\mathbf{R} - \mathbf{R}^T = 2\sin\theta \cdot [\mathbf{n}]_\times$

---

## 4. 四元数

### 4.1 四元数定义

**四元数**：$q = w + xi + yj + zk$

其中 $w$ 是实部，$(x, y, z)$ 是虚部。

**单位四元数**：$\|q\| = \sqrt{w^2 + x^2 + y^2 + z^2} = 1$

### 4.2 四元数运算

**乘法**：

$$q_1 \otimes q_2 = \begin{bmatrix} w_1w_2 - x_1x_2 - y_1y_2 - z_1z_2 \\ w_1x_2 + x_1w_2 + y_1z_2 - z_1y_2 \\ w_1y_2 - x_1z_2 + y_1w_2 + z_1x_2 \\ w_1z_2 + x_1y_2 - y_1x_2 + z_1w_2 \end{bmatrix}$$

**共轭**：$q^* = w - xi - yj - zk$

**逆**：$q^{-1} = \frac{q^*}{\|q\|^2}$

### 4.3 四元数与旋转矩阵的转换

**四元数 → 旋转矩阵**：

$$\mathbf{R} = \begin{bmatrix} 1-2y^2-2z^2 & 2xy-2zw & 2xz+2yw \\ 2xy+2zw & 1-2x^2-2z^2 & 2yz-2xw \\ 2xz-2yw & 2yz+2xw & 1-2x^2-2y^2 \end{bmatrix}$$

**旋转矩阵 → 四元数**：

$$w = \frac{1}{2}\sqrt{1 + r_{11} + r_{22} + r_{33}}$$
$$x = \frac{r_{32} - r_{23}}{4w}$$
$$y = \frac{r_{13} - r_{31}}{4w}$$
$$z = \frac{r_{21} - r_{12}}{4w}$$

### 4.4 四元数插值（SLERP）

**球面线性插值**：

$$q(t) = \frac{\sin((1-t)\theta)}{\sin\theta}q_0 + \frac{\sin(t\theta)}{\sin\theta}q_1$$

其中 $\theta = \arccos(q_0 \cdot q_1)$。

---

## 5. 欧拉角

### 5.1 定义

**Z-Y-X 顺序（航空顺序）**：

$$\mathbf{R} = \mathbf{R}_z(\psi) \mathbf{R}_y(\theta) \mathbf{R}_x(\phi)$$

其中：
- $\phi$：滚转角（roll）
- $\theta$：俯仰角（pitch）
- $\psi$：偏航角（yaw）

### 5.2 万向锁问题

当 $\theta = \pm 90^\circ$ 时，$\phi$ 和 $\psi$ 变得不确定。

**避免万向锁**：使用四元数或轴角表示。

### 5.3 欧拉角与旋转矩阵的转换

**欧拉角 → 旋转矩阵**：

$$\mathbf{R} = \begin{bmatrix}
\cos\psi\cos\theta & \cos\psi\sin\theta\sin\phi - \sin\psi\cos\phi & \cos\psi\sin\theta\cos\phi + \sin\psi\sin\phi \\
\sin\psi\cos\theta & \sin\psi\sin\theta\sin\phi + \cos\psi\cos\phi & \sin\psi\sin\theta\cos\phi - \cos\psi\sin\phi \\
-\sin\theta & \cos\theta\sin\phi & \cos\theta\cos\phi
\end{bmatrix}$$

---

## 6. 坐标变换

### 6.1 齐次变换矩阵

**变换矩阵**：

$$\mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{p} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

其中 $\mathbf{R}$ 是旋转矩阵，$\mathbf{p}$ 是平移向量。

### 6.2 点的变换

**从坐标系A到坐标系B**：

$$\mathbf{p}_B = \mathbf{R}_{BA} \mathbf{p}_A + \mathbf{p}_{AB}$$

**齐次坐标形式**：

$$\tilde{\mathbf{p}}_B = \mathbf{T}_{BA} \tilde{\mathbf{p}}_A$$

### 6.3 变换的复合

$$\mathbf{T}_{CA} = \mathbf{T}_{CB} \mathbf{T}_{BA}$$

---

## 7. 在具身智能中的应用

### 7.1 机器人定位

**位姿估计**：确定机器人在世界坐标系中的位置和姿态。

**传感器融合**：结合多种传感器数据估计位姿。

### 7.2 机械臂运动

**正运动学**：从关节角度计算末端执行器位姿。

**逆运动学**：从末端执行器位姿计算关节角度。

### 7.3 SLAM

**状态估计**：估计机器人位姿和地图。

**帧间匹配**：使用位姿变换匹配不同时刻的观测。

### 7.4 相机标定

**外参标定**：确定相机相对于世界坐标系的位姿。

**手眼标定**：确定相机相对于机器人末端的位姿。

---

## 8. 相关论文

### 经典文献

1. **"Introduction to Robotics"** - Craig (2005)
   - 机器人学经典教材

2. **"Quaternion kinematics for the error-state Kalman filter"** - Solà (2017)
   - 四元数在状态估计中的应用

### 应用相关

3. **"Multiple View Geometry in Computer Vision"** - Hartley & Zisserman (2004)
   - 多视图几何

4. **"Visual SLAM"** - Mur-Artal et al. (2015)
   - ORB-SLAM

---

## 9. 实践练习

### 练习1：旋转矩阵

```python
import numpy as np

def rotate_x(theta):
    """绕x轴旋转"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotate_y(theta):
    """绕y轴旋转"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotate_z(theta):
    """绕z轴旋转"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

# 示例：复合旋转
theta_x = np.pi / 4
theta_y = np.pi / 6
theta_z = np.pi / 3

Rx = rotate_x(theta_x)
Ry = rotate_y(theta_y)
Rz = rotate_z(theta_z)

# 旋转顺序：先绕x，再绕y，再绕z
R = Rz @ Ry @ Rx

print("旋转矩阵:")
print(R)

# 验证正交性
print("\n验证正交性:")
print("R^T @ R = I:", np.allclose(R.T @ R, np.eye(3)))
print("det(R) = 1:", np.isclose(np.linalg.det(R), 1))

# 旋转一个点
p = np.array([1, 0, 0])
p_rot = R @ p
print("\n原始点:", p)
print("旋转后:", p_rot)
```

### 练习2：四元数运算

```python
import numpy as np

class Quaternion:
    def __init__(self, w=1, x=0, y=0, z=0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        n = self.norm()
        self.w /= n
        self.x /= n
        self.y /= n
        self.z /= n
        return self
    
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def multiply(self, q):
        w = self.w*q.w - self.x*q.x - self.y*q.y - self.z*q.z
        x = self.w*q.x + self.x*q.w + self.y*q.z - self.z*q.y
        y = self.w*q.y - self.x*q.z + self.y*q.w + self.z*q.x
        z = self.w*q.z + self.x*q.y - self.y*q.x + self.z*q.w
        return Quaternion(w, x, y, z)
    
    def to_rotation_matrix(self):
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
        ])

# 示例：四元数旋转
q = Quaternion(np.cos(np.pi/8), np.sin(np.pi/8), 0, 0)  # 绕x轴旋转45度
q.normalize()

print("四元数:")
print(f"  w = {q.w:.4f}")
print(f"  x = {q.x:.4f}")
print(f"  y = {q.y:.4f}")
print(f"  z = {q.z:.4f}")

# 转换为旋转矩阵
R = q.to_rotation_matrix()
print("\n对应的旋转矩阵:")
print(R)

# 验证
p = np.array([0, 1, 0])
p_rot = R @ p
print("\n原始点:", p)
print("旋转后:", p_rot)
```

### 练习3：欧拉角与旋转矩阵转换

```python
import numpy as np

def euler_to_rotation(roll, pitch, yaw):
    """Z-Y-X顺序的欧拉角转旋转矩阵"""
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    return R

def rotation_to_euler(R):
    """旋转矩阵转欧拉角（Z-Y-X顺序）"""
    pitch = np.arcsin(-R[2, 0])
    
    if np.cos(pitch) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    
    return roll, pitch, yaw

# 示例
roll = np.pi / 4
pitch = np.pi / 6
yaw = np.pi / 3

R = euler_to_rotation(roll, pitch, yaw)
print("旋转矩阵:")
print(R)

roll2, pitch2, yaw2 = rotation_to_euler(R)
print("\n恢复的欧拉角:")
print(f"  roll: {roll2:.4f} (原始: {roll:.4f})")
print(f"  pitch: {pitch2:.4f} (原始: {pitch:.4f})")
print(f"  yaw: {yaw2:.4f} (原始: {yaw:.4f})")

# 验证万向锁
print("\n--- 万向锁测试 ---")
R_singular = euler_to_rotation(0, np.pi/2, 0)
roll_s, pitch_s, yaw_s = rotation_to_euler(R_singular)
print(f"万向锁状态下的欧拉角:")
print(f"  roll: {roll_s:.4f}")
print(f"  pitch: {pitch_s:.4f}")
print(f"  yaw: {yaw_s:.4f}")
```

### 练习4：齐次变换矩阵

```python
import numpy as np

def create_transform(R, p):
    """创建齐次变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def transform_point(T, p):
    """使用齐次变换矩阵变换点"""
    p_homogeneous = np.append(p, 1)
    p_transformed = T @ p_homogeneous
    return p_transformed[:3]

# 示例：机器人基座到末端的变换
R_base_to_arm = rotate_z(np.pi/2)
p_base_to_arm = np.array([0.5, 0, 0.3])
T_base_to_arm = create_transform(R_base_to_arm, p_base_to_arm)

R_arm_to_end = rotate_x(np.pi/4)
p_arm_to_end = np.array([0, 0, 0.5])
T_arm_to_end = create_transform(R_arm_to_end, p_arm_to_end)

# 复合变换
T_base_to_end = T_arm_to_end @ T_base_to_arm

print("基座到末端的变换矩阵:")
print(T_base_to_end)

# 末端执行器的位姿
p_end = np.array([0, 0, 0])
p_end_in_base = transform_point(T_base_to_end, p_end)
print("\n末端执行器在基座坐标系中的位置:")
print(p_end_in_base)

# 反向变换
T_end_to_base = np.linalg.inv(T_base_to_end)
print("\n末端到基座的逆变换矩阵:")
print(T_end_to_base)
```

### 练习5：四元数插值（SLERP）

```python
import numpy as np
import matplotlib.pyplot as plt

def quaternion_slerp(q0, q1, t):
    """球面线性插值"""
    # 确保四元数同向
    if np.dot(q0, q1) < 0:
        q1 = -q1
    
    # 计算角度
    theta = np.arccos(np.dot(q0, q1))
    
    if np.sin(theta) < 1e-6:
        return q0
    
    # SLERP公式
    s0 = np.sin((1-t)*theta) / np.sin(theta)
    s1 = np.sin(t*theta) / np.sin(theta)
    
    return s0 * q0 + s1 * q1

# 示例：从初始姿态到目标姿态的插值
q0 = np.array([1, 0, 0, 0])  # 初始姿态（单位四元数）
q1 = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0])  # 绕x轴旋转90度

# 插值
times = np.linspace(0, 1, 10)
quaternions = [quaternion_slerp(q0, q1, t) for t in times]

# 转换为旋转矩阵并提取欧拉角
euler_angles = []
for q in quaternions:
    w, x, y, z = q
    R = np.array([
        [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
    ])
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-R[2, 0])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    euler_angles.append([roll, pitch, yaw])

euler_angles = np.array(euler_angles)

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(times, np.degrees(euler_angles[:, 0]), label='roll')
plt.plot(times, np.degrees(euler_angles[:, 1]), label='pitch')
plt.plot(times, np.degrees(euler_angles[:, 2]), label='yaw')
plt.xlabel('时间')
plt.ylabel('角度 (°)')
plt.title('四元数SLERP插值')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

**下一步**：[机器人运动学](02-robot-kinematics.md)