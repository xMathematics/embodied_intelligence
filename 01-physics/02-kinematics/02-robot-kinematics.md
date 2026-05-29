# 2.2 机器人运动学

## 目录

- [1. 正运动学](#1-正运动学)
- [2. DH参数](#2-dh参数)
- [3. 逆运动学](#3-逆运动学)
- [4. Jacobian矩阵](#4-jacobian矩阵)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 正运动学

### 1.1 定义

**正运动学**：已知关节角度，计算末端执行器的位姿。

**公式**：

$$\mathbf{T}(\mathbf{q}) = \mathbf{T}_1(q_1) \mathbf{T}_2(q_2) \cdots \mathbf{T}_n(q_n)$$

其中 $\mathbf{q} = [q_1, q_2, \ldots, q_n]^T$ 是关节角度向量。

### 1.2 串联机械臂

**n自由度机械臂**：

$$\mathbf{T}_{0n} = \prod_{i=1}^{n} \mathbf{T}_{i-1,i}$$

其中 $\mathbf{T}_{i-1,i}$ 是从坐标系 $i-1$ 到坐标系 $i$ 的变换矩阵。

---

## 2. DH参数

### 2.1 DH参数定义

**Denavit-Hartenberg参数**：

| 参数 | 含义 | 符号 |
|------|------|------|
| 连杆长度 | 沿x轴的距离 | $a_i$ |
| 连杆扭转角 | 绕x轴的角度 | $\alpha_i$ |
| 关节偏移 | 沿z轴的距离 | $d_i$ |
| 关节角度 | 绕z轴的角度 | $\theta_i$ |

### 2.2 DH变换矩阵

**从坐标系i到坐标系i+1的变换**：

$$\mathbf{T}_{i,i+1} = \mathbf{R}_z(\theta_{i+1}) \mathbf{D}_z(d_{i+1}) \mathbf{D}_x(a_i) \mathbf{R}_x(\alpha_i)$$

展开为：

$$\mathbf{T}_{i,i+1} = \begin{bmatrix}
\cos\theta_{i+1} & -\sin\theta_{i+1} & 0 & a_i \\
\sin\theta_{i+1}\cos\alpha_i & \cos\theta_{i+1}\cos\alpha_i & -\sin\alpha_i & -d_{i+1}\sin\alpha_i \\
\sin\theta_{i+1}\sin\alpha_i & \cos\theta_{i+1}\sin\alpha_i & \cos\alpha_i & d_{i+1}\cos\alpha_i \\
0 & 0 & 0 & 1
\end{bmatrix}$$

### 2.3 DH参数表

| 连杆i | $a_{i-1}$ | $\alpha_{i-1}$ | $d_i$ | $\theta_i$ |
|-------|-----------|----------------|-------|-----------|
| 1 | 0 | 0 | $d_1$ | $\theta_1$ |
| 2 | $a_1$ | $\alpha_1$ | $d_2$ | $\theta_2$ |
| ... | ... | ... | ... | ... |
| n | $a_{n-1}$ | $\alpha_{n-1}$ | $d_n$ | $\theta_n$ |

---

## 3. 逆运动学

### 3.1 定义

**逆运动学**：已知末端执行器的位姿，计算关节角度。

**问题**：可能有多个解或无解。

### 3.2 封闭解

**条件**：
- 3自由度平面机械臂
- 6自由度球腕机械臂（腕关节三个轴交于一点）

**方法**：
- 几何方法
- 代数方法

### 3.3 数值解法

**梯度下降法**：

$$\mathbf{q}_{k+1} = \mathbf{q}_k + \alpha \mathbf{J}^+(\mathbf{q}_k) \Delta \mathbf{x}$$

其中 $\mathbf{J}^+$ 是伪逆，$\Delta \mathbf{x}$ 是位姿误差。

**Newton-Raphson法**：

$$\mathbf{q}_{k+1} = \mathbf{q}_k + \mathbf{J}^+(\mathbf{q}_k) (\mathbf{x}_{\text{des}} - \mathbf{x}(\mathbf{q}_k))$$

### 3.4 奇异性

**定义**：Jacobian矩阵不满秩的状态。

**影响**：
- 无法沿某些方向运动
- 关节速度趋向无穷大

---

## 4. Jacobian矩阵

### 4.1 定义

**Jacobian矩阵**：描述关节速度与末端执行器速度之间的关系。

$$\mathbf{v} = \mathbf{J}(\mathbf{q}) \dot{\mathbf{q}}$$

其中 $\mathbf{v} = [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]^T$ 是末端执行器的速度向量。

### 4.2 Jacobian矩阵的计算

**几何方法**：

$$\mathbf{J}(\mathbf{q}) = \begin{bmatrix}
\mathbf{z}_0 \times \mathbf{r}_n^0 & \mathbf{z}_1 \times \mathbf{r}_n^1 & \cdots & \mathbf{z}_{n-1} \times \mathbf{r}_n^{n-1} \\
\mathbf{z}_0 & \mathbf{z}_1 & \cdots & \mathbf{z}_{n-1}
\end{bmatrix}$$

其中 $\mathbf{z}_i$ 是第i个关节轴的单位向量，$\mathbf{r}_n^i$ 是从第i个坐标系原点到末端的向量。

### 4.3 雅可比伪逆

**Moore-Penrose伪逆**：

$$\mathbf{J}^+ = \mathbf{J}^T (\mathbf{J} \mathbf{J}^T)^{-1}$$

**最小二乘解**：

$$\dot{\mathbf{q}} = \mathbf{J}^+ \mathbf{v}$$

---

## 5. 在具身智能中的应用

### 5.1 机械臂控制

**轨迹规划**：生成关节空间或笛卡尔空间的轨迹。

**速度控制**：基于Jacobian的速度控制。

### 5.2 机器人操作

**抓取规划**：计算抓取物体所需的关节角度。

**避障**：在运动过程中避开障碍物。

### 5.3 移动机器人

**运动学建模**：建立移动机器人的运动学模型。

**轨迹跟踪**：跟踪期望轨迹。

### 5.4 双臂协调

**协调控制**：控制两个机械臂协同工作。

**力控制**：控制末端执行器的力。

---

## 6. 相关论文

### 经典文献

1. **"Introduction to Robotics"** - Craig (2005)
   - 机器人学经典教材

2. **"Robotics: Modelling, Planning and Control"** - Siciliano et al. (2010)
   - 机器人建模、规划与控制

### 应用相关

3. **"Planning Algorithms"** - LaValle (2006)
   - 运动规划算法

4. **"Numerical Methods for Robotics"** - Murray et al. (1994)
   - 机器人数值方法

---

## 7. 实践练习

### 练习1：2自由度机械臂正运动学

```python
import numpy as np

def dh_transform(theta, d, a, alpha):
    """DH变换矩阵"""
    c = np.cos(theta)
    s = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    T = np.array([
        [c, -s*ca, s*sa, a*c],
        [s, c*ca, -c*sa, a*s],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])
    return T

def forward_kinematics_2dof(q, L1, L2):
    """2自由度平面机械臂正运动学"""
    theta1, theta2 = q
    
    # DH参数
    # 连杆1: a=0, alpha=0, d=0, theta=theta1
    T1 = dh_transform(theta1, 0, 0, 0)
    
    # 连杆2: a=L1, alpha=0, d=0, theta=theta2
    T2 = dh_transform(theta2, 0, L1, 0)
    
    # 末端执行器: a=L2, alpha=0, d=0, theta=0
    T3 = dh_transform(0, 0, L2, 0)
    
    # 总变换
    T = T1 @ T2 @ T3
    
    # 末端位置
    x = T[0, 3]
    y = T[1, 3]
    
    return x, y, T

# 参数
L1 = 1.0  # 连杆1长度
L2 = 1.0  # 连杆2长度

# 关节角度
theta1 = np.pi / 4
theta2 = np.pi / 3

# 计算正运动学
x, y, T = forward_kinematics_2dof([theta1, theta2], L1, L2)

print("2自由度机械臂正运动学:")
print(f"关节角度: theta1={np.degrees(theta1):.1f}°, theta2={np.degrees(theta2):.1f}°")
print(f"末端位置: ({x:.4f}, {y:.4f})")
print("\n变换矩阵:")
print(T)
```

### 练习2：3自由度机械臂正运动学

```python
import numpy as np

def forward_kinematics_3dof(q, L1, L2, L3):
    """3自由度机械臂正运动学"""
    theta1, theta2, theta3 = q
    
    # 基座到关节1
    T0_1 = dh_transform(theta1, L1, 0, np.pi/2)
    
    # 关节1到关节2
    T1_2 = dh_transform(theta2, 0, L2, 0)
    
    # 关节2到末端
    T2_3 = dh_transform(theta3, 0, L3, 0)
    
    # 总变换
    T = T0_1 @ T1_2 @ T2_3
    
    return T

# 参数
L1 = 0.5  # 基座高度
L2 = 1.0  # 连杆1长度
L3 = 1.0  # 连杆2长度

# 关节角度
theta1 = np.pi / 4
theta2 = np.pi / 6
theta3 = np.pi / 3

# 计算正运动学
T = forward_kinematics_3dof([theta1, theta2, theta3], L1, L2, L3)

print("3自由度机械臂正运动学:")
print(f"关节角度: theta1={np.degrees(theta1):.1f}°, theta2={np.degrees(theta2):.1f}°, theta3={np.degrees(theta3):.1f}°")
print("\n末端位置:")
print(f"  x = {T[0, 3]:.4f}")
print(f"  y = {T[1, 3]:.4f}")
print(f"  z = {T[2, 3]:.4f}")
print("\n旋转矩阵:")
print(T[:3, :3])
```

### 练习3：2自由度机械臂逆运动学

```python
import numpy as np

def inverse_kinematics_2dof(x, y, L1, L2):
    """2自由度平面机械臂逆运动学"""
    # 计算手腕位置
    r = np.sqrt(x**2 + y**2)
    
    if r > L1 + L2 or r < np.abs(L1 - L2):
        return None  # 无解
    
    # 计算theta2
    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)  # 数值稳定性
    
    # 两个解
    theta2_sol1 = np.arccos(cos_theta2)
    theta2_sol2 = -np.arccos(cos_theta2)
    
    # 计算theta1
    theta1_sol1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2_sol1), L1 + L2 * np.cos(theta2_sol1))
    theta1_sol2 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2_sol2), L1 + L2 * np.cos(theta2_sol2))
    
    return [(theta1_sol1, theta2_sol1), (theta1_sol2, theta2_sol2)]

# 参数
L1 = 1.0
L2 = 1.0

# 目标位置
x_target = 1.5
y_target = 0.5

# 计算逆运动学
solutions = inverse_kinematics_2dof(x_target, y_target, L1, L2)

if solutions is None:
    print("无解")
else:
    print("2自由度机械臂逆运动学解:")
    for i, (theta1, theta2) in enumerate(solutions):
        print(f"解{i+1}: theta1={np.degrees(theta1):.1f}°, theta2={np.degrees(theta2):.1f}°")
        
        # 验证
        x, y, _ = forward_kinematics_2dof([theta1, theta2], L1, L2)
        print(f"  验证位置: ({x:.4f}, {y:.4f})")
```

### 练习4：Jacobian矩阵计算

```python
import numpy as np

def jacobian_2dof(q, L1, L2):
    """2自由度平面机械臂Jacobian矩阵"""
    theta1, theta2 = q
    
    J = np.array([
        [-L1*np.sin(theta1) - L2*np.sin(theta1+theta2), -L2*np.sin(theta1+theta2)],
        [L1*np.cos(theta1) + L2*np.cos(theta1+theta2), L2*np.cos(theta1+theta2)],
        [0, 0],
        [0, 0],
        [0, 0],
        [1, 1]
    ])
    return J

# 参数
L1 = 1.0
L2 = 1.0

# 关节角度
theta1 = np.pi / 4
theta2 = np.pi / 3

# 计算Jacobian矩阵
J = jacobian_2dof([theta1, theta2], L1, L2)

print("2自由度机械臂Jacobian矩阵:")
print(J)

# 计算奇异值
svd = np.linalg.svd(J, compute_uv=False)
print("\n奇异值:", svd)

# 条件数
cond_num = svd[0] / svd[-1] if svd[-1] > 1e-10 else np.inf
print(f"条件数: {cond_num:.2f}")
```

### 练习5：数值逆运动学

```python
import numpy as np

def numerical_inverse_kinematics(target_pos, q_init, L1, L2, max_iter=100, tol=1e-6):
    """数值逆运动学（Newton-Raphson法）"""
    q = np.array(q_init, dtype=float)
    
    for i in range(max_iter):
        # 当前末端位置
        x, y, _ = forward_kinematics_2dof(q, L1, L2)
        
        # 位置误差
        error = np.array([target_pos[0] - x, target_pos[1] - y])
        
        if np.linalg.norm(error) < tol:
            print(f"收敛于第{i+1}次迭代")
            return q
        
        # Jacobian矩阵（2x2简化版）
        J = jacobian_2dof(q, L1, L2)[:2, :]
        
        # 伪逆
        J_pinv = np.linalg.pinv(J)
        
        # 更新关节角度
        q += J_pinv @ error
    
    print("未收敛")
    return q

# 参数
L1 = 1.0
L2 = 1.0

# 目标位置
target_pos = [1.5, 0.5]

# 初始猜测
q_init = [0.1, 0.1]

# 数值求解
q_solution = numerical_inverse_kinematics(target_pos, q_init, L1, L2)

print("\n数值逆运动学解:")
print(f"关节角度: theta1={np.degrees(q_solution[0]):.4f}°, theta2={np.degrees(q_solution[1]):.4f}°")

# 验证
x, y, _ = forward_kinematics_2dof(q_solution, L1, L2)
print(f"末端位置: ({x:.4f}, {y:.4f})")
print(f"目标位置: ({target_pos[0]}, {target_pos[1]})")
print(f"误差: {np.linalg.norm([x-target_pos[0], y-target_pos[1]]):.6f}")
```

---

**运动学部分完成！** 🎉

**下一步**：[动力学](../dynamics/README.md)