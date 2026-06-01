# 6.5 机械臂规划

## 目录

- [1. 引言](#1-引言)
- [2. 正运动学](#2-正运动学)
- [3. 逆运动学](#3-逆运动学)
- [4. 机械臂规划器](#4-机械臂规划器)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 机械臂规划挑战

- 高维构型空间（6-7 DoF）
- 复杂碰撞检查
- 奇异性问题
- 运动学和动力学约束

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

---

## 2. 正运动学

### 2.1 连杆变换

```python
class Link:
    """连杆"""
    
    def __init__(self, a, alpha, d, theta_offset):
        self.a = a
        self.alpha = alpha
        self.d = d
        self.theta_offset = theta_offset
    
    def transform(self, theta):
        """DH变换矩阵"""
        t = theta + self.theta_offset
        ct = np.cos(t)
        st = np.sin(t)
        ca = np.cos(self.alpha)
        sa = np.sin(self.alpha)
        
        return np.array([
            [ct, -st*ca,  st*sa, self.a*ct],
            [st,  ct*ca, -ct*sa, self.a*st],
            [0,     sa,     ca,    self.d],
            [0,      0,      0,        1]
        ])


class SerialManipulator:
    """串联机械臂"""
    
    def __init__(self, links):
        self.links = links
        self.dof = len(links)
    
    def forward_kinematics(self, q, return_all=False):
        """正运动学"""
        T = np.eye(4)
        transforms = [T.copy()]
        
        for i, theta in enumerate(q):
            Ti = self.links[i].transform(theta)
            T = T @ Ti
            if return_all:
                transforms.append(T.copy())
        
        if return_all:
            return T, transforms
        return T
    
    def get_joint_positions(self, q):
        """获取关节位置"""
        T, transforms = self.forward_kinematics(q, return_all=True)
        positions = []
        for T in transforms:
            positions.append(T[:3, 3])
        return positions
```

### 2.2 2-DoF planar arm示例

```python
def create_planar_2dof(l1=1.0, l2=1.0):
    """创建2D平面臂"""
    # DH参数简化
    links = [
        Link(a=l1, alpha=0, d=0, theta_offset=0),
        Link(a=l2, alpha=0, d=0, theta_offset=0),
    ]
    return SerialManipulator(links)
```

---

## 3. 逆运动学

### 3.1 解析逆运动学（2-DoF）

```python
def ik_2dof(x, y, l1=1.0, l2=1.0):
    """2-DoF平面臂解析逆运动学"""
    d = np.sqrt(x**2 + y**2)
    
    if d > l1 + l2 or d < abs(l1 - l2):
        return None  # 不可达
    
    # 余弦定理
    cos_theta2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1, 1)
    theta2_a = np.arccos(cos_theta2)
    theta2_b = -theta2_a
    
    # theta1
    k1 = l1 + l2 * cos_theta2
    k2 = l2 * np.sin(theta2_a)
    theta1_a = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    k2 = l2 * np.sin(theta2_b)
    theta1_b = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    return [np.array([theta1_a, theta2_a]), np.array([theta1_b, theta2_b])]
```

### 3.2 数值逆运动学（雅可比转置法）

```python
class NumericalIK:
    """数值逆运动学"""
    
    def __init__(self, manipulator):
        self.manipulator = manipulator
    
    def jacobian(self, q, delta=1e-6):
        """数值雅可比"""
        n = len(q)
        J = np.zeros((6, n))
        
        T_base = self.manipulator.forward_kinematics(q)
        
        for i in range(n):
            # 扰动
            q_plus = q.copy()
            q_plus[i] += delta
            T_plus = self.manipulator.forward_kinematics(q_plus)
            
            q_minus = q.copy()
            q_minus[i] -= delta
            T_minus = self.manipulator.forward_kinematics(q_minus)
            
            # 有限差分
            dT = (T_plus - T_minus) / (2 * delta)
            
            # 线性部分
            J[:3, i] = dT[:3, 3]
            
            # 角速度（简化）
            R_diff = dT[:3, :3] @ T_base[:3, :3].T
            J[3, i] = R_diff[2, 1]
            J[4, i] = R_diff[0, 2]
            J[5, i] = R_diff[1, 0]
        
        return J
    
    def solve(self, q_init, target_pos, target_rot=None, 
              max_iter=1000, step_size=0.5, tol=1e-4):
        """解逆运动学"""
        q = np.array(q_init)
        
        for i in range(max_iter):
            T = self.manipulator.forward_kinematics(q)
            
            # 位置误差
            error_pos = target_pos - T[:3, 3]
            
            # 雅可比
            J = self.jacobian(q)
            
            # 仅考虑位置
            J_pos = J[:3, :]
            
            # 伪逆
            q_dot = J_pos.T @ error_pos
            q = q + step_size * q_dot
            
            if np.linalg.norm(error_pos) < tol:
                return q, True
        
        return q, False
```

---

## 4. 机械臂规划器

### 4.1 简单碰撞检查

```python
class ArmCollisionChecker:
    """机械臂碰撞检查"""
    
    def __init__(self, arm, obstacles):
        self.arm = arm
        self.obstacles = obstacles
    
    def check(self, q):
        """检查构型q是否碰撞"""
        joint_pos = self.arm.get_joint_positions(q)
        
        # 检查连杆
        for i in range(len(joint_pos) - 1):
            p1 = joint_pos[i]
            p2 = joint_pos[i + 1]
            
            # 线段采样检查
            for t in np.linspace(0, 1, 10):
                p = p1 + t * (p2 - p1)
                
                for obs in self.obstacles:
                    if obs.is_in_collision(p):
                        return True
        
        return False
```

---

## 5. 实践练习

### 练习1：正运动学演示

```python
def exercise_forward_kinematics():
    """正运动学演示"""
    print("=== 正运动学 ===")
    
    arm = create_planar_2dof(l1=1.0, l2=0.8)
    
    q = [np.pi/4, -np.pi/3]
    
    joint_pos = arm.get_joint_positions(q)
    joint_pos = np.array(joint_pos)
    
    plt.figure(figsize=(10, 10))
    
    # 画臂
    plt.plot(joint_pos[:, 0], joint_pos[:, 1], 'bo-', linewidth=5, markersize=10)
    
    # 画工作空间
    theta1 = np.linspace(0, 2*np.pi, 100)
    theta2 = np.linspace(0, 2*np.pi, 100)
    T1, T2 = np.meshgrid(theta1, theta2)
    X = np.cos(T1) + 0.8*np.cos(T1+T2)
    Y = np.sin(T1) + 0.8*np.sin(T1+T2)
    plt.plot(X.flatten(), Y.flatten(), 'r,', alpha=0.3, label='Workspace')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Forward Kinematics')
    plt.legend()
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.savefig('forward_kinematics.png')
    print("正运动学演示图已保存")

# exercise_forward_kinematics()
```

### 练习2：逆运动学演示

```python
def exercise_inverse_kinematics():
    """逆运动学演示"""
    print("=== 逆运动学 ===")
    
    arm = create_planar_2dof(l1=1.0, l2=0.8)
    
    target = np.array([1.2, 1.0])
    
    # 解析解
    solutions = ik_2dof(target[0], target[1], l1=1.0, l2=0.8)
    
    plt.figure(figsize=(10, 10))
    
    if solutions:
        colors = ['blue', 'green']
        labels = ['Elbow Up', 'Elbow Down']
        for i, q in enumerate(solutions):
            joint_pos = arm.get_joint_positions(q)
            joint_pos = np.array(joint_pos)
            plt.plot(joint_pos[:, 0], joint_pos[:, 1], 'o-', 
                    linewidth=4, markersize=10, color=colors[i], 
                    label=labels[i])
    
    plt.plot(target[0], target[1], 'r*', markersize=20, label='Target')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Inverse Kinematics')
    plt.legend()
    plt.axis('equal')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.savefig('inverse_kinematics.png')
    print("逆运动学演示图已保存")

# exercise_inverse_kinematics()
```

---

恭喜！你已经完成了运动规划部分的全部内容！

---

## 参考文献

1. Craig, J. J. (2005). Introduction to Robotics: Mechanics and Control (3rd ed.).
2. Siciliano, B., et al. (2009). Robotics: Modelling, Planning and Control.
3. Murray, R. M., et al. (1994). A Mathematical Introduction to Robotic Manipulation.
4. Choset, H., et al. (2005). Principles of Robot Motion: Theory, Algorithms, and Implementations.
5. Bohren, J., et al. (2011). Towards Fully Autonomous Operations of the PR2: Lessons from the Darpa Robotics Challenge.
