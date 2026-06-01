# 3.3 位姿估计

## 目录

- [1. 引言](#1-引言)
- [2. 旋转表示](#2-旋转表示)
- [3. PnP位姿估计](#3-pnp位姿估计)
- [4. ICP点云配准](#4-icp点云配准)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 位姿估计的重要性

**位姿估计**确定物体或传感器在空间中的位置和姿态（旋转+平移），是机器人、AR/VR、自动驾驶等领域的关键技术。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **AR应用** | 虚拟物体放置 | 物体识别与跟踪 |
| **机器人操作** | 机械臂抓取 | 目标物体位姿估计 |
| **自主导航** | 机器人定位 | 地图匹配定位 |
| **运动捕捉** | 人体/物体跟踪 | 动画制作、体育分析 |

---

## 2. 旋转表示

### 2.1 欧拉角、旋转矩阵、四元数

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Rotation:
    """旋转表示与转换"""
    
    @staticmethod
    def euler_to_rotmat(euler, order='xyz'):
        """
        欧拉角转旋转矩阵
        
        参数:
            euler: (roll, pitch, yaw)
            order: 旋转顺序
        
        返回:
            R: 3x3旋转矩阵
        """
        roll, pitch, yaw = euler
        
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        if order == 'xyz':
            return Rz @ Ry @ Rx
        elif order == 'zyx':
            return Rx @ Ry @ Rz
        else:
            raise ValueError(f"未知旋转顺序: {order}")
    
    @staticmethod
    def rotmat_to_euler(R, order='xyz'):
        """旋转矩阵转欧拉角"""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    @staticmethod
    def rotmat_to_quaternion(R):
        """旋转矩阵转四元数"""
        tr = np.trace(R)
        
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        
        return np.array([qw, qx, qy, qz])
    
    @staticmethod
    def quaternion_to_rotmat(q):
        """四元数转旋转矩阵"""
        qw, qx, qy, qz = q
        
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        
        return R
    
    @staticmethod
    def quaternion_multiply(q1, q2):
        """四元数乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    @staticmethod
    def slerp(q1, q2, t):
        """球面线性插值"""
        dot = np.dot(q1, q2)
        
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        theta0 = np.arccos(dot)
        theta = theta0 * t
        
        q3 = q2 - q1 * dot
        q3 = q3 / np.linalg.norm(q3)
        
        return q1 * np.cos(theta) + q3 * np.sin(theta)
```

### 2.2 旋转向量与轴角

```python
class AxisAngle:
    """轴角表示"""
    
    @staticmethod
    def rotvec_to_rotmat(rotvec):
        """旋转向量转旋转矩阵"""
        theta = np.linalg.norm(rotvec)
        if theta < 1e-10:
            return np.eye(3)
        
        k = rotvec / theta
        kx, ky, kz = k
        
        K = np.array([
            [0, -kz, ky],
            [kz, 0, -kx],
            [-ky, kx, 0]
        ])
        
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R
    
    @staticmethod
    def rotmat_to_rotvec(R):
        """旋转矩阵转旋转向量"""
        theta = np.arccos((np.trace(R) - 1) / 2)
        
        if theta < 1e-10:
            return np.zeros(3)
        
        k = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(theta))
        
        return k * theta
```

---

## 3. PnP位姿估计

### 3.1 PnP算法

```python
class PnPEstimator:
    """PnP位姿估计器"""
    
    def __init__(self, camera_matrix, dist_coeffs=None):
        """
        初始化PnP估计器
        
        参数:
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(4)
    
    def solve_pnp(self, object_points, image_points, method='solvepnp'):
        """
        求解PnP问题
        
        参数:
            object_points: Nx3 3D点
            image_points: Nx2 2D点
            method: 方法: 'solvepnp', 'ransac', 'epnp'
        
        返回:
            success: 是否成功
            rvec: 旋转向量
            tvec: 平移向量
        """
        object_points = object_points.astype(np.float32)
        image_points = image_points.astype(np.float32)
        
        if method == 'solvepnp':
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                self.camera_matrix, self.dist_coeffs
            )
        elif method == 'ransac':
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points,
                self.camera_matrix, self.dist_coeffs
            )
        elif method == 'epnp':
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points,
                self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP
            )
        else:
            raise ValueError(f"未知方法: {method}")
        
        return success, rvec, tvec
    
    def project_points(self, object_points, rvec, tvec):
        """
        投影3D点到图像平面
        
        参数:
            object_points: Nx3 3D点
            rvec: 旋转向量
            tvec: 平移向量
        
        返回:
            image_points: Nx2 投影点
        """
        image_points, _ = cv2.projectPoints(
            object_points, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
        )
        return image_points.reshape(-1, 2)
    
    def compute_reprojection_error(self, object_points, image_points, rvec, tvec):
        """计算重投影误差"""
        projected = self.project_points(object_points, rvec, tvec)
        errors = np.linalg.norm(projected - image_points, axis=1)
        return np.mean(errors), errors
    
    def visualize_result(self, img, object_points, image_points, rvec, tvec, axis_len=0.1):
        """
        可视化PnP结果
        
        参数:
            img: 图像
            object_points: 3D点
            image_points: 2D点
            rvec: 旋转向量
            tvec: 平移向量
            axis_len: 坐标轴长度
        
        返回:
            vis_img: 可视化图像
        """
        vis_img = img.copy()
        
        # 绘制坐标轴
        axis_points = np.float32([
            [0, 0, 0],
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len]
        ])
        
        axis_image_points = self.project_points(axis_points, rvec, tvec)
        
        cv2.line(vis_img, tuple(axis_image_points[0].astype(int)),
                 tuple(axis_image_points[1].astype(int)), (0, 0, 255), 3)
        cv2.line(vis_img, tuple(axis_image_points[0].astype(int)),
                 tuple(axis_image_points[2].astype(int)), (0, 255, 0), 3)
        cv2.line(vis_img, tuple(axis_image_points[0].astype(int)),
                 tuple(axis_image_points[3].astype(int)), (255, 0, 0), 3)
        
        # 绘制重投影点
        projected = self.project_points(object_points, rvec, tvec)
        for p in projected:
            cv2.circle(vis_img, tuple(p.astype(int)), 3, (255, 0, 255), -1)
        
        return vis_img
```

### 3.2 立方体跟踪示例

```python
def cube_tracking_demo():
    """立方体跟踪演示"""
    print("=== 立方体跟踪演示 ===")
    
    # 相机内参
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])
    
    # 立方体3D点
    cube_points = np.array([
        [-0.5, -0.5, 0], [0.5, -0.5, 0],
        [0.5, 0.5, 0], [-0.5, 0.5, 0],
        [-0.5, -0.5, 1], [0.5, -0.5, 1],
        [0.5, 0.5, 1], [-0.5, 0.5, 1]
    ])
    
    # 模拟真实位姿
    R_true = Rotation.euler_to_rotmat([0.2, 0.3, 0.4])
    t_true = np.array([0, 0, 3])
    
    # 创建PnP估计器
    estimator = PnPEstimator(K)
    
    # 投影到图像
    rvec_true, _ = cv2.Rodrigues(R_true)
    image_points = estimator.project_points(cube_points, rvec_true, t_true)
    
    # 添加噪声
    image_points_noisy = image_points + np.random.normal(0, 2, image_points.shape)
    
    # 求解PnP
    success, rvec, tvec = estimator.solve_pnp(cube_points, image_points_noisy, method='ransac')
    
    if success:
        # 计算误差
        R_est, _ = cv2.Rodrigues(rvec)
        t_est = tvec.flatten()
        
        t_error = np.linalg.norm(t_est - t_true)
        
        angle_err = np.arccos((np.trace(R_true.T @ R_est) - 1) / 2) * 180 / np.pi
        
        print(f"平移误差: {t_error:.4f} m")
        print(f"角度误差: {angle_err:.4f} deg")
        
        # 重投影误差
        mean_error, _ = estimator.compute_reprojection_error(
            cube_points, image_points_noisy, rvec, tvec
        )
        print(f"重投影误差: {mean_error:.4f} pixels")

# cube_tracking_demo()
```

---

## 4. ICP点云配准

### 4.1 ICP实现

```python
class ICP:
    """迭代最近点算法"""
    
    def __init__(self, max_iterations=50, tolerance=1e-6):
        """
        初始化ICP
        
        参数:
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def find_closest_points(self, source, target):
        """
        寻找最近点
        
        参数:
            source: Nx3 源点云
            target: Mx3 目标点云
        
        返回:
            matched_source: 匹配后的源点
            matched_target: 匹配后的目标点
        """
        from scipy.spatial import KDTree
        
        tree = KDTree(target)
        distances, indices = tree.query(source)
        
        return source, target[indices]
    
    def compute_rigid_transform(self, source, target):
        """
        计算刚性变换
        
        参数:
            source: Nx3 源点
            target: Nx3 目标点
        
        返回:
            R: 旋转矩阵
            t: 平移向量
        """
        assert source.shape == target.shape
        
        # 计算中心
        centroid_source = np.mean(source, axis=0)
        centroid_target = np.mean(target, axis=0)
        
        # 去中心
        source_centered = source - centroid_source
        target_centered = target - centroid_target
        
        # SVD
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        
        R = Vt.T @ U.T
        
        # 处理反射
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = centroid_target - R @ centroid_source
        
        return R, t
    
    def compute_error(self, source, target, R, t):
        """计算配准误差"""
        transformed = (R @ source.T).T + t
        return np.mean(np.linalg.norm(transformed - target, axis=1))
    
    def register(self, source, target, R_init=None, t_init=None):
        """
        配准点云
        
        参数:
            source: Nx3 源点云
            target: Mx3 目标点云
            R_init: 初始旋转
            t_init: 初始平移
        
        返回:
            R: 最终旋转矩阵
            t: 最终平移向量
            errors: 误差历史
        """
        if R_init is None:
            R = np.eye(3)
        else:
            R = R_init.copy()
        
        if t_init is None:
            t = np.zeros(3)
        else:
            t = t_init.copy()
        
        errors = []
        
        for i in range(self.max_iterations):
            # 变换源点云
            source_transformed = (R @ source.T).T + t
            
            # 找最近点
            src_matched, target_matched = self.find_closest_points(source_transformed, target)
            
            # 计算变换
            delta_R, delta_t = self.compute_rigid_transform(source_transformed, target_matched)
            
            # 更新变换
            R = delta_R @ R
            t = delta_R @ t + delta_t
            
            # 计算误差
            error = self.compute_error(source, target, R, t)
            errors.append(error)
            
            # 检查收敛
            if i > 0 and abs(errors[-2] - errors[-1]) < self.tolerance:
                print(f"ICP在第 {i+1} 次迭代收敛")
                break
        
        return R, t, errors
```

### 4.2 ICP演示

```python
def icp_demo():
    """ICP演示"""
    print("=== ICP配准演示 ===")
    
    # 生成点云
    np.random.seed(42)
    num_points = 100
    
    # 源点云
    source = np.random.randn(num_points, 3)
    
    # 目标点云：添加变换和噪声
    R_true = Rotation.euler_to_rotmat([0.3, 0.4, 0.5])
    t_true = np.array([0.5, 0.3, 0.2])
    
    target = (R_true @ source.T).T + t_true + np.random.randn(num_points, 3) * 0.02
    
    # ICP配准
    icp = ICP(max_iterations=50)
    R, t, errors = icp.register(source, target)
    
    # 计算误差
    angle_err = np.arccos((np.trace(R_true.T @ R) - 1) / 2) * 180 / np.pi
    t_err = np.linalg.norm(t - t_true)
    
    print(f"旋转误差: {angle_err:.4f} deg")
    print(f"平移误差: {t_err:.4f} m")
    print(f"最终配准误差: {errors[-1]:.6f}")
    
    # 绘制误差曲线
    plt.figure(figsize=(8, 4))
    plt.plot(errors)
    plt.xlabel('迭代次数')
    plt.ylabel('误差')
    plt.title('ICP收敛曲线')
    plt.grid(True)
    plt.savefig('icp_errors.png')
    print("误差曲线已保存")

# icp_demo()
```

---

## 5. 实践练习

### 练习1：旋转表示转换

```python
def exercise_rotation_representation():
    """旋转表示练习"""
    print("=== 旋转表示练习 ===")
    
    # 欧拉角
    euler = np.array([0.2, 0.3, 0.4])
    print(f"欧拉角: {euler}")
    
    # 转旋转矩阵
    R = Rotation.euler_to_rotmat(euler)
    print(f"旋转矩阵:\n{R}")
    
    # 转回欧拉角
    euler_back = Rotation.rotmat_to_euler(R)
    print(f"恢复欧拉角: {euler_back}")
    
    # 转四元数
    q = Rotation.rotmat_to_quaternion(R)
    print(f"四元数: {q}")
    
    # 转回旋转矩阵
    R_back = Rotation.quaternion_to_rotmat(q)
    print(f"恢复旋转矩阵:\n{R_back}")
    
    # 旋转向量
    rotvec = AxisAngle.rotmat_to_rotvec(R)
    print(f"旋转向量: {rotvec}")
    
    # 转回旋转矩阵
    R_back2 = AxisAngle.rotvec_to_rotmat(rotvec)
    print(f"恢复旋转矩阵:\n{R_back2}")

# exercise_rotation_representation()
```

### 练习2：PnP合成数据

```python
def exercise_pnp_synthetic():
    """PnP合成数据练习"""
    print("=== PnP合成数据练习 ===")
    
    # 相机内参
    K = np.array([
        [1000, 0, 320],
        [0, 1000, 240],
        [0, 0, 1]
    ])
    
    # 生成3D点
    num_points = 20
    object_points = np.random.randn(num_points, 3) * 0.5
    object_points[:, 2] = object_points[:, 2] + 2  # 确保在前方
    
    # 真实位姿
    R_true = Rotation.euler_to_rotmat([0.1, 0.2, 0.3])
    t_true = np.array([0.1, 0.2, 2.0])
    
    # 创建估计器
    estimator = PnPEstimator(K)
    
    # 投影
    rvec_true, _ = cv2.Rodrigues(R_true)
    image_points = estimator.project_points(object_points, rvec_true, t_true)
    
    # 添加噪声
    noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
    methods = ['solvepnp', 'ransac', 'epnp']
    
    results = {}
    
    for noise in noise_levels:
        results[noise] = {}
        for method in methods:
            image_points_noisy = image_points + np.random.normal(0, noise, image_points.shape)
            success, rvec, tvec = estimator.solve_pnp(object_points, image_points_noisy, method=method)
            
            if success:
                t_est = tvec.flatten()
                t_error = np.linalg.norm(t_est - t_true)
                results[noise][method] = t_error
            else:
                results[noise][method] = float('inf')
    
    print("\n平移误差 (m):")
    print("噪声\tSolvePnP\tRANSAC\tEPNP")
    for noise in noise_levels:
        print(f"{noise}\t{results[noise]['solvepnp']:.4f}\t"
              f"{results[noise]['ransac']:.4f}\t{results[noise]['epnp']:.4f}")

# exercise_pnp_synthetic()
```

### 练习3：ICP鲁棒性

```python
def exercise_icp_robustness():
    """ICP鲁棒性练习"""
    print("=== ICP鲁棒性练习 ===")
    
    np.random.seed(42)
    
    # 源点云
    source = np.random.randn(200, 3)
    
    # 真实变换
    R_true = Rotation.euler_to_rotmat([0.2, 0.3, 0.4])
    t_true = np.array([0.5, 0.3, 0.2])
    
    # 测试不同噪声水平
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    outlier_ratios = [0, 0.1, 0.2, 0.3]
    
    for noise in noise_levels:
        for outlier in outlier_ratios:
            # 生成目标点云
            target = (R_true @ source.T).T + t_true + np.random.randn(*source.shape) * noise
            
            # 添加外点
            num_outliers = int(len(source) * outlier)
            if num_outliers > 0:
                target[-num_outliers:] = np.random.randn(num_outliers, 3) * 10
            
            # ICP配准
            icp = ICP(max_iterations=100)
            R, t, errors = icp.register(source, target)
            
            # 计算误差
            angle_err = np.arccos((np.trace(R_true.T @ R) - 1) / 2) * 180 / np.pi
            t_err = np.linalg.norm(t - t_true)
            
            print(f"噪声={noise:.2f}, 外点={outlier:.2f}: "
                  f"旋转误差={angle_err:.4f}°, 平移误差={t_err:.4f}m")

# exercise_icp_robustness()
```

---

**下一节**：[SLAM基础](04-slam-basics.md)

---

## 参考文献

1. Fischler, M. A., & Bolles, R. C. (1981). Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography.
2. Besl, P. J., & McKay, N. D. (1992). A Method for Registration of 3-D Shapes.
3. Lepetit, V., et al. (2009). EPnP: An Accurate O(n) Solution to the PnP Problem.
4. Hartley, R., & Zisserman, A. (2004). Multiple View Geometry in Computer Vision.
