# 1.2 对极几何与本质矩阵

## 目录

- [1. 问题背景](#1-问题背景)
- [2. 对极几何基础](#2-对极几何基础)
- [3. 本质矩阵与基础矩阵](#3-本质矩阵与基础矩阵)
- [4. 估计方法](#4-估计方法)
- [5. 重要论文详解](#5-重要论文详解)
- [6. 在SLAM中的应用](#6-在slam中的应用)
- [7. 实践练习](#7-实践练习)

---

## 1. 问题背景

### 1.1 为什么需要对极几何？

在双目视觉或连续帧之间，我们需要建立对应点之间的几何关系：

1. **相机运动估计**：通过对应点计算相机的位姿变化
2. **深度估计**：利用对极约束估计场景深度
3. **消除歧义**：对极几何提供了强几何约束，可以排除错误匹配

### 1.2 核心问题

给定两幅图像中的对应点，如何确定：
- 相机之间的相对位姿
- 3D点的位置

---

## 2. 对极几何基础

### 2.1 基本概念

**对极点（Epipole）**：
- 相机光心在另一幅图像上的投影
- 所有对极线都经过对极点

**对极线（Epipolar Line）**：
- 空间点在一幅图像上的投影与对极点的连线
- 空间点在另一幅图像上的投影必在对应的对极线上

**对极约束（Epipolar Constraint）**：
- 对于任意一对对应点 $x_1$ 和 $x_2$，必须满足对极约束
- 这是一个强几何约束，用于验证匹配的正确性

### 2.2 对极几何关系

**几何关系**：

设两个相机的光心分别为 $O_1$ 和 $O_2$，空间点为 $P$。

- $x_1$ 是 $P$ 在图像1上的投影
- $x_2$ 是 $P$ 在图像2上的投影
- $e_1$ 是 $O_2$ 在图像1上的投影（对极点）
- $e_2$ 是 $O_1$ 在图像2上的投影（对极点）

对极约束的几何意义：点 $x_1$、$e_1$、$O_1$、$O_2$、$x_2$、$e_2$ 共面。

---

## 3. 本质矩阵与基础矩阵

### 3.1 本质矩阵（Essential Matrix）

**定义**：描述两个相机之间的旋转和平移关系。

**公式**：

$$\mathbf{x}_2^T \mathbf{E} \mathbf{x}_1 = 0$$

其中 $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$。

**性质**：
- 本质矩阵是一个3x3矩阵
- 本质矩阵有5个自由度（旋转3个，平移2个）
- 本质矩阵的奇异值为 $[\sigma, \sigma, 0]$

**从本质矩阵恢复位姿**：

$$\mathbf{E} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

其中 $\mathbf{\Sigma} = \text{diag}(\sigma, \sigma, 0)$。

可能的旋转和平移组合：
$$\mathbf{R}_1 = \mathbf{U} \mathbf{R}_z^T(\pi/2) \mathbf{V}^T$$
$$\mathbf{R}_2 = \mathbf{U} \mathbf{R}_z(\pi/2) \mathbf{V}^T$$
$$\mathbf{t}_1 = \mathbf{U}_{:,3}$$
$$\mathbf{t}_2 = -\mathbf{U}_{:,3}$$

需要通过三角化验证选择正确的解。

### 3.2 基础矩阵（Fundamental Matrix）

**定义**：描述两个图像平面之间的投影关系，考虑了相机内参。

**公式**：

$$\mathbf{x}_2^T \mathbf{F} \mathbf{x}_1 = 0$$

**与本质矩阵的关系**：

$$\mathbf{F} = \mathbf{K}_2^{-T} \mathbf{E} \mathbf{K}_1^{-1}$$

其中 $\mathbf{K}_1$ 和 $\mathbf{K}_2$ 分别是两个相机的内参矩阵。

---

## 4. 估计方法

### 4.1 八点法（Eight-Point Algorithm）

**核心思想**：使用8对对应点估计基础矩阵。

**步骤**：

1. **归一化**：对图像坐标进行归一化处理
2. **构建方程**：每对对应点提供一个方程 $\mathbf{x}_2^T \mathbf{F} \mathbf{x}_1 = 0$
3. **求解**：使用SVD求解线性方程组
4. **约束**：强制基础矩阵满足秩2约束

**代码流程**：
```python
# 构建9xN的系数矩阵A
for i in range(8):
    u1, v1 = x1[i]
    u2, v2 = x2[i]
    A[i] = [u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2, 1]

# SVD分解
U, S, V = np.linalg.svd(A)
F = V[-1].reshape(3, 3)

# 强制秩2约束
U, S, V = np.linalg.svd(F)
S[2] = 0
F = U @ np.diag(S) @ V
```

### 4.2 五点法（Five-Point Algorithm）

**核心思想**：使用5对对应点估计本质矩阵。

**优势**：
- 所需点数更少
- 精度更高
- 能处理退化情况

**难点**：
- 非线性求解
- 可能有多个解（最多10个）

### 4.3 RANSAC方法

**核心思想**：通过随机采样和验证来鲁棒估计模型。

**步骤**：

1. **随机采样**：选择最小样本集（如8对点）
2. **估计模型**：使用八点法估计基础矩阵
3. **验证**：计算所有点对的重投影误差
4. **迭代**：重复多次，选择内点最多的模型

---

## 5. 重要论文详解

### 5.1 An efficient solution to the five-point relative pose problem (2003)

**作者**：David Nister

**发表会议**：PAMI

**问题背景**：
- 传统的八点法需要8对对应点
- 在特征点较少时无法使用
- 需要一种更高效的方法

**核心贡献**：
1. **五点算法**：首次提出使用5对点估计本质矩阵
2. **代数几何方法**：将问题转化为多项式方程组求解
3. **高效求解**：使用Gröbner基方法求解，最多产生10个解

**为什么重要**：
- 将所需点数从8个减少到5个
- 提高了在特征点稀少场景下的鲁棒性
- 成为后续SLAM系统的标准方法

**算法流程**：
1. 建立以本质矩阵元素为变量的多项式方程
2. 使用Gröbner基求解方程组
3. 验证每个解，选择正确的一个

### 5.2 A computer algorithm for reconstructing a scene from two projections (1981)

**作者**：Longuet-Higgins

**发表期刊**：Nature

**问题背景**：
- 早期计算机视觉中缺乏可靠的立体匹配方法
- 需要一种从两幅图像恢复3D结构的方法

**核心贡献**：
1. **本质矩阵概念**：首次提出本质矩阵的概念
2. **对极几何理论**：建立了对极约束的数学基础
3. **八点法**：提出了估计基础矩阵的八点法

**为什么重要**：
- 奠定了立体视觉的理论基础
- 本质矩阵成为计算机视觉的核心概念
- 开创了从多视图恢复3D结构的研究方向

### 5.3 Multiple View Geometry in Computer Vision (2004)

**作者**：Richard Hartley, Andrew Zisserman

**发表形式**：书籍

**问题背景**：
- 需要一本系统讲解多视图几何的教科书
- 整合当时分散的研究成果

**核心贡献**：
1. **系统整理**：全面整理了多视图几何的理论体系
2. **统一框架**：将各种几何概念统一到射影几何框架下
3. **实用算法**：提供了大量实用的算法和实现细节

**为什么重要**：
- 成为计算机视觉领域的经典教材
- 被广泛用于研究生教学和科研参考
- 对后续SLAM和三维重建研究产生深远影响

---

## 6. 在SLAM中的应用

### 6.1 视觉里程计

**初始化**：使用对极几何估计初始相机运动
**帧间跟踪**：通过对极约束验证匹配的正确性

### 6.2 三角化

**深度估计**：使用两帧图像和本质矩阵计算3D点位置

### 6.3 回环检测

**验证**：使用对极几何验证候选回环的正确性

---

## 7. 实践练习

### 练习1：使用八点法估计基础矩阵

```python
import numpy as np
from scipy.linalg import svd

def eight_point_algorithm(x1, x2):
    """
    使用八点法估计基础矩阵
    x1, x2: 对应点对，形状为(N, 2)
    """
    N = x1.shape[0]
    
    # 构建系数矩阵
    A = np.zeros((N, 9))
    for i in range(N):
        u1, v1 = x1[i]
        u2, v2 = x2[i]
        A[i] = [u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2, 1]
    
    # SVD分解
    _, _, V = svd(A)
    F = V[-1].reshape(3, 3)
    
    # 强制秩2约束
    U, S, V = svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V
    
    return F

# 示例：生成模拟数据
np.random.seed(42)
N = 10
x1 = np.random.rand(N, 2) * 100
x2 = x1 + np.random.randn(N, 2) * 2  # 简单的平移

# 估计基础矩阵
F = eight_point_algorithm(x1, x2)

print("估计的基础矩阵:")
print(F)

# 验证对极约束
for i in range(N):
    x1_h = np.append(x1[i], 1)
    x2_h = np.append(x2[i], 1)
    error = x2_h @ F @ x1_h
    print(f"点对{i}的对极约束误差: {error:.6f}")
```

### 练习2：从本质矩阵恢复位姿

```python
import numpy as np

def recover_pose_from_E(E):
    """
    从本质矩阵恢复旋转和平移
    """
    # SVD分解
    U, S, V = np.linalg.svd(E)
    
    # 构建可能的解
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    
    R1 = U @ W @ V.T
    R2 = U @ W.T @ V.T
    t1 = U[:, 2]
    t2 = -U[:, 2]
    
    # 确保旋转矩阵行列式为1
    if np.linalg.det(R1) < 0:
        R1 = -R1
        t1 = -t1
    if np.linalg.det(R2) < 0:
        R2 = -R2
        t2 = -t2
    
    # 返回四种可能的解
    return [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]

# 示例：创建本质矩阵
R_true = np.array([[1, 0, 0], [0, 0.866, -0.5], [0, 0.5, 0.866]])  # 绕x轴旋转30度
t_true = np.array([0.5, 0, 0])

# 计算反对称矩阵
def skew_symmetric(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

E = skew_symmetric(t_true) @ R_true

print("原始本质矩阵:")
print(E)

# 恢复位姿
solutions = recover_pose_from_E(E)

print("\n四种可能的解:")
for i, (R, t) in enumerate(solutions):
    print(f"\n解{i+1}:")
    print("旋转矩阵:")
    print(R)
    print("平移向量:", t)
```

### 练习3：对极线绘制

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_epipolar_lines(img1, img2, F, x1, x2):
    """
    绘制对极线
    """
    # 获取图像尺寸
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    # 计算对极点
    _, _, V = np.linalg.svd(F)
    e1 = V[-1]
    e1 = e1 / e1[2]
    
    _, _, V = np.linalg.svd(F.T)
    e2 = V[-1]
    e2 = e2 / e2[2]
    
    # 绘制图像1和对极线
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    
    axes[0].imshow(img1, cmap='gray')
    axes[0].scatter(x1[:, 0], x1[:, 1], c='red', s=50)
    axes[0].scatter(e1[0], e1[1], c='blue', s=100, marker='x', label='对极点')
    
    # 绘制对极线
    for i in range(len(x2)):
        x2_h = np.append(x2[i], 1)
        line = F.T @ x2_h
        # 计算对极线与图像边界的交点
        y0 = (-line[2] - line[0] * 0) / line[1]
        y1 = (-line[2] - line[0] * w1) / line[1]
        axes[0].plot([0, w1], [y0, y1], 'g-', alpha=0.5)
    
    axes[0].set_title('图像1与对极线')
    axes[0].legend()
    
    # 绘制图像2和对极线
    axes[1].imshow(img2, cmap='gray')
    axes[1].scatter(x2[:, 0], x2[:, 1], c='red', s=50)
    axes[1].scatter(e2[0], e2[1], c='blue', s=100, marker='x', label='对极点')
    
    for i in range(len(x1)):
        x1_h = np.append(x1[i], 1)
        line = F @ x1_h
        y0 = (-line[2] - line[0] * 0) / line[1]
        y1 = (-line[2] - line[0] * w2) / line[1]
        axes[1].plot([0, w2], [y0, y1], 'g-', alpha=0.5)
    
    axes[1].set_title('图像2与对极线')
    axes[1].legend()
    
    plt.show()

# 示例：使用模拟数据
img1 = np.random.rand(200, 300) * 255
img2 = np.random.rand(200, 300) * 255

# 模拟对应点
x1 = np.array([[50, 50], [100, 80], [150, 120], [200, 60], [250, 150]])
x2 = np.array([[60, 55], [110, 85], [160, 125], [210, 65], [260, 155]])

# 使用八点法估计基础矩阵
F = eight_point_algorithm(x1, x2)

# 绘制对极线
draw_epipolar_lines(img1, img2, F, x1, x2)
```

### 练习4：使用OpenCV估计本质矩阵

```python
import cv2
import numpy as np

# 读取图像
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# 检测ORB特征
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 匹配特征
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)[:100]

# 提取对应点
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 估计本质矩阵
E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1.0)

print("估计的本质矩阵:")
print(E)

# 从本质矩阵恢复位姿
_, R, t, mask = cv2.recoverPose(E, pts1, pts2)

print("\n恢复的旋转矩阵:")
print(R)
print("恢复的平移向量:", t.flatten())
```

---

**下一步**：[直接法视觉里程计](1.3-direct-method-vo.md)