# 1.3 直接法视觉里程计

## 目录

- [1. 问题背景](#1-问题背景)
- [2. 直接法原理](#2-直接法原理)
- [3. 光流估计](#3-光流估计)
- [4. 直接法与稀疏法的对比](#4-直接法与稀疏法的对比)
- [5. 重要论文详解](#5-重要论文详解)
- [6. 在SLAM中的应用](#6-在slam中的应用)
- [7. 实践练习](#7-实践练习)

---

## 1. 问题背景

### 1.1 稀疏特征法的局限性

传统的稀疏特征法（如ORB-SLAM）存在以下问题：

1. **特征提取耗时**：检测和匹配特征点需要大量计算
2. **信息损失**：只使用少量特征点，丢弃了大部分图像信息
3. **纹理缺失场景**：在低纹理区域难以提取足够的特征点
4. **动态物体影响**：特征点可能落在动态物体上

### 1.2 直接法的优势

直接法直接使用图像像素信息进行相机运动估计：

1. **无需特征提取**：直接利用所有像素
2. **更高的鲁棒性**：在纹理缺失场景下表现更好
3. **更密集的地图**：可以构建半稠密或稠密地图
4. **计算效率高**：避免了特征匹配的开销

---

## 2. 直接法原理

### 2.1 亮度恒假设

**核心假设**：同一空间点在不同图像中的亮度保持不变。

$$I_1(\mathbf{x}) = I_2(\mathbf{x} + \Delta \mathbf{x})$$

### 2.2 图像梯度

**泰勒展开**：

$$I_2(\mathbf{x} + \Delta \mathbf{x}) \approx I_2(\mathbf{x}) + \nabla I_2(\mathbf{x}) \cdot \Delta \mathbf{x}$$

**误差方程**：

$$e = I_1(\mathbf{x}) - I_2(\mathbf{x}) - \nabla I_2(\mathbf{x}) \cdot \Delta \mathbf{x}$$

### 2.3 相机运动与像素位移

**投影方程**：

$$\mathbf{x}_2 = \pi(\mathbf{R} \pi^{-1}(\mathbf{x}_1, d) + \mathbf{t})$$

其中 $\pi$ 是投影函数，$\pi^{-1}$ 是反投影函数，$d$ 是深度。

**链式法则**：

$$\Delta \mathbf{x} = \frac{\partial \mathbf{x}_2}{\partial \xi} \delta \xi$$

其中 $\xi$ 是相机位姿的李代数表示。

### 2.4 雅可比矩阵

**图像雅可比矩阵**：

$$\mathbf{J} = \nabla I \cdot \frac{\partial \mathbf{x}_2}{\partial \xi}$$

**最小化光度误差**：

$$\min_{\xi} \sum_{i=1}^{N} e_i^T e_i = \min_{\xi} \sum_{i=1}^{N} (I_1(\mathbf{x}_i) - I_2(\mathbf{x}_i + \mathbf{J}_i \delta \xi))^2$$

---

## 3. 光流估计

### 3.1 光流的定义

**光流**：图像中像素点的瞬时运动。

**亮度恒假设**：

$$I(x, y, t) = I(x + dx, y + dy, t + dt)$$

### 3.2 Horn-Schunck光流算法

**变分方法**：

$$\min \iint (I_x u + I_y v + I_t)^2 dxdy + \lambda \iint (u_x^2 + u_y^2 + v_x^2 + v_y^2) dxdy$$

其中 $u$ 和 $v$ 是光流的x和y分量，$\lambda$ 是平滑项权重。

### 3.3 Lucas-Kanade光流算法

**窗口方法**：

在局部窗口内最小化亮度变化：

$$\min \sum_{(x,y) \in W} [I(x + u, y + v) - I(x, y)]^2$$

**求解**：

$$\begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} \sum I_x I_t \\ \sum I_y I_t \end{bmatrix}$$

---

## 4. 直接法与稀疏法的对比

| 方面 | 稀疏特征法 | 直接法 |
|------|-----------|--------|
| **特征提取** | 需要 | 不需要 |
| **匹配** | 需要 | 不需要 |
| **鲁棒性** | 依赖特征质量 | 依赖图像梯度 |
| **纹理要求** | 高 | 低 |
| **计算量** | 中等 | 较高 |
| **地图密度** | 稀疏 | 稠密/半稠密 |
| **动态物体** | 敏感 | 相对不敏感 |

---

## 5. 重要论文详解

### 5.1 LSD-SLAM: Large-Scale Direct Monocular SLAM (2014)

**作者**：Jakob Engel, Thomas Schops, Daniel Cremers

**发表会议**：ECCV

**问题背景**：
- 传统SLAM系统依赖特征点，在低纹理场景下性能下降
- 需要一种不依赖特征的SLAM方法

**核心贡献**：
1. **直接法SLAM**：首次将直接法应用于大规模SLAM
2. **半稠密地图**：构建半稠密的深度地图
3. **关键帧选择**：基于信息增益的关键帧选择策略
4. **位姿优化**：使用直接法进行位姿优化

**算法流程**：
1. **深度估计**：使用多视角立体视觉估计深度
2. **位姿跟踪**：使用直接法跟踪相机位姿
3. **关键帧插入**：当当前帧与最近关键帧差异较大时插入关键帧
4. **优化**：对关键帧进行Bundle Adjustment

**为什么重要**：
- 开创了直接法SLAM的先河
- 在低纹理场景下表现出色
- 为后续的直接法SLAM系统奠定了基础

### 5.2 DTAM: Dense Tracking and Mapping (2011)

**作者**：Richard Newcombe, Steven Lovegrove, Andrew Davison

**发表会议**：ICCV

**问题背景**：
- 需要一种实时的稠密重建方法
- 传统方法无法在手持设备上实时运行

**核心贡献**：
1. **稠密跟踪**：使用直接法进行稠密像素跟踪
2. **在线重建**：实时构建稠密深度图
3. **GPU加速**：利用GPU实现实时性能

**为什么重要**：
- 首次实现实时稠密SLAM
- 展示了直接法在稠密重建中的潜力
- 影响了后续的KinectFusion等工作

### 5.3 DSO: Direct Sparse Odometry (2016)

**作者**：Jakob Engel, Vladlen Koltun, Daniel Cremers

**发表会议**：IEEE Transactions on Pattern Analysis and Machine Intelligence

**详细分析**：[查看论文详解](../papers/dso.md)

**问题背景**：
- LSD-SLAM虽然有效，但计算复杂度较高
- 需要一种更高效的直接法SLAM系统

**核心贡献**：
1. **稀疏直接法**：只选择有信息量的像素进行优化
2. **分层优化**：使用粗到细的优化策略
3. **关键帧管理**：高效的关键帧选择和删除策略
4. **光度标定**：考虑曝光时间和增益的变化

**为什么重要**：
- 在保持精度的同时提高了效率
- 成为直接法SLAM的基准系统
- 在TUM数据集上达到了当时的最佳性能

### 5.4 Direct Visual Odometry with a Single Camera (2011)

**作者**：Jakob Engel, Jörg Stückler, Daniel Cremers

**发表会议**：BMVC

**问题背景**：
- 早期直接法主要用于稠密重建
- 需要一种轻量级的直接法视觉里程计

**核心贡献**：
1. **直接法视觉里程计**：首次提出直接法视觉里程计
2. **半稠密深度估计**：使用半稠密方法估计深度
3. **高效优化**：使用高斯-牛顿法进行优化

**为什么重要**：
- 奠定了直接法视觉里程计的基础
- 为后续的LSD-SLAM和DSO提供了思路

---

## 6. 在SLAM中的应用

### 6.1 视觉里程计

**直接法VO**：使用直接法估计相机运动

### 6.2 稠密重建

**DTAM/KinectFusion**：实时稠密重建

### 6.3 混合方法

**ORB-SLAM3**：结合特征法和直接法的优势

---

## 7. 实践练习

### 练习1：Lucas-Kanade光流估计

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img1 = cv2.imread('frame1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('frame2.jpg', cv2.IMREAD_GRAYSCALE)

# 检测特征点
corners = cv2.goodFeaturesToTrack(img1, maxCorners=100, qualityLevel=0.01, minDistance=10)
corners = np.int0(corners)

# Lucas-Kanade光流参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 计算光流
p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, corners, None, **lk_params)

# 选择好的跟踪点
good_new = p1[st == 1]
good_old = corners[st == 1]

print(f"跟踪到的特征点数量: {len(good_new)}")

# 绘制结果
plt.figure(figsize=(15, 10))
plt.subplot(121)
plt.imshow(img1, cmap='gray')
plt.scatter(good_old[:, 0], good_old[:, 1], c='red', s=50)
plt.title('帧1特征点')

plt.subplot(122)
plt.imshow(img2, cmap='gray')
plt.scatter(good_new[:, 0], good_new[:, 1], c='blue', s=50)
plt.title('帧2跟踪结果')

plt.show()

# 绘制光流向量
plt.figure(figsize=(10, 8))
plt.imshow(img2, cmap='gray')
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    plt.arrow(c, d, a - c, b - d, color='green', length_includes_head=True, head_width=2)
plt.title('光流向量')
plt.show()
```

### 练习2：直接法位姿估计

```python
import numpy as np
from scipy.optimize import least_squares

def project(points3d, K, R, t):
    """投影3D点到图像平面"""
    points_cam = R @ points3d.T + t.reshape(3, 1)
    points_img = K @ points_cam
    points_img = points_img[:2] / points_img[2]
    return points_img.T

def photometric_error(x, img1, img2, points3d, K):
    """光度误差"""
    # 从x中提取位姿参数（旋转向量和平移）
    rot_vec = x[:3]
    t = x[3:]
    
    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rot_vec)
    
    # 投影3D点
    points_proj = project(points3d, K, R, t)
    
    # 计算光度误差
    error = []
    for i, (u, v) in enumerate(points_proj):
        if 0 <= u < img2.shape[1] and 0 <= v < img2.shape[0]:
            # 双线性插值获取像素值
            u0, v0 = int(u), int(v)
            du, dv = u - u0, v - v0
            
            # 获取img1中对应点的像素值
            u1, v1 = int(points3d[i, 0]), int(points3d[i, 1])
            if 0 <= u1 < img1.shape[1] and 0 <= v1 < img1.shape[0]:
                I1 = img1[v1, u1]
                I2 = (1-du)*(1-dv)*img2[v0, u0] + du*(1-dv)*img2[v0, min(u0+1, img2.shape[1]-1)] + \
                     (1-du)*dv*img2[min(v0+1, img2.shape[0]-1), u0] + du*dv*img2[min(v0+1, img2.shape[0]-1), min(u0+1, img2.shape[1]-1)]
                error.append(I1 - I2)
    
    return np.array(error)

# 示例使用
# 假设有img1, img2, points3d和K
# x0 = np.zeros(6)  # 初始位姿估计
# result = least_squares(photometric_error, x0, args=(img1, img2, points3d, K))
```

### 练习3：直接法与特征法对比

```python
import cv2
import numpy as np
import time

# 读取图像
img1 = cv2.imread('frame1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('frame2.jpg', cv2.IMREAD_GRAYSCALE)

# 特征法
start_time = time.time()
orb = cv2.ORB_create(nfeatures=500)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
feature_time = time.time() - start_time

print(f"特征法耗时: {feature_time:.4f}秒")
print(f"检测到的特征点: {len(kp1)}")
print(f"匹配数: {len(matches)}")

# 直接法（光流）
start_time = time.time()
corners = cv2.goodFeaturesToTrack(img1, maxCorners=100, qualityLevel=0.01, minDistance=10)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, corners, None, **lk_params)
flow_time = time.time() - start_time

print(f"\n直接法耗时: {flow_time:.4f}秒")
print(f"跟踪到的特征点: {np.sum(st)}")

# 对比
print(f"\n直接法比特征法快 {feature_time/flow_time:.2f} 倍")
```

---

**下一步**：[SLAM算法基础](2.1-slam-algorithms-intro.md)