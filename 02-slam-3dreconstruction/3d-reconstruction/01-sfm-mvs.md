# 3.1 三维重建基础

## 目录

- [1. 结构从运动（SfM）](#1-结构从运动sfm)
- [2. 多视图立体视觉（MVS）](#2-多视图立体视觉mvs)
- [3. 深度估计](#3-深度估计)
- [4. 表面重建](#4-表面重建)
- [5. 重要论文详解](#5-重要论文详解)
- [6. 实践练习](#6-实践练习)

---

## 1. 结构从运动（SfM）

### 1.1 问题定义

**结构从运动**：从一组无序图像中恢复相机位姿和场景的三维结构。

**核心步骤**：
1. **特征提取与匹配**：在图像之间找到对应点
2. **相机位姿估计**：使用对极几何估计相机位姿
3. **三角化**：计算3D点的位置
4. **Bundle Adjustment**：优化相机位姿和3D点位置

### 1.2 经典SfM流程

**增量式SfM**：
1. **初始化**：选择一对图像进行初始化
2. **增量扩展**：逐步添加新图像
3. **BA优化**：定期进行Bundle Adjustment

**全局式SfM**：
1. **全局旋转估计**：估计所有相机之间的相对旋转
2. **全局平移估计**：估计相机之间的平移
3. **优化**：全局BA优化

### 1.3 COLMAP

**COLMAP**是目前最流行的SfM开源工具：
- 支持增量式和全局式SfM
- 具有良好的鲁棒性和精度
- 支持多种图像格式

---

## 2. 多视图立体视觉（MVS）

### 2.1 问题定义

**多视图立体视觉**：从多个视角重建场景的稠密深度图。

**核心步骤**：
1. **深度假设生成**：生成可能的深度值
2. **视图选择**：选择用于验证的视图
3. **深度验证**：验证深度假设的正确性
4. **深度融合**：融合多个视图的深度信息

### 2.2 稠密重建方法

**基于块的方法**：
- 将图像分成小块
- 对每个块进行深度估计
- 优点：简单直观
- 缺点：计算量大

**基于体素的方法**：
- 将场景表示为体素网格
- 使用截断符号距离函数（TSDF）
- 优点：可以处理复杂场景
- 缺点：内存占用大

**基于学习的方法**：
- 使用深度学习估计深度
- 优点：精度高
- 缺点：需要大量训练数据

---

## 3. 深度估计

### 3.1 单目深度估计

**基于传统方法**：
- 使用几何约束
- 使用纹理信息

**基于深度学习的方法**：
- 使用卷积神经网络
- 从单张图像预测深度

### 3.2 双目深度估计

**立体匹配**：
- 在左右图像之间找到对应点
- 计算视差
- 转换为深度

**方法**：
- BM（Block Matching）
- SGM（Semi-Global Matching）
- GC（Graph Cuts）

### 3.3 RGB-D深度估计

**直接使用深度传感器**：
- Kinect
- RealSense
- 优点：直接获取深度
- 缺点：传感器噪声

---

## 4. 表面重建

### 4.1 点云处理

**点云配准**：
- ICP（Iterative Closest Point）
- 特征匹配

**点云滤波**：
- 去除噪声点
- 下采样

### 4.2 网格重建

**泊松重建**：
- 使用泊松方程重建表面
- 生成光滑的网格

**Marching Cubes**：
- 从体素数据生成网格
- 适合稠密重建

### 4.3 纹理映射

**纹理提取**：
- 从图像中提取纹理
- 映射到网格表面

**纹理融合**：
- 融合多个视图的纹理
- 处理光照变化

---

## 5. 重要论文详解

### 5.1 Structure from Motion Revisited (2016)

**作者**：Johannes L. Schönberger, Jan-Michael Frahm

**发表会议**：CVPR

**问题背景**：
- 传统SfM方法在处理大规模数据集时效率低下
- 需要一种更高效的全局SfM方法

**核心贡献**：
1. **全局旋转估计**：使用旋转平均方法估计全局旋转
2. **全局平移估计**：使用Chordal距离估计平移
3. **高效BA**：使用稀疏线性系统求解器

**为什么重要**：
- 大幅提高了SfM的效率
- 成为COLMAP的核心算法
- 推动了大规模三维重建的发展

### 5.2 Pixelwise View Selection for Unstructured Multi-View Stereo (2017)

**作者**：Johannes L. Schönberger, Enliang Zheng, Jan-Michael Frahm

**发表会议**：ECCV

**问题背景**：
- 传统MVS方法在处理无序图像时性能较差
- 需要一种鲁棒的视图选择方法

**核心贡献**：
1. **像素级视图选择**：为每个像素选择最佳的参考视图
2. **自适应权重**：根据图像质量调整权重
3. **深度融合**：融合多个视图的深度信息

**为什么重要**：
- 提高了MVS的鲁棒性
- 成为COLMAP MVS的核心算法

### 5.3 Poisson Surface Reconstruction (2006)

**作者**：Michael Kazhdan, Matthew Bolitho, Hugues Hoppe

**发表会议**：SIGGRAPH

**问题背景**：
- 从点云重建高质量表面是一个挑战
- 需要一种能够生成光滑表面的方法

**核心贡献**：
1. **泊松方程**：使用泊松方程重建表面
2. **指示函数**：将点云转换为指示函数
3. **隐式表面**：生成隐式表面表示

**为什么重要**：
- 生成的表面质量高
- 成为点云重建的标准方法

### 5.4 COLMAP: Structure-from-Motion and Multi-View Stereo Reconstruction (2016)

**作者**：Johannes L. Schönberger, Jan-Michael Frahm

**发表会议**：Technical Report

**问题背景**：
- 需要一个开源的、高效的三维重建工具
- 整合SfM和MVS的最新研究成果

**核心贡献**：
1. **统一框架**：整合SfM和MVS
2. **高效实现**：使用多线程和GPU加速
3. **鲁棒性**：处理各种类型的图像

**为什么重要**：
- 成为三维重建的标准工具
- 被广泛用于学术研究和工业应用

---

## 6. 实践练习

### 练习1：使用COLMAP进行SfM重建

```bash
# 安装COLMAP
sudo apt-get install colmap

# 创建项目目录
mkdir -p colmap_project
cd colmap_project

# 准备图像（将图像放在images目录）
mkdir images
# 将图像复制到images目录

# 运行COLMAP
colmap feature_extractor \
    --database_path database.db \
    --image_path images

colmap exhaustive_matcher \
    --database_path database.db

mkdir sparse
colmap mapper \
    --database_path database.db \
    --image_path images \
    --output_path sparse

# 可视化结果
colmap model_analyzer \
    --input_path sparse/0
```

### 练习2：使用Open3D进行点云处理

```python
import open3d as o3d
import numpy as np

# 读取点云
pcd = o3d.io.read_point_cloud('point_cloud.pcd')

# 可视化原始点云
print("原始点云统计:")
print(pcd)
print(np.asarray(pcd.points))

# 点云滤波
pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
print("\n下采样后的点云:")
print(pcd_down)

# 去除离群点
cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd_clean = pcd_down.select_by_index(ind)
print("\n去除离群点后的点云:")
print(pcd_clean)

# 法线估计
pcd_clean.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# 可视化
o3d.visualization.draw_geometries([pcd_clean], point_show_normal=True)

# 泊松重建
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_clean, depth=9)

# 可视化网格
o3d.visualization.draw_geometries([mesh])
```

### 练习3：深度图生成

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取左右图像
img_left = cv2.imread('left.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('right.jpg', cv2.IMREAD_GRAYSCALE)

# 创建SGBM立体匹配器
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*5,
    blockSize=15,
    P1=8*3*15**2,
    P2=32*3*15**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# 计算视差图
disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

print("视差图形状:", disparity.shape)
print("视差范围:", np.min(disparity), "-", np.max(disparity))

# 转换为深度图（需要相机参数）
# depth = (f * baseline) / disparity
f = 525.0  # 焦距
baseline = 0.537  # 基线距离
depth = (f * baseline) / (disparity + 1e-6)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_left, cmap='gray')
axes[0].set_title('左图像')

axes[1].imshow(disparity, cmap='jet')
axes[1].set_title('视差图')

axes[2].imshow(depth, cmap='jet')
axes[2].set_title('深度图')

plt.show()
```

---

**下一步**：[神经辐射场（NeRF）](3.2-nerf.md)