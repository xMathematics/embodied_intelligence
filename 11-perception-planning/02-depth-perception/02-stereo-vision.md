# 2.2 双目立体视觉

## 目录

- [1. 引言](#1-引言)
- [2. 双目立体视觉概述](#2-双目立体视觉概述)
- [3. 极线几何](#3-极线几何)
- [4. 立体匹配算法](#4-立体匹配算法)
- [5. 深度学习方法](#5-深度学习方法)
- [6. 实践练习](#6-实践练习)

---

## 1. 引言

### 1.1 双目立体视觉的重要性

**双目立体视觉**模拟人类双眼视觉，通过两个摄像头从不同视角观察同一场景，计算视差并恢复深度信息，是最经典的深度感知技术之一。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **自动驾驶** | 车辆前方障碍物测距 | 立体相机感知系统 |
| **工业检测** | 产品三维测量 | 尺寸检测、缺陷定位 |
| **机器人抓取** | 物体姿态和位置估计 | 机械臂视觉引导 |
| **3D重建** | 场景三维建模 | 文物数字化、室内地图 |

---

## 2. 双目立体视觉概述

### 2.1 基本原理

**双目视觉原理**：通过两个平行放置的相机，同一物点在左右图像上的投影位置不同，形成视差，利用三角测量原理计算深度。

**三角测量公式**：
```
Z = f * B / d
```
其中：
- `Z`：深度（距离相机的距离）
- `f`：相机焦距
- `B`：基线（两相机光心之间的距离）
- `d`：视差（左右图像中对应点的水平位移）

### 2.2 系统组成

```python
import numpy as np
import cv2

class StereoCameraSystem:
    def __init__(self, B=0.1, f=500, img_size=(640, 480)):
        """
        初始化双目相机系统
        
        参数:
            B: 基线距离 (米)
            f: 焦距 (像素)
            img_size: 图像尺寸
        """
        self.B = B
        self.f = f
        self.img_size = img_size
        
        # 相机内参
        cx, cy = img_size[0] / 2, img_size[1] / 2
        self.K_left = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        
        self.K_right = self.K_left.copy()
        
        # 相机外参 (右相机相对于左相机的变换)
        self.R = np.eye(3)
        self.t = np.array([-B, 0, 0])
    
    def project_3d_to_2d(self, points_3d, camera='left'):
        """将3D点投影到2D图像"""
        K = self.K_left if camera == 'left' else self.K_right
        R = np.eye(3) if camera == 'left' else self.R
        t = np.zeros(3) if camera == 'left' else self.t
        
        points_2d = []
        for p in points_3d:
            p_cam = R @ p + t
            p_proj = K @ p_cam
            p_proj = p_proj[:2] / p_proj[2]
            points_2d.append(p_proj)
        
        return np.array(points_2d)
    
    def compute_depth(self, disparity):
        """根据视差计算深度"""
        depth = (self.f * self.B) / (disparity + 1e-6)
        return depth
```

---

## 3. 极线几何

### 3.1 基本概念

```python
class EpipolarGeometry:
    def __init__(self, K1, K2, R, t):
        """
        初始化极线几何
        
        参数:
            K1, K2: 左右相机内参
            R, t: 右相机相对于左相机的旋转和平移
        """
        self.K1 = K1
        self.K2 = K2
        self.R = R
        self.t = t
        
        # 计算本质矩阵
        self.E = self._compute_essential_matrix()
        
        # 计算基础矩阵
        self.F = self._compute_fundamental_matrix()
    
    def _compute_essential_matrix(self):
        """计算本质矩阵"""
        t_cross = np.array([
            [0, -self.t[2], self.t[1]],
            [self.t[2], 0, -self.t[0]],
            [-self.t[1], self.t[0], 0]
        ])
        E = t_cross @ self.R
        return E
    
    def _compute_fundamental_matrix(self):
        """计算基础矩阵"""
        F = np.linalg.inv(self.K2).T @ self.E @ np.linalg.inv(self.K1)
        return F
    
    def compute_epipolar_line(self, point, camera='left'):
        """计算极线"""
        if camera == 'left':
            line = self.F.T @ np.append(point, 1)
        else:
            line = self.F @ np.append(point, 1)
        return line
    
    def draw_epipolar_line(self, img, line):
        """在图像上绘制极线"""
        h, w = img.shape[:2]
        
        # 计算极线与图像边界的交点
        x0, y0 = 0, int(-line[2] / line[1])
        x1, y1 = w, int(-(line[2] + line[0] * w) / line[1])
        
        # 绘制极线
        img_line = img.copy()
        cv2.line(img_line, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        return img_line
```

### 3.2 立体校正

```python
class StereoRectification:
    def __init__(self, K1, K2, R, t, img_size):
        """
        立体校正
        
        参数:
            K1, K2: 左右相机内参
            R, t: 右相机相对于左相机的旋转和平移
            img_size: 图像尺寸
        """
        self.K1 = K1
        self.K2 = K2
        self.R = R
        self.t = t
        self.img_size = img_size
        
        # 计算校正变换
        self.R1, self.R2, self.P1, self.P2, self.Q = self._compute_rectification()
    
    def _compute_rectification(self):
        """计算校正变换"""
        # 使用OpenCV的立体校正
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.K1, None,
            self.K2, None,
            self.img_size,
            self.R, self.t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0
        )
        
        return R1, R2, P1, P2, Q
    
    def get_undistort_rectify_map(self):
        """获取校正映射"""
        map1_x, map1_y = cv2.initUndistortRectifyMap(
            self.K1, None, self.R1, self.P1,
            self.img_size, cv2.CV_32FC1
        )
        
        map2_x, map2_y = cv2.initUndistortRectifyMap(
            self.K2, None, self.R2, self.P2,
            self.img_size, cv2.CV_32FC1
        )
        
        return (map1_x, map1_y), (map2_x, map2_y)
    
    def rectify_images(self, img_left, img_right):
        """校正左右图像"""
        (map1_x, map1_y), (map2_x, map2_y) = self.get_undistort_rectify_map()
        
        img_left_rect = cv2.remap(img_left, map1_x, map1_y, cv2.INTER_LINEAR)
        img_right_rect = cv2.remap(img_right, map2_x, map2_y, cv2.INTER_LINEAR)
        
        return img_left_rect, img_right_rect
```

---

## 4. 立体匹配算法

### 4.1 基于块匹配的方法

```python
class BlockMatching:
    def __init__(self, block_size=15, num_disparities=64):
        """
        块匹配立体匹配
        
        参数:
            block_size: 匹配块大小
            num_disparities: 最大视差
        """
        self.block_size = block_size
        self.num_disparities = num_disparities
        self.half_block = block_size // 2
    
    def compute_sad(self, block1, block2):
        """计算绝对误差和 (SAD)"""
        return np.sum(np.abs(block1 - block2))
    
    def compute_ssd(self, block1, block2):
        """计算平方误差和 (SSD)"""
        return np.sum((block1 - block2) ** 2)
    
    def compute_ncc(self, block1, block2):
        """计算归一化互相关 (NCC)"""
        mean1 = np.mean(block1)
        mean2 = np.mean(block2)
        
        num = np.sum((block1 - mean1) * (block2 - mean2))
        den = np.sqrt(np.sum((block1 - mean1) ** 2) * np.sum((block2 - mean2) ** 2))
        
        return num / (den + 1e-6)
    
    def match(self, img_left, img_right):
        """计算视差图"""
        h, w = img_left.shape[:2]
        
        # 转换为灰度图
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right
        
        disparity_map = np.zeros((h, w), dtype=np.float32)
        
        # 遍历每个像素
        for y in range(self.half_block, h - self.half_block):
            for x in range(self.half_block, w - self.half_block):
                # 左图块
                block_left = gray_left[
                    y - self.half_block : y + self.half_block + 1,
                    x - self.half_block : x + self.half_block + 1
                ]
                
                best_d = 0
                min_cost = float('inf')
                
                # 在右图搜索最佳匹配
                max_d = min(self.num_disparities, x - self.half_block)
                for d in range(max_d):
                    x_right = x - d
                    if x_right - self.half_block < 0:
                        continue
                    
                    block_right = gray_right[
                        y - self.half_block : y + self.half_block + 1,
                        x_right - self.half_block : x_right + self.half_block + 1
                    ]
                    
                    cost = self.compute_sad(block_left, block_right)
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_d = d
                
                disparity_map[y, x] = best_d
        
        return disparity_map
```

### 4.2 Semi-Global Matching (SGM)

```python
class SemiGlobalMatching:
    def __init__(self, num_disparities=64, P1=10, P2=100):
        """
        半全局匹配
        
        参数:
            num_disparities: 最大视差
            P1, P2: 平滑惩罚参数
        """
        self.num_disparities = num_disparities
        self.P1 = P1
        self.P2 = P2
    
    def compute_cost_volume(self, img_left, img_right):
        """计算代价卷"""
        h, w = img_left.shape[:2]
        
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right
        
        cost_volume = np.zeros((h, w, self.num_disparities), dtype=np.float32)
        
        for d in range(self.num_disparities):
            # 平移右图
            shifted = np.zeros_like(gray_right)
            shifted[:, d:] = gray_right[:, :-d] if d > 0 else gray_right
            
            # 计算绝对差
            cost = np.abs(gray_left.astype(float) - shifted.astype(float))
            cost_volume[:, :, d] = cost
        
        return cost_volume
    
    def aggregate_cost(self, cost_volume):
        """代价聚合"""
        h, w, d = cost_volume.shape
        aggregated = cost_volume.copy()
        
        # 方向: 左->右, 右->左, 上->下, 下->上
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dy, dx in directions:
            path_cost = np.zeros_like(cost_volume)
            
            # 确定起始位置
            if dy == 0 and dx == 1:
                y_start, y_end = 0, h
                x_start, x_end = 1, w
            elif dy == 0 and dx == -1:
                y_start, y_end = 0, h
                x_start, x_end = w - 2, -1
            elif dy == 1 and dx == 0:
                y_start, y_end = 1, h
                x_start, x_end = 0, w
            else:
                y_start, y_end = h - 2, -1
                x_start, x_end = 0, w
            
            # 动态规划
            for y in range(y_start, y_end, dy if dy != 0 else 1):
                for x in range(x_start, x_end, dx if dx != 0 else 1):
                    prev_y = y - dy
                    prev_x = x - dx
                    
                    prev_costs = path_cost[prev_y, prev_x, :]
                    min_prev = np.min(prev_costs)
                    
                    for d_curr in range(self.num_disparities):
                        # 计算惩罚
                        penalty = np.zeros(self.num_disparities)
                        for d_prev in range(self.num_disparities):
                            if d_prev == d_curr:
                                penalty[d_prev] = 0
                            elif abs(d_prev - d_curr) == 1:
                                penalty[d_prev] = self.P1
                            else:
                                penalty[d_prev] = self.P2
                        
                        # 更新路径代价
                        path_cost[y, x, d_curr] = (
                            cost_volume[y, x, d_curr] +
                            np.min(prev_costs + penalty) - min_prev
                        )
            
            aggregated += path_cost
        
        return aggregated
    
    def compute_disparity(self, cost_volume):
        """WTA选择最佳视差"""
        return np.argmin(cost_volume, axis=2)
    
    def match(self, img_left, img_right):
        """完整SGM流程"""
        cost_volume = self.compute_cost_volume(img_left, img_right)
        aggregated = self.aggregate_cost(cost_volume)
        disparity = self.compute_disparity(aggregated)
        return disparity.astype(np.float32)
```

### 4.3 使用OpenCV的立体匹配

```python
class OpenCVStereoMatcher:
    def __init__(self, method='bm', num_disparities=64, block_size=15):
        """
        使用OpenCV的立体匹配
        
        参数:
            method: 'bm' (块匹配) 或 'sgbm' (半全局匹配)
            num_disparities: 最大视差
            block_size: 匹配块大小
        """
        self.method = method
        self.num_disparities = num_disparities
        self.block_size = block_size
        
        if method == 'bm':
            self.matcher = cv2.StereoBM_create(
                numDisparities=num_disparities,
                blockSize=block_size
            )
        elif method == 'sgbm':
            self.matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=8 * 3 * block_size ** 2,
                P2=32 * 3 * block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32
            )
    
    def match(self, img_left, img_right):
        """计算视差图"""
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right
        
        disparity = self.matcher.compute(gray_left, gray_right)
        
        # 转换为浮点型
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
```

---

## 5. 深度学习方法

### 5.1 PSMNet (Pyramid Stereo Matching Network)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PSMNet(nn.Module):
    """金字塔立体匹配网络"""
    def __init__(self, max_disparity=192):
        super().__init__()
        self.max_disparity = max_disparity
        
        # 特征提取
        self.feature_extraction = self._build_feature_extractor()
        
        # 空间金字塔池化
        self.spp = self._build_spp()
        
        # 成本量构建和3D卷积
        self.cost_volume = CostVolume(max_disparity)
        self.cost_aggregation = self._build_cost_aggregation()
        
        # 视差回归
        self.disparity_regression = DisparityRegression(max_disparity)
    
    def _build_feature_extractor(self):
        """构建特征提取器"""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
    
    def _build_spp(self):
        """构建空间金字塔池化"""
        return nn.ModuleList([
            nn.AdaptiveAvgPool2d(64),
            nn.AdaptiveAvgPool2d(32),
            nn.AdaptiveAvgPool2d(16),
            nn.AdaptiveAvgPool2d(8)
        ])
    
    def _build_cost_aggregation(self):
        """构建代价聚合"""
        return nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, img_left, img_right):
        """前向传播"""
        # 提取特征
        feat_left = self.feature_extraction(img_left)
        feat_right = self.feature_extraction(img_right)
        
        # 构建成本量
        cost = self.cost_volume(feat_left, feat_right)
        
        # 代价聚合
        cost = self.cost_aggregation(cost)
        cost = cost.squeeze(1)
        
        # 上采样
        cost = F.interpolate(cost, scale_factor=2, mode='bilinear')
        
        # 视差回归
        disparity = self.disparity_regression(cost)
        
        return disparity

class CostVolume(nn.Module):
    """成本量构建"""
    def __init__(self, max_disparity):
        super().__init__()
        self.max_disparity = max_disparity // 2
    
    def forward(self, feat_left, feat_right):
        B, C, H, W = feat_left.shape
        cost = torch.zeros(B, C * 2, self.max_disparity, H, W, device=feat_left.device)
        
        for d in range(self.max_disparity):
            if d > 0:
                cost[:, :C, d, :, d:] = feat_left[:, :, :, d:]
                cost[:, C:, d, :, d:] = feat_right[:, :, :, :-d]
            else:
                cost[:, :C, d, :, :] = feat_left
                cost[:, C:, d, :, :] = feat_right
        
        return cost

class DisparityRegression(nn.Module):
    """视差回归"""
    def __init__(self, max_disparity):
        super().__init__()
        self.max_disparity = max_disparity
    
    def forward(self, cost):
        """Soft-Argmax"""
        prob = F.softmax(-cost, dim=1)
        
        disp_values = torch.arange(0, self.max_disparity, dtype=torch.float32, device=cost.device)
        disp_values = disp_values.view(1, -1, 1, 1)
        
        disparity = torch.sum(prob * disp_values, dim=1, keepdim=True)
        
        return disparity

# 测试
model = PSMNet()
img_left = torch.randn(1, 3, 256, 512)
img_right = torch.randn(1, 3, 256, 512)
disparity = model(img_left, img_right)
print(f"视差图形状: {disparity.shape}")
```

### 5.2 GANet (Guided Aggregation Network)

```python
class GANet(nn.Module):
    """引导聚合网络"""
    def __init__(self, max_disparity=192):
        super().__init__()
        self.max_disparity = max_disparity
        
        # 特征提取
        self.feature_extraction = self._build_feature_extractor()
        
        # 引导聚合
        self.guided_aggregation = self._build_guided_aggregation()
        
        # 视差回归
        self.disparity_regression = DisparityRegression(max_disparity)
    
    def _build_feature_extractor(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
    
    def _build_guided_aggregation(self):
        return nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, img_left, img_right):
        feat_left = self.feature_extraction(img_left)
        feat_right = self.feature_extraction(img_right)
        
        # 构建成本量
        B, C, H, W = feat_left.shape
        cost = torch.zeros(B, C * 2, self.max_disparity // 2, H, W, device=feat_left.device)
        
        for d in range(self.max_disparity // 2):
            if d > 0:
                cost[:, :C, d, :, d:] = feat_left[:, :, :, d:]
                cost[:, C:, d, :, d:] = feat_right[:, :, :, :-d]
            else:
                cost[:, :C, d, :, :] = feat_left
                cost[:, C:, d, :, :] = feat_right
        
        # 聚合
        cost = self.guided_aggregation(cost)
        cost = cost.squeeze(1)
        
        # 上采样
        cost = F.interpolate(cost, scale_factor=2, mode='bilinear')
        
        # 回归
        disparity = self.disparity_regression(cost)
        
        return disparity
```

---

## 6. 实践练习

### 练习1：实现简单的立体匹配

```python
class SimpleStereoMatcher:
    def __init__(self, max_disparity=64, window_size=5):
        self.max_disparity = max_disparity
        self.window_size = window_size
        self.half_window = window_size // 2
    
    def match(self, img_left, img_right):
        """简单的SAD匹配"""
        h, w = img_left.shape[:2]
        
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY).astype(float)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY).astype(float)
        else:
            gray_left = img_left.astype(float)
            gray_right = img_right.astype(float)
        
        disparity = np.zeros((h, w), dtype=np.float32)
        
        for y in range(self.half_window, h - self.half_window):
            for x in range(self.half_window, w - self.half_window):
                best_d = 0
                min_sad = float('inf')
                
                # 左图窗口
                win_left = gray_left[
                    y - self.half_window : y + self.half_window + 1,
                    x - self.half_window : x + self.half_window + 1
                ]
                
                # 搜索视差
                for d in range(min(self.max_disparity, x - self.half_window)):
                    x_right = x - d
                    win_right = gray_right[
                        y - self.half_window : y + self.half_window + 1,
                        x_right - self.half_window : x_right + self.half_window + 1
                    ]
                    
                    sad = np.sum(np.abs(win_left - win_right))
                    
                    if sad < min_sad:
                        min_sad = sad
                        best_d = d
                
                disparity[y, x] = best_d
        
        return disparity

# 测试
matcher = SimpleStereoMatcher()
# disparity = matcher.match(img_left, img_right)
```

### 练习2：视差图后处理

```python
class DisparityPostProcessor:
    def __init__(self):
        pass
    
    def median_filter(self, disparity, kernel_size=5):
        """中值滤波"""
        return cv2.medianBlur(disparity.astype(np.uint8), kernel_size).astype(float)
    
    def bilateral_filter(self, disparity, d=9, sigma_color=75, sigma_space=75):
        """双边滤波"""
        return cv2.bilateralFilter(disparity.astype(np.float32), d, sigma_color, sigma_space)
    
    def fill_holes(self, disparity):
        """填充空洞"""
        # 使用形态学操作
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(disparity.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        return closed.astype(float)
    
    def compute_depth(self, disparity, B, f):
        """视差转深度"""
        depth = (f * B) / (disparity + 1e-6)
        return depth
    
    def postprocess(self, disparity):
        """完整后处理流程"""
        # 中值滤波
        disp_filtered = self.median_filter(disparity)
        
        # 双边滤波
        disp_filtered = self.bilateral_filter(disp_filtered)
        
        # 填充空洞
        disp_filled = self.fill_holes(disp_filtered)
        
        return disp_filled
```

### 练习3：立体匹配评估

```python
class StereoEvaluator:
    def __init__(self):
        pass
    
    def compute_error_map(self, pred, gt):
        """计算误差图"""
        return np.abs(pred - gt)
    
    def compute_d1_all(self, pred, gt, threshold=3):
        """计算D1-all指标"""
        error = np.abs(pred - gt)
        bad_pixels = (error > threshold) | (error > 0.05 * gt)
        return np.mean(bad_pixels)
    
    def compute_epe(self, pred, gt):
        """计算端点误差 (EPE)"""
        return np.mean(np.abs(pred - gt))
    
    def evaluate(self, pred, gt):
        """综合评估"""
        metrics = {
            'epe': self.compute_epe(pred, gt),
            'd1_all': self.compute_d1_all(pred, gt)
        }
        return metrics

# 测试
evaluator = StereoEvaluator()
# metrics = evaluator.evaluate(pred_disparity, gt_disparity)
# print(metrics)
```

---

**下一节**：[深度补全](03-depth-completion.md)

---

## 参考文献

1. Scharstein, D., & Szeliski, R. (2002). A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms.
2. Hirschmüller, H. (2008). Stereo Processing by Semiglobal Matching and Mutual Information.
3. Chang, J. R., & Chen, Y. S. (2018). Pyramid Stereo Matching Network.
4. Zhang, F., et al. (2019). GA-Net: Guided Aggregation Net for End-to-End Stereo Matching.
