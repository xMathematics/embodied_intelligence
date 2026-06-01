# 2.3 深度补全

## 目录

- [1. 引言](#1-引言)
- [2. 深度补全概述](#2-深度补全概述)
- [3. 传统插值方法](#3-传统插值方法)
- [4. 深度学习方法](#4-深度学习方法)
- [5. RGB引导的深度补全](#5-rgb引导的深度补全)
- [6. 实践练习](#6-实践练习)

---

## 1. 引言

### 1.1 深度补全的重要性

**深度补全**是将稀疏的深度测量（如LiDAR点云）转换为完整稠密深度图的技术，是连接传感器硬件和高层视觉任务的关键桥梁。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **自动驾驶** | 增强LiDAR深度感知 | 稀疏点云补全为稠密深度图 |
| **3D重建** | 提高点云密度 | 物体表面重建 |
| **SLAM** | 改进状态估计 | 更精确的深度信息 |
| **AR/VR** | 虚实融合 | 更准确的深度交互 |

---

## 2. 深度补全概述

### 2.1 问题定义

**深度补全**：给定一张RGB图像和对应的稀疏深度图（只有部分像素有深度值），预测出完整的稠密深度图。

**形式化表达**：
```
D_dense = f(I, D_sparse; θ)
```
其中：
- `I`：RGB图像
- `D_sparse`：稀疏深度图
- `D_dense`：稠密深度图
- `θ`：模型参数

### 2.2 稀疏深度生成

```python
import numpy as np
import cv2

class SparseDepthGenerator:
    def __init__(self):
        pass
    
    def generate_random_sparse(self, dense_depth, sparsity=0.95):
        """
        随机生成稀疏深度图
        
        参数:
            dense_depth: 稠密深度图
            sparsity: 稀疏率 (0-1)，值越大越稀疏
        """
        h, w = dense_depth.shape
        mask = np.random.random((h, w)) > sparsity
        
        sparse_depth = np.zeros_like(dense_depth)
        sparse_depth[mask] = dense_depth[mask]
        
        return sparse_depth, mask
    
    def generate_lidar_like(self, dense_depth, num_lines=64, points_per_line=1024):
        """
        生成类似LiDAR的稀疏深度图
        
        参数:
            dense_depth: 稠密深度图
            num_lines: 线数
            points_per_line: 每线点数
        """
        h, w = dense_depth.shape
        sparse_depth = np.zeros_like(dense_depth)
        mask = np.zeros_like(dense_depth, dtype=bool)
        
        # 模拟LiDAR扫描线
        line_indices = np.linspace(0, h - 1, num_lines, dtype=int)
        
        for y in line_indices:
            # 每线采样点
            x_indices = np.linspace(0, w - 1, points_per_line, dtype=int)
            for x in x_indices:
                if 0 <= y < h and 0 <= x < w:
                    sparse_depth[y, x] = dense_depth[y, x]
                    mask[y, x] = True
        
        return sparse_depth, mask
    
    def add_noise(self, sparse_depth, sigma=0.05):
        """添加高斯噪声"""
        noisy = sparse_depth.copy()
        mask = sparse_depth > 0
        noise = np.random.normal(0, sigma, sparse_depth.shape)
        noisy[mask] += noise[mask]
        noisy[mask] = np.clip(noisy[mask], 0, None)
        return noisy
```

---

## 3. 传统插值方法

### 3.1 基础插值方法

```python
class TraditionalDepthCompletion:
    def __init__(self):
        pass
    
    def bilinear_interpolation(self, sparse_depth, mask):
        """双线性插值"""
        h, w = sparse_depth.shape
        
        # 获取有效点坐标
        y_coords, x_coords = np.where(mask)
        valid_depths = sparse_depth[mask]
        
        if len(valid_depths) == 0:
            return np.zeros_like(sparse_depth)
        
        from scipy.interpolate import griddata
        
        # 创建网格
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        
        # 插值
        dense_depth = griddata(
            (y_coords, x_coords),
            valid_depths,
            (y_grid, x_grid),
            method='linear'
        )
        
        # 填充边界
        dense_depth = np.nan_to_num(dense_depth)
        
        return dense_depth
    
    def nearest_neighbor(self, sparse_depth, mask):
        """最近邻插值"""
        h, w = sparse_depth.shape
        
        y_coords, x_coords = np.where(mask)
        valid_depths = sparse_depth[mask]
        
        if len(valid_depths) == 0:
            return np.zeros_like(sparse_depth)
        
        from scipy.interpolate import griddata
        
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        
        dense_depth = griddata(
            (y_coords, x_coords),
            valid_depths,
            (y_grid, x_grid),
            method='nearest'
        )
        
        return dense_depth
    
    def inpainting(self, sparse_depth, mask, method='telea'):
        """
        使用OpenCV的图像修复
        
        参数:
            method: 'telea' 或 'ns' (Navier-Stokes)
        """
        # 准备输入
        sparse_float = sparse_depth.astype(np.float32)
        mask_uint8 = (mask == 0).astype(np.uint8) * 255
        
        if method == 'telea':
            flags = cv2.INPAINT_TELEA
        else:
            flags = cv2.INPAINT_NS
        
        dense_depth = cv2.inpaint(sparse_float, mask_uint8, 3, flags)
        
        return dense_depth
```

### 3.2 基于优化的方法

```python
class OptimizationBasedCompletion:
    def __init__(self, lambda_smooth=0.1):
        self.lambda_smooth = lambda_smooth
    
    def solve_laplacian(self, sparse_depth, mask, max_iter=1000):
        """
        基于拉普拉斯平滑的深度补全
        
        最小化: ||D - D_sparse||_M^2 + λ ||∇D||^2
        """
        h, w = sparse_depth.shape
        D = sparse_depth.copy()
        
        for it in range(max_iter):
            D_new = D.copy()
            
            # 拉普拉斯平滑
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if mask[y, x]:
                        # 保持已知值
                        D_new[y, x] = sparse_depth[y, x]
                    else:
                        # 拉普拉斯更新
                        D_new[y, x] = 0.25 * (
                            D[y - 1, x] + D[y + 1, x] +
                            D[y, x - 1] + D[y, x + 1]
                        )
            
            D = D_new
            
            if it % 100 == 0:
                print(f"Iteration {it}")
        
        return D
    
    def total_variation(self, sparse_depth, mask, lambda_tv=0.1, max_iter=100):
        """
        全变分(TV)正则化深度补全
        """
        h, w = sparse_depth.shape
        D = sparse_depth.copy()
        
        for it in range(max_iter):
            # 计算梯度
            grad_x = np.zeros_like(D)
            grad_y = np.zeros_like(D)
            
            grad_x[:, :-1] = D[:, 1:] - D[:, :-1]
            grad_y[:-1, :] = D[1:, :] - D[:-1, :]
            
            # TV梯度
            tv_grad_x = np.sign(grad_x)
            tv_grad_y = np.sign(grad_y)
            
            # 散度
            div = np.zeros_like(D)
            div[:, 1:] += tv_grad_x[:, :-1]
            div[:, :-1] -= tv_grad_x[:, :-1]
            div[1:, :] += tv_grad_y[:-1, :]
            div[:-1, :] -= tv_grad_y[:-1, :]
            
            # 更新
            step_size = 0.1
            D[~mask] = D[~mask] - step_size * (
                lambda_tv * div[~mask]
            )
            
            D[mask] = sparse_depth[mask]
        
        return D
```

---

## 4. 深度学习方法

### 4.1 编码器-解码器架构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthCompletionNet(nn.Module):
    """基础深度补全网络"""
    def __init__(self):
        super().__init__()
        
        # 深度编码器
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, sparse_depth):
        """前向传播"""
        # 编码
        features = self.depth_encoder(sparse_depth)
        
        # 解码
        dense_depth = self.decoder(features)
        
        # 确保非负
        dense_depth = F.relu(dense_depth)
        
        return dense_depth

# 测试
model = DepthCompletionNet()
sparse_depth = torch.randn(1, 1, 256, 256)
dense_depth = model(sparse_depth)
print(f"输入形状: {sparse_depth.shape}")
print(f"输出形状: {dense_depth.shape}")
```

### 4.2 CSPN (Convolutional Spatial Propagation Network)

```python
class CSPN(nn.Module):
    """卷积空间传播网络"""
    def __init__(self, kernel_size=3, num_iter=8):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_iter = num_iter
        self.half_kernel = kernel_size // 2
        
        # 亲和度预测网络
        self.affinity_net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, kernel_size * kernel_size - 1, kernel_size=3, padding=1)
        )
    
    def forward(self, img, sparse_depth, mask):
        """
        前向传播
        
        参数:
            img: RGB图像
            sparse_depth: 稀疏深度
            mask: 有效深度掩码
        """
        B, C, H, W = img.shape
        
        # 预测亲和度
        affinity_input = torch.cat([img, sparse_depth], dim=1)
        affinity = self.affinity_net(affinity_input)
        
        # 归一化亲和度
        affinity = torch.sigmoid(affinity)
        
        # 初始化深度
        depth = sparse_depth.clone()
        
        # 迭代传播
        for _ in range(self.num_iter):
            depth = self.propagate(depth, affinity, mask)
        
        return depth
    
    def propagate(self, depth, affinity, mask):
        """空间传播"""
        B, C, H, W = depth.shape
        
        new_depth = depth.clone()
        
        # 展开邻域
        pad = self.half_kernel
        depth_pad = F.pad(depth, (pad, pad, pad, pad), mode='constant')
        
        # 遍历邻域
        idx = 0
        for ky in range(self.kernel_size):
            for kx in range(self.kernel_size):
                if ky == self.half_kernel and kx == self.half_kernel:
                    continue
                
                # 提取偏移的深度图
                depth_shifted = depth_pad[:, :, ky:ky+H, kx:kx+W]
                
                # 应用亲和度
                new_depth = new_depth + affinity[:, idx:idx+1, :, :] * depth_shifted
                idx += 1
        
        # 保持已知深度
        new_depth[mask] = depth[mask]
        
        return new_depth
```

### 4.3 SparseConv 稀疏卷积

```python
class SparseConvNet(nn.Module):
    """稀疏卷积深度补全网络"""
    def __init__(self):
        super().__init__()
        
        # 稀疏卷积层
        self.enc1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # 解码器
        self.dec1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
    
    def forward(self, sparse_depth, mask):
        """前向传播"""
        # 输入: 拼接深度和掩码
        x = torch.cat([sparse_depth, mask.float()], dim=1)
        
        # 编码
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1))
        x3 = F.relu(self.enc3(x2))
        
        # 解码
        x = F.relu(self.dec1(x3))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        
        # 非负
        x = F.relu(x)
        
        return x
```

---

## 5. RGB引导的深度补全

### 5.1 多模态融合网络

```python
class RGBGuidedCompletion(nn.Module):
    """RGB引导的深度补全"""
    def __init__(self):
        super().__init__()
        
        # RGB编码器
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 深度编码器
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, img, sparse_depth):
        """前向传播"""
        # RGB特征
        rgb_feat = self.rgb_encoder(img)
        
        # 深度特征
        depth_feat = self.depth_encoder(sparse_depth)
        
        # 融合
        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        fused = self.fusion(fused)
        
        # 解码
        dense_depth = self.decoder(fused)
        dense_depth = F.relu(dense_depth)
        
        return dense_depth

# 测试
model = RGBGuidedCompletion()
img = torch.randn(1, 3, 256, 256)
sparse_depth = torch.randn(1, 1, 256, 256)
dense_depth = model(img, sparse_depth)
print(f"输出形状: {dense_depth.shape}")
```

### 5.2 注意力引导融合

```python
class AttentionGuidedCompletion(nn.Module):
    """注意力引导的深度补全"""
    def __init__(self):
        super().__init__()
        
        # RGB编码器
        self.rgb_encoder = nn.ModuleList([
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        ])
        
        # 深度编码器
        self.depth_encoder = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        ])
        
        # 注意力模块
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(32),
            AttentionBlock(64),
            AttentionBlock(128)
        ])
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, img, sparse_depth):
        rgb_feats = []
        depth_feats = []
        
        # 编码
        x_rgb = img
        x_depth = sparse_depth
        
        for i in range(3):
            x_rgb = F.relu(self.rgb_encoder[i](x_rgb))
            x_depth = F.relu(self.depth_encoder[i](x_depth))
            
            # 注意力融合
            fused = self.attention_blocks[i](x_rgb, x_depth)
            x_depth = fused
            
            rgb_feats.append(x_rgb)
            depth_feats.append(x_depth)
        
        # 解码
        dense_depth = self.decoder(depth_feats[-1])
        dense_depth = F.relu(dense_depth)
        
        return dense_depth

class AttentionBlock(nn.Module):
    """注意力模块"""
    def __init__(self, channels):
        super().__init__()
        
        self.rgb_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.depth_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)
    
    def forward(self, rgb_feat, depth_feat):
        # 计算注意力
        rgb_att = self.rgb_attention(rgb_feat)
        depth_att = self.depth_attention(depth_feat)
        
        # 应用注意力
        rgb_weighted = rgb_feat * rgb_att
        depth_weighted = depth_feat * depth_att
        
        # 融合
        fused = torch.cat([rgb_weighted, depth_weighted], dim=1)
        fused = self.fusion(fused)
        
        return fused + depth_feat  # 残差连接
```

---

## 6. 实践练习

### 练习1：实现简单的深度补全网络

```python
class SimpleCompletionNet(nn.Module):
    """简单的深度补全网络"""
    def __init__(self):
        super().__init__()
        
        # 编码器
        self.enc1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        
        # 解码器
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dec3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
    
    def forward(self, sparse_depth, mask):
        # 输入拼接
        x = torch.cat([sparse_depth, mask.float()], dim=1)
        
        # 编码
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1))
        x3 = F.relu(self.enc3(x2))
        
        # 解码
        x = F.relu(self.dec1(x3))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        
        # 非负
        x = F.relu(x)
        
        return x

# 测试
model = SimpleCompletionNet()
sparse_depth = torch.randn(1, 1, 128, 128)
mask = torch.rand(1, 1, 128, 128) > 0.9
dense_depth = model(sparse_depth, mask)
print(f"输出形状: {dense_depth.shape}")
```

### 练习2：损失函数

```python
class DepthCompletionLoss(nn.Module):
    """深度补全损失函数"""
    def __init__(self):
        super().__init__()
    
    def l1_loss(self, pred, gt, mask):
        """L1损失"""
        loss = torch.abs(pred[mask] - gt[mask])
        return torch.mean(loss)
    
    def l2_loss(self, pred, gt, mask):
        """L2损失"""
        loss = (pred[mask] - gt[mask]) ** 2
        return torch.mean(loss)
    
    def smooth_l1_loss(self, pred, gt, mask, beta=1.0):
        """Smooth L1损失"""
        diff = torch.abs(pred[mask] - gt[mask])
        loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        return torch.mean(loss)
    
    def gradient_loss(self, pred, gt, mask):
        """梯度损失"""
        # x方向梯度
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
        
        # y方向梯度
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
        
        loss_x = torch.mean(torch.abs(pred_dx - gt_dx))
        loss_y = torch.mean(torch.abs(pred_dy - gt_dy))
        
        return loss_x + loss_y
    
    def forward(self, pred, gt, mask):
        """总损失"""
        l1 = self.l1_loss(pred, gt, mask)
        grad = self.gradient_loss(pred, gt, mask)
        
        total_loss = l1 + 0.1 * grad
        
        return total_loss

# 测试
loss_fn = DepthCompletionLoss()
pred = torch.randn(1, 1, 128, 128)
gt = torch.randn(1, 1, 128, 128)
mask = torch.rand(1, 1, 128, 128) > 0.5
loss = loss_fn(pred, gt, mask)
print(f"Loss: {loss.item():.4f}")
```

### 练习3：深度补全评估

```python
class DepthCompletionEvaluator:
    """深度补全评估器"""
    def __init__(self):
        pass
    
    def compute_rmse(self, pred, gt, mask):
        """RMSE"""
        error = pred[mask] - gt[mask]
        return torch.sqrt(torch.mean(error ** 2))
    
    def compute_mae(self, pred, gt, mask):
        """MAE"""
        return torch.mean(torch.abs(pred[mask] - gt[mask]))
    
    def compute_abs_rel(self, pred, gt, mask):
        """绝对相对误差"""
        return torch.mean(torch.abs(pred[mask] - gt[mask]) / gt[mask])
    
    def compute_delta(self, pred, gt, mask, threshold=1.25):
        """delta指标"""
        ratio = torch.max(pred[mask] / gt[mask], gt[mask] / pred[mask])
        return torch.mean((ratio < threshold).float())
    
    def evaluate(self, pred, gt, mask):
        """综合评估"""
        metrics = {
            'rmse': self.compute_rmse(pred, gt, mask).item(),
            'mae': self.compute_mae(pred, gt, mask).item(),
            'abs_rel': self.compute_abs_rel(pred, gt, mask).item(),
            'delta1': self.compute_delta(pred, gt, mask, 1.25).item(),
            'delta2': self.compute_delta(pred, gt, mask, 1.25 ** 2).item(),
            'delta3': self.compute_delta(pred, gt, mask, 1.25 ** 3).item()
        }
        return metrics

# 测试
evaluator = DepthCompletionEvaluator()
pred = torch.rand(1, 1, 128, 128) * 10 + 0.1
gt = torch.rand(1, 1, 128, 128) * 10 + 0.1
mask = torch.rand(1, 1, 128, 128) > 0.5
metrics = evaluator.evaluate(pred, gt, mask)
print("评估指标:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
```

---

**下一节**：[LiDAR点云](04-lidar-pointcloud.md)

---

## 参考文献

1. Uhrig, J., et al. (2017). Sparsity Invariant CNNs.
2. Cheng, X., et al. (2018). Depth Estimation via Affinity Learned with Convolutional Spatial Propagation Network.
3. Qiu, J., et al. (2019). DeepLiDAR: Deep Surface Normal Guided Depth Prediction for Outdoor Scene from Sparse LiDAR Data and Single Color Image.
4. Tang, J., et al. (2019). Learning Joint 2D-3D Representations for Depth Completion.
