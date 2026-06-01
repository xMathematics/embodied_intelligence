# 2.1 单目深度估计

## 目录

- [1. 引言](#1-引言)
- [2. 单目深度估计概述](#2-单目深度估计概述)
- [3. 传统方法](#3-传统方法)
- [4. 深度学习方法](#4-深度学习方法)
- [5. 自监督学习](#5-自监督学习)
- [6. 实践练习](#6-实践练习)

---

## 1. 引言

### 1.1 单目深度估计的重要性

**单目深度估计**是指仅使用单一摄像头从二维图像中恢复三维深度信息的技术，是计算机视觉和机器人感知中的关键任务。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **自动驾驶** | 从单目相机估计障碍物距离 | 车辆、行人距离感知 |
| **AR/VR** | 增强现实中的深度感知 | 虚拟物体放置 |
| **机器人导航** | 室内环境理解 | 避障、路径规划 |
| **人机交互** | 手势识别和距离感知 | 交互界面控制 |

---

## 2. 单目深度估计概述

### 2.1 定义

**单目深度估计**：给定一张 RGB 图像，预测每个像素点对应的深度值（距离相机的距离）。

**形式化表达**：
```
D = f(I; θ)
```
其中 `I` 是输入图像，`θ` 是模型参数，`D` 是深度图。

### 2.2 挑战

- 尺度歧义：单目图像无法直接确定绝对尺度
- 纹理缺失区域：低纹理区域难以估计深度
- 运动模糊：动态场景下的模糊影响估计
- 光照变化：不同光照条件下的外观变化

---

## 3. 传统方法

### 3.1 基于运动的方法 (SfM)

```python
import numpy as np
import cv2

class StructureFromMotion:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def extract_features(self, image):
        """提取ORB特征"""
        kp, des = self.orb.detectAndCompute(image, None)
        return kp, des
    
    def match_features(self, des1, des2):
        """特征匹配"""
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches
    
    def estimate_pose(self, kp1, kp2, matches, K):
        """估计相机位姿"""
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, K)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
        
        return R, t
    
    def triangulate_points(self, kp1, kp2, matches, K, R, t):
        """三角化计算3D点"""
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))
        
        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts3d = pts4d[:3] / pts4d[3]
        
        return pts3d.T
```

### 3.2 基于景深的方法

```python
class DepthFromDefocus:
    def __init__(self):
        pass
    
    def compute_blur_kernel(self, image, aperture=2.8):
        """计算模糊核"""
        # 简化的模糊核估计
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        blurred = cv2.filter2D(gray, -1, kernel)
        return blurred
    
    def estimate_depth_from_blur(self, sharp_img, blur_img):
        """从模糊估计深度"""
        diff = np.abs(sharp_img.astype(float) - blur_img.astype(float))
        blur_map = np.mean(diff, axis=2)
        
        # 假设模糊程度与深度成反比
        depth = 1.0 / (blur_map + 1e-6)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        return depth
```

---

## 4. 深度学习方法

### 4.1 编码器-解码器架构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthEncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Conv2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Conv3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Conv4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            # Upconv1
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Upconv2
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Upconv3
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Upconv4
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # 最终层
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        """前向传播"""
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth

# 测试
model = DepthEncoderDecoder()
input = torch.randn(1, 3, 256, 256)
output = model(input)
print(f"输入形状: {input.shape}")
print(f"输出深度图形状: {output.shape}")
```

### 4.2 Monodepth 系列

```python
class Monodepth2(nn.Module):
    """Monodepth2: 自监督深度估计"""
    def __init__(self, num_layers=18):
        super().__init__()
        
        # 使用ResNet作为编码器
        from torchvision import models
        if num_layers == 18:
            self.encoder = models.resnet18(pretrained=True)
        elif num_layers == 50:
            self.encoder = models.resnet50(pretrained=True)
        
        # 移除最后的全连接层
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        
        # 解码器
        self.decoder = self._build_decoder()
        
        # 视差预测头
        self.disp_head = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    
    def _build_decoder(self):
        """构建解码器"""
        return nn.ModuleDict({
            'upconv1': nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            'iconv1': nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            'upconv2': nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            'iconv2': nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            'upconv3': nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            'iconv3': nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        })
    
    def forward(self, x):
        """前向传播"""
        # 编码器
        x1 = self.encoder[0](x)    # 64
        x2 = self.encoder[1](x1)   # 64
        x3 = self.encoder[2](x2)   # 64
        x4 = self.encoder[3](x3)   # 128
        x5 = self.encoder[4](x4)   # 256
        x6 = self.encoder[5](x5)   # 512
        x7 = self.encoder[6](x6)   # 512
        
        # 解码器
        x = self.decoder['upconv1'](x7)
        x = torch.cat([x, x6], dim=1)
        x = F.relu(self.decoder['iconv1'](x))
        
        x = self.decoder['upconv2'](x)
        x = torch.cat([x, x4], dim=1)
        x = F.relu(self.decoder['iconv2'](x))
        
        x = self.decoder['upconv3'](x)
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.decoder['iconv3'](x))
        
        # 视差预测
        disp = self.disp_head(x)
        disp = F.sigmoid(disp)
        
        # 视差转深度
        depth = 1.0 / (disp + 1e-6)
        
        return depth, disp

# 测试
model = Monodepth2()
input = torch.randn(1, 3, 256, 512)
depth, disp = model(input)
print(f"深度图形状: {depth.shape}")
print(f"视差图形状: {disp.shape}")
```

### 4.3 DepthFormer (Transformer-based)

```python
class DepthFormer(nn.Module):
    """基于Transformer的深度估计"""
    def __init__(self, img_size=256, patch_size=16, embed_dim=512):
        super().__init__()
        
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        # Patch嵌入
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=2048
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Patch嵌入
        x = self.patch_embed(x)  # [B, E, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # 重塑回图像格式
        x = x.transpose(1, 2)  # [B, E, N]
        x = x.view(B, -1, H//self.patch_size, W//self.patch_size)
        
        # 解码
        depth = self.decoder(x)
        
        return depth

# 测试
model = DepthFormer()
input = torch.randn(1, 3, 256, 256)
output = model(input)
print(f"DepthFormer输出形状: {output.shape}")
```

---

## 5. 自监督学习

### 5.1 光度损失

```python
class SelfSupervisedDepthLoss(nn.Module):
    """自监督深度估计损失函数"""
    def __init__(self):
        super().__init__()
        self.ssim_weight = 0.85
        self.L1_weight = 0.15
    
    def ssim(self, x, y):
        """计算SSIM损失"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        
        sigma_x = F.avg_pool2d(x**2, 3, 1, 1) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, 3, 1, 1) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
        
        ssim_map = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
                   ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        
        return ssim_map
    
    def warp_image(self, img, depth, K, R, t):
        """根据深度图和位姿扭曲图像"""
        # 简化的图像扭曲实现
        B, C, H, W = img.shape
        
        # 生成像素坐标
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
        coords = torch.stack([x, y, torch.ones_like(x)], dim=0).float()
        
        # 投影到3D
        K_inv = torch.inverse(K)
        points_3d = depth * (K_inv @ coords.view(3, -1))
        
        # 应用变换
        points_3d_transformed = R @ points_3d + t
        
        # 投影回2D
        points_2d = K @ points_3d_transformed
        points_2d = points_2d[:2] / points_2d[2:]
        
        # 重采样
        warped = F.grid_sample(img, points_2d.view(1, H, W, 2), align_corners=True)
        
        return warped
    
    def forward(self, img1, img2, depth1, depth2, pose):
        """计算损失"""
        # 光度损失
        warped1 = self.warp_image(img2, depth1, ...)
        warped2 = self.warp_image(img1, depth2, ...)
        
        loss_recon1 = self.compute_recon_loss(img1, warped1)
        loss_recon2 = self.compute_recon_loss(img2, warped2)
        
        # 平滑损失
        loss_smooth1 = self.compute_smooth_loss(depth1)
        loss_smooth2 = self.compute_smooth_loss(depth2)
        
        total_loss = loss_recon1 + loss_recon2 + 0.1 * (loss_smooth1 + loss_smooth2)
        
        return total_loss
    
    def compute_recon_loss(self, pred, target):
        """计算重建损失"""
        ssim = self.ssim(pred, target)
        loss_ssim = torch.mean(1 - ssim)
        loss_l1 = torch.mean(torch.abs(pred - target))
        
        return self.ssim_weight * loss_ssim + self.L1_weight * loss_l1
    
    def compute_smooth_loss(self, depth):
        """计算深度平滑损失"""
        grad_x = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        grad_y = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        
        return torch.mean(grad_x) + torch.mean(grad_y)
```

---

## 6. 实践练习

### 练习1：实现简单的深度估计网络

```python
class SimpleDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 解码器
        self.dec_conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.dec_conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.dec_conv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # 输出层
        self.output = nn.Conv2d(16, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        # 编码
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(x1)
        x3 = self.enc_conv3(x2)
        
        # 解码
        x = self.dec_conv1(x3)
        x = self.dec_conv2(x)
        x = self.dec_conv3(x)
        
        # 输出深度
        depth = self.output(x)
        depth = F.relu(depth)  # 深度非负
        
        return depth

# 测试
model = SimpleDepthNet()
input = torch.randn(1, 3, 128, 128)
output = model(input)
print(f"输出深度图形状: {output.shape}")
```

### 练习2：深度可视化

```python
import matplotlib.pyplot as plt

class DepthVisualizer:
    def __init__(self):
        pass
    
    def normalize_depth(self, depth):
        """归一化深度图用于可视化"""
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        return depth_normalized
    
    def apply_colormap(self, depth_normalized):
        """应用伪彩色"""
        # 使用JET colormap
        depth_8bit = (depth_normalized * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        return colormap
    
    def visualize(self, image, depth):
        """可视化图像和深度图"""
        depth_normalized = self.normalize_depth(depth)
        depth_colored = self.apply_colormap(depth_normalized)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('输入图像')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(depth_normalized, cmap='gray')
        plt.title('深度图(灰度)')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
        plt.title('深度图(伪彩色)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# 测试
visualizer = DepthVisualizer()
# visualizer.visualize(image, depth)
```

### 练习3：评估指标

```python
class DepthEvaluator:
    def __init__(self):
        pass
    
    def compute_abs_rel(self, pred, gt):
        """计算绝对相对误差"""
        return torch.mean(torch.abs(pred - gt) / (gt + 1e-6))
    
    def compute_sq_rel(self, pred, gt):
        """计算平方相对误差"""
        return torch.mean((pred - gt) ** 2 / (gt + 1e-6))
    
    def compute_rmse(self, pred, gt):
        """计算均方根误差"""
        return torch.sqrt(torch.mean((pred - gt) ** 2))
    
    def compute_rmse_log(self, pred, gt):
        """计算对数均方根误差"""
        return torch.sqrt(torch.mean((torch.log(pred + 1e-6) - torch.log(gt + 1e-6)) ** 2))
    
    def compute_delta(self, pred, gt, threshold=1.25):
        """计算delta指标"""
        ratio = torch.max(pred / (gt + 1e-6), gt / (pred + 1e-6))
        delta1 = torch.mean((ratio < threshold).float())
        delta2 = torch.mean((ratio < threshold**2).float())
        delta3 = torch.mean((ratio < threshold**3).float())
        return delta1, delta2, delta3
    
    def evaluate(self, pred, gt):
        """综合评估"""
        metrics = {
            'abs_rel': self.compute_abs_rel(pred, gt).item(),
            'sq_rel': self.compute_sq_rel(pred, gt).item(),
            'rmse': self.compute_rmse(pred, gt).item(),
            'rmse_log': self.compute_rmse_log(pred, gt).item()
        }
        
        delta1, delta2, delta3 = self.compute_delta(pred, gt)
        metrics.update({
            'delta1': delta1.item(),
            'delta2': delta2.item(),
            'delta3': delta3.item()
        })
        
        return metrics

# 测试
evaluator = DepthEvaluator()
pred_depth = torch.rand(1, 1, 128, 128) * 10 + 0.1
gt_depth = torch.rand(1, 1, 128, 128) * 10 + 0.1
metrics = evaluator.evaluate(pred_depth, gt_depth)
print("评估指标:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
```

---

**下一节**：[双目立体视觉](02-stereo-vision.md)

---

## 参考文献

1. Godard, C., et al. (2017). Unsupervised Monocular Depth Estimation with Left-Right Consistency.
2. Godard, C., et al. (2019). Digging Into Self-Supervised Monocular Depth Estimation.
3. Ranftl, R., et al. (2021). Vision Transformers for Dense Prediction.
