# 7.6 RGB-D稠密重建系统

## 1. 概述

RGB-D相机（如Kinect、RealSense）的出现推动了实时稠密重建的巨大进步。本章介绍从KinectFusion到BundleFusion等代表性实时RGB-D重建系统。

## 2. KinectFusion

**Newcombe et al., ISMAR 2011** - 开创性的实时RGB-D重建系统。

### 2.1 系统架构

1. **深度图预处理**：双边滤波去噪
2. **相机跟踪**：点到面ICP
3. **TSDF融合**：将深度融合到全局TSDF
4. **表面预测**：光线投射生成参考深度图

### 2.2 关键技术

**点到面ICP**：
$$ \mathbf{T}^* = \arg\min_{\mathbf{T}} \sum_i \|(\mathbf{T} \mathbf{v}_i - \mathbf{u}_i)^T \mathbf{n}_i\|^2 $$

**TSDF增量更新**：
$$ F_{k+1}(\mathbf{v}) = \frac{W_k(\mathbf{v}) F_k(\mathbf{v}) + w_{k+1}(\mathbf{v}) f_{k+1}(\mathbf{v})}{W_k(\mathbf{v}) + w_{k+1}(\mathbf{v})} $$

**光线投射渲染**：沿光线采样TSDF，查找零交叉点。

### 2.3 贡献与局限

**贡献**：
- 首个实时稠密SLAM系统
- GPU加速的TSDF融合
- 高质量重建结果

**局限**：
- 仅限于小场景（GPU内存限制）
- 无回环检测
- 相机跟踪可能丢失

## 3. DynamicFusion

**Newcombe et al., CVPR 2015** - 扩展KinectFusion到动态场景。

**核心创新**：
- 非刚体变形场（warp field）
- 每帧动态估计
- 可重建非刚性运动的物体

**变形场表示**：
$$ \mathbf{v}' = \sum_{k \in \mathcal{N}(\mathbf{v})} w_k(\mathbf{v}) (\mathbf{R}_k (\mathbf{v} - \mathbf{g}_k) + \mathbf{g}_k + \mathbf{t}_k) $$

## 4. ElasticFusion

**Whelan et al., IJRR 2016** - 支持回环的稠密SLAM。

**核心创新**：
- 基于面元的变形图（Deformation Graph）
- 非刚体回环校正
- 长期漂移处理

**面元地图**：每个面元包含位置、法线、颜色、权重、半径。

## 5. BundleFusion

**Dai et al., ACM SIGGRAPH 2017** - 全局一致的实时3D重建。

**核心创新**：
- 在线全局BA
- 分层优化（局部+全局）
- 基于SIFT特征的稀疏对齐
- 重新融合校正TSDF

**系统流程**：
```
深度帧 → [局部ICP跟踪] → [分层全局BA] → [TSDF重新融合] → 高质量模型
```

## 6. 实时稠密重建系统对比

| 系统 | 年份 | 回环 | 动态 | 场景大小 | 实时 | 特点 |
|------|------|------|------|---------|------|------|
| KinectFusion | 2011 | 无 | 静态 | 小 | 是 | 开创性 |
| Kintinuous | 2015 | 有限 | 静态 | 大 | 近实时 | 大场景 |
| DynamicFusion | 2015 | 无 | 动态 | 小 | 是 | 动态重建 |
| ElasticFusion | 2016 | 有 | 静态 | 中 | 是 | 非刚体回环 |
| BundleFusion | 2017 | 有 | 静态 | 大 | 是 | 全局一致 |
| Voxblox | 2017 | 无 | 静态 | 大 | 是 | 导航用 |

## 7. 参考文献

1. Newcombe, R. A., et al. (2011). KinectFusion: Real-time dense surface mapping and tracking. *ISMAR*.
2. Newcombe, R. A., Fox, D., & Seitz, S. M. (2015). DynamicFusion: Reconstruction and tracking of non-rigid scenes in real-time. *CVPR*.
3. Whelan, T., et al. (2016). ElasticFusion: Dense SLAM without a pose graph. *Robotics: Science and Systems*.
4. Dai, A., Nießner, M., Zollhöfer, M., Izadi, S., & Theobalt, C. (2017). BundleFusion: Real-time globally consistent 3D reconstruction using on-the-fly surface re-integration. *ACM Transactions on Graphics*, 36(4).
