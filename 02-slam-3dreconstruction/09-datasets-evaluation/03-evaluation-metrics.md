# 9.3 评估指标

## 1. 概述

标准化的评估指标对于比较SLAM和三维重建系统的性能至关重要。本章介绍最常用的评估指标及其数学定义。

## 2. SLAM评估指标

### 2.1 绝对轨迹误差（ATE）

$$ \text{ATE}(\mathbf{P}, \mathbf{Q}) = \left( \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{P}_i - \mathbf{Q}_i\|^2 \right)^{1/2} $$

其中 $\mathbf{P}$ 和 $\mathbf{Q}$ 是经过对齐后的估计轨迹和真实轨迹。

**对齐**：使用Sim(3)对齐估计轨迹与真值

$$ \mathbf{s}, \mathbf{R}, \mathbf{t} = \arg\min \sum_i \|\mathbf{P}_i - s\mathbf{R}\mathbf{Q}_i - \mathbf{t}\|^2 $$

### 2.2 相对位姿误差（RPE）

$$ \text{RPE} = \frac{1}{N} \sum_{i=1}^{N} \|\ln(\mathbf{T}_{i,i+\Delta}^{-1} \mathbf{T}_{i,i+\Delta}^{\text{gt}})^\vee\| $$

RPE衡量固定时间间隔 $\Delta$ 内的局部轨迹漂移。

### 2.3 ATE vs RPE

| 指标 | 衡量 | 适用 |
|------|------|------|
| ATE | 全局轨迹精度 | 整体评估 |
| RPE | 局部漂移 | 里程计评估 |

## 3. 三维重建评估指标

### 3.1 精确度（Accuracy）

$$ \text{Acc} = \text{mean}_{p \in P} \min_{q \in Q} \|p - q\| $$

重建点云到真值点云的距离。

### 3.2 完整度（Completeness）

$$ \text{Comp} = \text{mean}_{q \in Q} \min_{p \in P} \|p - q\| $$

真值点云到重建点云的距离。

### 3.3 F1分数

$$ F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$

- **Precision**：距离 < 阈值的重建点比例
- **Recall**：距离 < 阈值的真值点比例

### 3.4 Chamfer距离

$$ d_{\text{CD}}(P, Q) = \frac{1}{|P|} \sum_{p \in P} \min_{q \in Q} \|p - q\| + \frac{1}{|Q|} \sum_{q \in Q} \min_{p \in P} \|p - q\| $$

## 4. 新视图合成评估指标

| 指标 | 全称 | 范围 | 最佳值 |
|------|------|------|--------|
| PSNR | 峰值信噪比 | $[0, \infty)$ | 越大越好 |
| SSIM | 结构相似性 | $[-1, 1]$ | 越接近1越好 |
| LPIPS | 学习感知相似性 | $[0, \infty)$ | 越小越好 |

### 4.1 PSNR

$$ \text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}} \right) $$

其中 $\text{MAX}$ 是最大像素值（通常为255）。

### 4.2 SSIM

$$ \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)} $$

## 5. 其他指标

| 指标 | 应用 | 公式 |
|------|------|------|
| 回环检测精度 | 回环检测 | $\frac{TP}{TP+FP}$ |
| 回环检测召回率 | 回环检测 | $\frac{TP}{TP+FN}$ |
| 计算时间 | 效率评估 | 毫秒/帧 |
| 内存占用 | 效率评估 | MB |

## 6. 参考文献

1. Sturm, J., et al. (2012). A benchmark for the evaluation of RGB-D SLAM systems. *IROS*.
2. Seitz, S. M., et al. (2006). A comparison and evaluation of multi-view stereo reconstruction algorithms. *CVPR*.
3. Wang, Z., et al. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4), 600-612.
4. Zhang, R., et al. (2018). The unreasonable effectiveness of deep features as a perceptual metric. *CVPR*.
