# 5.4 几何验证与时序一致性

## 1. 概述

几何验证和时序一致性检查是回环检测的最后防线，用于剔除错误回环。仅有外观相似度（如BoW或NetVLAD）不足以可靠地确认回环。

## 2. 几何验证

### 2.1 基础矩阵验证

对匹配点对估计基础矩阵 $F$：

$$ \mathbf{x}_2^T \mathbf{F} \mathbf{x}_1 = 0 $$

**验证标准**：
- RANSAC内点比例 > 阈值
- Sampson距离 < 阈值

### 2.2 单应矩阵验证

对匹配点对估计单应矩阵 $H$（适用于平面场景或纯旋转）：

$$ \mathbf{x}_2 = \mathbf{H} \mathbf{x}_1 $$

### 2.3 PnP几何验证

如果当前帧有3D信息，使用PnP验证：

1. 基于候选回环建立3D-2D对应关系
2. 使用EPnP估计相机位姿
3. 计算重投影误差
4. 检查内点比例

## 3. 时序一致性

### 3.1 连续帧验证

回环检测不应该基于单帧，而应该验证连续多帧的一致性。

**方法**：
- 检测到候选回环后，检查后续连续帧是否也能与候选回环的连续帧匹配
- 使用贝叶斯滤波累积回环置信度

### 3.2 延迟决策

不立即确认回环，而是在一段时间内积累证据：

$$ P(\text{loop} \mid \mathbf{z}_{1:k}) = \frac{P(\mathbf{z}_k \mid \text{loop}) P(\text{loop} \mid \mathbf{z}_{1:k-1})}{P(\mathbf{z}_k)} $$

### 3.3 一致性检查组

同时检查多个候选帧组：
- 当前帧与候选回环帧
- 当前帧-1与候选回环帧-1
- 当前帧+1与候选回环帧+1（如果有）

## 4. 回环校正

### 4.1 位姿图优化

确认回环后，添加回环边到位姿图中：

$$ \mathbf{e}_{ij} = \ln(\mathbf{T}_{ij}^{-1} \mathbf{T}_i^{-1} \mathbf{T}_j)^\vee $$

### 4.2 全局BA

在回环校正后进行全局BA，调整所有位姿和地图点：

$$ \min \sum_{(i,j)} \|\mathbf{e}_{ij}\|^2_{\mathbf{\Sigma}_{ij}} + \sum_{(i,k)} \|\mathbf{e}_{ik}^{\text{obs}}\|^2_{\mathbf{\Sigma}_{ik}} $$

## 5. 鲁棒回环

当存在误检测时，使用鲁棒优化方法：

- **DCS（Dynamic Covariance Scaling）**
- **Switchable Constraints**
- **Max-Mixture**
- **R-Graph**

## 6. 参考文献

1. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision*. Cambridge University Press.
2. Sünderhauf, N., & Protzel, P. (2012). Switchable constraints for robust pose graph SLAM. *IROS*.
3. Agarwal, P., Tipaldi, G. D., Spinello, L., Stachniss, C., & Burgard, W. (2013). Robust map optimization using dynamic covariance scaling. *ICRA*.
4. Olson, E. (2009). Recognizing places using spectrally clustered local matches. *Robotics and Autonomous Systems*, 57(12), 1157-1172.
