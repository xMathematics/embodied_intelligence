# 6.4 语义地图

## 1. 概述

语义地图在几何地图基础上加入语义信息（物体类别、属性、关系），使机器人能够理解环境的高层语义。语义地图是实现智能机器人交互和任务执行的关键。

## 2. 语义SLAM

### 2.1 语义信息注入

语义SLAM将语义分割或目标检测结果融入SLAM系统：

```mermaid
输入帧 → [语义分割/检测] → 语义标签
   ↓                              ↓
[几何SLAM] → [语义-几何融合] → [语义地图]
```

### 2.2 语义SLAM的优势

- **鲁棒数据关联**：利用语义标签提高匹配可靠性
- **动态物体处理**：识别并忽略动态物体
- **重定位辅助**：语义物体作为鲁棒的landmark
- **高层任务支持**：语义信息支持任务规划

### 2.3 代表系统

| 系统 | 语义来源 | 地图形式 | 特点 |
|------|----------|----------|------|
| SemanticFusion | CNN分割 | 稠密+语义标签 | 实时 |
| Kimera | 实例分割 | 度量+语义网格 | VIO+语义 |
| CubeSLAM | 3D物体检测 | 物体级语义地图 | 物体SLAM |
| DROID-SLAM | 语义分割 | 稠密深度+语义 | 学习型SLAM |

## 3. 物体级地图

### 3.1 物体SLAM

以物体（Object）而非点特征作为基本地图元素：

**物体表示**：
- 3D包围盒（3D Bounding Box）
- 物体类别和属性
- 物体位姿和尺寸

### 3.2 物体SLAM的优势

- **高层语义**：更接近人类对环境的理解
- **紧凑表示**：一个物体替代数百个特征点
- **长期稳定性**：物体相对于点特征更稳定
- **任务相关**：直接支持抓取、操作等任务

### 3.3 物体检测与SLAM的耦合

- **耦合方式**：检测引导SLAM / SLAM辅助跟踪
- **数据关联**：基于IoU/特征/位姿的物体关联
- **优化**：联合优化相机位姿、物体位姿和点特征

## 4. 场景图（Scene Graph）

### 4.1 定义

场景图是结构化的语义地图表示：

$$ \mathcal{G} = (\mathcal{V}, \mathcal{E}) $$

- **节点 $\mathcal{V}$**：物体实例、房间、区域等
- **边 $\mathcal{E}$**：空间关系（在...上、在...里、旁边）、属性关系

### 4.2 场景图SLAM

**Hydra**：使用场景图表示的多机器人SLAM
**GraphSLAM++**：语义增强的图SLAM

**建图流程**：
1. 检测图像中的物体及其关系
2. 将2D检测关联到3D地图
3. 构建场景图
4. 利用场景图优化地图

## 5. 动态语义地图

处理环境中的动态变化：

- **可移动物体**：椅子、箱子等
- **状态变化**：门开关、灯开关
- **长期变化**：季节、装修

**方法**：
- 基于变化的更新策略
- 多层地图（静态层+动态层）
- 时间一致性检查

## 6. 语义地图的未来

- **大规模语言模型集成**：LLM读取语义地图进行任务规划
- **开放词汇语义**：突破固定类别限制
- **交互式建图**：人机协作完善语义地图

## 7. 参考文献

1. McCormac, J., Handa, A., Davison, A., & Leutenegger, S. (2017). SemanticFusion: Dense 3D semantic mapping with convolutional neural networks. *ICRA*.
2. Rosinol, A., et al. (2019). Kimera: an open-source library for real-time metric-semantic localization and mapping. *ICRA*.
3. Yang, S., & Scherer, S. (2019). CubeSLAM: Monocular 3D object SLAM. *IEEE Transactions on Robotics*, 35(4), 925-938.
4. Wald, J., et al. (2020). Learning end-to-end scene graph generation for 3D scene understanding. *CVPR*.
