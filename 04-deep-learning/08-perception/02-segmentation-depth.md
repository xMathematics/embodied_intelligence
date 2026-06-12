# 8.2 语义分割与深度估计

## 1. 为什么需要密集预测

**目标检测**只给出边界框，但很多任务需要像素级的理解：
- **分割**：哪些像素属于哪个物体
- **深度**：每个像素的距离

## 2. 语义分割

**FCN**（Long et al., 2015）：全卷积网络，编码器-解码器结构。

**DeepLab系列**：空洞卷积保持分辨率，ASPP多尺度特征。

**Mask2Former**（2022）：统一语义/实例/全景分割的Transformer方法。

## 3. 单目深度估计

**问题**：从单张图像估计每个像素的深度——本质上是病态问题。

**MiDaS**（Ranftl et al., 2019）：混合多数据集训练。

**Depth Anything**（Yang et al., 2024）：大规模无标注训练，当前最强。

## 4. 在具身智能中的应用

- **抓取规划**：分割识别物体区域，深度提供3D信息
- **避障导航**：深度图提供障碍物距离
- **场景理解**：语义分割理解环境构成
- **操作目标**：分割目标物体用于精确操作

## 5. 参考文献

1. Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. *CVPR*.
2. Chen, L. C., et al. (2017). DeepLab: Semantic image segmentation with deep convolutional nets. *IEEE TPAMI*.
3. Cheng, B., et al. (2022). Masked-attention mask transformer for universal image segmentation. *CVPR*.
4. Yang, L., et al. (2024). Depth anything: Unleashing the power of large-scale unlabeled data. *arXiv*.
