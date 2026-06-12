# 2.2 经典CNN架构演进

## 1. LeNet-5 → AlexNet → VGG → GoogLeNet → ResNet

### 1.1 LeNet-5 (1998)

**为什么提出**：手写支票数字识别的实际需求。

**架构**：Conv→Pool→Conv→Pool→FC→FC→FC

**贡献**：确立了CNN的基本模式。

### 1.2 AlexNet (2012)

**论文**：Krizhevsky, Sutskever, Hinton — NeurIPS 2012

**为什么提出**：在ImageNet大规模图像分类中证明CNN的潜力。

**架构**：5个卷积层 + 3个全连接层

**关键创新**：
- 使用ReLU激活函数（比Sigmoid训练快6倍）
- GPU并行训练（两块GTX 580）
- 重叠最大池化
- Dropout正则化
- 数据增强

**影响**：ImageNet Top-5错误率从26.2%降到15.3%，引爆深度学习革命。

### 1.3 VGG (2014)

**论文**：Simonyan & Zisserman — ICLR 2015

**为什么提出**：探索网络深度对性能的影响。

**核心设计**：全部使用3×3卷积，更深（16-19层）。

| 版本 | 层数 | 参数量 | Top-5错误率 |
|------|------|--------|-------------|
| VGG-16 | 16 | 138M | 7.4% |
| VGG-19 | 19 | 144M | 7.3% |

**局限**：参数量巨大（FC层占89%）。

### 1.4 GoogLeNet / Inception (2014)

**论文**：Szegedy et al. — CVPR 2015

**核心创新**：Inception模块——同一层使用多种尺寸卷积核。

**优势**：比VGG参数量少12倍（6.8M vs 138M），性能更好。

### 1.5 ResNet (2015)

**论文**：He et al. — CVPR 2016 (Best Paper)

**为什么提出**：实证发现网络深度增加时精度反而下降（退化问题）。

**残差块**：

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

**为什么有效**：
- 梯度可以直接流过跳跃连接
- 网络只需学习残差（输出与输入的差异）
- 允许训练极深网络（152层）

**影响**：ResNet-152在ImageNet达到3.57% Top-5错误率（超越人类）。

## 2. 后续架构

| 架构 | 年份 | 核心创新 | 参数量 | ImageNet Top-1 |
|------|------|----------|--------|---------------|
| ResNeXt | 2017 | 分组卷积 | 25M | 77.8% |
| DenseNet | 2017 | 密集连接 | 8M | 77.3% |
| SENet | 2018 | 通道注意力 | 28M | 79.3% |
| EfficientNet | 2019 | 复合缩放 | 5.3M | 80.1% |
| ConvNeXt | 2022 | 现代化CNN | 28M | 82.0% |

### 2.1 SENet (2017)

**Squeeze-and-Excitation**：显式建模通道间依赖关系。

**局限**：增加少量计算开销。

### 2.2 EfficientNet (2019)

**复合缩放**：同时缩放深度、宽度和分辨率。

**NAS搜索**：使用神经架构搜索找到最优缩放策略。

## 3. 在具身智能中的应用

- **ResNet**是机器人感知系统的标准骨干网络
- **EfficientNet**在边缘设备（如Jetson）上高效运行
- **ConvNeXt**被用于最新VLA模型中的视觉编码器
- 特征金字塔（FPN）用于机器人抓取中的多尺度特征

## 4. 参考文献

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS*.
2. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *ICLR*.
3. He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
4. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
5. Hu, J., et al. (2018). Squeeze-and-excitation networks. *CVPR*.
6. Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR*.
