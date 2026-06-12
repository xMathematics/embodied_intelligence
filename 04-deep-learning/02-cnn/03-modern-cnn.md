# 2.3 现代CNN与轻量化设计

## 1. 轻量化CNN的背景

**为什么需要**：移动端、嵌入式设备（机器人）计算资源有限，需要实时推理。

**核心权衡**：精度 vs 速度 vs 参数量。

## 2. 深度可分离卷积

**标准卷积**：计算量 = $K \times K \times C_{\text{in}} \times C_{\text{out}} \times H \times W$

**深度可分离卷积**：
1. Depthwise Conv：每个通道独立卷积
2. Pointwise Conv：1×1卷积融合通道

**计算量比**：$\frac{1}{C_{\text{out}}} + \frac{1}{K^2}$

当 $K=3, C_{\text{out}}=64$ 时，计算量约为标准卷积的 $\frac{1}{8}$。

## 3. MobileNet系列

| 版本 | 核心创新 | ImageNet Top-1 | 参数量 | 特点 |
|------|----------|---------------|--------|------|
| V1 | 深度可分离卷积 | 70.6% | 4.2M | 奠基 |
| V2 | 倒残差+线性瓶颈 | 72.0% | 3.4M | 更好特征流 |
| V3 | NAS+SE+Swish | 75.2% | 5.4M | 最优效率 |
| V4 | 通用高效架构 | 78.0% | 6.9M | 最新 |

## 4. ConvNeXt：现代化的CNN

**为什么提出**：ViT在视觉任务上超越CNN后，探究CNN能否通过"现代化"缩小差距。

**改进措施**：
| 设计 | 从Swin借鉴 | ConvNeXt配置 |
|------|-----------|-------------|
| 训练策略 | AdamW, large epoch | 300 epoch |
| 激活函数 | GELU | ReLU→GELU |
| 归一化 | LayerNorm | BN→LN |
| 下采样 | Patch Merging | 非重叠 Conv |
| 卷积核 | 7×7 depthwise | 大核深度可分离 |
| 微观设计 | Inverted Bottleneck | 4:1比例 |

**结果**：ConvNeXt在ImageNet上达到82.0% Top-1，与Swin Transformer相当。

## 5. 注意力增强CNN

| 方法 | 类型 | 描述 |
|------|------|------|
| SE-Net | 通道注意力 | Squeeze-and-Excitation |
| CBAM | 通道+空间 | Convolutional Block Attention Module |
| Non-local | 全局依赖 | 自注意力式非局部操作 |
| Coordinate Attention | 位置感知 | 编码位置信息的注意力 |

## 6. 在具身智能中的应用

- **MobileNet**用于机器人上的实时目标检测（如Jetson Nano）
- **ConvNeXt**作为RT-2等VLA模型的视觉骨干
- **高效CNN**使无人机、四足机器人上的实时视觉感知成为可能
- **注意力增强CNN**提高机器人抓取检测的精度

## 7. 参考文献

1. Howard, A. G., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. *arXiv:1704.04861*.
2. Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*.
3. Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV*.
4. Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR*.
5. Woo, S., et al. (2018). CBAM: Convolutional block attention module. *ECCV*.
