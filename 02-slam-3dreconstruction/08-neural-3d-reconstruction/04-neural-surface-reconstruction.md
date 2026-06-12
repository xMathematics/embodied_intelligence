# 8.4 神经表面重建

## 1. 概述

神经表面重建使用神经网络学习场景表面的隐式表示。基于有符号距离函数（SDF）和占用场（Occupancy Field）的方法是其中的主流。

## 2. 占用网络（Occupancy Networks）

**Mescheder et al., CVPR 2019**

### 2.1 核心思想

学习一个二分类函数：给定空间点，判断其是否在物体内部：

$$ f_{\Theta}(\mathbf{p}) = \begin{cases} 1 & \text{点在物体内部} \\ 0 & \text{点在物体外部} \end{cases} $$

### 2.2 损失函数

使用二值交叉熵损失：

$$ \mathcal{L} = \sum_{i} [y_i \log f_{\Theta}(\mathbf{p}_i) + (1-y_i) \log(1-f_{\Theta}(\mathbf{p}_i))] $$

## 3. DeepSDF

**Park et al., CVPR 2019**

### 3.1 核心思想

学习一个函数，输出点到物体表面的有符号距离：

$$ f_{\Theta}(\mathbf{p}) = \text{SDF}(\mathbf{p}) $$

- SDF = 0：点在表面
- SDF > 0：点在表面外部
- SDF < 0：点在表面内部

### 3.2 潜在编码

DeepSDF使用条件网络，每个物体有一个潜在编码 $\mathbf{z}$：

$$ f_{\Theta}(\mathbf{p}, \mathbf{z}) = \text{SDF}(\mathbf{p}) $$

## 4. NeuS

**Wang et al., NeurIPS 2021**

### 4.1 核心创新

将SDF表示与体素渲染结合，实现从多视图图像进行神经表面重建。

**SDF到体素密度的转换**：

$$ \sigma(\mathbf{x}) = \max\left(-\frac{d\Phi_s}{dt}(\text{SDF}(\mathbf{x})), 0\right) $$

其中 $\Phi_s$ 是logistic函数：

$$ \Phi_s(t) = (1 + \exp(-st))^{-1} $$

### 4.2 损失函数

- 渲染损失（与NeRF相同）
- Eikonal正则化：$\|\nabla \text{SDF}(\mathbf{x})\| = 1$
- 法向量一致性

## 5. VolSDF

**Yariv et al., NeurIPS 2021**

### 5.1 核心思想

将SDF转换为体素密度用于渲染：

$$ \sigma(\mathbf{x}) = \frac{1}{\beta} \Psi_{\beta}(-\text{SDF}(\mathbf{x})) $$

其中 $\Psi_{\beta}$ 是累计分布函数。

### 5.2 几何正则化

- 法向量平滑
- 曲率正则化
- 二进制交叉熵损失

## 6. Neural implicit surface方法对比

| 方法 | 年份 | 表示 | 输入 | 特点 |
|------|------|------|------|------|
| DeepSDF | 2019 | SDF | 3D(带编码) | 单物体泛化 |
| Occupancy Networks | 2019 | 占用 | 3D | 简单有效 |
| NeuS | 2021 | SDF | 多视图图像 | 高质量表面 |
| VolSDF | 2021 | SDF | 多视图图像 | 几何精度高 |
| NeuralWarp | 2022 | SDF | 多视图图像 | 匹配代价引导 |
| MonoSDF | 2022 | SDF | 单目(辅助) | 单目深度辅助 |

## 7. 神经表面重建流程

```
多视图图像 → [特征提取] → [SDF网络] → [体素渲染] → 新视图
                              ↓
                    [Marching Cubes] → 三角网格
```

## 8. 参考文献

1. Mescheder, L., Oechsle, M., Niemeyer, M., Nowozin, S., & Geiger, A. (2019). Occupancy networks: Learning 3D reconstruction in function space. *CVPR*.
2. Park, J. J., Florence, P., Straub, J., Newcombe, R., & Lovegrove, S. (2019). DeepSDF: Learning continuous signed distance functions for shape representation. *CVPR*.
3. Wang, P., Liu, L., Liu, Y., Theobalt, C., Komura, T., & Wang, W. (2021). NeuS: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. *NeurIPS*.
4. Yariv, L., et al. (2021). Volume rendering of neural implicit surfaces. *NeurIPS*.
