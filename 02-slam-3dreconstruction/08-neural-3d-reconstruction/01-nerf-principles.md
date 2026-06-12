# 8.1 NeRF原理

## 1. 概述

神经辐射场（Neural Radiance Fields, NeRF）是2020年由Mildenhall等人提出的开创性3D表示方法。NeRF使用多层感知机（MLP）隐式表示3D场景，能够从稀疏输入图像生成高质量的新视角图像。

## 2. NeRF核心原理

### 2.1 辐射场的定义

NeRF将场景表示为一个连续函数：

$$ F_{\Theta}: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma) $$

其中：
- $\mathbf{x} = (x, y, z)$：3D空间位置
- $\mathbf{d} = (\theta, \phi)$：视角方向
- $\mathbf{c} = (r, g, b)$：辐射颜色
- $\sigma$：体积密度（衡量该点的"不透明度"）

### 2.2 体素渲染

沿穿过场景的光线进行积分：

$$ C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) dt $$

其中透射率（Transmittance）：

$$ T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) ds\right) $$

### 2.3 离散化

在采样点处数值近似积分：

$$ \hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \alpha_i \mathbf{c}_i $$

$$ T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right) $$

$$ \alpha_i = 1 - \exp(-\sigma_i \delta_i) $$

其中 $\delta_i = t_{i+1} - t_i$ 是采样间隔。

## 3. 网络设计

### 3.1 网络架构

```
输入: (x, y, z) → [FC(256)]*8 → σ
                              ↘ [FC(128)]*1 → 级联d → [FC(128)]*1 → (r, g, b)
```

网络被设计为两个分支：
- **粗分支**：仅根据位置 $\mathbf{x}$ 预测密度 $\sigma$
- **精分支**：结合位置 $\mathbf{x}$ 和方向 $\mathbf{d}$ 预测颜色 $\mathbf{c}$

### 3.2 位置编码

位置编码将连续坐标映射到高频特征空间，使网络能学习高频细节：

$$ \gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)) $$

- 位置 $\mathbf{x}$：$L=10$ → 编码维度 60
- 方向 $\mathbf{d}$：$L=4$ → 编码维度 24

### 3.3 分层采样

**粗网络**：均匀采样 $N_c$ 个点，预测粗密度分布

**精网络**：根据粗网络的密度分布进行重要性采样，在密度高的区域集中更多采样点。

$$ \hat{w}_i = \frac{w_i}{\sum w_j}, \quad w_i = \alpha_i T_i $$

## 4. 训练与损失

### 4.1 损失函数

$$ \mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left[ \|\hat{C}_c(\mathbf{r}) - C(\mathbf{r})\|^2_2 + \|\hat{C}_f(\mathbf{r}) - C(\mathbf{r})\|^2_2 \right] $$

同时优化粗网络和精网络的输出。

### 4.2 训练细节

- 每帧4096条光线
- 每条光线128个采样点（粗）+128个（精）
- Adam优化器，学习率5e-4
- 单个场景训练数小时

## 5. NeRF的优势与局限

**优势**：
- 高质量的连续场景表示
- 自然的视角依赖效果（反射、折射）
- 隐式表示可处理任意拓扑

**局限**：
- 训练和渲染速度慢
- 只能处理静态场景
- 需要大量已知位姿的图像
- 泛化能力有限（逐场景优化）

## 6. 参考文献

1. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing scenes as neural radiance fields for view synthesis. *ECCV*.
2. Tancik, M., et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. *NeurIPS*.
3. Max, N. (1995). Optical models for direct volume rendering. *IEEE Transactions on Visualization and Computer Graphics*, 1(2), 99-108.
