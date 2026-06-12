# 2.2 特征描述子

## 1. 概述

特征描述子是对特征点周围图像区域的编码表示，用于在不同图像之间建立特征对应关系。一个好的描述子应具有独特性、鲁棒性和高效性。本章介绍从经典浮点描述子到现代学习型描述子的全面内容。

## 2. 浮点型描述子

### 2.1 SIFT描述子

SIFT描述子（Lowe, 2004）是一个128维的梯度直方图向量。

**生成步骤**：
1. 将特征点邻域旋转到主方向
2. 取16×16邻域，分为4×4=16个子块
3. 每个子块计算8个方向的梯度直方图
4. 得到16×8=128维向量
5. 归一化以抵抗光照变化

**梯度计算**：

$$ m(x,y) = \sqrt{(L(x+1,y)-L(x-1,y))^2 + (L(x,y+1)-L(x,y-1))^2} $$
$$ \theta(x,y) = \tan^{-1}((L(x,y+1)-L(x,y-1))/(L(x+1,y)-L(x-1,y))) $$

### 2.2 SURF描述子

SURF描述子（Bay et al., 2006）使用Haar小波响应替代梯度计算，64维向量。

**特点**：
- 使用积分图像加速Haar小波计算
- 将20×20邻域分为4×4子区域
- 每个子区域计算 $\sum dx, \sum dy, \sum |dx|, \sum |dy|$

### 2.3 GLOH描述子

GLOH（Gradient Location and Orientation Histogram）是SIFT的改进，使用对数极坐标分块，输出128维或272维向量。

## 3. 二进制描述子

### 3.1 BRIEF描述子

BRIEF（Binary Robust Independent Elementary Features, 2010）使用像素强度比较生成二进制字符串。

**定义**：在特征点邻域内选择 $n$ 对像素点 $(p_i, q_i)$：

$$ \tau(\mathbf{p}_i; \mathbf{q}_i) = \begin{cases} 1 & I(\mathbf{p}_i) < I(\mathbf{q}_i) \\ 0 & \text{otherwise} \end{cases} $$

$$ \text{descriptor} = \sum_{i=1}^{n} 2^{i-1} \tau(\mathbf{p}_i, \mathbf{q}_i) $$

**像素对选择策略**：
- 均匀分布（原始BRIEF）
- 高斯分布（Gaussian BRIEF）
- 学习优化对（ORB中的rBRIEF）

### 3.2 ORB描述子（rBRIEF）

ORB将BRIEF改进为旋转不变的steered BRIEF，再通过学习得到rBRIEF。

**Steered BRIEF**：根据特征点方向 $\theta$ 旋转像素对坐标：

$$ \mathbf{S}_\theta = \mathbf{R}_\theta \begin{bmatrix} x_1 & x_2 & \cdots & x_n \\ y_1 & y_2 & \cdots & y_n \end{bmatrix} $$

**rBRIEF**：通过贪婪搜索从所有可能的像素对中选择方差最大、相关性最小的像素对。

### 3.3 BRISK描述子

BRISK使用固定的采样模式，在特征点周围的同心圆上采样。使用短距离点对计算描述子，长距离点对估计方向。

### 3.4 FREAK描述子

FREAK受视网膜生物结构启发，采样点密度从中心向边缘递减（类似于视网膜的神经节细胞分布）。

## 4. 学习型描述子

### 4.1 HardNet

HardNet使用三元组损失函数训练CNN描述子，强调困难负样本的挖掘。

**损失函数**：

$$ L = \max(0, 1 + d(a, p) - d(a, n_{\text{hard}})) $$

### 4.2 SuperPoint描述子

SuperPoint使用自监督学习框架，在合成数据上训练MagicPoint，然后通过Homography Adaptation适应真实图像。描述子维度为256。

### 4.3 NetVLAD

NetVLAD是VLAD（Vector of Locally Aggregated Descriptors）的可微版本，广泛用于图像检索和地点识别。它将CNN特征聚合成一个紧凑的全局描述子。

## 5. 描述子匹配距离度量

| 描述子类型 | 距离度量 | 计算复杂度 |
|-----------|----------|-----------|
| 浮点型(SIFT, SURF) | L2 欧氏距离 | O(N²D) |
| 二进制(ORB, BRIEF) | Hamming 距离（XOR + popcount） | O(N²D/64) |
| 学习型(SuperPoint) | L2 或余弦距离 | O(N²D) |

## 6. 描述子性能对比

| 描述子 | 维度 | 类型 | 匹配精度 | 速度 | 内存占用 |
|--------|------|------|----------|------|----------|
| SIFT | 128 | 浮点 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| SURF | 64 | 浮点 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| ORB | 256 | 二进制 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| BRISK | 64 | 二进制 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| HardNet | 128 | 浮点(学习) | ⭐⭐⭐⭐⭐ | ⭐⭐(GPU) | ⭐⭐ |
| SuperPoint | 256 | 浮点(学习) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐(GPU) | ⭐⭐ |

## 7. 参考文献

1. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *IJCV*, 60(2), 91-110.
2. Bay, H., Tuytelaars, T., & Van Gool, L. (2006). SURF: Speeded up robust features. *ECCV*.
3. Calonder, M., et al. (2010). BRIEF: Binary robust independent elementary features. *ECCV*.
4. Rublee, E., et al. (2011). ORB: An efficient alternative to SIFT or SURF. *ICCV*.
5. Mishchuk, A., et al. (2017). Working hard to know your neighbor's margins: Local descriptor learning loss. *NeurIPS*.
