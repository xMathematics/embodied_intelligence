
# 1.5 张量基础

## 目录

- [1. 张量的定义](#1-张量的定义)
- [2. 张量运算](#2-张量运算)
- [3. 张量分解](#3-张量分解)
- [4. 在深度学习中的应用](#4-在深度学习中的应用)
- [5. 相关论文](#5-相关论文)
- [6. 实践练习](#6-实践练习)

---

## 1. 张量的定义

### 1.1 直观理解

**张量**是多维数组的推广：
- 0阶张量（标量）：单个数值
- 1阶张量（向量）：一维数组
- 2阶张量（矩阵）：二维数组
- 3阶及以上张量：多维数组

### 1.2 数学定义

在坐标变换下，张量的分量按照特定规则变换。

**协变张量**：分量随基向量协变变换
**逆变张量**：分量随基向量逆变变换
**混合张量**：同时包含协变和逆变指标

### 1.3 张量的表示

**指标表示法**：

$$T^{i_1 i_2 \cdots i_p}_{j_1 j_2 \cdots j_q}$$

表示 $(p, q)$ 型张量，$p$ 个逆变指标，$q$ 个协变指标。

**示例**：
- 向量：$v^i$（逆变）或 $v_i$（协变）
- 矩阵：$A^i_j$（(1,1)型）或 $A_{ij}$（(0,2)型）
- 三阶张量：$T_{ijk}$

---

## 2. 张量运算

### 2.1 张量加法

同型张量可以相加：

$$(\mathbf{A} + \mathbf{B})_{ijk} = A_{ijk} + B_{ijk}$$

### 2.2 张量积（外积）

$$C_{i_1 \cdots i_p j_1 \cdots j_q} = A_{i_1 \cdots i_p} \cdot B_{j_1 \cdots j_q}$$

**示例**：两个向量的外积产生矩阵

$$C_{ij} = a_i \cdot b_j$$

### 2.3 缩并（Contraction）

对指定指标求和：

$$C_{jk} = \sum_i A_{ijk}$$

**示例**：矩阵的迹是缩并运算

$$\text{tr}(\mathbf{A}) = \sum_i A^i_i$$

### 2.4 爱因斯坦求和约定

重复的指标表示求和：

$$C_{ij} = A_{ik} B_{kj} \equiv \sum_k A_{ik} B_{kj}$$

**规则**：
- 同一项中同一指标出现两次表示求和（哑指标）
- 只出现一次的指标是自由指标

### 2.5 常用张量运算

#### 矩阵乘法

$$C_{ij} = A_{ik} B_{kj}$$

#### 批量矩阵乘法

$$C_{bij} = A_{bik} B_{bkj}$$

#### 转置

$$A^T_{ij} = A_{ji}$$

---

## 3. 张量分解

### 3.1 CP分解（CANDECOMP/PARAFAC）

将张量分解为秩-1张量的和：

$$\mathcal{X} \approx \sum_{r=1}^{R} \lambda_r \mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r$$

其中 $\circ$ 表示外积。

**元素形式**：

$$x_{ijk} \approx \sum_{r=1}^{R} \lambda_r a_{ir} b_{jr} c_{kr}$$

### 3.2 Tucker分解

$$\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{A} \times_2 \mathbf{B} \times_3 \mathbf{C}$$

其中：
- $\mathcal{G}$：核心张量
- $\mathbf{A}, \mathbf{B}, \mathbf{C}$：因子矩阵
- $\times_n$：n-模乘积

### 3.3 张量网络

**矩阵乘积态（MPS）/ 张量链（TT）**：

$$\mathcal{X}_{i_1 i_2 \cdots i_d} = \sum_{\alpha_1, \ldots, \alpha_{d-1}} G^{(1)}_{i_1 \alpha_1} G^{(2)}_{\alpha_1 i_2 \alpha_2} \cdots G^{(d)}_{\alpha_{d-1} i_d}$$

**优点**：
- 参数数量从 $O(n^d)$ 减少到 $O(dnr^2)$
- 适合高维数据

---

## 4. 在深度学习中的应用

### 4.1 神经网络中的张量

**数据表示**：
- 图像：$\text{(batch, height, width, channels)}$
- 视频：$\text{(batch, time, height, width, channels)}$
- 点云：$\text{(batch, num_points, features)}$

**权重张量**：
- 卷积核：$\text{(out_channels, in_channels, kernel_h, kernel_w)}$
- 全连接：$\text{(out_features, in_features)}$

### 4.2 卷积运算

$$Y_{b,o,h,w} = \sum_{i,k_h,k_w} X_{b,i,h+k_h,w+k_w} \cdot W_{o,i,k_h,k_w}$$

### 4.3 注意力机制

**自注意力**：

$$\text{Attention}(Q, K, V)_{b,t,d} = \sum_{t'} \text{softmax}\left(\frac{Q_{b,t,:} \cdot K_{b,t',:}}{\sqrt{d_k}}\right) V_{b,t',d}$$

### 4.4 张量网络压缩

**神经网络压缩**：
- 使用TT分解压缩全连接层
- 使用CP分解压缩卷积核

**优势**：
- 减少参数量
- 加速推理
- 保持性能

---

## 5. 相关论文

### 张量分解

1. **"Tensor Decompositions and Applications"** - Kolda & Bader (2009)
   - 张量分解综述

2. **"A New Introduction to Multiple Time Series Analysis"** - Lütkepohl
   - 多变量时间序列分析

### 深度学习应用

3. **"Tensorizing Neural Networks"** - Novikov et al. (2015)
   - 使用TT分解压缩神经网络

4. **"Compressing Deep Convolutional Networks using Vector Quantization"** - Gong et al.
   - 神经网络压缩

### 张量网络

5. **"Hand-waving and Interpretive Dance: An Introductory Course on Tensor Networks"** - Bridgeman & Chubb
   - 张量网络入门

6. **"Tensor Networks for Dimensionality Reduction and Large-scale Optimization"** - Cichocki et al.
   - 张量网络在优化中的应用

---

## 6. 实践练习

### 练习1：张量基础操作

```python
import numpy as np

# 创建张量
scalar = np.array(5)  # 0阶
vector = np.array([1, 2, 3])  # 1阶
matrix = np.array([[1, 2], [3, 4], [5, 6]])  # 2阶
tensor_3d = np.random.rand(3, 4, 5)  # 3阶
tensor_4d = np.random.rand(2, 3, 4, 5)  # 4阶

print(f"标量形状: {scalar.shape}")
print(f"向量形状: {vector.shape}")
print(f"矩阵形状: {matrix.shape}")
print(f"3D张量形状: {tensor_3d.shape}")
print(f"4D张量形状: {tensor_4d.shape}")

# 张量索引
print(f"\n3D张量[0, 1, 2] = {tensor_3d[0, 1, 2]}")
print(f"3D张量[0, :, :] 形状: {tensor_3d[0, :, :].shape}")
```

### 练习2：爱因斯坦求和

```python
import numpy as np

# 矩阵乘法: C_ij = A_ik * B_kj
A = np.random.rand(3, 4)
B = np.random.rand(4, 5)
C = np.einsum('ik,kj->ij', A, B)
C_verify = A @ B
print(f"矩阵乘法结果一致: {np.allclose(C, C_verify)}")

# 批量矩阵乘法: C_bij = A_bik * B_bkj
A_batch = np.random.rand(10, 3, 4)
B_batch = np.random.rand(10, 4, 5)
C_batch = np.einsum('bik,bkj->bij', A_batch, B_batch)
C_batch_verify = np.matmul(A_batch, B_batch)
print(f"批量矩阵乘法结果一致: {np.allclose(C_batch, C_batch_verify)}")

# 迹: tr(A) = A_ii
tr_A = np.einsum('ii->', A[:3, :3])
tr_A_verify = np.trace(A[:3, :3])
print(f"迹计算结果一致: {np.allclose(tr_A, tr_A_verify)}")

# 外积: C_ij = a_i * b_j
a = np.array([1, 2, 3])
b = np.array([4, 5])
C_outer = np.einsum('i,j->ij', a, b)
C_outer_verify = np.outer(a, b)
print(f"外积结果一致: {np.allclose(C_outer, C_outer_verify)}")

# 张量缩并: C_jk = A_ijk * B_i
A_tensor = np.random.rand(3, 4, 5)
B_vector = np.random.rand(3)
C_contract = np.einsum('ijk,i->jk', A_tensor, B_vector)
print(f"缩并结果形状: {C_contract.shape}")
```

### 练习3：图像张量操作

```python
import numpy as np

# 模拟图像数据 (batch, height, width, channels)
images = np.random.rand(32, 64, 64, 3)  # 32张64x64的RGB图像

# 1. 提取所有图像的红色通道
red_channel = images[:, :, :, 0]
print(f"红色通道形状: {red_channel.shape}")

# 2. 提取第一张图像
first_image = images[0]
print(f"第一张图像形状: {first_image.shape}")

# 3. 提取图像中心区域
center_crop = images[:, 16:48, 16:48, :]
print(f"中心裁剪形状: {center_crop.shape}")

# 4. 计算每个通道的均值
channel_mean = np.mean(images, axis=(0, 1, 2))
print(f"各通道均值: {channel_mean}")

# 5. 转置为 (batch, channels, height, width) - PyTorch格式
images_torch = np.transpose(images, (0, 3, 1, 2))
print(f"PyTorch格式形状: {images_torch.shape}")
```

### 练习4：简单的张量分解

```python
import numpy as np
from scipy.linalg import svd

def simple_cp_decomposition(tensor, rank):
    """
    简化的CP分解（使用交替最小二乘法）
    """
    # 获取张量形状
    I, J, K = tensor.shape
    
    # 初始化因子矩阵
    A = np.random.rand(I, rank)
    B = np.random.rand(J, rank)
    C = np.random.rand(K, rank)
    
    # 迭代优化
    for iteration in range(100):
        # 更新A
        tensor_unfold = tensor.reshape(I, J*K)
        B_khatri = np.kron(B, np.ones((K, rank))) * np.kron(np.ones((J, rank)), C)
        A = tensor_unfold @ B_khatri @ np.linalg.inv(B_khatri.T @ B_khatri + 0.01*np.eye(rank))
        
        # 更新B
        tensor_unfold = tensor.transpose(1, 0, 2).reshape(J, I*K)
        A_khatri = np.kron(A, np.ones((K, rank))) * np.kron(np.ones((I, rank)), C)
        B = tensor_unfold @ A_khatri @ np.linalg.inv(A_khatri.T @ A_khatri + 0.01*np.eye(rank))
        
        # 更新C
        tensor_unfold = tensor.transpose(2, 0, 1).reshape(K, I*J)
        A_khatri = np.kron(A, np.ones((J, rank))) * np.kron(np.ones((I, rank)), B)
        C = tensor_unfold @ A_khatri @ np.linalg.inv(A_khatri.T @ A_khatri + 0.01*np.eye(rank))
    
    return A, B, C

# 测试
np.random.seed(42)
tensor = np.random.rand(10, 10, 10)
rank = 3

A, B, C = simple_cp_decomposition(tensor, rank)

# 重构张量
tensor_reconstructed = np.zeros_like(tensor)
for r in range(rank):
    tensor_reconstructed += np.outer(A[:, r], np.outer(B[:, r], C[:, r]).flatten()).reshape(10, 10, 10)

# 计算重构误差
error = np.linalg.norm(tensor - tensor_reconstructed) / np.linalg.norm(tensor)
print(f"相对重构误差: {error:.4f}")
```

---

**下一步**：[微积分 - 一元微积分](../calculus/01-single-variable.md)
