# 7.5 表面重建

## 1. 概述

表面重建是从点云、深度图或体素数据生成连续表面的过程。高质量的表面重建是三维重建的最后关键步骤。

## 2. 泊松表面重建

### 2.1 核心思想

泊松重建（Kazhdan et al., 2006）将表面重建转化为求解泊松方程问题。

**基本思路**：点云的法向量定义了一个向量场 $\vec{V}$，表面隐式函数 $\chi$ 满足：

$$ \nabla \chi = \vec{V} $$

求解泊松方程：

$$ \Delta \chi = \nabla \cdot \vec{V} $$

### 2.2 算法步骤

1. **构建八叉树**：自适应细分空间
2. **定义向量场**：将点云法向量插值到八叉树节点
3. **求解泊松方程**：使用多重网格法求解
4. **提取等值面**：使用Marching Cubes提取 $\chi = 0$ 的表面

### 2.3 带约束的泊松重建

**Screened Poisson Reconstruction**（Kazhdan & Hoppe, 2013）：
增加位置约束，使表面更贴合输入点云：

$$ \min_{\chi} \|\nabla \chi - \vec{V}\|^2 + \alpha \sum_p \|\chi(p)\|^2 $$

## 3. Marching Cubes

### 3.1 基本原理

Marching Cubes（Lorensen & Cline, 1987）从体素数据中提取等值面：

1. 遍历每个体素
2. 根据8个顶点的符号确定体素配置（256种，对称简化为15种）
3. 在体素边界上插值等值点
4. 连接等值点生成三角面片

### 3.2 Marching Cubes变体

| 变体 | 特点 |
|------|------|
| Marching Tetrahedra | 更简单，无歧义 |
| Dual Marching Cubes | 更好的拓扑 |
| Cubical Marching Squares | 自适应分辨率 |
| GPU Marching Cubes | 实时性能 |

## 4. 网格简化与优化

### 4.1 网格简化

- **边塌缩（Edge Collapse）**：通过QEM（Quadric Error Metric）
- **顶点聚类**：合并相邻顶点
- **渐进网格（Progressive Meshes）**

### 4.2 网格优化

- **拉普拉斯平滑**：$v_i' = v_i + \lambda \sum_{j \in N(i)} (v_j - v_i)$
- **各向异性扩散**：保持边缘的去噪
- **惠特克平滑（Taubin平滑）**：防止收缩

## 5. 纹理映射

### 5.1 纹理生成

从多幅图像中为三角网格生成纹理：

1. **可见性分析**：确定每个三角面的最佳纹理来源
2. **纹理展开**：将三角网格UV展开到纹理空间
3. **纹理融合**：融合多视点的纹理颜色
4. **颜色校正**：消除不同图像间的颜色差异

### 5.2 纹理拼接

**方法**：
- **顶点颜色**：直接为顶点赋值颜色
- **纹理包（Texture Atlas）**：将网格分割为连续区域
- **无缝纹理**：使用梯度域融合

## 6. 参考文献

1. Kazhdan, M., Bolitho, M., & Hoppe, H. (2006). Poisson surface reconstruction. *SIGGRAPH*.
2. Kazhdan, M., & Hoppe, H. (2013). Screened Poisson surface reconstruction. *ACM Transactions on Graphics*, 32(3), 1-13.
3. Lorensen, W. E., & Cline, H. E. (1987). Marching cubes: A high resolution 3D surface construction algorithm. *ACM SIGGRAPH*.
4. Waechter, M., Moehrle, N., & Goesele, M. (2014). Let there be color! Large-scale texturing of 3D reconstructions. *ECCV*.
