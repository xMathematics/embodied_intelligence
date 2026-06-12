# 2.6 PnP算法

## 1. 概述

PnP（Perspective-n-Point）问题指已知 $n$ 个3D点及其在图像中的2D投影位置，求解相机在世界坐标系中的位姿（旋转 $\mathbf{R}$ 和平移 $\mathbf{t}$）。PnP是视觉SLAM中位姿估计的核心算法之一。

## 2. P3P算法

P3P使用3对3D-2D对应点求解相机位姿，是PnP问题的闭合形式解法。

**几何关系**：3个世界点 $\mathbf{A}, \mathbf{B}, \mathbf{C}$ 和相机光心 $\mathbf{O}$ 构成四面体。

**余弦定理**：

$$ \begin{aligned} d_{AB}^2 &= d_{OA}^2 + d_{OB}^2 - 2 d_{OA} d_{OB} \cos\langle \mathbf{OA}, \mathbf{OB} \rangle \\ d_{BC}^2 &= d_{OB}^2 + d_{OC}^2 - 2 d_{OB} d_{OC} \cos\langle \mathbf{OB}, \mathbf{OC} \rangle \\ d_{AC}^2 &= d_{OA}^2 + d_{OC}^2 - 2 d_{OA} d_{OC} \cos\langle \mathbf{OA}, \mathbf{OC} \rangle \end{aligned} $$

令 $x = d_{OA}, y = d_{OB}, z = d_{OC}$，上述方程可转化为4次多项式，最多4个解，需要第4个点验证。

**局限性**：
- 对噪声敏感
- 只有4个解，可能都不可靠
- 需要额外的验证点

## 3. EPnP算法

EPnP（Efficient PnP, Lepetit et al., 2009）将 $n$ 个3D点表示为4个虚拟控制点的加权和，将PnP问题转化为控制点在相机坐标系下的坐标估计。

**控制点表示**：

$$ \mathbf{p}_i^w = \sum_{j=1}^{4} \alpha_{ij} \mathbf{c}_j^w, \quad \sum_{j=1}^{4} \alpha_{ij} = 1 $$

**相机坐标下的表示**：

$$ \mathbf{p}_i^c = \sum_{j=1}^{4} \alpha_{ij} \mathbf{c}_j^c $$

**投影约束**：

$$ \begin{bmatrix} u_i \\ v_i \end{bmatrix} = \frac{1}{z_i^c} \begin{bmatrix} x_i^c \\ y_i^c \end{bmatrix} = \frac{1}{\sum_{j} \alpha_{ij} z_j^c} \begin{bmatrix} \sum_{j} \alpha_{ij} x_j^c \\ \sum_{j} \alpha_{ij} y_j^c \end{bmatrix} $$

**求解步骤**：
1. 计算控制点的齐次重心坐标 $\alpha_{ij}$
2. 建立线性方程组求解 $\mathbf{c}_j^c$
3. 从 $\mathbf{p}_i^w$ 和 $\mathbf{p}_i^c$ 恢复 $\mathbf{R}, \mathbf{t}$
4. 使用正交Procrustes问题求解旋转

**复杂度**：$O(n)$，适合大规模点集。

## 4. UPnP算法

UPnP（Universal PnP, Kneip et al., 2014）在不知道相机内参的情况下同时估计位姿和内参。

**核心思想**：将未知的焦距 $f$ 也作为优化变量引入PnP问题。

## 5. DLS算法

DLS（Direct Least-Squares, Hesch & Roumeliotis, 2011）将PnP问题转化为一个4阶多项式的求解：

$$ \mathbf{r}^* = \arg\min_{\mathbf{r} \in \mathbb{S}^3} \mathbf{r}^T \mathbf{M}(\mathbf{r}) \mathbf{r} $$

其中 $\mathbf{r}$ 是旋转的四元数表示。

## 6. MLPnP

MLPnP（Maximum Likelihood PnP, 2017）考虑每对匹配点的不确定性不同，使用最大似然估计：

$$ \mathbf{R}^*, \mathbf{t}^* = \arg\min \sum_{i=1}^{n} \|\mathbf{e}_i\|^2_{\mathbf{\Sigma}_i} $$

其中 $\mathbf{\Sigma}_i$ 是第 $i$ 个点的观测协方差矩阵。

## 7. PnP算法对比

| 算法 | 复杂度 | 精度 | 鲁棒性 | 最少点数 | 是否需内参 |
|------|--------|------|--------|---------|-----------|
| P3P | $O(1)$ | 低 | 低 | 3+1 | 是 |
| EPnP | $O(n)$ | 高 | 中 | 4 | 是 |
| DLS | $O(1)$ | 高 | 高 | 4 | 是 |
| UPnP | $O(n)$ | 中 | 中 | 5 | 否 |
| MLPnP | $O(n)$ | 高 | 高 | 4 | 是 |
| PnP+BA | $O(n^3)$ | 最高 | 最高 | 4 | 是 |

## 8. 参考文献

1. Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009). EPnP: An accurate O(n) solution to the PnP problem. *International Journal of Computer Vision*, 81(2), 155-166.
2. Kneip, L., Scaramuzza, D., & Siegwart, R. (2011). A novel parametrization of the perspective-three-point problem for a direct computation of absolute camera position and orientation. *CVPR*.
3. Hesch, J. A., & Roumeliotis, S. I. (2011). A direct least-squares (DLS) method for PnP. *ICCV*.
4. Kneip, L., Li, H., & Seo, Y. (2014). UPnP: An optimal O(n) solution to the absolute pose problem with universal applicability. *ECCV*.
