# NeuralBundler-位姿图优化VO

## 基本信息
- **英文标题**: Pose Graph Optimization for Unsupervised Monocular Visual Odometry
- **作者**: Yang Li, Yoshitaka Ushiku, Tatsuya Harada
- **发表会议/期刊**: ICRA 2019
- **关键词**: 图优化,闭环检测,无监督学习,混合VO

## 研究背景（前提）
无监督学习VO因缺乏漂移校正技术，精度远低于几何方法。现有方法仅关注帧到帧VO估计，无法进行全局优化。

## 问题提出（由什么问题引出）
SLAM通过图优化和闭环成功减少累积漂移。能否将图优化技术应用于基于学习的VO？如何设计统一框架？

## 要解决的问题
如何将深度学习VO与经典图优化技术结合，利用闭环检测减少累积漂移，实现大规模场景下的高精度位姿估计？

---

## 采用的方法
提出NeuralBundler(无监督VO前端)+位姿图优化后端。前端在滑动窗口内生成多视图6DoF约束，提出位姿循环一致性损失；后端构建全局位姿图(含局部和闭环约束)，在SE(3)上优化。

## 理论依据
图SLAM理论：节点表示相机位姿，边表示测量约束。通过非线性最小二乘(Gauss-Newton/LM)找到与所有约束最大一致的节点配置。

## 核心公式推导
- **光度损失**: $L_{pho}= \sum [(1-\alpha)L_{l1} + \alpha L_{SSIM}]$
  - 重建与原始图像的联合损失

- **投影**: $p_j = K T_{ij} D_i K^{-1} p_i$
  - 像素从视图i投影到j

- **循环一致性**: $L_{cyc} = \sum ||T_{ij}T_{jk} - T_{ik}||$
  - 窗口内位姿图环路约束

- **图优化**: $X^* = \arg\min \sum ||e_{ij}||^2_{\Sigma_{ij}}$
  - 最小化所有约束误差



---

## 实验结果
NeuralBundler达无监督单目VO的SOTA。结合图优化和闭环后，平移精度媲美单目SLAM系统。

## 尚未解决的问题（后续方向）
闭环检测仍依赖传统词袋方法；实时性受图优化计算量影响。

---

## 原始摘要
odometry (VO) has lately drawn signiﬁcant attention for its
potential in label-free leaning ability and robustness to camera
parameters and environmental variations. However, partially
due to the lack of drift correction technique, these methods are
still by far less accurate than geometric approaches for large-
scale odometry estimation. In this paper, we propose to leverage
graph optimization and loop closure detection to overcome
limitations of unsupervised learning based monocular visual
odometry. To this end, we propose a hybrid VO system which
combines an unsupervised monocular VO called NeuralBundler
with a pose graph optimization back-end. NeuralBundler is
a neural network architecture that uses temporal and spatial
photometric loss as main supervision and generates a windowed
pose graph consists of multi-view 6DoF constraints. We propose
a novel pose cycle consistency loss to relieve the tensions in the
windowed pose graph, leading to improved performance and
robustness. In the back-end, a global pose graph is built from
local and loop 6DoF constraints estimated by NeuralBundler,
and is optimized over SE(3). Empirical evaluation on the KITTI
odometry dataset demonstrates that 1) NeuralBundler achieves
state-of-the-art performance on unsupervised monocular VO
estimation, and 2) our whole approach can achieve efﬁcient
loop closing and show favorable overall translational accuracy
compared to established monocular SLAM systems.
I. INTRODUCTION
Nowadays, monocular visual
