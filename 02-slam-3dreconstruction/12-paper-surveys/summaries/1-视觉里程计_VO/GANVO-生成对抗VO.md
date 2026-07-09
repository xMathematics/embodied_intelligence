# GANVO-生成对抗VO

## 基本信息
- **英文标题**: GANVO: Unsupervised Deep Monocular Visual Odometry and Depth Estimation with Generative Adversarial Networks
- **作者**: Yasin Almalioglu, Muhamad Risqi U. Saputra, Pedro P. B. de Gusmão, Andrew Markham, Niki Trigoni
- **发表会议/期刊**: ICRA 2019
- **关键词**: 生成对抗网络,无监督学习,深度估计,位姿回归

## 研究背景（前提）
有监督深度VO需大量标注数据。现有无监督方法用编码器-解码器生成深度图，倾向于产生过于平滑的图像。

## 问题提出（由什么问题引出）
如何利用GAN的无监督学习能力，在没有深度真值时生成更清晰准确的深度图，同时联合估计6-DoF相机运动？

## 要解决的问题
在无监督框架下联合估计单目相机6-DoF位姿和场景深度，利用GAN生成更高质量的深度估计。

---

## 采用的方法
包含深度生成器(GAN-based:编码器E+生成器G+判别器D)和位姿回归器(CNN-RNN)。通过视图重建和对抗训练实现无监督学习。

## 理论依据
利用GAN对抗训练生成更真实深度图；视图重建通过估计深度和位姿从源视图合成目标视图，光度一致性作为自监督信号。

## 核心公式推导
- **视图重建损失**: $L_g = \sum_s \sum_p |I_t(p) - \hat{I}_s(p)|$
  - 目标视图与合成视图的光度差异

- **投影变换**: $p_s \sim K \hat{T}_{t\to s} \hat{D}_t(p_t) K^{-1} p_t$
  - 像素从目标投影到源视图

- **GAN损失**: $L_d = \min_G\max_D [\log D(I) + \log(1-D(G(z)))]$
  - 生成器与判别器对抗优化

- **总损失**: $L_{final} = L_g + \beta L_d$
  - 重建损失与对抗损失加权和



---

## 实验结果
在KITTI和Cityscapes上优于传统和无监督深度VO方法。GAN生成了更清晰的深度图。

## 尚未解决的问题（后续方向）
训练计算量大；GAN训练不稳定；极端光照和快速运动下性能下降。

---

## 原始摘要
proaches have been extensively employed in visual odometry
(VO) applications, which is not feasible in environments where
labelled data is not abundant. On the other hand, unsupervised
deep learning approaches for localization and mapping in
unknown environments from unlabelled data have received
comparatively less attention in VO research. In this study,
we propose a generative unsupervised learning framework that
predicts 6-DoF pose camera motion and monocular depth map
of the scene from unlabelled RGB image sequences, using deep
convolutional Generative Adversarial Networks (GANs). We
create a supervisory signal by warping view sequences and
assigning the re-projection minimization to the objective loss
function that is adopted in multi-view pose estimation and
single-view depth generation network. Detailed quantitative
and qualitative evaluations of the proposed framework on
the KITTI [1] and Cityscapes [2] datasets show that the
proposed method outperforms both existing traditional and
unsupervised deep VO methods providing better results for
both pose estimation and depth recovery.
I. INTRODUCTION
Visual odometry (VO) and depth recovery are essential
modules of simultaneous localization and mapping (SLAM)
applications. In the last few decades, VO systems have
attracted a substantial amount of attention, enabling robust
localization and accurate depth map reconstruction. Monoc-
ular VO is confronted with numerous challenges such as
large scale drift, the need for hand-cr
