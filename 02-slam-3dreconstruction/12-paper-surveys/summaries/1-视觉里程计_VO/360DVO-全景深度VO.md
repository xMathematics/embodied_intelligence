# 360DVO-全景深度VO

## 基本信息
- **英文标题**: 360DVO: Deep Visual Odometry for Monocular 360-Degree Camera
- **作者**: Xiaopeng Guo, Yinzhe Xu, Huajian Huang, Sai-Kit Yeung
- **发表会议/期刊**: IEEE RA-L 2026
- **关键词**: 全景相机,畸变感知,可微分BA

## 研究背景（前提）
全景相机克服了透视VO的视野限制。但现有方法依赖手工特征或光度目标，在剧烈运动和光照变化中缺乏鲁棒性。

## 问题提出（由什么问题引出）
首个基于深度学习的全景VO框架。如何设计针对360度图像畸变感知的特征提取器？

## 要解决的问题
如何针对360度全景相机的畸变特性设计深度学习VO框架？

---

## 采用的方法
提出360DVO：1)畸变感知球面特征提取器(DAS-Feat)自适应学习抗畸变特征；2)全景可微分BA(ODBA)模块进行位姿优化；3)贡献真实世界OVO基准。

## 理论依据
全景图像的等距柱状投影导致不同纬度区域的畸变程度不同。球面特征提取需考虑畸变模型。可微分BA将传统优化嵌入网络。

## 核心公式推导
- **球面特征**: $f_{sphere} = f_{DAS}(I_{360})$
  - 畸变感知的球面特征提取

- **可微分BA**: $\min \sum ||\pi_{360}(T, X) - x||^2$
  - 全景相机的可微分光束法平差



---

## 实验结果
在真实和合成数据集上超越SOTA基线(360VO/OpenVSLAM)，鲁棒性提升50%，精度提升37.5%。

## 尚未解决的问题（后续方向）
全景相机分辨率受限；等距柱状投影导致极地区域过采样。

---

## 原始摘要
systems leverage 360-degree cameras to overcome field-of-view
limitations of perspective VO systems. However, existing methods,
reliant on handcrafted features or photometric objectives, often
lack robustness in challenging scenarios, such as aggressive
motion and varying illumination. To address this, we present
360DVO, the first deep learning-based OVO framework. Our
approach introduces a distortion-aware spherical feature ex-
tractor (DAS-Feat) that adaptively learns distortion-resistant
features from 360-degree images. These sparse feature patches
are then used to establish constraints for effective pose estimation
within a novel omnidirectional differentiable bundle adjustment
(ODBA) module. To facilitate evaluation in realistic settings, we
also contribute a new real-world OVO benchmark. Extensive
experiments on this benchmark and public synthetic datasets
(TartanAir V2 and 360VO) demonstrate that 360DVO surpasses
state-of-the-art baselines (including 360VO and OpenVSLAM),
improving robustness by 50% and accuracy by 37.5%. Home-
page: https://chris1004336379.github.io/360DVO-homepage
Index Terms—visual odometry, omnidirectional vision
I. INTRODUCTION
V
ISUAL odometry (VO) and simultaneous localization
and mapping (VSLAM) estimate agent’s ego-motion
from image sequences which enable various applications,
including autonomous navigation and augmented reality. 360-
degree cameras capture omnidirectional field-of-view (FoV)
information and produce full-sphere images in the 
