# 9.2 三维重建数据集

## 1. 概述

三维重建数据集为评估SfM、MVS和神经重建方法提供了标准化的测试平台。

## 2. DTU MVS数据集

**典型场景**：室内多视图重建
**网址**：https://roboimagedata.compute.dtu.dk/

| 项目 | 详情 |
|------|------|
| 场景数 | 80个（49训练，15验证，16测试） |
| 每场景图像 | 49-64幅 |
| 分辨率 | 1200×1600 |
| 光照条件 | 7种光照 |
| 真值 | 结构光扫描 |

**评估指标**：精确度（Accuracy）、完整度（Completeness）、F1分数

## 3. Middlebury MVS

**常见版本**：
- Middlebury 2006：经典MVS基准
- Middlebury 2014：高分辨率立体

**特点**：
- 使用高精度结构光扫描获取真值
- 室内场景
- 不同分辨率级别

## 4. NeRF合成数据集

**Mildenhall et al., 2020**：
- 8个物体场景（乐高、鼓、椅子等）
- 每场景100张训练图像，200张测试
- 渲染图像，分辨率800×800

## 5. 其他重建数据集

| 数据集 | 年份 | 场景 | 规模 |
|--------|------|------|------|
| ETH3D | 2017 | 室内+室外 | 25训练，25测试 |
| Tanks and Temples | 2017 | 大场景 | 6+6场景 |
| ScanNet | 2017 | 室内RGB-D | 1513扫描 |
| BlendedMVS | 2019 | 各种 | 113个场景 |
| MegaDepth | 2018 | 室外大规模 | 100K+ |

## 6. 新兴数据集

| 数据集 | 年份 | 特点 |
|--------|------|------|
| ARKitScenes | 2021 | Apple LiDAR扫描 |
| KITTI-360 | 2022 | 自主驾驶+地面重建 |
| 3D Scene Graph | 2022 | 语义+几何 |
| Replica | 2019 | 高质量室内合成 |

## 7. 参考文献

1. Jensen, R., Dahl, A., Vogiatzis, G., Tola, E., & Aanæs, H. (2014). Large scale multi-view stereopsis evaluation. *CVPR*.
2. Knapitsch, A., et al. (2017). Tanks and temples: Benchmarking large-scale scene reconstruction. *ACM Transactions on Graphics*, 36(4).
3. Dai, A., et al. (2017). ScanNet: Richly-annotated 3D reconstructions of indoor scenes. *CVPR*.
4. Schönberger, J. L., et al. (2016). Comparative evaluation of hand-crafted and learned local features. *CVPR*.
