# 5.6 3D生成模型

## 1. 为什么需要3D生成

### 1.1 问题

手工制作3D模型成本极高，且数量有限。3D生成可以从文本或图像自动创建3D资产，对具身智能仿真至关重要。

### 1.2 核心挑战

| 挑战 | 描述 |
|------|------|
| 几何一致性 | 从多视角投影一致 |
| 纹理质量 | 高分辨率纹理 |
| 生成速度 | 快速资产创建 |
| 可交互性 | 物理属性正确 |

## 2. 主要方法

**DreamFusion**（Poole et al., 2022）：使用SDS损失从预训练2D扩散蒸馏3D知识。

**Zero123**（Liu et al., 2023）：新视角条件扩散。

**TripoSR**：快速前馈3D重建。

## 3. 在具身智能中的应用

- **仿真场景创建**：快速生成机器人训练场景
- **物体中心表示**：可交互的3D物体表示
- **Sim-to-Real**：生成多样化环境提高泛化

## 4. 参考文献

1. Poole, B., et al. (2022). DreamFusion: Text-to-3D using 2D diffusion. *ICLR*.
2. Liu, R., et al. (2023). Zero-1-to-3: Zero-shot one image to 3D object. *ICCV*.
3. Tochilkin, D., et al. (2024). TripoSR: Fast 3D object reconstruction from a single image. *arXiv*.
