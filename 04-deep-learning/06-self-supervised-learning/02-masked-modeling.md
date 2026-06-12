# 6.2 掩码自编码器（MAE）

## 1. 为什么提出MAE

**问题**：对比学习需要精心设计的数据增强和负样本。

**解决方案**：BERT的视觉版本——掩码建模。

## 2. MAE原理

**论文**：He et al., 2022 — CVPR

**关键设计**：
1. **高掩码率75%**：只编码25%的可见patch
2. **非对称编码器-解码器**：编码器轻量（仅可见），解码器稍重
3. **重构目标**：归一化像素值

**为什么编码效率高**：编码器只需处理25%的输入。

## 3. DINOv2

**Meta 2023**：大规模自蒸馏，无需微调即为SOTA。

**方法**：
- ViT-g/14编码器
- 教师-学生自蒸馏
- 1.42亿图像预训练

**影响**：成为视觉基础模型的标准。

## 4. 在具身智能中的应用

- **MAE预训练**：从机器人视频中学习视觉表征
- **DINOv2特征**：RT-2等VLA模型使用DINOv2作为视觉编码器
- **鲁棒匹配**：自监督特征对光照和视角变化鲁棒

## 5. 参考文献

1. He, K., et al. (2022). Masked autoencoders are scalable vision learners. *CVPR*.
2. Oquab, M., et al. (2023). DINOv2: Learning robust visual features without supervision. *arXiv*.
3. Caron, M., et al. (2021). Emerging properties in self-supervised vision transformers. *ICCV*.
