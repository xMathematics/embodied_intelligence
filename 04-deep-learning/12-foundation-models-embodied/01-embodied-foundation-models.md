# 12.1 具身基础模型

## 1. 为什么需要具身基础模型

**问题**：传统机器人策略针对单一任务设计，无法泛化。具身基础模型在**大规模多任务数据**上预训练，可以快速适配新任务。

## 2. 视觉基础模型

### 2.1 DINOv2

Meta自监督视觉模型，无需微调即为SOTA特征提取器。

**在具身中的应用**：RT-2等VLA模型的视觉骨干。

### 2.2 SAM（Segment Anything）

通用的分割基础模型，零样本分割一切。

**在具身中的应用**：
- 机器人抓取：SAM分割目标物体
- 场景理解：SAM提供精细分割
- 操作约束：分割物体边界用于操作规划

## 3. VLA模型（视觉-语言-动作）

### 3.1 RT-2（Robotic Transformer 2）

**论文**：Brohan et al., 2023 — Google DeepMind

**核心**：在PaLM-E中增加机器人动作token。

**能力**：
- 互联网知识迁移到机器人控制
- 语义操作："捡起已灭绝的动物"→捡起恐龙玩具
- 符号推理："将香蕉放在苹果上面"的叠加顺序推理

### 3.2 Octo

开源的通用机器人策略：800个机器人数据集预训练。

### 3.3 π0（Physical Intelligence, 2024）

通用操作基础模型，使用流匹配生成动作。

## 4. 具身基础模型对比

| 模型 | 类型 | 参数量 | 数据量 | 开源 |
|------|------|--------|--------|------|
| RT-2 | VLA | 55B | 130K演示 | ❌ |
| Octo | VLA | 1B | 800数据集 | ✅ |
| OpenVLA | VLA | 7B | 60K演示 | ✅ |
| π0 | 操作基座 | - | 多源 | ❌ |
| GATO | 通用Agent | 1.2B | 604任务 | ❌ |

## 5. 参考文献

1. Brohan, A., et al. (2023). RT-2: Vision-language-action models transfer web knowledge to robotic control. *arXiv*.
2. Team, O. M. (2024). Octo: An open-source generalist robot policy. *arXiv*.
3. Black, K., et al. (2024). OpenVLA: An open-source vision-language-action model. *arXiv*.
4. Kirillov, A., et al. (2023). Segment anything. *ICCV*.
5. Oquab, M., et al. (2023). DINOv2: Learning robust visual features without supervision. *arXiv*.
6. Reed, S., et al. (2022). A generalist agent (GATO). *arXiv*.
