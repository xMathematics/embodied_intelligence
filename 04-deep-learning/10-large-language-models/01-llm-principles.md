# 10.1 大语言模型原理

## 1. 为什么需要大语言模型

**问题**：小模型（<1B）能力有限，无法涌现推理、上下文学习等能力。

**scaling law**（Kaplan 2020 / Chinchilla 2022）：模型性能随参数量和训练数据量呈幂律增长。

## 2. 训练流程

### 2.1 预训练（Pretraining）

在海量文本上做next-token prediction。

**数据**：CommonCrawl, 维基百科, 书籍, GitHub代码等。

### 2.2 指令微调（SFT）

让模型学会遵循人类指令格式。

### 2.3 RLHF

**3阶段**：
1. SFT微调
2. 训练奖励模型 $r_\phi$
3. PPO优化策略

$$\max_{\pi_\theta} \mathbb{E}_{x, y \sim \pi_\theta}[r_\phi(x, y)] - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

**DPO**（Rafailov 2023）：不需要显式奖励模型，直接优化偏好。

## 3. 推理能力

| 能力 | 方法 | 原理 |
|------|------|------|
| In-Context Learning | Few-shot prompting | 示例模式识别 |
| Chain-of-Thought | "Let's think step by step" | 分解推理步骤 |
| Self-Consistency | 多次采样+投票 | 一致性提升 |
| Tree-of-Thoughts | 树搜索 | 探索多种推理路径 |

## 4. 局限

| 问题 | 描述 | 解决方案 |
|------|------|----------|
| 幻觉 | 生成不真实内容 | RAG, 验证 |
| 事实错误 | 知识过时或不准确 | 检索增强 |
| 推理不可靠 | 复杂推理出错 | CoT, Tool use |
| 偏见 | 训练数据中的偏见 | 对齐训练 |

## 5. 在具身智能中的应用

- **SayCan**：LLM进行任务规划
- **Code as Policies**：LLM生成机器人控制代码
- **CLIPort**：语言条件操作策略
- **PaLM-E**：具身多模态LLM

## 6. 参考文献

1. Brown, T. B., et al. (2020). Language models are few-shot learners. *NeurIPS*.
2. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*.
3. Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*.
4. Rafailov, R., et al. (2023). Direct preference optimization. *NeurIPS*.
5. Ahn, M., et al. (2022). Do as I can, not as I say: Grounding language in robotic affordances. *CoRL*.
