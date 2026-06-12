# 7.1 多模态学习

## 1. 为什么需要多模态学习

### 1.1 问题

单一模态信息有限：
- 图像：缺乏语义描述
- 文本：缺乏视觉细节
- 语音：缺乏上下文

### 1.2 多模态的优势

- **信息互补**：不同模态提供不同信息
- **鲁棒性**：某一模态缺失时其他模态补充
- **丰富交互**：人机交互需要多模态理解

## 2. CLIP——图文对齐

**论文**：Radford et al., 2021 — ICML

**核心**：4亿图文对的对比学习。

$$\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_j\exp(\text{sim}(I_i, T_j)/\tau)} + \log\frac{\exp(\text{sim}(T_i, I_i)/\tau)}{\sum_j\exp(\text{sim}(T_i, I_j)/\tau)}\right]$$

**能力**：Zero-shot分类、跨模态检索。

**局限**：对细粒度理解不够、计数能力差。

## 3. 视觉语言模型（VLM）

| 模型 | 连接方式 | 基座LLM | 特点 |
|------|----------|----------|------|
| BLIP-2 | Q-Former | 冻结LLM | 高效 |
| LLaVA | 线性投影 | LLaMA/Vicuna | 简单有效 |
| Flamingo | Perceiver Resampler | Chinchilla | 少样本 |
| GPT-4V | 原生多模态 | GPT-4 | 最强通用 |
| Gemini | 原生多模态 | Gemini | 百万token |

## 4. 在具身智能中的应用

- **语言条件操作**：自然语言指令→机器人动作
- **视觉导航**：语言目标的视觉导航
- **场景理解**：CLIP提供开放词汇感知
- **人机交互**：多模态对话理解用户意图

## 5. 参考文献

1. Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
2. Li, J., et al. (2023). BLIP-2: Bootstrapping language-image pre-training with frozen image encoders. *ICML*.
3. Liu, H., et al. (2023). Visual instruction tuning. *NeurIPS*.
4. Alayrac, J. B., et al. (2022). Flamingo: a visual language model for few-shot learning. *NeurIPS*.
