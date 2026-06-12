# 9.1 模仿学习与行为克隆

## 1. 为什么需要模仿学习

**问题**：强化学习需要大量试错，样本效率低。人类可以快速通过演示学习——模仿学习就是让机器人"看"人类演示学会任务。

## 2. 行为克隆（BC）

**核心**：监督学习从状态到动作的映射。

$$\pi_\theta(a|s) = \arg\min_\theta \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{演示}}}[\mathcal{L}(\pi_\theta(s), a)]$$

**局限**：分布偏移——机器人遇到未见过状态时性能大幅下降。

## 3. DAgger

**改进**：在机器人执行过程中，请求专家纠正动作，逐步扩展到新状态。

## 4. 逆强化学习（IRL）

从演示中学习奖励函数，然后用RL学习策略——比BC更鲁棒。

## 5. 在具身智能中的应用

- **操作技能学习**：从人类演示学习抓取、放置等技能
- **遥控操作学习**：从遥操作数据学习策略
- **视觉模仿**：从视频（非动作标签）学习任务

## 6. 参考文献

1. Pomerleau, D. A. (1989). ALVINN: An autonomous land vehicle in a neural network. *NeurIPS*.
2. Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. *AISTATS*.
3. Zare, M., et al. (2024). Diffusion policy: Visuomotor policy learning via action diffusion. *RSS*.
