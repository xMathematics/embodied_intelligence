# 9.2 世界模型与决策Transformer

## 1. 为什么需要世界模型

**人脑**可以在行动前"想象"后果——世界模型让机器人也拥有这种能力。

## 2. Dreamer系列

**Dreamer**（Hafner et al., 2021）：
1. **RSSM**（循环状态空间模型）学习世界动力学
2. 在潜在空间"想象"轨迹
3. 优化动作以最大化奖励

**DreamerV2**使用分类潜在变量，**DreamerV3**使用固定超参数。

## 3. 决策Transformer（DT）

**论文**：Chen et al., 2021 — NeurIPS

**核心创新**：将强化学习重新定义为**序列建模问题**。

$$\text{轨迹: } [R_1, s_1, a_1, R_2, s_2, a_2, \ldots]$$

**为什么有效**：因果Transformer可以捕获轨迹中的长程依赖。

**局限**：DT不能很好地处理离策略数据和次优演示。

## 4. 在具身智能中的应用

- **DayDreamer**：Dreamer在真实机器人上的部署
- **机械臂操作**：学习长期操作任务
- **导航规划**：世界模型预测导航后果
- **技能链**：DT将技能组成长程任务

## 5. 参考文献

1. Hafner, D., et al. (2021). Dream to control: Learning behaviors by latent imagination. *ICLR*.
2. Hafner, D., et al. (2023). Mastering diverse domains through world models. *arXiv*.
3. Chen, L., et al. (2021). Decision transformer: Reinforcement learning via sequence modeling. *NeurIPS*.
