# 强化学习模块

## 模块概述

强化学习是机器学习的重要分支，通过智能体与环境的交互学习最优策略。本模块从RL基础开始，全面覆盖经典算法、深度强化学习、高级主题和应用领域，为具身智能提供决策能力基础。

## 学习目标

完成本模块学习后，您将能够：
- 理解强化学习的基本概念和原理
- 掌握Markov决策过程和值函数
- 理解经典RL算法（Q-Learning、策略梯度）
- 掌握深度强化学习算法（DQN、PPO、SAC）
- 了解离线RL、分层RL、多智能体RL等高级主题
- 能够使用Gym/Stable Baselines3实现RL算法
- 了解RL在机器人控制、游戏AI等领域的应用

---

## 模块结构

```
05-reinforcement-learning/
├── 01-fundamentals/             # RL基础
├── 02-classical-algorithms/     # 经典算法
├── 03-deep-rl/                  # 深度强化学习
├── 04-offline-rl/               # 离线强化学习
├── 05-hierarchical-rl/          # 分层强化学习
├── 06-multi-agent/              # 多智能体强化学习
├── 07-model-based-rl/           # 基于模型的RL
├── 08-safe-rl/                  # 安全强化学习
├── 09-applications/             # 应用领域
└── 10-paper-surveys/            # 论文详解
```

---

## 学习路径

### 第一部分：RL基础（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 1.1 | RL概述 | ⭐⭐⭐ |
| 1.2 | Markov决策过程 | ⭐⭐⭐⭐ |
| 1.3 | 值函数与Bellman方程 | ⭐⭐⭐⭐ |
| 1.4 | 策略评估与改进 | ⭐⭐⭐⭐ |
| 1.5 | 探索与利用 | ⭐⭐⭐ |

**核心内容**：
- 强化学习基本概念（智能体、环境、奖励、回报）
- MDP定义、状态转移、策略、价值函数
- Bellman方程、最优值函数、最优策略
- 策略迭代、值迭代算法
- ε-greedy、UCB、Thompson采样

---

### 第二部分：经典算法（2周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 2.1 | 基于值的方法 | ⭐⭐⭐⭐ |
| 2.2 | 时序差分学习 | ⭐⭐⭐⭐ |
| 2.3 | Q-Learning | ⭐⭐⭐⭐ |
| 2.4 | SARSA | ⭐⭐⭐⭐ |
| 2.5 | 策略梯度方法 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- DP方法、蒙特卡洛方法
- TD(0)、TD(λ)、资格迹
- Q-Learning、Watkins-Q
- SARSA、Expected SARSA
- REINFORCE、策略梯度定理

---

### 第三部分：深度强化学习（3周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 3.1 | DQN及其变体 | ⭐⭐⭐⭐ |
| 3.2 | 策略梯度方法 | ⭐⭐⭐⭐⭐ |
| 3.3 | Actor-Critic方法 | ⭐⭐⭐⭐⭐ |
| 3.4 | PPO算法 | ⭐⭐⭐⭐⭐ |
| 3.5 | 连续控制算法 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- DQN、Double DQN、Dueling DQN、PER
- TRPO、PPO、GAE
- A2C、A3C、DDPG、TD3、SAC
- 连续动作空间处理
- 深度Q网络训练技巧

---

### 第四部分：离线强化学习（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 4.1 | 离线RL概述 | ⭐⭐⭐⭐ |
| 4.2 | 行为克隆 | ⭐⭐⭐⭐ |
| 4.3 | 离线策略评估 | ⭐⭐⭐⭐⭐ |
| 4.4 | 离线策略学习 | ⭐⭐⭐⭐⭐ |
| 4.5 | 决策Transformer | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 离线RL挑战（分布偏移）
- BC、DAgger、IL
- OPE方法、重要性采样
- CQL、BCQ、TD3+BC
- Decision Transformer、Trajectory Transformer

---

### 第五部分：分层强化学习（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 5.1 | 分层RL概述 | ⭐⭐⭐⭐ |
| 5.2 | 选项框架 | ⭐⭐⭐⭐⭐ |
| 5.3 | 层次化策略 | ⭐⭐⭐⭐⭐ |
| 5.4 | 元RL与迁移 | ⭐⭐⭐⭐⭐ |
| 5.5 | 课程学习 | ⭐⭐⭐⭐ |

**核心内容**：
- 时间抽象、选项学习
- MAXQ、HRL层次结构
- FeUdal Networks、HIRO
- 元强化学习、MAML
- 课程学习、自动课程

---

### 第六部分：多智能体强化学习（2周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 6.1 | MARL概述 | ⭐⭐⭐⭐ |
| 6.2 | 合作式MARL | ⭐⭐⭐⭐⭐ |
| 6.3 | 竞争式MARL | ⭐⭐⭐⭐⭐ |
| 6.4 | 混合MARL | ⭐⭐⭐⭐⭐ |
| 6.5 | 多智能体通信 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 多智能体MDP、Dec-POMDP
- 合作学习、团队奖励
- 零和博弈、纳什均衡
- 混合动机、社交困境
- 通信机制、注意力机制

---

### 第七部分：基于模型的RL（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 7.1 | 模型学习 | ⭐⭐⭐⭐ |
| 7.2 | 世界模型 | ⭐⭐⭐⭐⭐ |
| 7.3 | 模型预测控制 | ⭐⭐⭐⭐⭐ |
| 7.4 | 想象增强学习 | ⭐⭐⭐⭐⭐ |
| 7.5 | 模型不确定性 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 动力学模型学习
- World Models、Dreamer
- MPC、PILCO、PETS
- 想象轨迹优化
- 模型偏差、不确定性估计

---

### 第八部分：安全强化学习（1周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 8.1 | 安全RL概述 | ⭐⭐⭐⭐ |
| 8.2 | 约束优化 | ⭐⭐⭐⭐⭐ |
| 8.3 | 风险敏感RL | ⭐⭐⭐⭐⭐ |
| 8.4 | 鲁棒RL | ⭐⭐⭐⭐⭐ |
| 8.5 | 可验证RL | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 安全约束、代价函数
- CPO、TRPO约束版
- CVaR、风险度量
- 对抗训练、分布鲁棒性
- 形式验证、证书学习

---

### 第九部分：应用领域（2周）

| 章节 | 内容 | 难度 |
|------|------|------|
| 9.1 | 机器人控制 | ⭐⭐⭐⭐⭐ |
| 9.2 | 游戏AI | ⭐⭐⭐⭐ |
| 9.3 | 推荐系统 | ⭐⭐⭐⭐ |
| 9.4 | 金融交易 | ⭐⭐⭐⭐⭐ |
| 9.5 | 具身智能 | ⭐⭐⭐⭐⭐ |

**核心内容**：
- 机械臂控制、导航、操作
- Atari、围棋、星际争霸
- 个性化推荐、内容优化
- 投资组合、交易策略
- 具身导航、操作、交互

---

### 第十部分：论文详解（3周）

| 章节 | 论文 | 发表年份 |
|------|------|----------|
| 10.1 | "Human-Level Control Through Deep Reinforcement Learning" (DQN) | 2015 |
| 10.2 | "Trust Region Policy Optimization" (TRPO) | 2015 |
| 10.3 | "Proximal Policy Optimization Algorithms" (PPO) | 2017 |
| 10.4 | "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning" | 2018 |
| 10.5 | "DDPG: Continuous Control with Deep Reinforcement Learning" | 2016 |
| 10.6 | "Decision Transformer: Reinforcement Learning via Sequence Modeling" | 2021 |
| 10.7 | "World Models" | 2018 |
| 10.8 | "DreamerV2: Mastering Atari with Discrete World Models" | 2020 |
| 10.9 | "CQL: Conservative Q-Learning for Offline Reinforcement Learning" | 2020 |
| 10.10 | "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (MADDPG) | 2017 |

---

## 推荐学习资源

### 书籍
1. 《Reinforcement Learning: An Introduction》- Sutton & Barto
2. 《Deep Reinforcement Learning Hands-On》- Maxim Lapan
3. 《强化学习》- 周志华
4. 《Foundations of Deep Reinforcement Learning》- Liu et al.
5. 《Multi-Agent Reinforcement Learning》- Busoniu et al.

### 在线课程
1. **CS285** - UC Berkeley Deep RL
2. **CS234** - Stanford RL
3. **David Silver's RL Course** - UCL
4. **DeepLearning.AI RL Specialization**
5. **MIT 6.S094** - Deep Learning for Self-Driving Cars

### 工具与框架
1. **OpenAI Gym** - 环境库
2. **Stable Baselines3** - RL算法库
3. **Ray RLLib** - 分布式RL
4. **PyTorch RL** - PyTorch官方RL库
5. **DeepMind Lab** - 3D环境

---

## 学习评估

### 自测题
1. 解释强化学习与监督学习的区别
2. 什么是Markov决策过程？为什么它重要？
3. 解释Bellman方程的含义
4. 比较基于值的方法和基于策略的方法
5. 什么是经验回放？为什么DQN需要它？
6. 离线RL的主要挑战是什么？
7. 什么是PPO的核心思想？
8. 分层RL解决什么问题？

### 实践项目
1. 使用Q-Learning解决FrozenLake
2. 实现DQN玩Atari游戏
3. 使用PPO训练CartPole
4. 实现CQL进行离线RL
5. 使用SAC训练机械臂控制
6. 实现简单的多智能体环境
7. 在Habitat中使用RL进行导航

---

## 模块关系

```
具身智能
├── 03-machine-learning (机器学习) ← 前置模块
├── 04-deep-learning (深度学习) ← 前置模块
├── 05-reinforcement-learning (强化学习) ← 当前模块
├── 07-robot-control (机器人控制) ← 相关模块
└── 12-embodied-systems (具身系统) ← 综合应用
```

**学习顺序建议**：
1. 先学习机器学习模块（03-machine-learning）
2. 再学习深度学习模块（04-deep-learning）
3. 最后学习本模块（强化学习）

**前置知识**：
- 机器学习基础（监督学习、无监督学习）
- 深度学习基础（神经网络、反向传播）
- Python编程基础

---

## 重要论文列表

| 论文标题 | 作者 | 发表期刊/会议 | 年份 | 核心贡献 |
|----------|------|---------------|------|----------|
| Human-Level Control Through Deep RL | Mnih et al. | Nature | 2015 | DQN算法 |
| Trust Region Policy Optimization | Schulman et al. | ICML | 2015 | TRPO算法 |
| Proximal Policy Optimization | Schulman et al. | arXiv | 2017 | PPO算法 |
| Soft Actor-Critic | Haarnoja et al. | ICML | 2018 | SAC算法 |
| DDPG | Lillicrap et al. | arXiv | 2016 | 连续控制DQN |
| Decision Transformer | Chen et al. | NeurIPS | 2021 | RL序列建模 |
| World Models | Ha & Schmidhuber | arXiv | 2018 | 世界模型概念 |
| DreamerV2 | Hafner et al. | NeurIPS | 2020 | 离散世界模型 |
| CQL | Kumar et al. | NeurIPS | 2020 | 保守Q学习 |
| MADDPG | Lowe et al. | NIPS | 2017 | 多智能体AC |

---

**上一个模块**：[深度学习](../04-deep-learning/README.md)
**下一个模块**：[机器人学](../06-robotics/README.md)