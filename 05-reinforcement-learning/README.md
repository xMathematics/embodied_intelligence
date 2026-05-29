
# 强化学习模块

## 模块概述

强化学习是具身智能的核心学习范式，本模块涵盖RL基础、经典算法、进阶算法等内容。

## 学习目标

完成本模块学习后，您将能够：
- 理解马尔可夫决策过程
- 掌握经典RL算法
- 能够实现和训练强化学习模型

## 学习路径

### 第一部分：RL基础（2周）

#### 1.1 马尔可夫决策过程

**核心概念：**
- 状态、动作、奖励
- 马尔可夫性质
- 值函数与策略

**推荐资源：**
- 《Reinforcement Learning: An Introduction》- Sutton & Barto
- 《Deep Reinforcement Learning Hands-On》- Lapan

**实践练习：**
- 理解MDP的基本概念
- 计算值函数

#### 1.2 动态规划

**核心概念：**
- 策略评估
- 策略迭代
- 值迭代

**实践练习：**
- 实现值迭代算法
- 求解简单MDP

---

### 第二部分：经典RL算法（2周）

#### 2.1 值基于方法

**核心概念：**
- Q-learning
- SARSA
- Deep Q-Network (DQN)

**推荐论文：**
- "Human-Level Control through Deep Reinforcement Learning" (DQN)

**实践练习：**
- 实现Q-learning
- 训练DQN玩Atari游戏

#### 2.2 策略梯度方法

**核心概念：**
- REINFORCE算法
- Actor-Critic
- Advantage Actor-Critic (A2C)

**推荐论文：**
- "Policy Gradient Methods for Reinforcement Learning with Function Approximation"

**实践练习：**
- 实现REINFORCE
- 训练策略梯度模型

---

### 第三部分：进阶RL算法（2周）

#### 3.1 近端策略优化（PPO）

**核心概念：**
- 信任区域优化
- PPO裁剪目标
- 优势函数

**推荐论文：**
- "Proximal Policy Optimization Algorithms" (PPO)

**实践练习：**
- 实现PPO算法
- 在连续控制任务上测试

#### 3.2 深度确定性策略梯度（DDPG/SAC）

**核心概念：**
- 确定性策略
- 演员-评论家架构
- 软演员-评论家（SAC）

**推荐论文：**
- "Deep Deterministic Policy Gradients" (DDPG)
- "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning" (SAC)

**实践练习：**
- 实现SAC算法
- 解决连续控制问题

---

## 推荐学习资源

### 书籍
1. 《Reinforcement Learning: An Introduction》- Sutton & Barto
2. 《Deep Reinforcement Learning Hands-On》- Lapan
3. 《Foundations of Deep Reinforcement Learning》- Liu

### 在线课程
1. CS234 - Reinforcement Learning (Stanford)
2. Deep RL Bootcamp

### 工具
- OpenAI Gym - 强化学习环境
- Stable Baselines3 - RL算法库
- PyTorch RL - 自定义实现

## 学习评估

### 自测题
1. 解释Q-learning与SARSA的区别
2. 为什么PPO比传统策略梯度更稳定？
3. 解释SAC中的熵正则化

### 实践项目
1. 使用DQN玩CartPole
2. 使用PPO训练机器人手臂
3. 使用SAC解决连续控制任务

---

**下一个模块：** [机器人学基础](../06-robotics/README.md)
