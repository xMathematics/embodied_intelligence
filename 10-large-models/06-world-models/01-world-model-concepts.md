# 6.1 世界模型概念

## 目录

- [1. 引言](#1-引言)
- [2. 世界模型概述](#2-世界模型概述)
- [3. 世界模型分类](#3-世界模型分类)
- [4. 世界模型架构](#4-世界模型架构)
- [5. 代表性模型](#5-代表性模型)
- [6. 实践练习](#6-实践练习)

---

## 1. 引言

### 1.1 世界模型的重要性

**世界模型**是指智能体对环境进行建模的内部表示，能够预测未来状态和奖励。这是实现通用人工智能的关键组成部分，也是强化学习领域的重要研究方向。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **自动驾驶** | 预测道路环境和交通 | 车辆轨迹预测 |
| **机器人控制** | 理解物理世界 | 机械臂操作 |
| **游戏AI** | 理解游戏环境 | Atari游戏 |
| **科学仿真** | 模拟物理系统 | 分子动力学 |

---

## 2. 世界模型概述

### 2.1 定义

**世界模型**：智能体对外部环境的内部表示和预测能力。

**核心组件**：
```
世界模型 = 感知模块 + 记忆模块 + 预测模块
```

### 2.2 世界模型的特点

| 特点 | 描述 |
|------|------|
| **抽象性** | 对环境的高度抽象表示 |
| **预测性** | 预测未来状态的能力 |
| **可学习性** | 从经验中学习 |
| **泛化性** | 推广到新场景 |

---

## 3. 世界模型分类

### 3.1 基于观测的世界模型

**定义**：仅基于观测序列建模

**示例**：
```python
import torch
import torch.nn as nn

class ObservationWorldModel(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.transition = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def forward(self, obs, action):
        """预测下一观测"""
        obs_enc = self.encoder(obs)
        combined = torch.cat([obs_enc, action], dim=-1)
        next_obs_enc = self.transition(combined)
        next_obs_pred = self.decoder(next_obs_enc)
        return next_obs_pred
```

### 3.2 基于状态的世界模型

**定义**：建模隐状态空间

**示例**：
```python
class LatentWorldModel(nn.Module):
    def __init__(self, obs_dim, latent_dim, action_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim)
        )
    
    def predict_next_state(self, state, action):
        """预测下一状态"""
        combined = torch.cat([state, action], dim=-1)
        next_state = self.dynamics(combined)
        return next_state
    
    def reconstruct(self, state):
        """从状态重建观测"""
        return self.decoder(state)
```

### 3.3 组合世界模型

**定义**：结合多个世界模型

**示例**：
```python
class CompositionWorldModel:
    def __init__(self, models):
        """
        组合多个世界模型
        
        参数:
            models: 子模型字典
        """
        self.models = models
    
    def predict(self, obs, action):
        """组合预测"""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(obs, action)
        return predictions
```

---

## 4. 世界模型架构

### 4.1 编码器-解码器架构

**结构**：
```
观测 → 编码器 → 隐状态 → 动态模型 → 下一状态 → 解码器 → 下一观测
```

### 4.2 循环世界模型

**结构**：
```
隐状态_t → 动态模型 → 隐状态_{t+1}
     ↓
观测_t → 编码器 → 隐状态_t
```

### 4.3 变分世界模型

**结构**：
```
观测 → 编码器 → μ, σ → 采样 → 隐状态 → 解码器 → 重建观测
```

```python
class VariationalWorldModel(nn.Module):
    def __init__(self, obs_dim, latent_dim, action_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim)
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, obs, action):
        """前向传播"""
        # 编码
        h = self.encoder(obs)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        
        # 动态预测
        next_z = self.dynamics(torch.cat([z, action], dim=-1))
        
        # 解码
        next_obs_pred = self.decoder(next_z)
        
        return next_obs_pred, mu, logvar, z
```

---

## 5. 代表性模型

### 5.1 Dreamer

**论文**：Dream to Control: Learning Behaviors by Latent Imagination (Hafner et al., 2020)

**核心特点**：
- 基于变分自编码器的世界模型
- 在隐空间进行规划
- 高效的想象 rollout

### 5.2 World Models

**论文**：World Models (Ha & Schmidhuber, 2018)

**核心特点**：
- 早期的世界模型研究
- 压缩环境经验
- 循环神经网络建模

### 5.3 SimPLe

**论文**：Model-Based Reinforcement Learning for Atari (Kaiser et al., 2019)

**核心特点**：
- 用于Atari游戏的世界模型
- 像素级预测
- 模型预测控制

---

## 6. 实践练习

### 练习1：实现简单的世界模型

```python
import torch
import torch.nn as nn

class SimpleWorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 观测编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 动态模型
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 观测解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def forward(self, obs, action):
        """
        前向传播：预测下一观测
        
        参数:
            obs: 当前观测 [batch, obs_dim]
            action: 动作 [batch, action_dim]
        
        返回:
            下一观测预测 [batch, obs_dim]
        """
        z = self.encoder(obs)
        z_next = self.dynamics(torch.cat([z, action], dim=-1))
        obs_next_pred = self.decoder(z_next)
        return obs_next_pred
    
    def imagine_rollout(self, obs0, actions, horizon):
        """
        想象 rollout
        
        参数:
            obs0: 初始观测
            actions: 动作序列
            horizon: 预测步数
        
        返回:
            预测的观测序列
        """
        obs_predictions = [obs0]
        current_obs = obs0
        
        for t in range(horizon):
            action = actions[t] if t < len(actions) else torch.zeros(1, self.action_dim)
            next_obs_pred = self.forward(current_obs, action)
            obs_predictions.append(next_obs_pred)
            current_obs = next_obs_pred
        
        return torch.stack(obs_predictions)

# 测试
model = SimpleWorldModel(obs_dim=4, action_dim=2)
obs0 = torch.randn(1, 4)
actions = [torch.randn(1, 2) for _ in range(5)]
rollout = model.imagine_rollout(obs0, actions, horizon=5)
print(f"Rollout shape: {rollout.shape}")  # [6, 1, 4]
```

### 练习2：实现Dreamer风格的世界模型

```python
class DreamerWorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=32, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 观测编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # RSSM动态模型
        self.rssm = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # 观测解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        # 奖励预测器
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, obs):
        """编码观测到隐状态"""
        h = self.encoder(obs)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def dynamics(self, z_prev, action):
        """RSSM动态模型"""
        h = self.rssm(torch.cat([z_prev, action], dim=-1))
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        """从隐状态解码"""
        return self.decoder(z)
    
    def predict_reward(self, z):
        """预测奖励"""
        return self.reward_predictor(z)

# 测试
model = DreamerWorldModel(obs_dim=4, action_dim=2)
obs = torch.randn(1, 4)
action = torch.randn(1, 2)

z, mu, logvar = model.encode(obs)
z_next, _, _ = model.dynamics(z, action)
obs_pred = model.decode(z_next)
reward_pred = model.predict_reward(z_next)

print(f"隐状态形状: {z.shape}")           # [1, 32]
print(f"观测预测形状: {obs_pred.shape}")    # [1, 4]
print(f"奖励预测形状: {reward_pred.shape}") # [1, 1]
```

### 练习3：世界模型评估

```python
import numpy as np

def evaluate_world_model(model, env, num_episodes=10, horizon=50):
    """
    评估世界模型的预测准确性
    
    参数:
        model: 世界模型
        env: 环境
        num_episodes: 评估的episode数量
        horizon: 每episode的预测步数
    
    返回:
        评估指标
    """
    mse_errors = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_errors = []
        
        for step in range(horizon):
            action = env.action_space.sample()
            
            # 世界模型预测
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_tensor = torch.FloatTensor(action).unsqueeze(0)
                obs_pred = model(obs_tensor, action_tensor).numpy()[0]
            
            # 环境实际转移
            obs_next, reward, done, _ = env.step(action)
            
            # 计算预测误差
            mse = np.mean((obs_pred - obs_next) ** 2)
            episode_errors.append(mse)
            
            obs = obs_next
            
            if done:
                break
        
        mse_errors.append(np.mean(episode_errors))
    
    return {
        'mean_mse': np.mean(mse_errors),
        'std_mse': np.std(mse_errors),
        'per_episode_mse': mse_errors
    }

# 示例：使用随机环境
# import gym
# env = gym.make('Pendulum-v1')
# metrics = evaluate_world_model(model, env)
# print(f"Mean MSE: {metrics['mean_mse']:.4f}")
```

---

**下一节**：[动态模型学习](02-dynamics-learning.md)

---

## 参考文献

1. Ha, D., & Schmidhuber, J. (2018). World Models.
2. Hafner, D., et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination.
3. Kaiser, L., et al. (2019). Model-Based Reinforcement Learning for Atari.
