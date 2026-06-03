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

## 7. 世界模型的设计原则

### 7.1 模块性原则

世界模型应该具备良好的模块性，各个组件可以独立设计和优化：

```python
class ModularWorldModel:
    """模块化世界模型设计"""
    
    def __init__(self, encoder, dynamics, decoder, predictor=None):
        self.encoder = encoder      # 感知模块
        self.dynamics = dynamics    # 动态模型
        self.decoder = decoder      # 生成模块
        self.predictor = predictor  # 预测模块（可选）
    
    def forward(self, obs, action):
        """前向传播"""
        z = self.encoder(obs)
        z_next = self.dynamics(z, action)
        
        outputs = {
            'state': z_next,
            'obs_pred': self.decoder(z_next)
        }
        
        if self.predictor:
            outputs.update(self.predictor(z_next))
        
        return outputs

# 示例：构建模块化世界模型
encoder = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 64)
)

dynamics = nn.Sequential(
    nn.Linear(64 + 10, 256),
    nn.ReLU(),
    nn.Linear(256, 64)
)

decoder = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 100)
)

model = ModularWorldModel(encoder, dynamics, decoder)
```

### 7.2 可扩展性原则

好的世界模型设计应该易于扩展：

```python
class ExtensibleWorldModel(nn.Module):
    """可扩展的世界模型"""
    
    def __init__(self, base_config, extensions=None):
        super().__init__()
        
        # 基础组件
        self.encoder = self._build_encoder(base_config['obs_dim'], base_config['latent_dim'])
        self.dynamics = self._build_dynamics(base_config['latent_dim'], base_config['action_dim'])
        self.decoder = self._build_decoder(base_config['latent_dim'], base_config['obs_dim'])
        
        # 扩展组件
        self.extensions = nn.ModuleDict()
        if extensions:
            for name, ext_config in extensions.items():
                self.extensions[name] = self._build_extension(name, ext_config)
    
    def _build_encoder(self, obs_dim, latent_dim):
        """构建编码器"""
        return nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def _build_dynamics(self, latent_dim, action_dim):
        """构建动态模型"""
        return nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
    
    def _build_decoder(self, latent_dim, obs_dim):
        """构建解码器"""
        return nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim)
        )
    
    def _build_extension(self, name, config):
        """构建扩展组件"""
        if name == 'reward_predictor':
            return nn.Sequential(
                nn.Linear(config['input_dim'], 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        elif name == 'termination_predictor':
            return nn.Sequential(
                nn.Linear(config['input_dim'], 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        elif name == 'value_predictor':
            return nn.Sequential(
                nn.Linear(config['input_dim'], 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        else:
            raise ValueError(f"Unknown extension: {name}")
    
    def forward(self, obs, action):
        """前向传播"""
        z = self.encoder(obs)
        z_next = self.dynamics(torch.cat([z, action], dim=-1))
        
        outputs = {
            'state': z_next,
            'obs_pred': self.decoder(z_next)
        }
        
        # 应用扩展
        for name, ext in self.extensions.items():
            outputs[name.replace('_predictor', '')] = ext(z_next)
        
        return outputs

# 使用示例
config = {
    'obs_dim': 100,
    'action_dim': 10,
    'latent_dim': 64
}

extensions = {
    'reward_predictor': {'input_dim': 64},
    'termination_predictor': {'input_dim': 64},
    'value_predictor': {'input_dim': 64}
}

model = ExtensibleWorldModel(config, extensions)
obs = torch.randn(32, 100)
action = torch.randn(32, 10)
output = model(obs, action)

print("输出键:", list(output.keys()))  # ['state', 'obs_pred', 'reward', 'termination', 'value']
```

### 7.3 鲁棒性原则

世界模型需要对噪声和不确定性具有鲁棒性：

```python
class RobustWorldModel(nn.Module):
    """鲁棒世界模型"""
    
    def __init__(self, obs_dim, action_dim, latent_dim=64):
        super().__init__()
        
        # 噪声注入层
        self.noise_injector = GaussianNoiseInjector(std=0.01)
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # 动态模型（带dropout）
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim)
        )
    
    def forward(self, obs, action, training=True):
        """前向传播"""
        # 在训练时注入噪声
        if training:
            obs = self.noise_injector(obs)
        
        z = self.encoder(obs)
        z_next = self.dynamics(torch.cat([z, action], dim=-1))
        obs_pred = self.decoder(z_next)
        
        return obs_pred, z_next

class GaussianNoiseInjector(nn.Module):
    """高斯噪声注入器"""
    
    def __init__(self, std=0.01):
        super().__init__()
        self.std = std
    
    def forward(self, x):
        """注入高斯噪声"""
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x
```

---

## 8. 世界模型的评估指标

### 8.1 预测准确性指标

```python
class PredictionMetrics:
    """预测准确性指标计算"""
    
    @staticmethod
    def mse(pred, target):
        """均方误差"""
        return torch.mean((pred - target) ** 2)
    
    @staticmethod
    def rmse(pred, target):
        """均方根误差"""
        return torch.sqrt(torch.mean((pred - target) ** 2))
    
    @staticmethod
    def mae(pred, target):
        """平均绝对误差"""
        return torch.mean(torch.abs(pred - target))
    
    @staticmethod
    def cosine_similarity(pred, target):
        """余弦相似度"""
        pred_norm = pred / pred.norm(dim=-1, keepdim=True)
        target_norm = target / target.norm(dim=-1, keepdim=True)
        return torch.mean(torch.sum(pred_norm * target_norm, dim=-1))
    
    @staticmethod
    def psnr(pred, target, max_val=1.0):
        """峰值信噪比（用于图像）"""
        mse_val = PredictionMetrics.mse(pred, target)
        return 10 * torch.log10((max_val ** 2) / mse_val)

# 使用示例
pred = torch.randn(32, 100)
target = torch.randn(32, 100)

print(f"MSE: {PredictionMetrics.mse(pred, target):.4f}")
print(f"RMSE: {PredictionMetrics.rmse(pred, target):.4f}")
print(f"MAE: {PredictionMetrics.mae(pred, target):.4f}")
print(f"Cosine Similarity: {PredictionMetrics.cosine_similarity(pred, target):.4f}")
```

### 8.2 规划性能指标

```python
class PlanningMetrics:
    """规划性能指标"""
    
    @staticmethod
    def episode_reward(rewards):
        """总回报"""
        return sum(rewards)
    
    @staticmethod
    def average_reward(rewards):
        """平均回报"""
        return sum(rewards) / len(rewards) if rewards else 0
    
    @staticmethod
    def success_rate(episodes, success_threshold=0.9):
        """成功率"""
        successes = sum(1 for ep in episodes if ep['reward'] >= success_threshold)
        return successes / len(episodes)
    
    @staticmethod
    def planning_efficiency(model, planner, env, num_episodes=10):
        """规划效率（每步平均回报）"""
        total_reward = 0
        total_steps = 0
        
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            
            while not done:
                action = planner.select_action(model, obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                total_steps += 1
        
        return total_reward / total_steps if total_steps > 0 else 0
    
    @staticmethod
    def sample_efficiency(returns, sample_counts):
        """样本效率（每样本回报）"""
        efficiency = []
        for ret, count in zip(returns, sample_counts):
            efficiency.append(ret / count)
        return efficiency
```

### 8.3 模型质量指标

```python
class ModelQualityMetrics:
    """模型质量指标"""
    
    @staticmethod
    def latent_diversity(model, dataset, sample_size=1000):
        """隐状态多样性"""
        latents = []
        for i in range(min(sample_size, len(dataset))):
            obs = dataset[i]['obs']
            z = model.encoder(torch.FloatTensor(obs).unsqueeze(0))
            latents.append(z.detach().numpy())
        
        latents = np.concatenate(latents, axis=0)
        # 计算PCA后的方差
        pca = PCA(n_components=min(10, latents.shape[1]))
        pca.fit(latents)
        return np.sum(pca.explained_variance_ratio_)
    
    @staticmethod
    def model_complexity(model):
        """模型复杂度（参数数量）"""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def inference_speed(model, input_shape, num_runs=100):
        """推理速度"""
        dummy_input = torch.randn(input_shape)
        action = torch.randn(input_shape[:-1] + (10,))
        
        # 预热
        for _ in range(10):
            model(dummy_input, action)
        
        # 计时
        start = time.time()
        for _ in range(num_runs):
            model(dummy_input, action)
        end = time.time()
        
        return num_runs / (end - start)  # samples per second
```

---

## 9. 世界模型的训练策略

### 9.1 监督学习训练

```python
class SupervisedTrainer:
    """监督学习训练器"""
    
    def __init__(self, model, lr=1e-3, weight_decay=1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.loss_fn = nn.MSELoss()
    
    def train_step(self, obs, action, next_obs):
        """单步训练"""
        self.model.train()
        
        # 前向传播
        obs_pred = self.model(obs, action)
        
        # 计算损失
        loss = self.loss_fn(obs_pred, next_obs)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            obs, action, next_obs = batch
            loss = self.train_step(obs, action, next_obs)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches
```

### 9.2 自监督学习训练

```python
class SelfSupervisedTrainer:
    """自监督学习训练器"""
    
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def train_step(self, obs_seq, action_seq):
        """序列训练"""
        self.model.train()
        
        batch_size, seq_len = obs_seq.size(0), obs_seq.size(1)
        total_loss = 0
        
        for t in range(seq_len - 1):
            obs = obs_seq[:, t]
            action = action_seq[:, t]
            next_obs = obs_seq[:, t+1]
            
            # 前向传播
            obs_pred = self.model(obs, action)
            
            # 计算损失
            loss = nn.MSELoss()(obs_pred, next_obs)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / (seq_len - 1)
```

### 9.3 强化学习训练

```python
class RLBasedTrainer:
    """强化学习训练器"""
    
    def __init__(self, model, policy, env, config):
        self.model = model
        self.policy = policy
        self.env = env
        self.config = config
        
        self.model_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['model_lr']
        )
        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config['policy_lr']
        )
        
        self.replay_buffer = []
    
    def collect_experience(self, num_steps=1000):
        """收集经验"""
        obs = self.env.reset()
        
        for _ in range(num_steps):
            action = self.policy.select_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            
            self.replay_buffer.append({
                'obs': obs,
                'action': action,
                'next_obs': next_obs,
                'reward': reward,
                'done': done
            })
            
            obs = next_obs
            if done:
                obs = self.env.reset()
        
        # 限制缓冲区大小
        if len(self.replay_buffer) > self.config['buffer_size']:
            self.replay_buffer = self.replay_buffer[-self.config['buffer_size']:]
    
    def train_model(self, batch_size=32):
        """训练世界模型"""
        self.model.train()
        
        # 采样批次
        batch = random.sample(self.replay_buffer, min(batch_size, len(self.replay_buffer)))
        
        obs = torch.FloatTensor([b['obs'] for b in batch])
        action = torch.FloatTensor([b['action'] for b in batch])
        next_obs = torch.FloatTensor([b['next_obs'] for b in batch])
        
        # 前向传播
        obs_pred = self.model(obs, action)
        
        # 计算损失
        loss = nn.MSELoss()(obs_pred, next_obs)
        
        # 反向传播
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        
        return loss.item()
    
    def imagine_and_learn(self, horizon=10):
        """想象训练"""
        self.model.eval()
        
        # 采样初始状态
        initial_transition = random.choice(self.replay_buffer)
        obs = torch.FloatTensor(initial_transition['obs']).unsqueeze(0)
        
        # 想象轨迹
        rewards = []
        current_obs = obs
        
        for t in range(horizon):
            action = self.policy.select_action(current_obs.squeeze(0).numpy())
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            
            with torch.no_grad():
                next_obs_pred = self.model(current_obs, action_tensor)
            
            # 预测奖励
            reward = self.policy.predict_reward(next_obs_pred)
            rewards.append(reward.item())
            
            current_obs = next_obs_pred
        
        # 使用想象奖励更新策略
        returns = self.compute_returns(rewards)
        self.update_policy(returns)
    
    def compute_returns(self, rewards, gamma=0.99):
        """计算回报"""
        returns = []
        running_sum = 0
        for reward in reversed(rewards):
            running_sum = reward + gamma * running_sum
            returns.insert(0, running_sum)
        return returns
    
    def update_policy(self, returns):
        """更新策略"""
        # 简化的策略更新
        pass
```

---

## 10. 世界模型的应用实践

### 10.1 机器人控制应用

```python
class RobotControlWorldModel(nn.Module):
    """机器人控制世界模型"""
    
    def __init__(self, obs_dim, action_dim, latent_dim=64):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # 动态模型
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, obs_dim)
        )
    
    def forward(self, obs, action):
        z = self.encoder(obs)
        z_next = self.dynamics(torch.cat([z, action], dim=-1))
        return self.decoder(z_next), z_next

# 机器人控制流程
class RobotController:
    """机器人控制器"""
    
    def __init__(self, world_model, controller, env):
        self.world_model = world_model
        self.controller = controller
        self.env = env
    
    def plan(self, initial_obs, goal, horizon=10):
        """规划路径"""
        current_obs = initial_obs
        plan = []
        
        for _ in range(horizon):
            # 使用控制器选择动作
            action = self.controller.select_action(current_obs, goal)
            plan.append(action)
            
            # 使用世界模型预测下一状态
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(current_obs).unsqueeze(0)
                action_tensor = torch.FloatTensor(action).unsqueeze(0)
                next_obs_pred = self.world_model(obs_tensor, action_tensor)[0].numpy()[0]
            
            current_obs = next_obs_pred
            
            # 检查是否到达目标
            if self.is_goal(current_obs, goal):
                break
        
        return plan
    
    def is_goal(self, obs, goal, threshold=0.1):
        """检查是否到达目标"""
        return np.linalg.norm(obs - goal) < threshold

# 使用示例
model = RobotControlWorldModel(obs_dim=12, action_dim=3)
controller = ModelPredictiveController(model, horizon=5)
env = RobotEnvironment()

obs = env.reset()
goal = np.array([0.5, 0.5, 0.0])
plan = RobotController(model, controller, env).plan(obs, goal)

print(f"规划的动作序列长度: {len(plan)}")
```

### 10.2 游戏AI应用

```python
class GameWorldModel(nn.Module):
    """游戏世界模型"""
    
    def __init__(self, obs_dim, action_dim, latent_dim=128):
        super().__init__()
        
        # 卷积编码器（处理像素输入）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
        
        # 动态模型
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=8, stride=4),
            nn.Sigmoid()
        )
    
    def forward(self, obs, action):
        """前向传播"""
        z = self.encoder(obs)
        z_next = self.dynamics(torch.cat([z, action], dim=-1))
        obs_pred = self.decoder(z_next)
        return obs_pred, z_next

# 游戏AI训练
class GameAITrainer:
    """游戏AI训练器"""
    
    def __init__(self, model, policy, env):
        self.model = model
        self.policy = policy
        self.env = env
        self.model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-5)
    
    def train(self, num_iterations=1000):
        """训练循环"""
        for iteration in range(num_iterations):
            # 收集数据
            obs = self.env.reset()
            done = False
            
            while not done:
                action = self.policy.select_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                
                # 训练世界模型
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).permute(0, 3, 1, 2)
                action_tensor = torch.FloatTensor(action).unsqueeze(0)
                next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).permute(0, 3, 1, 2)
                
                obs_pred, _ = self.model(obs_tensor, action_tensor)
                loss = nn.MSELoss()(obs_pred, next_obs_tensor)
                
                self.model_optimizer.zero_grad()
                loss.backward()
                self.model_optimizer.step()
                
                obs = next_obs
            
            # 定期评估
            if iteration % 100 == 0:
                reward = self.evaluate()
                print(f"Iteration {iteration}, Reward: {reward}")
    
    def evaluate(self, num_episodes=5):
        """评估"""
        total_reward = 0
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = False
            
            while not done:
                action = self.policy.select_action(obs)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
        
        return total_reward / num_episodes
```

### 10.3 模拟与仿真应用

```python
class SimulationWorldModel(nn.Module):
    """仿真世界模型"""
    
    def __init__(self, state_dim, action_dim, latent_dim=256):
        super().__init__()
        
        # 状态编码器
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        # 动态模型（使用GRU）
        self.dynamics = nn.GRU(
            input_size=latent_dim + action_dim,
            hidden_size=latent_dim,
            batch_first=True
        )
        
        # 状态解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim)
        )
    
    def forward(self, state_seq, action_seq):
        """序列预测"""
        # 编码状态序列
        batch_size, seq_len = state_seq.size(0), state_seq.size(1)
        z_seq = self.encoder(state_seq)
        
        # 动态预测
        gru_input = torch.cat([z_seq, action_seq], dim=-1)
        z_next_seq, _ = self.dynamics(gru_input)
        
        # 解码
        state_pred = self.decoder(z_next_seq)
        
        return state_pred

# 仿真应用示例
class PhysicsSimulator:
    """物理仿真器"""
    
    def __init__(self, world_model):
        self.world_model = world_model
        self.world_model.eval()
    
    def simulate(self, initial_state, action_seq):
        """模拟物理过程"""
        states = [initial_state]
        current_state = initial_state
        
        for t in range(len(action_seq)):
            action = action_seq[t]
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0).unsqueeze(1)
                action_tensor = torch.FloatTensor(action).unsqueeze(0).unsqueeze(1)
                
                next_state_pred = self.world_model(state_tensor, action_tensor)
                next_state = next_state_pred.squeeze(0).squeeze(0).numpy()
            
            states.append(next_state)
            current_state = next_state
        
        return np.array(states)

# 使用示例
model = SimulationWorldModel(state_dim=10, action_dim=3)
simulator = PhysicsSimulator(model)

initial_state = np.random.randn(10)
action_seq = [np.random.randn(3) for _ in range(100)]
states = simulator.simulate(initial_state, action_seq)

print(f"模拟状态序列形状: {states.shape}")  # [101, 10]
```

---

## 11. 世界模型的挑战与解决方案

### 11.1 长期预测挑战

```python
class LongHorizonPredictor(nn.Module):
    """长期预测模型"""
    
    def __init__(self, latent_dim, action_dim, horizon=100):
        super().__init__()
        self.horizon = horizon
        
        # 分层预测模型
        self.short_term = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        self.medium_term = nn.Sequential(
            nn.Linear(latent_dim * 5 + action_dim * 5, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        self.long_term = nn.Sequential(
            nn.Linear(latent_dim * 20 + action_dim * 20, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
    
    def forward(self, initial_state, actions):
        """分层长期预测"""
        states = [initial_state]
        current_state = initial_state
        
        for t in range(self.horizon):
            action = actions[:, t]
            
            if t < 5:
                # 短期预测
                current_state = self.short_term(
                    torch.cat([current_state, action], dim=-1)
                )
            elif t < 20:
                # 中期预测（使用历史信息）
                history = torch.cat(states[-5:], dim=-1)
                history_actions = torch.cat(actions[:, t-5:t], dim=-1)
                current_state = self.medium_term(
                    torch.cat([history, history_actions], dim=-1)
                )
            else:
                # 长期预测
                history = torch.cat(states[-20:], dim=-1)
                history_actions = torch.cat(actions[:, t-20:t], dim=-1)
                current_state = self.long_term(
                    torch.cat([history, history_actions], dim=-1)
                )
            
            states.append(current_state)
        
        return torch.stack(states, dim=1)
```

### 11.2 模型偏差问题

```python
class BiasCorrectedWorldModel(nn.Module):
    """偏差校正世界模型"""
    
    def __init__(self, base_model, correction_weight=0.1):
        super().__init__()
        self.base_model = base_model
        self.correction_weight = correction_weight
        
        # 偏差估计器
        self.bias_estimator = nn.Sequential(
            nn.Linear(base_model.latent_dim + base_model.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, base_model.latent_dim)
        )
    
    def forward(self, obs, action):
        """带偏差校正的前向传播"""
        # 基础预测
        obs_pred, z = self.base_model(obs, action)
        
        # 估计偏差
        bias = self.bias_estimator(torch.cat([z, action], dim=-1))
        
        # 校正预测
        corrected_z = z + self.correction_weight * bias
        corrected_obs_pred = self.base_model.decoder(corrected_z)
        
        return corrected_obs_pred, corrected_z
    
    def update_bias(self, obs, action, true_next_obs):
        """更新偏差估计器"""
        # 获取基础预测
        with torch.no_grad():
            obs_pred, z = self.base_model(obs, action)
        
        # 计算真实偏差
        true_z = self.base_model.encoder(true_next_obs)
        actual_bias = true_z - z
        
        # 更新偏差估计器
        predicted_bias = self.bias_estimator(torch.cat([z, action], dim=-1))
        loss = nn.MSELoss()(predicted_bias, actual_bias)
        
        # 反向传播（仅更新偏差估计器）
        optimizer = torch.optim.Adam(self.bias_estimator.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
```

### 11.3 计算复杂度优化

```python
class EfficientWorldModel(nn.Module):
    """高效世界模型"""
    
    def __init__(self, obs_dim, action_dim, latent_dim=64):
        super().__init__()
        
        # 轻量级编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # 线性动态模型（减少计算）
        self.dynamics = nn.Linear(latent_dim + action_dim, latent_dim)
        
        # 轻量级解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim)
        )
    
    def forward(self, obs, action):
        """快速前向传播"""
        z = self.encoder(obs)
        z_next = self.dynamics(torch.cat([z, action], dim=-1))
        obs_pred = self.decoder(z_next)
        return obs_pred

# 知识蒸馏加速
class DistilledWorldModel:
    """蒸馏世界模型"""
    
    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model
    
    def distill(self, dataloader, num_epochs=10, temperature=2.0):
        """蒸馏训练"""
        optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-3)
        
        for epoch in range(num_epochs):
            for obs, action, next_obs in dataloader:
                # 教师预测
                with torch.no_grad():
                    teacher_pred = self.teacher(obs, action)
                
                # 学生预测
                student_pred = self.student(obs, action)
                
                # 蒸馏损失
                loss = nn.MSELoss()(student_pred / temperature, teacher_pred / temperature)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

---

## 12. 世界模型的理论基础

### 12.1 马尔可夫性质

```python
class MarkovChecker:
    """马尔可夫性质验证器"""
    
    @staticmethod
    def check_markov_property(transition_data, state_dim, threshold=0.01):
        """
        检查数据是否满足马尔可夫性质
        
        P(s_{t+1} | s_t, a_t) ≈ P(s_{t+1} | s_t, a_t, s_{t-1}, ...)
        """
        # 简化检查：比较一步和多步条件下的预测误差
        mse_single = []
        mse_multi = []
        
        for i in range(len(transition_data) - 2):
            # 单步预测
            s_t = transition_data[i]['state']
            a_t = transition_data[i]['action']
            s_t1 = transition_data[i+1]['state']
            
            # 多步预测（使用额外历史）
            s_t_1 = transition_data[i-1]['state'] if i > 0 else s_t
            
            # 计算误差（假设已有模型）
            # ... 实际实现需要训练模型
            
            pass
        
        # 比较误差
        return np.mean(mse_single), np.mean(mse_multi)
```

### 12.2 信息论视角

```python
class InformationTheoryAnalysis:
    """信息论分析"""
    
    @staticmethod
    def entropy(x, bins=10):
        """计算熵"""
        hist, _ = np.histogram(x, bins=bins)
        p = hist / hist.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))
    
    @staticmethod
    def mutual_information(x, y, bins=10):
        """计算互信息"""
        joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
        p_xy = joint_hist / joint_hist.sum()
        
        p_x = p_xy.sum(axis=1, keepdims=True)
        p_y = p_xy.sum(axis=0, keepdims=True)
        
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mi
    
    @staticmethod
    def information_bottleneck(obs_seq, latent_seq):
        """计算信息瓶颈"""
        # I(X;Z) - beta * I(Z;Y)
        mi_obs_latent = InformationTheoryAnalysis.mutual_information(
            obs_seq.flatten(), 
            latent_seq.flatten()
        )
        
        mi_latent_next = InformationTheoryAnalysis.mutual_information(
            latent_seq[:-1].flatten(),
            latent_seq[1:].flatten()
        )
        
        return mi_obs_latent, mi_latent_next
```

---

## 13. 前沿研究方向

### 13.1 基于大语言模型的世界模型

```python
class LLMEnhancedWorldModel:
    """基于大语言模型的世界模型"""
    
    def __init__(self, llm_model, neural_model):
        self.llm = llm_model
        self.neural_model = neural_model
    
    def predict(self, obs, action):
        """混合预测"""
        # 神经预测
        neural_pred = self.neural_model.predict(obs, action)
        
        # 语言描述
        obs_desc = self.obs_to_text(obs)
        action_desc = self.action_to_text(action)
        
        # LLM预测
        prompt = f"""
        当前状态: {obs_desc}
        执行动作: {action_desc}
        
        请描述接下来可能发生的情况：
        """
        llm_pred_desc = self.llm.generate(prompt)
        
        # 融合预测
        return {
            'neural': neural_pred,
            'llm_description': llm_pred_desc,
            'fusion': self.fuse_predictions(neural_pred, llm_pred_desc)
        }
    
    def obs_to_text(self, obs):
        """观测转文本"""
        return f"观测: 维度={obs.shape}, 均值={obs.mean():.2f}"
    
    def action_to_text(self, action):
        """动作转文本"""
        return f"动作: 维度={action.shape}"
    
    def fuse_predictions(self, neural_pred, llm_desc):
        """融合预测结果"""
        # 简化融合
        return neural_pred
```

### 13.2 神经符号世界模型

```python
class NeuroSymbolicWorldModel(nn.Module):
    """神经符号世界模型"""
    
    def __init__(self, neural_model, symbolic_rules):
        super().__init__()
        self.neural_model = neural_model
        self.symbolic_rules = symbolic_rules
    
    def forward(self, obs, action):
        """神经符号推理"""
        # 神经感知
        z = self.neural_model.encoder(obs)
        
        # 符号提取
        symbols = self.extract_symbols(z)
        
        # 符号推理
        next_symbols = self.apply_rules(symbols, action)
        
        # 神经生成
        next_z = self.neural_model.dynamics(z, action)
        obs_pred = self.neural_model.decoder(next_z)
        
        return obs_pred, next_symbols
    
    def extract_symbols(self, z):
        """从隐状态提取符号"""
        # 简化：阈值判断
        return {
            'is_goal': z[0] > 0.5,
            'is_dangerous': z[1] < -0.5
        }
    
    def apply_rules(self, symbols, action):
        """应用符号规则"""
        next_symbols = symbols.copy()
        
        if symbols['is_dangerous'] and action == 'move_forward':
            next_symbols['is_dangerous'] = True
            next_symbols['warning'] = True
        
        return next_symbols
```

---

## 14. 总结与展望

世界模型是实现通用人工智能的关键技术。通过学习环境的动态规律，智能体可以：

1. **高效学习**：在想象中进行大量试错
2. **安全探索**：在执行前模拟可能的结果
3. **长期规划**：支持多步预测和目标达成
4. **知识迁移**：将学到的模型应用到新任务

### 未来研究方向

| 方向 | 描述 | 挑战 |
|------|------|------|
| **长期预测** | 解决预测误差累积 | 模型偏差、计算复杂度 |
| **自适应模型** | 根据环境变化调整 | 在线学习、持续适应 |
| **多模态融合** | 整合多种传感器输入 | 模态对齐、信息融合 |
| **神经符号结合** | 结合神经网络和符号推理 | 表示转换、推理效率 |
| **大模型整合** | 与大语言模型结合 | 知识对齐、推理一致性 |

---

**下一节**：[动态模型学习](02-dynamics-learning.md)

---

## 参考文献

1. Ha, D., & Schmidhuber, J. (2018). World Models. arXiv preprint arXiv:1803.10122.
2. Hafner, D., et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination. NeurIPS.
3. Kaiser, L., et al. (2019). Model-Based Reinforcement Learning for Atari. arXiv preprint arXiv:1903.00374.
4. Watter, M., et al. (2015). Embed to Control: A Locally Linear Latent Dynamics Model. NIPS.
5. Janner, M., et al. (2019). When to Trust Your Model: Model-Based Policy Optimization. NeurIPS.
