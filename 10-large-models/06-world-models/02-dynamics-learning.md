# 6.2 动态模型学习

## 目录

- [1. 引言](#1-引言)
- [2. 动态模型概述](#2-动态模型概述)
- [3. 动态模型类型](#3-动态模型类型)
- [4. 动态模型训练方法](#4-动态模型训练方法)
- [5. 动态模型高级架构](#5-动态模型高级架构)
- [6. 动态模型优化策略](#6-动态模型优化策略)
- [7. 动态模型评估指标](#7-动态模型评估指标)
- [8. 工程实践与部署](#8-工程实践与部署)
- [9. 理论基础与分析](#9-理论基础与分析)
- [10. 前沿研究方向](#10-前沿研究方向)
- [11. 实践练习](#11-实践练习)

---

## 1. 引言

### 1.1 动态模型的重要性

**动态模型**是世界模型的核心组成部分，它学习环境的状态转移规律，能够预测在给定动作下环境如何变化。这对于智能体进行规划、决策和控制至关重要。

在强化学习中，动态模型可以：
- 加速策略学习
- 减少与真实环境的交互次数
- 支持离线策略优化
- 实现想象增强的学习

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **机器人控制** | 预测机器人动作效果 | 机械臂运动规划、无人机导航 |
| **游戏AI** | 预测游戏状态变化 | 围棋、象棋、Atari游戏 |
| **自动驾驶** | 预测交通场景演化 | 车辆轨迹预测、避障 |
| **流体仿真** | 预测流体运动 | 天气预报、海洋环流 |
| **经济预测** | 预测市场变化 | 股票价格预测、供需分析 |

### 1.3 动态模型的挑战

动态模型学习面临以下核心挑战：

| 挑战 | 描述 |
|------|------|
| **长期预测** | 误差累积导致长期预测精度下降 |
| **模型偏差** | 训练数据与真实环境存在分布差异 |
| **计算复杂度** | 高维状态空间的计算开销 |
| **不确定性建模** | 需要准确估计预测的不确定性 |

---

## 2. 动态模型概述

### 2.1 定义

**动态模型**：学习环境状态转移函数的模型。

**形式化表达**：
```
s_{t+1} = f(s_t, a_t; θ)
```

其中：
- $s_t$：时刻t的状态
- $a_t$：时刻t的动作
- $f$：动态模型
- $θ$：模型参数

### 2.2 动态模型的特点

| 特点 | 描述 |
|------|------|
| **马尔可夫性** | 假设状态转移仅依赖当前状态和动作 |
| **确定性/随机性** | 可分为确定性动态和随机动态 |
| **连续/离散** | 可处理连续或离散状态空间 |
| **线性/非线性** | 可建模线性或非线性关系 |

### 2.3 动态模型的分类体系

```
动态模型
├── 按输出类型
│   ├── 确定性动态模型
│   ├── 随机动态模型
│   └── 混合动态模型
├── 按架构
│   ├── 前馈神经网络
│   ├── 循环神经网络
│   ├── 变分自编码器
│   └── 图神经网络
└── 按学习方式
    ├── 监督学习
    ├── 强化学习
    ├── 自监督学习
    └── 对比学习
```

---

## 3. 动态模型类型

### 3.1 确定性动态模型

**定义**：给定状态和动作，输出确定性的下一状态。

**适用场景**：
- 环境噪声较小的场景
- 对预测精度要求高的任务
- 计算资源有限的应用

```python
class DeterministicDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        
        layers = []
        input_dim = state_dim + action_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, action):
        """
        前向传播
        
        参数:
            state: 当前状态 [batch, state_dim]
            action: 动作 [batch, action_dim]
        
        返回:
            下一状态预测 [batch, state_dim]
        """
        combined = torch.cat([state, action], dim=-1)
        next_state = self.network(combined)
        return next_state
    
    def predict_sequence(self, initial_state, actions):
        """
        预测状态序列
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            actions: 动作序列 [batch, seq_len, action_dim]
        
        返回:
            状态序列 [batch, seq_len+1, state_dim]
        """
        states = [initial_state]
        current_state = initial_state
        
        for t in range(actions.size(1)):
            action = actions[:, t]
            current_state = self.forward(current_state, action)
            states.append(current_state)
        
        return torch.stack(states, dim=1)
```

### 3.2 随机动态模型

**定义**：建模状态转移的分布，输出分布参数。

**适用场景**：
- 环境存在不确定性
- 需要估计预测置信度
- 强化学习中的策略评估

```python
class StochasticDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        
        self.feature_net = nn.Sequential()
        input_dim = state_dim + action_dim
        
        for i in range(num_layers):
            self.feature_net.add_module(f'linear_{i}', nn.Linear(input_dim, hidden_dim))
            self.feature_net.add_module(f'relu_{i}', nn.ReLU())
            self.feature_net.add_module(f'norm_{i}', nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        # 输出均值和方差
        self.mean_layer = nn.Linear(hidden_dim, state_dim)
        self.logvar_layer = nn.Linear(hidden_dim, state_dim)
        
        # 约束方差范围
        self.min_logvar = nn.Parameter(torch.tensor(-10.0))
        self.max_logvar = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, state, action):
        """
        前向传播
        
        返回:
            下一状态的均值和方差
        """
        combined = torch.cat([state, action], dim=-1)
        h = self.feature_net(combined)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        
        # 约束方差范围
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        
        return mean, logvar
    
    def sample(self, state, action, deterministic=False):
        """从分布中采样"""
        mean, logvar = self.forward(state, action)
        
        if deterministic:
            return mean
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def kl_divergence(self, state, action):
        """计算与标准正态分布的KL散度"""
        mean, logvar = self.forward(state, action)
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return kl
    
    def log_prob(self, state, action, next_state):
        """计算下一状态的对数概率"""
        mean, logvar = self.forward(state, action)
        std = torch.exp(0.5 * logvar)
        return torch.distributions.Normal(mean, std).log_prob(next_state).sum(dim=-1)
```

### 3.3 混合动态模型

**定义**：结合确定性预测和随机性建模，同时输出确定性成分和随机成分。

**适用场景**：
- 需要同时建模系统的确定性趋势和随机波动
- 复杂系统的精确建模

```python
class HybridDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, latent_dim=64):
        super().__init__()
        
        # 确定性特征提取
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 确定性预测头
        self.deterministic_head = nn.Linear(hidden_dim, state_dim)
        
        # 随机预测头
        self.stochastic_mean = nn.Linear(hidden_dim, latent_dim)
        self.stochastic_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 随机状态解码器
        self.stochastic_decoder = nn.Linear(latent_dim, state_dim)
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, state, action, return_distribution=False):
        """混合预测"""
        combined = torch.cat([state, action], dim=-1)
        features = self.feature_net(combined)
        
        # 确定性预测
        det_pred = self.deterministic_head(features)
        
        # 随机预测
        stochastic_mean = self.stochastic_mean(features)
        stochastic_logvar = self.stochastic_logvar(features)
        z = self.reparameterize(stochastic_mean, stochastic_logvar)
        stochastic_pred = self.stochastic_decoder(z)
        
        # 组合预测
        combined_pred = det_pred + stochastic_pred
        
        if return_distribution:
            return combined_pred, det_pred, stochastic_mean, stochastic_logvar
        else:
            return combined_pred
    
    def get_uncertainty(self, state, action, num_samples=100):
        """估计预测不确定性"""
        predictions = []
        for _ in range(num_samples):
            pred = self.forward(state, action)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
```

### 3.4 基于能量的动态模型

**定义**：使用能量函数来建模状态转移的概率分布。

```python
class EnergyBasedDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.energy_net = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def energy(self, state, action, next_state):
        """计算能量值"""
        combined = torch.cat([state, action, next_state], dim=-1)
        return self.energy_net(combined)
    
    def log_prob(self, state, action, next_state):
        """计算对数概率（基于能量的模型）"""
        return -self.energy(state, action, next_state)
    
    def sample(self, state, action, num_samples=1, temperature=1.0):
        """使用MCMC采样"""
        samples = []
        
        for _ in range(num_samples):
            # 从高斯分布初始化
            sample = torch.randn_like(state)
            
            # Langevin采样
            for _ in range(10):
                sample.requires_grad_(True)
                energy = self.energy(state, action, sample)
                grad = torch.autograd.grad(energy.sum(), sample)[0]
                sample = sample.detach() + temperature * grad + torch.randn_like(sample) * 0.01
            
            samples.append(sample)
        
        return torch.stack(samples, dim=0)
```

---

## 4. 动态模型训练方法

### 4.1 监督学习训练

**方法**：使用收集的经验数据进行训练，最小化预测误差。

```python
class SupervisedDynamicsTrainer:
    def __init__(self, dynamics_model, lr=1e-3, weight_decay=1e-5):
        self.model = dynamics_model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.loss_fn = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000
        )
    
    def train_step(self, states, actions, next_states):
        """
        单步训练
        
        参数:
            states: 状态序列 [batch, seq_len, state_dim]
            actions: 动作序列 [batch, seq_len, action_dim]
            next_states: 下一状态序列 [batch, seq_len, state_dim]
        
        返回:
            损失值
        """
        # 展平序列
        states_flat = states.view(-1, states.size(-1))
        actions_flat = actions.view(-1, actions.size(-1))
        next_states_flat = next_states.view(-1, next_states.size(-1))
        
        # 预测
        if hasattr(self.model, 'sample'):
            # 随机模型
            next_states_pred = self.model.sample(states_flat, actions_flat)
        else:
            # 确定性模型
            next_states_pred = self.model(states_flat, actions_flat)
        
        # 计算损失
        loss = self.loss_fn(next_states_pred, next_states_flat)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, device='cpu'):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for states, actions, next_states in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            
            loss = self.train_step(states, actions, next_states)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches
```

### 4.2 重建损失训练

**方法**：通过重建观测来辅助动态模型学习，同时优化编码器和动态模型。

```python
class ReconstructionDynamicsTrainer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def train_step(self, obs, actions, next_obs, alpha=1.0, beta=0.1):
        """
        单步训练
        
        参数:
            obs: 当前观测 [batch, obs_dim]
            actions: 动作 [batch, action_dim]
            next_obs: 下一观测 [batch, obs_dim]
            alpha: 重建损失权重
            beta: KL散度权重
        """
        self.optimizer.zero_grad()
        
        # 前向传播
        obs_recon, state, next_state = self.model(obs, actions)
        
        # 重建损失
        recon_loss = nn.functional.mse_loss(obs_recon, next_obs)
        
        # 如果是变分模型，添加KL散度
        kl_loss = 0.0
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'kl_loss'):
            kl_loss = self.model.encoder.kl_loss()
        
        # 总损失
        loss = alpha * recon_loss + beta * kl_loss
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item() if hasattr(self.model, 'encoder') else 0.0
        }
```

### 4.3 对比学习训练

**方法**：使用对比学习来学习状态表示，通过最大化正样本对的相似度，最小化负样本对的相似度。

```python
class ContrastiveDynamicsTrainer:
    def __init__(self, model, lr=1e-3, temperature=0.1):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.temperature = temperature
    
    def info_nce_loss(self, z1, z2, negatives):
        """
        计算InfoNCE损失
        
        参数:
            z1: 锚点特征 [batch, dim]
            z2: 正样本特征 [batch, dim]
            negatives: 负样本特征 [batch, num_neg, dim]
        """
        # 归一化特征
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        
        # 计算相似度
        pos_sim = torch.sum(z1 * z2, dim=-1, keepdim=True)  # [batch, 1]
        neg_sim = torch.bmm(z1.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1)  # [batch, num_neg]
        
        # 合并相似度
        logits = torch.cat([pos_sim, neg_sim], dim=-1) / self.temperature
        
        # 计算交叉熵损失
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=z1.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def train_step(self, obs1, actions, obs2, negative_obs):
        """
        单步训练
        
        参数:
            obs1: 当前观测 [batch, obs_dim]
            actions: 动作 [batch, action_dim]
            obs2: 下一观测（正样本）[batch, obs_dim]
            negative_obs: 负样本观测 [batch, num_neg, obs_dim]
        """
        self.optimizer.zero_grad()
        
        # 编码观测
        z1 = self.model.encoder(obs1)
        z2 = self.model.encoder(obs2)
        z_neg = self.model.encoder(negative_obs.view(-1, negative_obs.size(-1)))
        z_neg = z_neg.view(negative_obs.size(0), negative_obs.size(1), -1)
        
        # 动态预测
        z1_next = self.model.dynamics(torch.cat([z1, actions], dim=-1))
        
        # 计算对比损失
        loss = self.info_nce_loss(z1_next, z2, z_neg)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### 4.4 强化学习训练

**方法**：结合强化学习来训练动态模型，通过与环境交互来优化模型。

```python
class RLBasedDynamicsTrainer:
    def __init__(self, model, env, lr=1e-3, gamma=0.99):
        self.model = model
        self.env = env
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
    
    def collect_trajectories(self, policy, num_trajectories=10, max_steps=100):
        """收集轨迹数据"""
        trajectories = []
        
        for _ in range(num_trajectories):
            obs = self.env.reset()
            trajectory = []
            
            for _ in range(max_steps):
                action = policy(obs)
                next_obs, reward, done, _ = self.env.step(action)
                trajectory.append((obs, action, next_obs, reward, done))
                
                obs = next_obs
                if done:
                    break
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def train_step(self, trajectories):
        """
        单步训练，使用RL损失
        """
        self.optimizer.zero_grad()
        total_loss = 0.0
        
        for trajectory in trajectories:
            for obs, action, next_obs, reward, done in trajectory:
                # 转换为张量
                obs = torch.tensor(obs, dtype=torch.float32)
                action = torch.tensor(action, dtype=torch.float32)
                next_obs = torch.tensor(next_obs, dtype=torch.float32)
                
                # 预测下一观测
                pred_next_obs = self.model(obs, action)
                
                # 计算预测误差
                error = torch.norm(pred_next_obs - next_obs)
                
                # 使用奖励加权的损失
                loss = (1 - self.gamma) * error + self.gamma * reward
                
                total_loss += loss
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
```

---

## 5. 动态模型高级架构

### 5.1 分层动态模型

**定义**：将状态空间分解为不同层次的抽象，每层学习不同时间尺度的动态。

```python
class HierarchicalDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_levels=3):
        super().__init__()
        
        self.num_levels = num_levels
        self.level_dynamics = nn.ModuleList()
        
        for level in range(num_levels):
            level_dim = state_dim // (2 ** level)
            self.level_dynamics.append(
                nn.Sequential(
                    nn.Linear(level_dim + action_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, level_dim)
                )
            )
        
        # 跨层连接
        self.level_connections = nn.ModuleList()
        for level in range(num_levels - 1):
            in_dim = state_dim // (2 ** level)
            out_dim = state_dim // (2 ** (level + 1))
            self.level_connections.append(
                nn.Linear(in_dim, out_dim)
            )
    
    def forward(self, state, action, level=0):
        """
        分层前向传播
        
        参数:
            state: 当前状态
            action: 动作
            level: 层级（0为最高层）
        """
        # 提取当前层级的状态
        level_dim = state.size(-1) // (2 ** level)
        level_state = state[..., :level_dim]
        
        # 层级内动态
        next_level_state = self.level_dynamics[level](
            torch.cat([level_state, action], dim=-1)
        )
        
        # 跨层传播
        if level < self.num_levels - 1:
            next_state = self.level_connections[level](next_level_state)
            next_state = self.forward(next_state, action, level + 1)
        else:
            next_state = next_level_state
        
        return next_state
```

### 5.2 注意力动态模型

**定义**：使用注意力机制来建模状态转移，允许模型关注状态的不同部分。

```python
class AttentionDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=state_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(state_dim)
        self.norm2 = nn.LayerNorm(state_dim)
    
    def forward(self, state, action):
        """
        注意力动态模型前向传播
        
        参数:
            state: 当前状态 [batch, state_dim]
            action: 动作 [batch, action_dim]
        """
        # 添加序列维度
        state_seq = state.unsqueeze(1)
        
        # 自注意力
        attn_output, _ = self.attention(state_seq, state_seq, state_seq)
        attn_output = self.norm1(state_seq + attn_output)
        
        # 提取处理后的状态
        attended_state = attn_output.squeeze(1)
        
        # 前馈网络
        combined = torch.cat([attended_state, action], dim=-1)
        next_state = self.feed_forward(combined)
        next_state = self.norm2(attended_state + next_state)
        
        return next_state
```

### 5.3 图动态模型

**定义**：使用图神经网络来建模结构化状态的动态转移。

```python
class GraphDynamics(nn.Module):
    def __init__(self, node_dim, edge_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # 消息传递网络
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # 节点更新网络
        self.update_net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, nodes, edges, edge_index, action):
        """
        图动态模型前向传播
        
        参数:
            nodes: 节点特征 [num_nodes, node_dim]
            edges: 边特征 [num_edges, edge_dim]
            edge_index: 边索引 [2, num_edges]
            action: 动作 [action_dim]
        """
        # 消息传递
        src_idx, dst_idx = edge_index
        src_nodes = nodes[src_idx]
        dst_nodes = nodes[dst_idx]
        
        # 构造消息
        messages = torch.cat([
            src_nodes, 
            dst_nodes, 
            edges, 
            action.repeat(len(edges), 1)
        ], dim=-1)
        
        messages = self.message_net(messages)
        
        # 聚合消息
        aggregated = torch.zeros_like(nodes)
        aggregated.index_add_(0, dst_idx, messages)
        
        # 更新节点
        updated_nodes = self.update_net(torch.cat([nodes, aggregated], dim=-1))
        
        return updated_nodes
```

---

## 6. 动态模型优化策略

### 6.1 学习率调度

```python
class LearningRateScheduler:
    def __init__(self, optimizer, warmup_steps=100, max_lr=1e-3, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # 预热阶段：线性增长
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 (self.step_count / self.warmup_steps)
        else:
            # 余弦退火
            progress = (self.step_count - self.warmup_steps) / 10000
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
```

### 6.2 正则化技术

```python
class RegularizedDynamicsTrainer:
    def __init__(self, model, lr=1e-3, weight_decay=1e-5, dropout_rate=0.1):
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.dropout = nn.Dropout(dropout_rate)
    
    def train_step(self, states, actions, next_states):
        self.optimizer.zero_grad()
        
        # 前向传播（带dropout）
        self.model.train()
        combined = torch.cat([states, actions], dim=-1)
        combined = self.dropout(combined)
        
        next_states_pred = self.model(states, actions)
        
        # MSE损失
        mse_loss = nn.functional.mse_loss(next_states_pred, next_states)
        
        # 权重正则化
        reg_loss = sum(p.pow(2).sum() for p in self.model.parameters())
        
        # 总损失
        loss = mse_loss + 1e-4 * reg_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### 6.3 数据增强

```python
class DynamicsDataAugmenter:
    def __init__(self, noise_std=0.01, shift_range=0.1, scale_range=0.1):
        self.noise_std = noise_std
        self.shift_range = shift_range
        self.scale_range = scale_range
    
    def add_noise(self, tensor):
        """添加高斯噪声"""
        noise = torch.randn_like(tensor) * self.noise_std
        return tensor + noise
    
    def random_shift(self, tensor):
        """随机平移"""
        shift = (torch.rand_like(tensor) - 0.5) * 2 * self.shift_range
        return tensor + shift
    
    def random_scale(self, tensor):
        """随机缩放"""
        scale = 1 + (torch.rand_like(tensor) - 0.5) * 2 * self.scale_range
        return tensor * scale
    
    def augment(self, states, actions, next_states):
        """应用数据增强"""
        states = self.add_noise(states)
        actions = self.add_noise(actions)
        next_states = self.add_noise(next_states)
        
        # 随机选择增强方式
        if torch.rand(1) < 0.3:
            states = self.random_shift(states)
        if torch.rand(1) < 0.3:
            actions = self.random_scale(actions)
        
        return states, actions, next_states
```

---

## 7. 动态模型评估指标

### 7.1 预测准确性指标

```python
class DynamicsEvaluator:
    def __init__(self, model):
        self.model = model
    
    def mse_error(self, states, actions, next_states):
        """计算均方误差"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(states, actions)
            mse = nn.functional.mse_loss(predictions, next_states).item()
        return mse
    
    def rmse_error(self, states, actions, next_states):
        """计算均方根误差"""
        mse = self.mse_error(states, actions, next_states)
        return math.sqrt(mse)
    
    def mae_error(self, states, actions, next_states):
        """计算平均绝对误差"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(states, actions)
            mae = nn.functional.l1_loss(predictions, next_states).item()
        return mae
    
    def relative_error(self, states, actions, next_states):
        """计算相对误差"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(states, actions)
            abs_error = torch.abs(predictions - next_states)
            relative_error = torch.mean(abs_error / (torch.abs(next_states) + 1e-8))
        return relative_error.item()
    
    def evaluate(self, dataloader, device='cpu'):
        """综合评估"""
        metrics = {
            'mse': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'relative_error': 0.0
        }
        
        num_batches = 0
        
        for states, actions, next_states in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            
            metrics['mse'] += self.mse_error(states, actions, next_states)
            metrics['rmse'] += self.rmse_error(states, actions, next_states)
            metrics['mae'] += self.mae_error(states, actions, next_states)
            metrics['relative_error'] += self.relative_error(states, actions, next_states)
            
            num_batches += 1
        
        # 平均
        for key in metrics:
            metrics[key] /= num_batches
        
        return metrics
```

### 7.2 长期预测评估

```python
class LongHorizonEvaluator:
    def __init__(self, model, horizons=[1, 5, 10, 20, 50, 100]):
        self.model = model
        self.horizons = horizons
    
    def evaluate_long_horizon(self, initial_states, actions_sequence):
        """
        评估长期预测性能
        
        参数:
            initial_states: 初始状态 [batch, state_dim]
            actions_sequence: 动作序列 [batch, seq_len, action_dim]
        
        返回:
            各预测步数的误差
        """
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            current_states = initial_states
            predictions = [current_states]
            
            for t in range(actions_sequence.size(1)):
                action = actions_sequence[:, t]
                current_states = self.model(current_states, action)
                predictions.append(current_states)
            
            predictions = torch.stack(predictions, dim=1)
        
        for horizon in self.horizons:
            if horizon <= actions_sequence.size(1):
                # 计算该预测步数的误差
                pred = predictions[:, horizon]
                # 假设真实状态序列已知
                results[f'mse_h{horizon}'] = nn.functional.mse_loss(
                    pred, 
                    torch.randn_like(pred)  # 这里应该是真实状态
                ).item()
        
        return results
```

---

## 8. 工程实践与部署

### 8.1 模型压缩

```python
class DynamicsModelCompressor:
    def __init__(self, model):
        self.model = model
    
    def prune_weights(self, sparsity=0.5):
        """权重剪枝"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # 计算阈值
                threshold = torch.quantile(torch.abs(param.data), sparsity)
                # 剪枝
                param.data[torch.abs(param.data) < threshold] = 0.0
    
    def quantize_weights(self, bits=8):
        """权重量化"""
        scale = 2 ** bits - 1
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # 归一化到[-1, 1]
                max_val = torch.max(torch.abs(param.data))
                normalized = param.data / max_val
                # 量化
                quantized = torch.round(normalized * scale) / scale
                # 反归一化
                param.data = quantized * max_val
    
    def distill(self, student_model, dataloader, temperature=1.0, alpha=0.9):
        """知识蒸馏"""
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        
        self.model.eval()
        student_model.train()
        
        for states, actions, next_states in dataloader:
            optimizer.zero_grad()
            
            # 教师预测
            with torch.no_grad():
                teacher_pred = self.model(states, actions)
            
            # 学生预测
            student_pred = student_model(states, actions)
            
            # 蒸馏损失
            soft_target = torch.softmax(teacher_pred / temperature, dim=-1)
            soft_pred = torch.log_softmax(student_pred / temperature, dim=-1)
            distillation_loss = ce_loss(soft_pred, soft_target)
            
            # 硬目标损失
            hard_loss = mse_loss(student_pred, next_states)
            
            # 总损失
            loss = alpha * distillation_loss + (1 - alpha) * hard_loss
            
            loss.backward()
            optimizer.step()
        
        return student_model
```

### 8.2 分布式训练

```python
class DistributedDynamicsTrainer:
    def __init__(self, model, lr=1e-3):
        # 设置分布式
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        
        # 模型并行化
        self.model = torch.nn.parallel.DistributedDataParallel(model)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def train_step(self, states, actions, next_states):
        """分布式训练步骤"""
        # 同步数据
        states = states.to(f'cuda:{self.rank}')
        actions = actions.to(f'cuda:{self.rank}')
        next_states = next_states.to(f'cuda:{self.rank}')
        
        # 前向传播
        predictions = self.model(states, actions)
        
        # 计算损失
        loss = nn.functional.mse_loss(predictions, next_states)
        
        # 反向传播（自动处理梯度平均）
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### 8.3 模型部署

```python
class DynamicsModelDeployer:
    def __init__(self, model):
        self.model = model
    
    def export_to_onnx(self, input_shape, output_path):
        """导出为ONNX格式"""
        dummy_state = torch.randn(*input_shape)
        dummy_action = torch.randn(input_shape[0], 4)  # 假设action_dim=4
        
        torch.onnx.export(
            self.model,
            (dummy_state, dummy_action),
            output_path,
            opset_version=11,
            input_names=['state', 'action'],
            output_names=['next_state'],
            dynamic_axes={
                'state': {0: 'batch_size'},
                'action': {0: 'batch_size'},
                'next_state': {0: 'batch_size'}
            }
        )
    
    def optimize_for_inference(self):
        """优化推理性能"""
        self.model.eval()
        
        # 融合层
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        
        return self.model
    
    def create_triton_model(self, model_name, model_version=1):
        """创建Triton推理服务器模型配置"""
        config = {
            'name': model_name,
            'platform': 'pytorch_libtorch',
            'backend': 'pytorch',
            'max_batch_size': 64,
            'input': [
                {'name': 'state', 'data_type': 'FP32', 'dims': [-1, 32]},
                {'name': 'action', 'data_type': 'FP32', 'dims': [-1, 4]}
            ],
            'output': [
                {'name': 'next_state', 'data_type': 'FP32', 'dims': [-1, 32]}
            ]
        }
        
        return config
```

---

## 9. 理论基础与分析

### 9.1 动态模型的表达能力

**通用逼近定理**：具有足够隐藏层的神经网络可以逼近任何连续函数。对于动态模型：

$$\forall \epsilon > 0, \exists f_{\theta} \text{ s.t. } \|f_{\theta}(s, a) - f_{true}(s, a)\| < \epsilon$$

### 9.2 稳定性分析

```python
class StabilityAnalyzer:
    def __init__(self, model):
        self.model = model
    
    def compute_jacobian(self, state, action):
        """计算雅可比矩阵"""
        state.requires_grad_(True)
        action.requires_grad_(True)
        
        next_state = self.model(state, action)
        
        jacobian = []
        for i in range(next_state.size(-1)):
            grads = torch.autograd.grad(
                next_state[0, i], 
                [state, action],
                retain_graph=True
            )
            jacobian.append(torch.cat([grads[0].flatten(), grads[1].flatten()]))
        
        return torch.stack(jacobian)
    
    def spectral_radius(self, state, action):
        """计算谱半径（稳定性指标）"""
        jacobian = self.compute_jacobian(state, action)
        eigenvalues = torch.linalg.eigvals(jacobian)
        spectral_radius = torch.max(torch.abs(eigenvalues))
        
        return spectral_radius.item()
    
    def is_stable(self, state, action, threshold=1.0):
        """判断动态系统是否稳定"""
        sr = self.spectral_radius(state, action)
        return sr < threshold
```

### 9.3 误差传播分析

```python
class ErrorPropagationAnalyzer:
    def __init__(self, model):
        self.model = model
    
    def predict_with_error(self, initial_state, actions, initial_error=0.0):
        """
        预测误差传播
        
        参数:
            initial_state: 初始状态
            actions: 动作序列
            initial_error: 初始误差
        
        返回:
            误差序列
        """
        errors = [initial_error]
        current_state = initial_state
        current_error = initial_error
        
        for t in range(actions.size(0)):
            # 预测下一状态
            next_state = self.model(current_state, actions[t])
            
            # 计算雅可比矩阵范数（假设为常数）
            jacobian_norm = 0.8  # 简化假设
            
            # 误差传播
            current_error = jacobian_norm * current_error + 0.1  # 过程噪声
            
            errors.append(current_error)
            current_state = next_state
        
        return errors
```

---

## 10. 前沿研究方向

### 10.1 持续学习动态模型

```python
class ContinualLearningDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_tasks=10):
        super().__init__()
        
        # 共享特征提取器
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 任务特定头部
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, state_dim) for _ in range(num_tasks)
        ])
        
        # 弹性权重整合参数
        self.ewc_lambda = 1.0
        self.fisher_diagonals = None
    
    def forward(self, state, action, task_id=0):
        """前向传播"""
        features = self.shared_backbone(torch.cat([state, action], dim=-1))
        return self.task_heads[task_id](features)
    
    def compute_fisher_information(self, dataloader, task_id):
        """计算Fisher信息矩阵"""
        self.eval()
        fisher_diagonal = {}
        
        for param_name, param in self.named_parameters():
            fisher_diagonal[param_name] = torch.zeros_like(param)
        
        for states, actions, next_states in dataloader:
            self.zero_grad()
            predictions = self.forward(states, actions, task_id)
            loss = nn.functional.mse_loss(predictions, next_states)
            loss.backward()
            
            for param_name, param in self.named_parameters():
                if param.grad is not None:
                    fisher_diagonal[param_name] += param.grad.pow(2)
        
        # 归一化
        num_samples = len(dataloader.dataset)
        for key in fisher_diagonal:
            fisher_diagonal[key] /= num_samples
        
        self.fisher_diagonals = fisher_diagonal
    
    def ewc_loss(self, states, actions, next_states, task_id):
        """计算EWC损失"""
        # 标准损失
        predictions = self.forward(states, actions, task_id)
        mse_loss = nn.functional.mse_loss(predictions, next_states)
        
        # EWC正则化项
        ewc_loss = 0.0
        if self.fisher_diagonals is not None:
            for param_name, param in self.named_parameters():
                if param_name in self.fisher_diagonals:
                    ewc_loss += (self.fisher_diagonals[param_name] * 
                                (param - param.detach()).pow(2)).sum()
        
        return mse_loss + self.ewc_lambda * ewc_loss
```

### 10.2 元学习动态模型

```python
class MetaLearningDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # 元学习器
        self.meta_learner = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 动态模型参数生成器
        self.param_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * (state_dim + action_dim))
        )
    
    def generate_dynamics_params(self, context):
        """根据上下文生成动态模型参数"""
        features = self.meta_learner(context)
        params = self.param_generator(features)
        
        # 分解参数
        weight_dim = state_dim * (state_dim + action_dim)
        bias_dim = state_dim
        
        weight = params[:, :weight_dim].view(-1, state_dim, state_dim + action_dim)
        bias = params[:, weight_dim:weight_dim + bias_dim]
        
        return weight, bias
    
    def forward(self, state, action, context):
        """
        元学习前向传播
        
        参数:
            state: 当前状态
            action: 动作
            context: 上下文（任务信息）
        """
        # 生成动态模型参数
        weight, bias = self.generate_dynamics_params(context)
        
        # 使用生成的参数进行预测
        combined = torch.cat([state, action], dim=-1)
        next_state = torch.einsum('bni,bij->bnj', combined, weight) + bias
        
        return next_state
```

### 10.3 因果动态模型

```python
class CausalDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # 因果编码器
        self.causal_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 干预预测器
        self.intervention_predictor = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 因果图注意力
        self.causal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, state, action, intervention_mask=None):
        """
        因果动态模型前向传播
        
        参数:
            state: 当前状态
            action: 动作
            intervention_mask: 干预掩码（可选）
        """
        # 编码状态
        encoded_state = self.causal_encoder(state)
        
        # 应用因果注意力
        state_seq = encoded_state.unsqueeze(1)
        attended_state, _ = self.causal_attention(state_seq, state_seq, state_seq)
        attended_state = attended_state.squeeze(1)
        
        # 如果有干预，应用干预掩码
        if intervention_mask is not None:
            attended_state = attended_state * intervention_mask
        
        # 预测下一状态
        combined = torch.cat([attended_state, action], dim=-1)
        next_state = self.intervention_predictor(combined)
        
        return next_state
    
    def counterfactual_prediction(self, state, action, intervention):
        """
        反事实预测
        
        参数:
            state: 当前状态
            action: 实际动作
            intervention: 干预（假设的动作）
        """
        # 实际预测
        actual_pred = self.forward(state, action)
        
        # 反事实预测
        counterfactual_pred = self.forward(state, intervention)
        
        return actual_pred, counterfactual_pred
```

---

## 11. 实践练习

### 练习1：实现标准的动态模型

```python
class StandardDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        layers = []
        input_dim = state_dim + action_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, state, action):
        """
        预测下一状态
        
        参数:
            state: 当前状态
            action: 执行的动作
        
        返回:
            预测的下一状态
        """
        combined = torch.cat([state, action], dim=-1)
        return self.network(combined)
    
    def train_step(self, states, actions, next_states, optimizer=None):
        """单步训练"""
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        loss_fn = nn.MSELoss()
        
        # 前向传播
        next_states_pred = self.forward(states, actions)
        
        # 计算损失
        loss = loss_fn(next_states_pred, next_states)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()

# 测试
model = StandardDynamicsModel(state_dim=4, action_dim=2, hidden_dim=128)
states = torch.randn(32, 4)
actions = torch.randn(32, 2)
next_states = torch.randn(32, 4)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = model.train_step(states, actions, next_states, optimizer)
print(f"训练损失: {loss:.4f}")
```

### 练习2：实现变分动态模型

```python
class VariationalDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=16, hidden_dim=256):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # 动态模型
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, state, action):
        """
        前向传播
        
        返回:
            重建的状态, 均值, 方差
        """
        # 编码当前状态
        h = self.encoder(state)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        
        # 动态预测
        h_next = self.dynamics(torch.cat([z, action], dim=-1))
        mu_next, logvar_next = h_next.chunk(2, dim=-1)
        z_next = self.reparameterize(mu_next, logvar_next)
        
        # 解码
        state_recon = self.decoder(z_next)
        
        return state_recon, mu_next, logvar_next
    
    def elbo_loss(self, state, action, next_state, beta=1.0):
        """
        计算ELBO损失
        """
        recon, mu, logvar = self.forward(state, action)
        
        # 重建损失
        recon_loss = nn.functional.mse_loss(recon, next_state)
        
        # KL散度损失
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kl_loss

# 测试
model = VariationalDynamicsModel(state_dim=4, action_dim=2, latent_dim=8)
state = torch.randn(32, 4)
action = torch.randn(32, 2)
next_state = torch.randn(32, 4)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = model.elbo_loss(state, action, next_state)
print(f"ELBO损失: {loss.item():.4f}")
```

### 练习3：动态模型规划

```python
class ModelPredictiveController:
    def __init__(self, dynamics_model, planner, num_steps=10):
        """
        模型预测控制器
        
        参数:
            dynamics_model: 动态模型
            planner: 规划器
            num_steps: 预测步数
        """
        self.dynamics = dynamics_model
        self.planner = planner
        self.num_steps = num_steps
    
    def select_action(self, state, goal_state):
        """
        选择动作
        
        参数:
            state: 当前状态
            goal_state: 目标状态
        
        返回:
            选择的动作
        """
        # 使用规划器选择动作序列
        action_sequence = self.planner(
            self.dynamics,
            state,
            goal_state,
            self.num_steps
        )
        
        # 返回第一个动作
        return action_sequence[0] if len(action_sequence) > 0 else None


def simple_planner(dynamics, state, goal, horizon):
    """
    简单的规划器：直接朝目标方向移动
    
    参数:
        dynamics: 动态模型
        state: 当前状态
        goal: 目标状态
        horizon: 规划步数
    
    返回:
        动作序列
    """
    actions = []
    current_state = state.clone()
    
    for _ in range(horizon):
        # 计算到目标的方向
        direction = goal - current_state
        
        # 简单的动作：朝目标方向移动
        action = direction / (direction.norm() + 1e-8) * 0.1
        
        # 预测下一状态
        with torch.no_grad():
            current_state = dynamics(current_state, action.unsqueeze(0)).squeeze(0)
        
        actions.append(action)
        
        # 如果接近目标，停止
        if (goal - current_state).norm() < 0.1:
            break
    
    return actions


def random_shooting_planner(dynamics, state, goal, horizon, num_candidates=100):
    """
    随机射击规划器：采样多个动作序列，选择最优的
    
    参数:
        dynamics: 动态模型
        state: 当前状态
        goal: 目标状态
        horizon: 规划步数
        num_candidates: 候选动作序列数量
    
    返回:
        最优动作序列
    """
    best_actions = None
    best_cost = float('inf')
    
    for _ in range(num_candidates):
        actions = []
        current_state = state.clone()
        total_cost = 0.0
        
        for _ in range(horizon):
            # 随机采样动作
            action = torch.randn_like(state) * 0.1
            actions.append(action)
            
            # 预测下一状态
            with torch.no_grad():
                current_state = dynamics(current_state, action.unsqueeze(0)).squeeze(0)
            
            # 累计成本
            total_cost += (goal - current_state).norm()
        
        # 更新最优序列
        if total_cost < best_cost:
            best_cost = total_cost
            best_actions = actions
    
    return best_actions


# 测试
dynamics = StandardDynamicsModel(state_dim=4, action_dim=2)
mpc_simple = ModelPredictiveController(dynamics, simple_planner, num_steps=10)
mpc_shooting = ModelPredictiveController(dynamics, random_shooting_planner, num_steps=10)

state = torch.randn(4)
goal = torch.randn(4)

action_simple = mpc_simple.select_action(state, goal)
action_shooting = mpc_shooting.select_action(state, goal)

print(f"简单规划器选择动作: {action_simple}")
print(f"随机射击规划器选择动作: {action_shooting}")
```

### 练习4：实现基于Transformer的动态模型

```python
class TransformerDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3, num_heads=4):
        super().__init__()
        
        # 状态嵌入
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        
        # 动作嵌入
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, states, actions):
        """
        Transformer动态模型前向传播
        
        参数:
            states: 状态序列 [batch, seq_len, state_dim]
            actions: 动作序列 [batch, seq_len, action_dim]
        
        返回:
            下一状态序列 [batch, seq_len, state_dim]
        """
        # 嵌入
        state_emb = self.state_embedding(states)  # [batch, seq_len, hidden_dim]
        action_emb = self.action_embedding(actions)  # [batch, seq_len, hidden_dim]
        
        # 合并嵌入
        combined = state_emb + action_emb  # [batch, seq_len, hidden_dim]
        
        # Transformer处理
        output = self.transformer(combined)  # [batch, seq_len, hidden_dim]
        
        # 预测下一状态
        next_states = self.output_layer(output)  # [batch, seq_len, state_dim]
        
        return next_states
    
    def predict_single(self, state, action):
        """
        预测单个状态转移
        
        参数:
            state: 当前状态 [batch, state_dim]
            action: 动作 [batch, action_dim]
        
        返回:
            下一状态 [batch, state_dim]
        """
        # 添加序列维度
        states = state.unsqueeze(1)
        actions = action.unsqueeze(1)
        
        # 预测
        next_states = self.forward(states, actions)
        
        return next_states.squeeze(1)


# 测试
model = TransformerDynamics(state_dim=4, action_dim=2, hidden_dim=64, num_layers=2)
state = torch.randn(32, 4)
action = torch.randn(32, 2)

next_state = model.predict_single(state, action)
print(f"输入状态形状: {state.shape}")
print(f"输入动作形状: {action.shape}")
print(f"输出状态形状: {next_state.shape}")

# 序列预测测试
states_seq = torch.randn(32, 10, 4)
actions_seq = torch.randn(32, 10, 2)
next_states_seq = model.forward(states_seq, actions_seq)
print(f"序列输出形状: {next_states_seq.shape}")
```

---

**下一节**：[状态表示学习](03-state-representation.md)

---

## 参考文献

1. Hafner, D., et al. (2020). Learning Latent Dynamics for Planning from Pixels.
2. Khan, A., et al. (2021). Learning Visual Dynamics Models with Object-centric Representations.
3. Watter, M., et al. (2015). Embed to Control: A Locally Linear Latent Dynamics Model.
4. Chua, K., et al. (2018). Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning.
5. Srivastava, A., et al. (2015). Unsupervised Learning of Video Representations Using LSTMs.
6. Oh, J., et al. (2015). Action-Conditional Video Prediction using Deep Networks in Atari Games.
7. Kalchbrenner, N., et al. (2016). Neural Machine Translation in Linear Time.
8. Vaswani, A., et al. (2017). Attention is All You Need.
9. Finn, C., et al. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.
10. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks.
