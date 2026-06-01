# 6.2 动态模型学习

## 目录

- [1. 引言](#1-引言)
- [2. 动态模型概述](#2-动态模型概述)
- [3. 动态模型类型](#3-动态模型类型)
- [4. 动态模型训练](#4-动态模型训练)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 动态模型的重要性

**动态模型**是学习环境转移规律的模型，能够预测在给定动作下环境如何变化。这是世界模型的核心组成部分，对于规划和决策至关重要。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **机器人控制** | 预测机器人动作效果 | 机械臂运动规划 |
| **游戏AI** | 预测游戏状态变化 | 围棋、象棋 |
| **自动驾驶** | 预测交通场景演化 | 轨迹预测 |
| **流体仿真** | 预测流体运动 | 天气预报 |

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

---

## 3. 动态模型类型

### 3.1 确定性动态模型

**定义**：给定状态和动作，输出确定性的下一状态。

```python
class DeterministicDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
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
```

### 3.2 随机动态模型

**定义**：建模状态转移的分布，输出分布参数。

```python
class StochasticDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 输出均值和方差
        self.mean_layer = nn.Linear(hidden_dim, state_dim)
        self.logvar_layer = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, state, action):
        """
        前向传播
        
        返回:
            下一状态的均值和方差
        """
        combined = torch.cat([state, action], dim=-1)
        h = self.network(combined)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        return mean, logvar
    
    def sample(self, state, action):
        """从分布中采样"""
        mean, logvar = self.forward(state, action)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
```

### 3.3 混合动态模型

**定义**：结合确定性预测和随机性建模。

```python
class HybridDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # 确定性特征提取
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # 确定性预测
        self.deterministic_head = nn.Linear(hidden_dim, state_dim)
        # 随机预测
        self.stochastic_mean = nn.Linear(hidden_dim, state_dim)
        self.stochastic_logvar = nn.Linear(hidden_dim, state_dim)
    
    def forward(self, state, action):
        """混合预测"""
        combined = torch.cat([state, action], dim=-1)
        features = self.feature_net(combined)
        
        det_pred = self.deterministic_head(features)
        mean = self.stochastic_mean(features)
        logvar = self.stochastic_logvar(features)
        
        return det_pred, mean, logvar
```

---

## 4. 动态模型训练

### 4.1 监督学习训练

**方法**：使用收集的经验数据进行训练。

```python
class DynamicsTrainer:
    def __init__(self, dynamics_model, lr=1e-3):
        self.model = dynamics_model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
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
        self.optimizer.step()
        
        return loss.item()
```

### 4.2 重建损失

**方法**：重建观测来辅助动态模型学习。

```python
class ReconstructionDynamicsModel(nn.Module):
    def __init__(self, obs_dim, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # 动态模型
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        # 观测编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        # 观测解码器
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def forward(self, obs, action):
        """前向传播"""
        # 编码观测
        state = self.encoder(obs)
        # 预测下一状态
        next_state = self.dynamics(torch.cat([state, action], dim=-1))
        # 重建观测
        obs_recon = self.decoder(next_state)
        return obs_recon, state, next_state
    
    def compute_loss(self, obs, action, next_obs):
        """计算损失"""
        obs_recon, state, next_state = self.forward(obs, action)
        
        # 重建损失
        recon_loss = nn.functional.mse_loss(obs_recon, next_obs)
        
        return recon_loss
```

### 4.3 对比学习训练

**方法**：使用对比学习来学习状态表示。

```python
class ContrastiveDynamicsModel(nn.Module):
    def __init__(self, obs_dim, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.projector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def contrastive_loss(self, obs1, action, obs2):
        """
        计算对比损失
        
        正样本： (obs1, action) -> obs2
        负样本： (obs1, action) -> 其他观测
        """
        # 编码观测
        z1 = self.encoder(obs1)
        z2 = self.encoder(obs2)
        
        # 预测
        z1_next = self.dynamics(torch.cat([z1, action], dim=-1))
        
        # 计算相似度
        pos_sim = F.cosine_similarity(z1_next, z2, dim=-1)
        neg_sim = torch.mm(self.projector(z1_next), self.projector(z2).T)
        
        # InfoNCE损失
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(logits.size(0), dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        
        return loss
```

---

## 5. 实践练习

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
    
    def train_step(self, states, actions, next_states, lr=1e-3):
        """单步训练"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
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
model = StandardDynamicsModel(state_dim=4, action_dim=2)
states = torch.randn(32, 4)
actions = torch.randn(32, 2)
next_states = torch.randn(32, 4)

loss = model.train_step(states, actions, next_states)
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
    
    def elbo_loss(self, state, action, next_state):
        """
        计算ELBO损失
        """
        recon, mu, logvar = self.forward(state, action)
        
        # 重建损失
        recon_loss = nn.functional.mse_loss(recon, next_state)
        
        # KL散度损失
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + kl_loss

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

# 测试
dynamics = StandardDynamicsModel(state_dim=4, action_dim=2)
mpc = ModelPredictiveController(dynamics, simple_planner, num_steps=10)

state = torch.randn(4)
goal = torch.randn(4)
action = mpc.select_action(state, goal)
print(f"选择动作: {action}")
```

---

**下一节**：[状态表示学习](03-state-representation.md)

---

## 参考文献

1. Hafner, D., et al. (2020). Learning Latent Dynamics for Planning from Pixels.
2. Khan, A., et al. (2021). Learning Visual Dynamics Models with Object-centric Representations.
3. Watter, M., et al. (2015). Embed to Control: A Locally Linear Latent Dynamics Model.
