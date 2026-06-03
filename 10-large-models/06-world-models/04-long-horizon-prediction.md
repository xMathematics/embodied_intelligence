# 6.4 长期预测

## 目录

- [1. 引言](#1-引言)
- [2. 长期预测概述](#2-长期预测概述)
- [3. 长期预测方法](#3-长期预测方法)
- [4. 模型架构](#4-模型架构)
- [5. 长期预测评估指标](#5-长期预测评估指标)
- [6. 优化策略](#6-优化策略)
- [7. 工程实践](#7-工程实践)
- [8. 理论基础与分析](#8-理论基础与分析)
- [9. 前沿研究方向](#9-前沿研究方向)
- [10. 实践练习](#10-实践练习)

---

## 1. 引言

### 1.1 长期预测的重要性

**长期预测**是指对环境未来状态进行多步预测的能力。这对于长期规划和决策至关重要，也是评估世界模型质量的重要指标。

在强化学习和机器人领域，长期预测能力直接决定了智能体能否进行有效的规划和决策。一个好的世界模型应该能够准确预测未来几十步甚至上百步的状态演变。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **机器人规划** | 预测长期动作效果 | 复杂任务规划、导航 |
| **游戏AI** | 预测游戏未来状态 | 策略游戏、即时战略 |
| **交通预测** | 预测交通流 | 智慧城市、自动驾驶 |
| **天气预报** | 预测天气变化 | 气象预报、气候模拟 |
| **金融预测** | 预测市场趋势 | 股票价格、经济指标 |
| **健康监测** | 预测生理状态变化 | 疾病预测、健康管理 |

### 1.3 长期预测与短期预测的对比

| 特性 | 短期预测 | 长期预测 |
|------|----------|----------|
| **预测步数** | 1-5步 | 10步以上 |
| **主要挑战** | 建模精度 | 误差累积 |
| **计算复杂度** | 低 | 高 |
| **应用场景** | 实时控制 | 规划决策 |
| **评估重点** | 单步精度 | 累积误差 |

---

## 2. 长期预测概述

### 2.1 定义

**长期预测**：对环境进行多步未来状态预测。

**形式化表达**：
```
ŝ_{t:H} = Model(s_t, a_{t:H-1}; θ)
```

其中：
- $H$：预测 horizon
- $ŝ_{t:H}$：预测的未来状态序列
- $a_{t:H-1}$：动作序列
- $θ$：模型参数

### 2.2 长期预测的挑战

| 挑战 | 描述 | 影响 |
|------|------|------|
| **误差累积** | 单步预测误差会累积放大 | 长期预测迅速偏离真实轨迹 |
| **状态混淆** | 不同状态可能变得相似 | 模型无法区分相似但不同的状态 |
| **长程依赖** | 长期依赖难以捕捉 | 早期信息丢失 |
| **计算复杂度** | 多步预测计算量大 | 实时应用受限 |
| **模型漂移** | 模型预测与真实分布逐渐偏离 | 预测质量随时间下降 |
| **不确定性累积** | 每步的不确定性累积 | 长期预测置信度降低 |

### 2.3 长期预测的关键特性

一个优秀的长期预测模型应该具备以下特性：

1. **稳定性**：预测误差不会随步数呈指数增长
2. **一致性**：预测分布与真实分布保持一致
3. **计算效率**：能够在合理时间内完成长序列预测
4. **不确定性估计**：能够准确估计预测的不确定性
5. **可解释性**：能够解释预测结果的来源

---

## 3. 长期预测方法

### 3.1 开环预测

**方法**：直接多步预测，不使用反馈。

```python
def open_loop_prediction(model, initial_state, actions, horizon):
    """
    开环长期预测
    
    参数:
        model: 动态模型
        initial_state: 初始状态 [batch, state_dim]
        actions: 动作序列 [horizon, batch, action_dim]
        horizon: 预测步数
    
    返回:
        预测的状态序列 [horizon+1, batch, state_dim]
    """
    predictions = [initial_state]
    current_state = initial_state
    
    for t in range(horizon):
        action = actions[t] if t < len(actions) else torch.zeros_like(actions[0])
        next_state = model(current_state, action)
        predictions.append(next_state)
        current_state = next_state
    
    return torch.stack(predictions)

# 示例
class StandardDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        combined = torch.cat([state, action], dim=-1)
        return self.net(combined)

model = StandardDynamicsModel(state_dim=4, action_dim=2)
initial_state = torch.randn(32, 4)
actions = [torch.randn(32, 2) for _ in range(10)]

predictions = open_loop_prediction(model, initial_state, actions, horizon=10)
print(f"预测序列形状: {predictions.shape}")  # [11, 32, 4]
```

### 3.2 闭环预测

**方法**：在预测过程中使用预测状态作为反馈。

```python
def closed_loop_prediction(model, policy, initial_state, horizon):
    """
    闭环长期预测
    
    参数:
        model: 动态模型
        policy: 策略网络
        initial_state: 初始状态 [batch, state_dim]
        horizon: 预测步数
    
    返回:
        预测的状态和动作序列
    """
    states = [initial_state]
    actions = []
    current_state = initial_state
    
    for t in range(horizon):
        # 策略选择动作
        action = policy(current_state)
        actions.append(action)
        
        # 模型预测下一状态
        next_state = model(current_state, action)
        states.append(next_state)
        current_state = next_state
    
    return torch.stack(states), torch.stack(actions)

# 示例策略
class SimplePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)

policy = SimplePolicy(state_dim=4, action_dim=2)
states, actions = closed_loop_prediction(model, policy, initial_state, horizon=10)
print(f"状态序列形状: {states.shape}")  # [11, 32, 4]
print(f"动作序列形状: {actions.shape}")  # [10, 32, 2]
```

### 3.3 集成预测

**方法**：使用多个模型集成预测。

```python
class EnsembleDynamics:
    def __init__(self, models, aggregation='mean'):
        """
        集成动态模型
        
        参数:
            models: 模型列表
            aggregation: 聚合方式 ('mean', 'max', 'min', 'median')
        """
        self.models = models
        self.aggregation = aggregation
    
    def predict(self, state, action):
        """集成预测"""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(state, action)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        if self.aggregation == 'mean':
            return predictions.mean(dim=0)
        elif self.aggregation == 'max':
            return predictions.max(dim=0)[0]
        elif self.aggregation == 'min':
            return predictions.min(dim=0)[0]
        elif self.aggregation == 'median':
            return predictions.median(dim=0)[0]
        else:
            return predictions[0]
    
    def predict_uncertainty(self, state, action):
        """预测不确定性"""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(state, action)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # 计算均值和标准差
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        var = predictions.var(dim=0)
        
        # 计算最大最小范围
        min_pred = predictions.min(dim=0)[0]
        max_pred = predictions.max(dim=0)[0]
        
        return {
            'mean': mean,
            'std': std,
            'var': var,
            'min': min_pred,
            'max': max_pred,
            'ensemble_size': len(self.models)
        }
    
    def predict_with_dropout(self, state, action, num_samples=10):
        """
        使用 dropout 进行蒙特卡洛采样
        
        参数:
            state: 当前状态
            action: 动作
            num_samples: 采样次数
        """
        predictions = []
        for model in self.models:
            model.train()  # 启用 dropout
            for _ in range(num_samples // len(self.models)):
                pred = model(state, action)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        return predictions.mean(dim=0), predictions.std(dim=0)

# 示例
ensemble_models = [StandardDynamicsModel(4, 2) for _ in range(5)]
ensemble = EnsembleDynamics(ensemble_models, aggregation='mean')

state = torch.randn(32, 4)
action = torch.randn(32, 2)
prediction = ensemble.predict(state, action)
uncertainty = ensemble.predict_uncertainty(state, action)
print(f"预测形状: {prediction.shape}")
print(f"不确定性均值形状: {uncertainty['mean'].shape}")
print(f"不确定性标准差形状: {uncertainty['std'].shape}")
```

### 3.4 正则化预测

**方法**：通过正则化减少误差累积。

```python
class RegularizedDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # 标准动态模型
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 正则化网络：预测状态变化
        self.delta_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 状态约束
        self.state_bound = 5.0
    
    def forward(self, state, action):
        """前向传播"""
        combined = torch.cat([state, action], dim=-1)
        
        # 标准预测
        delta = self.delta_net(combined)
        
        # 限制变化幅度（梯度惩罚）
        delta = torch.clamp(delta, min=-1.0, max=1.0)
        
        next_state = state + delta
        
        # 状态边界约束
        next_state = torch.clamp(next_state, min=-self.state_bound, max=self.state_bound)
        
        return next_state
    
    def langevin_dynamics(self, state, action, noise_scale=0.1, steps=10):
        """
        朗之万动力学正则化
        
        参数:
            state: 当前状态
            action: 动作
            noise_scale: 噪声规模
            steps: 采样步数
        """
        current_state = state.clone()
        
        for _ in range(steps):
            # 添加噪声
            noise = torch.randn_like(current_state) * noise_scale
            noisy_state = current_state + noise
            
            # 预测
            next_state = self.forward(noisy_state, action)
            current_state = next_state
        
        return current_state
    
    def constrained_prediction(self, state, action, constraint_fn=None):
        """
        带约束的预测
        
        参数:
            state: 当前状态
            action: 动作
            constraint_fn: 约束函数
        """
        next_state = self.forward(state, action)
        
        # 应用约束
        if constraint_fn is not None:
            next_state = constraint_fn(next_state)
        
        return next_state

# 示例：约束函数
def velocity_constraint(state):
    """限制速度不超过某个阈值"""
    # 假设状态前半部分是位置，后半部分是速度
    velocity = state[:, 2:]
    max_velocity = 2.0
    velocity = torch.clamp(velocity, min=-max_velocity, max=max_velocity)
    new_state = torch.cat([state[:, :2], velocity], dim=-1)
    return new_state

model = RegularizedDynamicsModel(state_dim=4, action_dim=2)
state = torch.randn(32, 4)
action = torch.randn(32, 2)

# 标准预测
pred = model(state, action)
print(f"标准预测形状: {pred.shape}")

# 朗之万动力学预测
langevin_pred = model.langevin_dynamics(state, action)
print(f"朗之万预测形状: {langevin_pred.shape}")

# 带约束预测
constrained_pred = model.constrained_prediction(state, action, velocity_constraint)
print(f"约束预测形状: {constrained_pred.shape}")
```

### 3.5 分层预测

**方法**：在不同时间尺度上进行预测。

```python
class HierarchicalDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, levels=3):
        super().__init__()
        self.levels = levels
        
        # 每层的动态模型
        self.level_models = nn.ModuleList()
        for level in range(levels):
            model = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim)
            )
            self.level_models.append(model)
        
        # 时间步长映射
        self.time_scales = [2 ** level for level in range(levels)]
    
    def predict_at_level(self, state, action, level):
        """
        在指定层级进行预测
        
        参数:
            state: 当前状态
            action: 动作
            level: 层级（0=细粒度，越高越粗粒度）
        """
        model = self.level_models[level]
        combined = torch.cat([state, action], dim=-1)
        return model(combined)
    
    def hierarchical_predict(self, initial_state, actions, horizon):
        """
        分层长期预测
        
        参数:
            initial_state: 初始状态
            actions: 动作序列
            horizon: 预测步数
        
        返回:
            预测的状态序列
        """
        predictions = [initial_state]
        current_state = initial_state
        
        for t in range(horizon):
            # 根据当前步数选择层级
            level = min(self.levels - 1, t // 5)
            
            action = actions[t] if t < len(actions) else torch.zeros_like(actions[0])
            
            # 使用对应层级的模型
            next_state = self.predict_at_level(current_state, action, level)
            predictions.append(next_state)
            current_state = next_state
        
        return torch.stack(predictions)
    
    def coarse_to_fine_prediction(self, initial_state, horizon, num_coarse_steps=5):
        """
        粗到细预测：先进行粗粒度预测，再细化
        
        参数:
            initial_state: 初始状态
            horizon: 总预测步数
            num_coarse_steps: 粗粒度步数
        """
        # 粗粒度预测
        coarse_states = [initial_state]
        current_state = initial_state
        
        # 使用最高层级进行粗粒度预测
        coarse_action = torch.zeros(initial_state.shape[0], 2)  # 零动作
        
        for _ in range(num_coarse_steps):
            next_state = self.predict_at_level(current_state, coarse_action, self.levels - 1)
            coarse_states.append(next_state)
            current_state = next_state
        
        # 细粒度细化
        fine_predictions = [initial_state]
        current_state = initial_state
        
        for t in range(horizon):
            # 找到对应的粗粒度状态
            coarse_idx = min(t // (horizon // num_coarse_steps), num_coarse_steps)
            target_coarse = coarse_states[coarse_idx]
            
            # 使用细粒度模型
            action = torch.zeros(initial_state.shape[0], 2)
            next_state = self.predict_at_level(current_state, action, 0)
            
            # 向粗粒度目标调整
            next_state = 0.9 * next_state + 0.1 * target_coarse
            
            fine_predictions.append(next_state)
            current_state = next_state
        
        return torch.stack(fine_predictions)

# 示例
hierarchical_model = HierarchicalDynamicsModel(state_dim=4, action_dim=2, levels=3)
initial_state = torch.randn(32, 4)
actions = [torch.randn(32, 2) for _ in range(20)]

# 分层预测
predictions = hierarchical_model.hierarchical_predict(initial_state, actions, horizon=20)
print(f"分层预测形状: {predictions.shape}")  # [21, 32, 4]

# 粗到细预测
cf_predictions = hierarchical_model.coarse_to_fine_prediction(initial_state, horizon=20)
print(f"粗到细预测形状: {cf_predictions.shape}")  # [21, 32, 4]
```

---

## 4. 模型架构

### 4.1 循环预测架构

**使用 RNN 进行序列预测**。

```python
class RecurrentDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        # 输入嵌入
        self.input_embedding = nn.Linear(state_dim + action_dim, hidden_dim)
        
        # 动态模型 - 使用 GRU
        self.dynamics = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, state_dim)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, state_sequence, action_sequence):
        """
        序列预测
        
        参数:
            state_sequence: 状态序列 [batch, seq_len, state_dim]
            action_sequence: 动作序列 [batch, seq_len, action_dim]
        
        返回:
            预测的下一状态序列
        """
        # 拼接
        batch_size, seq_len, _ = state_sequence.shape
        combined = torch.cat([state_sequence, action_sequence], dim=-1)
        
        # 嵌入
        embedded = self.input_embedding(combined)
        embedded = self.layer_norm(embedded)
        
        # RNN前向传播
        output, hidden = self.dynamics(embedded)
        
        # 输出预测
        predictions = self.output_layer(output)
        
        return predictions, hidden
    
    def predict_step(self, state, action, hidden=None):
        """
        单步预测
        
        参数:
            state: 当前状态 [batch, state_dim]
            action: 当前动作 [batch, action_dim]
            hidden: 隐藏状态
        
        返回:
            预测状态和新的隐藏状态
        """
        combined = torch.cat([state, action], dim=-1).unsqueeze(1)  # [batch, 1, state_dim + action_dim]
        embedded = self.input_embedding(combined)
        
        if hidden is None:
            output, new_hidden = self.dynamics(embedded)
        else:
            output, new_hidden = self.dynamics(embedded, hidden)
        
        prediction = self.output_layer(output).squeeze(1)
        
        return prediction, new_hidden
    
    def rollout(self, initial_state, actions, initial_hidden=None):
        """
        多步预测
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            actions: 动作序列 [seq_len, batch, action_dim]
            initial_hidden: 初始隐藏状态
        
        返回:
            预测状态序列
        """
        predictions = [initial_state]
        current_state = initial_state
        hidden = initial_hidden
        
        for action in actions:
            next_state, hidden = self.predict_step(current_state, action, hidden)
            predictions.append(next_state)
            current_state = next_state
        
        return torch.stack(predictions)

# 示例
rnn_model = RecurrentDynamicsModel(state_dim=4, action_dim=2, hidden_dim=128)
state_seq = torch.randn(32, 10, 4)
action_seq = torch.randn(32, 10, 2)

predictions, hidden = rnn_model(state_seq, action_seq)
print(f"序列预测形状: {predictions.shape}")  # [32, 10, 4]

# 单步预测
state = torch.randn(32, 4)
action = torch.randn(32, 2)
step_pred, new_hidden = rnn_model.predict_step(state, action)
print(f"单步预测形状: {step_pred.shape}")  # [32, 4]

# 多步预测
actions = [torch.randn(32, 2) for _ in range(15)]
rollout_pred = rnn_model.rollout(state, actions)
print(f"多步预测形状: {rollout_pred.shape}")  # [16, 32, 4]
```

### 4.2 Transformer预测架构

**使用 Transformer 进行长期预测**。

```python
class TransformerDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # 嵌入层
        self.state_embedding = nn.Linear(state_dim, latent_dim)
        self.action_embedding = nn.Linear(action_dim, latent_dim)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, latent_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            batch_first=True,
            dropout=0.1,
            layer_norm_eps=1e-5
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, state_dim)
        )
        
        # 层归一化
        self.final_norm = nn.LayerNorm(state_dim)
    
    def forward(self, state_sequence, action_sequence, mask=None):
        """
        Transformer预测
        
        参数:
            state_sequence: 状态序列 [batch, seq_len, state_dim]
            action_sequence: 动作序列 [batch, seq_len, action_dim]
            mask: 注意力掩码
        
        返回:
            预测的下一状态序列
        """
        batch_size, seq_len, _ = state_sequence.shape
        
        # 嵌入
        state_emb = self.state_embedding(state_sequence)
        action_emb = self.action_embedding(action_sequence)
        combined = state_emb + action_emb
        
        # 添加位置编码
        if seq_len > self.pos_embedding.size(1):
            # 动态位置编码
            pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            pos_emb = self.pos_embedding[:, :seq_len, :]
        combined = combined + pos_emb
        
        # Transformer编码
        encoded = self.transformer(combined, mask=mask)
        
        # 输出预测
        predictions = self.output_layer(encoded)
        predictions = self.final_norm(predictions)
        
        return predictions
    
    def autoregressive_predict(self, initial_state, horizon):
        """
        自回归长期预测
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            horizon: 预测步数
        
        返回:
            预测状态序列 [horizon+1, batch, state_dim]
        """
        self.eval()
        predictions = [initial_state]
        current_state = initial_state
        
        with torch.no_grad():
            for t in range(horizon):
                # 构建输入序列（所有历史状态）
                state_seq = torch.stack(predictions).permute(1, 0, 2)  # [batch, t+1, state_dim]
                
                # 假设零动作
                action_seq = torch.zeros(state_seq.shape[0], state_seq.shape[1], self.action_dim)
                
                # 预测
                pred = self.forward(state_seq, action_seq)
                
                # 取最后一个预测
                next_state = pred[:, -1, :]
                predictions.append(next_state)
        
        return torch.stack(predictions)
    
    def predict_with_future_actions(self, initial_state, actions):
        """
        给定未来动作序列的预测
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            actions: 动作序列 [horizon, batch, action_dim]
        
        返回:
            预测状态序列 [horizon+1, batch, state_dim]
        """
        horizon = len(actions)
        predictions = [initial_state]
        current_state = initial_state
        
        with torch.no_grad():
            for t in range(horizon):
                # 构建输入序列
                state_seq = torch.stack(predictions).permute(1, 0, 2)
                action_seq = torch.stack(actions[:t+1]).permute(1, 0, 2)
                
                # 预测
                pred = self.forward(state_seq, action_seq)
                next_state = pred[:, -1, :]
                
                predictions.append(next_state)
        
        return torch.stack(predictions)

# 示例
transformer_model = TransformerDynamicsModel(
    state_dim=4, 
    action_dim=2, 
    latent_dim=128, 
    num_heads=4, 
    num_layers=3
)

# 序列预测
state_seq = torch.randn(32, 10, 4)
action_seq = torch.randn(32, 10, 2)
predictions = transformer_model(state_seq, action_seq)
print(f"Transformer预测形状: {predictions.shape}")  # [32, 10, 4]

# 自回归预测
initial_state = torch.randn(32, 4)
ar_predictions = transformer_model.autoregressive_predict(initial_state, horizon=15)
print(f"自回归预测形状: {ar_predictions.shape}")  # [16, 32, 4]

# 带动作序列的预测
actions = [torch.randn(32, 2) for _ in range(15)]
action_predictions = transformer_model.predict_with_future_actions(initial_state, actions)
print(f"带动作预测形状: {action_predictions.shape}")  # [16, 32, 4]
```

### 4.3 图神经网络预测

**使用 GNN 建模状态关系**。

```python
class GraphDynamicsModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        
        # 节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 边编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 图卷积层
        self.graph_conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            conv_layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.graph_conv_layers.append(conv_layer)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, node_features, edge_index, edge_features):
        """
        图神经网络预测
        
        参数:
            node_features: 节点特征 [batch, num_nodes, node_dim]
            edge_index: 边索引 [2, num_edges]
            edge_features: 边特征 [num_edges, edge_dim]
        
        返回:
            更新后的节点特征
        """
        batch_size, num_nodes, _ = node_features.shape
        num_edges = edge_index.size(1)
        
        # 编码
        h = self.node_encoder(node_features)  # [batch, num_nodes, hidden_dim]
        h_edge = self.edge_encoder(edge_features)  # [num_edges, hidden_dim]
        
        # 图卷积
        for conv in self.graph_conv_layers:
            # 消息传递
            messages = []
            for i in range(num_edges):
                src_idx = edge_index[0, i]
                dst_idx = edge_index[1, i]
                
                # 获取源节点和目标节点的特征
                src_feature = h[:, src_idx, :]  # [batch, hidden_dim]
                dst_feature = h[:, dst_idx, :]  # [batch, hidden_dim]
                
                # 组合特征
                combined = torch.cat([src_feature, h_edge[i].unsqueeze(0).repeat(batch_size, 1)], dim=-1)
                msg = conv(combined)  # [batch, hidden_dim]
                messages.append(msg)
            
            messages = torch.stack(messages, dim=1)  # [batch, num_edges, hidden_dim]
            
            # 聚合消息到目标节点
            h_new = torch.zeros_like(h)
            for i in range(num_edges):
                dst_idx = edge_index[1, i]
                h_new[:, dst_idx, :] += messages[:, i, :]
            
            # 残差连接
            h = h + h_new
        
        # 输出
        output = self.output_layer(h)
        
        return output
    
    def predict_temporal(self, node_features_seq, edge_index_seq, edge_features_seq):
        """
        时序图预测
        
        参数:
            node_features_seq: 节点特征序列 [batch, seq_len, num_nodes, node_dim]
            edge_index_seq: 边索引序列 [seq_len, 2, num_edges]
            edge_features_seq: 边特征序列 [seq_len, num_edges, edge_dim]
        
        返回:
            预测的节点特征序列
        """
        batch_size, seq_len, num_nodes, _ = node_features_seq.shape
        predictions = []
        
        current_features = node_features_seq[:, 0, :, :]
        predictions.append(current_features)
        
        for t in range(1, seq_len):
            edge_index = edge_index_seq[t]
            edge_features = edge_features_seq[t]
            
            next_features = self.forward(current_features, edge_index, edge_features)
            predictions.append(next_features)
            current_features = next_features
        
        return torch.stack(predictions, dim=1)

# 示例
graph_model = GraphDynamicsModel(node_dim=16, edge_dim=8, hidden_dim=128, num_layers=2)

# 单步图预测
node_features = torch.randn(32, 10, 16)  # [batch, num_nodes, node_dim]
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])  # [2, num_edges]
edge_features = torch.randn(10, 8)  # [num_edges, edge_dim]

predictions = graph_model(node_features, edge_index, edge_features)
print(f"图预测形状: {predictions.shape}")  # [32, 10, 16]

# 时序图预测
node_seq = torch.randn(32, 5, 10, 16)  # [batch, seq_len, num_nodes, node_dim]
edge_index_seq = edge_index.unsqueeze(0).repeat(5, 1, 1)  # [seq_len, 2, num_edges]
edge_features_seq = torch.randn(5, 10, 8)  # [seq_len, num_edges, edge_dim]

temporal_preds = graph_model.predict_temporal(node_seq, edge_index_seq, edge_features_seq)
print(f"时序图预测形状: {temporal_preds.shape}")  # [32, 5, 10, 16]
```

### 4.4 扩散模型预测

**使用扩散模型进行长期预测**。

```python
class DiffusionDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_steps=100):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_steps = num_steps
        
        # 噪声预测网络
        self.noise_net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 预计算扩散参数
        self.beta_schedule = self._cosine_beta_schedule(num_steps)
        self.alpha_schedule = 1.0 - self.beta_schedule
        self.alpha_bar_schedule = torch.cumprod(self.alpha_schedule, dim=0)
    
    def _cosine_beta_schedule(self, num_steps):
        """余弦调度的beta值"""
        steps = torch.arange(num_steps + 1)
        cosine = torch.cos((steps / num_steps + 0.008) / 1.008 * torch.pi / 2) ** 2
        beta = 1 - cosine[1:] / cosine[:-1]
        return torch.clamp(beta, min=0.0001, max=0.9999)
    
    def add_noise(self, state, t):
        """
        向状态添加噪声
        
        参数:
            state: 原始状态
            t: 时间步
        
        返回:
            加噪后的状态和噪声
        """
        alpha_bar = self.alpha_bar_schedule[t]
        noise = torch.randn_like(state)
        noisy_state = torch.sqrt(alpha_bar) * state + torch.sqrt(1 - alpha_bar) * noise
        return noisy_state, noise
    
    def forward(self, state, action, t):
        """
        预测噪声
        
        参数:
            state: 当前状态（可能已加噪）
            action: 动作
            t: 时间步
        
        返回:
            预测的噪声
        """
        # 将时间步嵌入
        t_embed = torch.ones(state.shape[0], 1) * t / self.num_steps
        
        # 组合输入
        combined = torch.cat([state, action, t_embed], dim=-1)
        
        # 预测噪声
        noise_pred = self.noise_net(combined)
        
        return noise_pred
    
    def denoise_step(self, noisy_state, action, t):
        """
        去噪单步
        
        参数:
            noisy_state: 带噪声的状态
            action: 动作
            t: 当前时间步
        
        返回:
            去噪后的状态
        """
        with torch.no_grad():
            # 预测噪声
            noise_pred = self.forward(noisy_state, action, t)
            
            # 获取参数
            alpha_t = self.alpha_schedule[t]
            alpha_bar_t = self.alpha_bar_schedule[t]
            
            # 计算均值
            mean = (noisy_state - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # 添加噪声（除了最后一步）
            if t > 0:
                beta_t = self.beta_schedule[t]
                noise = torch.randn_like(noisy_state)
                mean += torch.sqrt(beta_t) * noise
            
            return mean
    
    def predict(self, initial_state, action, horizon=1):
        """
        使用扩散模型预测
        
        参数:
            initial_state: 初始状态
            action: 动作
            horizon: 预测步数
        
        返回:
            预测的状态
        """
        # 从初始状态加噪到最大噪声水平
        noisy_state = initial_state
        
        for t in range(self.num_steps):
            noisy_state, _ = self.add_noise(noisy_state, t)
        
        # 去噪过程
        for t in reversed(range(self.num_steps)):
            noisy_state = self.denoise_step(noisy_state, action, t)
        
        return noisy_state
    
    def rollout(self, initial_state, actions):
        """
        多步扩散预测
        
        参数:
            initial_state: 初始状态
            actions: 动作序列
        
        返回:
            预测状态序列
        """
        predictions = [initial_state]
        current_state = initial_state
        
        for action in actions:
            next_state = self.predict(current_state, action)
            predictions.append(next_state)
            current_state = next_state
        
        return torch.stack(predictions)

# 示例
diffusion_model = DiffusionDynamicsModel(state_dim=4, action_dim=2, hidden_dim=128, num_steps=50)

# 单步预测
state = torch.randn(32, 4)
action = torch.randn(32, 2)
pred = diffusion_model.predict(state, action)
print(f"扩散预测形状: {pred.shape}")  # [32, 4]

# 多步预测
actions = [torch.randn(32, 2) for _ in range(5)]
rollout = diffusion_model.rollout(state, actions)
print(f"扩散多步预测形状: {rollout.shape}")  # [6, 32, 4]
```

---

## 5. 长期预测评估指标

### 5.1 预测准确性指标

```python
class PredictionMetrics:
    def __init__(self):
        self.metrics = {}
    
    def compute_mse(self, predicted, target):
        """计算均方误差"""
        return nn.functional.mse_loss(predicted, target, reduction='none').mean(dim=-1)
    
    def compute_mae(self, predicted, target):
        """计算平均绝对误差"""
        return nn.functional.l1_loss(predicted, target, reduction='none').mean(dim=-1)
    
    def compute_rmse(self, predicted, target):
        """计算均方根误差"""
        mse = self.compute_mse(predicted, target)
        return torch.sqrt(mse)
    
    def compute_cosine_similarity(self, predicted, target):
        """计算余弦相似度"""
        pred_norm = predicted / predicted.norm(dim=-1, keepdim=True)
        target_norm = target / target.norm(dim=-1, keepdim=True)
        return (pred_norm * target_norm).sum(dim=-1)
    
    def compute_r2_score(self, predicted, target):
        """计算R²分数"""
        ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=-1)
        ss_res = ((target - predicted) ** 2).sum(dim=-1)
        return 1 - (ss_res / ss_tot)
    
    def compute_all_metrics(self, predicted_states, true_states):
        """
        计算所有指标
        
        参数:
            predicted_states: 预测状态序列 [horizon+1, batch, state_dim]
            true_states: 真实状态序列 [horizon+1, batch, state_dim]
        
        返回:
            指标字典
        """
        horizon = predicted_states.size(0) - 1
        
        results = {
            'per_step': [],
            'final': {},
            'average': {},
            'cumulative': {}
        }
        
        # 每步指标
        for t in range(horizon + 1):
            pred = predicted_states[t]
            true = true_states[t]
            
            step_metrics = {
                'mse': self.compute_mse(pred, true).mean().item(),
                'mae': self.compute_mae(pred, true).mean().item(),
                'rmse': self.compute_rmse(pred, true).mean().item(),
                'cosine_sim': self.compute_cosine_similarity(pred, true).mean().item(),
                'r2': self.compute_r2_score(pred, true).mean().item()
            }
            results['per_step'].append(step_metrics)
        
        # 最终步指标
        final_pred = predicted_states[-1]
        final_true = true_states[-1]
        results['final'] = {
            'mse': self.compute_mse(final_pred, final_true).mean().item(),
            'mae': self.compute_mae(final_pred, final_true).mean().item(),
            'rmse': self.compute_rmse(final_pred, final_true).mean().item(),
            'cosine_sim': self.compute_cosine_similarity(final_pred, final_true).mean().item(),
            'r2': self.compute_r2_score(final_pred, final_true).mean().item()
        }
        
        # 平均指标
        mse_vals = torch.tensor([step['mse'] for step in results['per_step']])
        mae_vals = torch.tensor([step['mae'] for step in results['per_step']])
        rmse_vals = torch.tensor([step['rmse'] for step in results['per_step']])
        cosine_vals = torch.tensor([step['cosine_sim'] for step in results['per_step']])
        r2_vals = torch.tensor([step['r2'] for step in results['per_step']])
        
        results['average'] = {
            'mse': mse_vals.mean().item(),
            'mae': mae_vals.mean().item(),
            'rmse': rmse_vals.mean().item(),
            'cosine_sim': cosine_vals.mean().item(),
            'r2': r2_vals.mean().item()
        }
        
        # 累积误差（误差增长速率）
        results['cumulative'] = {
            'error_growth_rate': (mse_vals[-1] / mse_vals[0]).item() if mse_vals[0] > 0 else float('inf'),
            'average_growth_rate': ((mse_vals[1:] / mse_vals[:-1])).mean().item()
        }
        
        return results

# 示例
metrics = PredictionMetrics()

# 生成模拟数据
predicted = torch.randn(11, 32, 4)  # [horizon+1, batch, state_dim]
true = torch.randn(11, 32, 4)

# 计算指标
results = metrics.compute_all_metrics(predicted, true)
print(f"最终MSE: {results['final']['mse']:.4f}")
print(f"平均MSE: {results['average']['mse']:.4f}")
print(f"误差增长率: {results['cumulative']['error_growth_rate']:.4f}")
```

### 5.2 长期预测评估指标

```python
class LongHorizonMetrics:
    def __init__(self):
        self.metrics = {}
    
    def compute_temporal_consistency(self, predicted_states):
        """
        计算时间一致性
        
        参数:
            predicted_states: 预测状态序列 [horizon+1, batch, state_dim]
        
        返回:
            时间一致性分数
        """
        # 计算相邻状态的变化
        diffs = predicted_states[1:] - predicted_states[:-1]
        
        # 计算变化的平滑度
        diff_diffs = diffs[1:] - diffs[:-1]
        smoothness = 1 / (1 + diff_diffs.abs().mean())
        
        return smoothness.item()
    
    def compute_drift_score(self, predicted_states, initial_state):
        """
        计算漂移分数（衡量长期预测是否偏离初始状态太远）
        
        参数:
            predicted_states: 预测状态序列 [horizon+1, batch, state_dim]
            initial_state: 初始状态 [batch, state_dim]
        
        返回:
            漂移分数
        """
        # 计算每个时间步与初始状态的距离
        distances = torch.norm(predicted_states - initial_state.unsqueeze(0), dim=-1)
        
        # 漂移率：最终距离与初始距离的比值
        drift_rate = distances[-1].mean() / (distances[0].mean() + 1e-6)
        
        return drift_rate.item()
    
    def compute_entropy(self, predictions):
        """
        计算预测的熵（衡量不确定性）
        
        参数:
            predictions: 多个预测样本 [num_samples, horizon+1, batch, state_dim]
        
        返回:
            熵值
        """
        # 计算每个时间步的方差
        variances = predictions.var(dim=0)
        
        # 熵与方差成正比
        entropy = variances.mean().item()
        
        return entropy
    
    def compute_coverage(self, predicted_states, true_states, threshold=0.1):
        """
        计算覆盖度（预测覆盖真实状态的比例）
        
        参数:
            predicted_states: 预测状态序列 [horizon+1, batch, state_dim]
            true_states: 真实状态序列 [horizon+1, batch, state_dim]
            threshold: 距离阈值
        
        返回:
            覆盖度分数
        """
        distances = torch.norm(predicted_states - true_states, dim=-1)
        covered = (distances < threshold).float().mean().item()
        
        return covered
    
    def compute_long_horizon_score(self, predicted_states, true_states, initial_state):
        """
        计算综合长期预测分数
        
        参数:
            predicted_states: 预测状态序列 [horizon+1, batch, state_dim]
            true_states: 真实状态序列 [horizon+1, batch, state_dim]
            initial_state: 初始状态 [batch, state_dim]
        
        返回:
            综合分数
        """
        # 计算各项指标
        mse_metric = nn.functional.mse_loss(predicted_states, true_states).item()
        consistency = self.compute_temporal_consistency(predicted_states)
        drift = self.compute_drift_score(predicted_states, initial_state)
        coverage = self.compute_coverage(predicted_states, true_states)
        
        # 综合评分（权重可以调整）
        score = (
            0.4 * (1 / (1 + mse_metric)) +
            0.2 * consistency +
            0.2 * (1 / (1 + drift)) +
            0.2 * coverage
        )
        
        return score, {
            'mse': mse_metric,
            'consistency': consistency,
            'drift': drift,
            'coverage': coverage
        }

# 示例
long_metrics = LongHorizonMetrics()

# 生成模拟数据
predicted = torch.randn(21, 32, 4)  # [21 steps, batch, state_dim]
true = torch.randn(21, 32, 4)
initial_state = torch.randn(32, 4)

# 计算时间一致性
consistency = long_metrics.compute_temporal_consistency(predicted)
print(f"时间一致性: {consistency:.4f}")

# 计算漂移分数
drift = long_metrics.compute_drift_score(predicted, initial_state)
print(f"漂移分数: {drift:.4f}")

# 计算覆盖度
coverage = long_metrics.compute_coverage(predicted, true)
print(f"覆盖度: {coverage:.4f}")

# 计算综合评分
score, details = long_metrics.compute_long_horizon_score(predicted, true, initial_state)
print(f"综合评分: {score:.4f}")
print(f"详细指标: {details}")
```

---

## 6. 优化策略

### 6.1 学习率调度

```python
class LearningRateScheduler:
    def __init__(self, optimizer, initial_lr=1e-3):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
    
    def cosine_annealing(self, epoch, total_epochs, min_lr=1e-6):
        """余弦退火调度"""
        lr = min_lr + (self.initial_lr - min_lr) * (1 + torch.cos(torch.pi * epoch / total_epochs)) / 2
        self.set_lr(lr)
        return lr
    
    def step_decay(self, epoch, decay_epochs=10, decay_rate=0.5):
        """阶梯衰减调度"""
        lr = self.initial_lr * (decay_rate ** (epoch // decay_epochs))
        self.set_lr(lr)
        return lr
    
    def exponential_decay(self, epoch, decay_rate=0.99):
        """指数衰减调度"""
        lr = self.initial_lr * (decay_rate ** epoch)
        self.set_lr(lr)
        return lr
    
    def warmup_cosine(self, epoch, total_epochs, warmup_epochs=5):
        """预热+余弦退火"""
        if epoch < warmup_epochs:
            lr = self.initial_lr * (epoch + 1) / warmup_epochs
        else:
            lr = self.cosine_annealing(epoch - warmup_epochs, total_epochs - warmup_epochs)
        self.set_lr(lr)
        return lr
    
    def set_lr(self, lr):
        """设置学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# 示例
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = LearningRateScheduler(optimizer, initial_lr=1e-3)

# 使用调度器
total_epochs = 100
for epoch in range(total_epochs):
    lr = scheduler.warmup_cosine(epoch, total_epochs, warmup_epochs=10)
    # 训练逻辑...
    print(f"Epoch {epoch}, LR: {lr:.6f}")
```

### 6.2 正则化技术

```python
class RegularizationUtils:
    @staticmethod
    def l2_regularization(model, lambda_l2=1e-4):
        """L2正则化"""
        l2_loss = 0.0
        for param in model.parameters():
            l2_loss += torch.norm(param) ** 2
        return lambda_l2 * l2_loss
    
    @staticmethod
    def l1_regularization(model, lambda_l1=1e-4):
        """L1正则化"""
        l1_loss = 0.0
        for param in model.parameters():
            l1_loss += torch.norm(param, p=1)
        return lambda_l1 * l1_loss
    
    @staticmethod
    def weight_decay(optimizer, weight_decay=1e-4):
        """权重衰减"""
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = weight_decay
    
    @staticmethod
    def dropout_regularization(model, dropout_rate=0.2):
        """在模型中添加Dropout"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # 在线性层后添加Dropout
                module.register_forward_hook(
                    lambda m, inp, out: nn.functional.dropout(out, p=dropout_rate, training=m.training)
                )
    
    @staticmethod
    def gradient_clipping(optimizer, max_norm=1.0):
        """梯度裁剪"""
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm)

# 示例
reg_utils = RegularizationUtils()

# 在训练循环中使用
for epoch in range(100):
    optimizer.zero_grad()
    
    # 计算损失
    pred = model(state, action)
    mse_loss = nn.functional.mse_loss(pred, target)
    
    # 添加正则化
    l2_loss = reg_utils.l2_regularization(model)
    total_loss = mse_loss + l2_loss
    
    total_loss.backward()
    
    # 梯度裁剪
    reg_utils.gradient_clipping(optimizer)
    
    optimizer.step()
```

### 6.3 数据增强

```python
class DataAugmenter:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def add_gaussian_noise(self, state, noise_std=0.1):
        """添加高斯噪声"""
        noise = torch.randn_like(state) * noise_std
        return state + noise
    
    def add_action_noise(self, action, noise_std=0.1):
        """为动作添加噪声"""
        noise = torch.randn_like(action) * noise_std
        return action + noise
    
    def time_warping(self, state_sequence, warping_factor=0.1):
        """时间扭曲"""
        seq_len = state_sequence.size(1)
        warped_indices = torch.linspace(0, seq_len - 1, seq_len) * (1 + (torch.rand(1) - 0.5) * 2 * warping_factor)
        warped_indices = torch.clamp(warped_indices, 0, seq_len - 1)
        
        # 线性插值
        warped = torch.zeros_like(state_sequence)
        for i in range(seq_len):
            idx = warped_indices[i]
            lower = int(torch.floor(idx))
            upper = min(lower + 1, seq_len - 1)
            weight = idx - lower
            
            warped[:, i, :] = (1 - weight) * state_sequence[:, lower, :] + weight * state_sequence[:, upper, :]
        
        return warped
    
    def state_scaling(self, state, scale_range=(0.9, 1.1)):
        """状态缩放"""
        scale = torch.rand(1) * (scale_range[1] - scale_range[0]) + scale_range[0]
        return state * scale
    
    def action_masking(self, action, mask_prob=0.1):
        """随机掩盖动作"""
        mask = torch.rand_like(action) < mask_prob
        return action * (1 - mask.float())
    
    def augment_batch(self, states, actions, augment_prob=0.5):
        """对批次数据进行增强"""
        batch_size = states.size(0)
        
        # 随机选择要增强的样本
        augment_mask = torch.rand(batch_size) < augment_prob
        
        # 应用各种增强
        if augment_mask.any():
            states[augment_mask] = self.add_gaussian_noise(states[augment_mask])
            actions[augment_mask] = self.add_action_noise(actions[augment_mask])
        
        return states, actions

# 示例
augmenter = DataAugmenter(state_dim=4, action_dim=2)

# 增强数据
states = torch.randn(32, 4)
actions = torch.randn(32, 2)

augmented_states, augmented_actions = augmenter.augment_batch(states, actions)
print(f"原始状态形状: {states.shape}")
print(f"增强后状态形状: {augmented_states.shape}")
```

---

## 7. 工程实践

### 7.1 模型压缩与加速

```python
class ModelCompressor:
    def __init__(self, model):
        self.model = model
    
    def prune_model(self, pruning_ratio=0.3):
        """
        剪枝模型
        
        参数:
            pruning_ratio: 剪枝比例
        """
        # 获取所有参数的L1范数
        weights = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weights.append(param.data.abs().mean())
        
        # 按重要性排序
        sorted_weights = sorted(enumerate(weights), key=lambda x: x[1])
        
        # 剪枝最不重要的参数
        num_prune = int(len(sorted_weights) * pruning_ratio)
        prune_indices = [idx for idx, _ in sorted_weights[:num_prune]]
        
        # 置零
        param_idx = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if param_idx in prune_indices:
                    param.data.fill_(0)
                param_idx += 1
        
        return self.model
    
    def quantize_model(self, bits=8):
        """
        量化模型
        
        参数:
            bits: 量化位数
        """
        # 动态量化
        self.model = torch.ao.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8 if bits == 8 else torch.quint4x2
        )
        return self.model
    
    def export_to_onnx(self, input_shape, output_path):
        """
        导出为ONNX格式
        
        参数:
            input_shape: 输入形状
            output_path: 输出路径
        """
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    
    def optimize_for_inference(self):
        """优化推理性能"""
        self.model.eval()
        
        # 融合批归一化和卷积/线性层
        self.model = torch.nn.utils.fuse_conv_bn_weights(self.model)
        
        # 使用JIT编译
        self.model = torch.jit.script(self.model)
        
        return self.model

# 示例
compressor = ModelCompressor(model)

# 剪枝
pruned_model = compressor.prune_model(pruning_ratio=0.3)

# 量化
quantized_model = compressor.quantize_model(bits=8)

# 导出ONNX
compressor.export_to_onnx((32, 4), 'model.onnx')

# 优化推理
optimized_model = compressor.optimize_for_inference()
```

### 7.2 分布式训练

```python
class DistributedTrainer:
    def __init__(self, model, rank, world_size):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        
        # 初始化分布式环境
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # 模型并行化
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[rank],
            output_device=rank
        )
    
    def average_gradients(self):
        """平均梯度"""
        for param in self.model.parameters():
            if param.requires_grad:
                torch.distributed.all_reduce(
                    param.grad.data,
                    op=torch.distributed.ReduceOp.SUM,
                    world_size=self.world_size
                )
                param.grad.data /= self.world_size
    
    def broadcast_parameters(self):
        """广播参数"""
        for param in self.model.parameters():
            torch.distributed.broadcast(
                param.data,
                src=0
            )
    
    def save_checkpoint(self, path, epoch, optimizer):
        """保存检查点"""
        if self.rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'rank': self.rank,
                'world_size': self.world_size
            }, path)
    
    def load_checkpoint(self, path, optimizer):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

# 示例
# 在每个进程中
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

model = StandardDynamicsModel(state_dim=4, action_dim=2)
trainer = DistributedTrainer(model, rank, world_size)

optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)

# 训练循环
for epoch in range(100):
    trainer.model.train()
    
    # 训练逻辑...
    
    # 平均梯度
    trainer.average_gradients()
    
    optimizer.step()
    
    # 保存检查点
    if epoch % 10 == 0:
        trainer.save_checkpoint(f'checkpoint_{epoch}.pt', epoch, optimizer)
```

### 7.3 在线学习与增量更新

```python
class OnlineLearner:
    def __init__(self, model, learning_rate=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_history = []
    
    def update(self, state, action, next_state):
        """
        在线更新模型
        
        参数:
            state: 当前状态
            action: 动作
            next_state: 下一个状态
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 预测
        pred = self.model(state, action)
        
        # 计算损失
        loss = nn.functional.mse_loss(pred, next_state)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        # 记录损失
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def update_with_replay_buffer(self, replay_buffer, batch_size=32):
        """
        使用重放缓冲区更新
        
        参数:
            replay_buffer: 重放缓冲区
            batch_size: 批次大小
        
        返回:
            平均损失
        """
        if len(replay_buffer) < batch_size:
            return None
        
        # 采样批次
        batch = replay_buffer.sample(batch_size)
        states, actions, next_states = batch
        
        # 更新
        self.model.train()
        self.optimizer.zero_grad()
        
        pred = self.model(states, actions)
        loss = nn.functional.mse_loss(pred, next_states)
        
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
        return loss.item()
    
    def adaptive_learning_rate(self, window_size=10):
        """
        自适应学习率调整
        
        参数:
            window_size: 滑动窗口大小
        """
        if len(self.loss_history) < window_size:
            return
        
        # 计算最近窗口的损失变化
        recent_losses = self.loss_history[-window_size:]
        loss_std = torch.tensor(recent_losses).std().item()
        
        # 根据损失波动调整学习率
        if loss_std > 0.1:
            # 损失波动大，降低学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.9
        elif loss_std < 0.01:
            # 损失稳定，提高学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.1, 1e-2)

# 示例
online_learner = OnlineLearner(model)

# 在线更新
state = torch.randn(32, 4)
action = torch.randn(32, 2)
next_state = torch.randn(32, 4)

loss = online_learner.update(state, action, next_state)
print(f"在线更新损失: {loss:.4f}")

# 自适应学习率
online_learner.adaptive_learning_rate()
```

---

## 8. 理论基础与分析

### 8.1 预测误差传播分析

```python
class ErrorAnalysis:
    def __init__(self, model):
        self.model = model
    
    def compute_error_jacobian(self, state, action):
        """
        计算误差雅可比矩阵
        
        参数:
            state: 当前状态
            action: 动作
        
        返回:
            雅可比矩阵
        """
        state.requires_grad = True
        action.requires_grad = True
        
        pred = self.model(state, action)
        
        # 计算雅可比矩阵
        jacobian = torch.zeros(state.size(1), pred.size(1))
        for i in range(pred.size(1)):
            grads = torch.autograd.grad(pred[:, i], state, grad_outputs=torch.ones_like(pred[:, i]))[0]
            jacobian[:, i] = grads.mean(dim=0)
        
        return jacobian
    
    def compute_error_propagation(self, initial_error, jacobian, horizon):
        """
        计算误差传播
        
        参数:
            initial_error: 初始误差
            jacobian: 雅可比矩阵
            horizon: 预测步数
        
        返回:
            各步误差
        """
        errors = [initial_error]
        current_error = initial_error
        
        for t in range(horizon):
            # 误差传播：e_{t+1} = J * e_t
            current_error = jacobian @ current_error
            errors.append(current_error)
        
        return torch.stack(errors)
    
    def compute_spectral_radius(self, jacobian):
        """
        计算谱半径（衡量误差增长速率）
        
        参数:
            jacobian: 雅可比矩阵
        
        返回:
            谱半径
        """
        eigenvalues = torch.linalg.eigvals(jacobian)
        spectral_radius = torch.max(torch.abs(eigenvalues)).item()
        
        return spectral_radius
    
    def analyze_stability(self, state, action, horizon=10):
        """
        分析模型的稳定性
        
        参数:
            state: 当前状态
            action: 动作
            horizon: 预测步数
        
        返回:
            稳定性分析结果
        """
        # 计算雅可比矩阵
        jacobian = self.compute_error_jacobian(state, action)
        
        # 计算谱半径
        spectral_radius = self.compute_spectral_radius(jacobian)
        
        # 分析稳定性
        is_stable = spectral_radius < 1.0
        
        # 计算误差传播
        initial_error = torch.randn(state.size(1)) * 0.01
        errors = self.compute_error_propagation(initial_error, jacobian, horizon)
        
        return {
            'spectral_radius': spectral_radius,
            'is_stable': is_stable,
            'error_propagation': errors,
            'error_norm': torch.norm(errors, dim=1).tolist()
        }

# 示例
error_analyzer = ErrorAnalysis(model)
state = torch.randn(32, 4)
action = torch.randn(32, 2)

# 计算雅可比矩阵
jacobian = error_analyzer.compute_error_jacobian(state, action)
print(f"雅可比矩阵形状: {jacobian.shape}")

# 分析稳定性
stability = error_analyzer.analyze_stability(state, action, horizon=10)
print(f"谱半径: {stability['spectral_radius']:.4f}")
print(f"是否稳定: {stability['is_stable']}")
print(f"误差范数: {stability['error_norm']}")
```

### 8.2 长期预测理论保证

```python
class TheoreticalAnalysis:
    def __init__(self, model):
        self.model = model
    
    def compute_contractivity(self, state1, state2, action, epsilon=1e-6):
        """
        计算模型的收缩性
        
        参数:
            state1: 状态1
            state2: 状态2
            action: 动作
            epsilon: 扰动大小
        
        返回:
            收缩系数
        """
        # 计算两个状态的预测
        pred1 = self.model(state1, action)
        pred2 = self.model(state2, action)
        
        # 计算距离变化
        delta = torch.norm(state1 - state2)
        delta_pred = torch.norm(pred1 - pred2)
        
        # 收缩系数
        contractivity = delta_pred / (delta + epsilon)
        
        return contractivity.item()
    
    def compute_lipschitz_constant(self, state, action, num_samples=100):
        """
        估计Lipschitz常数
        
        参数:
            state: 当前状态
            action: 动作
            num_samples: 采样次数
        
        返回:
            估计的Lipschitz常数
        """
        lipschitz_values = []
        
        for _ in range(num_samples):
            # 随机扰动
            delta = torch.randn_like(state) * 0.1
            state_perturbed = state + delta
            
            # 计算预测
            pred = self.model(state, action)
            pred_perturbed = self.model(state_perturbed, action)
            
            # 计算Lipschitz比值
            ratio = torch.norm(pred_perturbed - pred) / torch.norm(delta)
            lipschitz_values.append(ratio.item())
        
        return max(lipschitz_values)
    
    def verify_robustness(self, state, action, noise_level=0.1, num_trials=10):
        """
        验证模型的鲁棒性
        
        参数:
            state: 当前状态
            action: 动作
            noise_level: 噪声水平
            num_trials: 试验次数
        
        返回:
            鲁棒性分析结果
        """
        results = []
        
        for _ in range(num_trials):
            # 添加噪声
            noisy_state = state + torch.randn_like(state) * noise_level
            
            # 预测
            clean_pred = self.model(state, action)
            noisy_pred = self.model(noisy_state, action)
            
            # 计算误差
            error = torch.norm(noisy_pred - clean_pred)
            results.append(error.item())
        
        return {
            'mean_error': sum(results) / len(results),
            'max_error': max(results),
            'min_error': min(results),
            'std_error': torch.tensor(results).std().item()
        }
    
    def analyze_generalization(self, train_states, test_states, actions):
        """
        分析模型的泛化能力
        
        参数:
            train_states: 训练状态
            test_states: 测试状态
            actions: 动作序列
        
        返回:
            泛化分析结果
        """
        # 训练集上的误差
        train_preds = self.model(train_states, actions)
        train_error = torch.norm(train_preds - train_states).item()
        
        # 测试集上的误差
        test_preds = self.model(test_states, actions)
        test_error = torch.norm(test_preds - test_states).item()
        
        # 泛化差距
        generalization_gap = test_error - train_error
        
        return {
            'train_error': train_error,
            'test_error': test_error,
            'generalization_gap': generalization_gap,
            'gap_ratio': generalization_gap / (train_error + 1e-6)
        }

# 示例
theoretical_analyzer = TheoreticalAnalysis(model)

# 计算收缩性
state1 = torch.randn(32, 4)
state2 = torch.randn(32, 4)
action = torch.randn(32, 2)
contractivity = theoretical_analyzer.compute_contractivity(state1, state2, action)
print(f"收缩系数: {contractivity:.4f}")

# 计算Lipschitz常数
lipschitz = theoretical_analyzer.compute_lipschitz_constant(state1, action)
print(f"Lipschitz常数: {lipschitz:.4f}")

# 验证鲁棒性
robustness = theoretical_analyzer.verify_robustness(state1, action)
print(f"鲁棒性分析: {robustness}")

# 分析泛化能力
train_states = torch.randn(100, 4)
test_states = torch.randn(50, 4)
actions = torch.randn(100, 2)
generalization = theoretical_analyzer.analyze_generalization(train_states, test_states, actions)
print(f"泛化分析: {generalization}")
```

---

## 9. 前沿研究方向

### 9.1 持续学习长期预测

```python
class ContinualLearningPredictor:
    def __init__(self, model, memory_size=1000):
        self.model = model
        self.memory = []
        self.memory_size = memory_size
    
    def add_to_memory(self, state, action, next_state):
        """
        添加样本到记忆
        
        参数:
            state: 当前状态
            action: 动作
            next_state: 下一个状态
        """
        self.memory.append((state, action, next_state))
        
        # 保持记忆大小
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def replay_from_memory(self, batch_size=32):
        """
        从记忆中重放
        
        参数:
            batch_size: 批次大小
        
        返回:
            重放批次
        """
        if len(self.memory) < batch_size:
            return None
        
        # 随机采样
        indices = torch.randperm(len(self.memory))[:batch_size]
        states = []
        actions = []
        next_states = []
        
        for idx in indices:
            s, a, ns = self.memory[idx]
            states.append(s)
            actions.append(a)
            next_states.append(ns)
        
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(next_states)
        )
    
    def incremental_update(self, new_state, new_action, new_next_state, replay_prob=0.5):
        """
        增量更新模型
        
        参数:
            new_state: 新状态
            new_action: 新动作
            new_next_state: 新的下一个状态
            replay_prob: 重放概率
        
        返回:
            更新损失
        """
        # 添加到记忆
        self.add_to_memory(new_state, new_action, new_next_state)
        
        # 混合新数据和重放数据
        loss = 0.0
        
        # 新数据更新
        pred = self.model(new_state, new_action)
        loss += nn.functional.mse_loss(pred, new_next_state)
        
        # 重放更新
        if torch.rand(1) < replay_prob:
            replay_batch = self.replay_from_memory()
            if replay_batch is not None:
                states, actions, next_states = replay_batch
                pred_replay = self.model(states, actions)
                loss += nn.functional.mse_loss(pred_replay, next_states)
        
        # 反向传播
        loss.backward()
        
        return loss.item()

# 示例
continual_learner = ContinualLearningPredictor(model, memory_size=500)

# 增量更新
new_state = torch.randn(32, 4)
new_action = torch.randn(32, 2)
new_next_state = torch.randn(32, 4)

loss = continual_learner.incremental_update(new_state, new_action, new_next_state)
print(f"增量更新损失: {loss:.4f}")
```

### 9.2 元学习长期预测

```python
class MetaLearningPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        
        # 基础模型
        self.base_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 元学习头部
        self.meta_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)  # 输出参数更新
        )
    
    def forward(self, state, action, context=None):
        """
        前向传播
        
        参数:
            state: 当前状态
            action: 动作
            context: 上下文信息（用于元学习）
        
        返回:
            预测的下一个状态
        """
        # 标准预测
        combined = torch.cat([state, action], dim=-1)
        pred = self.base_model(combined)
        
        # 如果有上下文，应用元学习调整
        if context is not None:
            # 计算参数更新
            context_input = torch.cat([state, action, context], dim=-1)
            param_update = self.meta_head(context_input)
            
            # 应用更新（简化版本）
            pred = pred + param_update[:, :state.size(1)]
        
        return pred
    
    def meta_learn(self, support_set, query_set):
        """
        元学习训练
        
        参数:
            support_set: 支持集 (states, actions, next_states)
            query_set: 查询集 (states, actions, next_states)
        
        返回:
            元学习损失
        """
        # 从支持集学习
        support_states, support_actions, support_next_states = support_set
        
        # 计算初始预测
        initial_pred = self.base_model(torch.cat([support_states, support_actions], dim=-1))
        
        # 计算元更新
        context = initial_pred - support_next_states  # 误差作为上下文
        
        # 在查询集上测试
        query_states, query_actions, query_next_states = query_set
        query_pred = self.forward(query_states, query_actions, context)
        
        # 计算损失
        loss = nn.functional.mse_loss(query_pred, query_next_states)
        
        return loss

# 示例
meta_predictor = MetaLearningPredictor(state_dim=4, action_dim=2)

# 元学习训练
support_states = torch.randn(10, 4)
support_actions = torch.randn(10, 2)
support_next_states = torch.randn(10, 4)
support_set = (support_states, support_actions, support_next_states)

query_states = torch.randn(5, 4)
query_actions = torch.randn(5, 2)
query_next_states = torch.randn(5, 4)
query_set = (query_states, query_actions, query_next_states)

meta_loss = meta_predictor.meta_learn(support_set, query_set)
print(f"元学习损失: {meta_loss:.4f}")
```

### 9.3 因果长期预测

```python
class CausalPredictor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # 因果图学习器
        self.graph_learner = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim * state_dim)
        )
        
        # 因果动态模型
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def learn_causal_graph(self, states, actions):
        """
        学习因果图
        
        参数:
            states: 状态序列
            actions: 动作序列
        
        返回:
            因果邻接矩阵
        """
        # 计算因果图
        combined = torch.cat([states, actions], dim=-1)
        graph_logits = self.graph_learner(combined)
        
        # 重塑为邻接矩阵
        batch_size = states.size(0)
        adjacency_matrix = graph_logits.view(batch_size, states.size(1), states.size(1))
        
        # 应用sigmoid得到概率
        adjacency_matrix = torch.sigmoid(adjacency_matrix)
        
        return adjacency_matrix
    
    def causal_intervention(self, state, action, intervention_mask):
        """
        执行因果干预
        
        参数:
            state: 当前状态
            action: 动作
            intervention_mask: 干预掩码
        
        返回:
            干预后的预测
        """
        # 标准预测
        pred = self.dynamics(torch.cat([state, action], dim=-1))
        
        # 应用干预
        pred = pred * (1 - intervention_mask) + intervention_mask * state
        
        return pred
    
    def counterfactual_prediction(self, state, action, alternative_action):
        """
        反事实预测
        
        参数:
            state: 当前状态
            action: 实际动作
            alternative_action: 替代动作
        
        返回:
            反事实预测结果
        """
        # 实际预测
        actual_pred = self.dynamics(torch.cat([state, action], dim=-1))
        
        # 反事实预测
        counterfactual_pred = self.dynamics(torch.cat([state, alternative_action], dim=-1))
        
        return {
            'actual': actual_pred,
            'counterfactual': counterfactual_pred,
            'difference': counterfactual_pred - actual_pred
        }

# 示例
causal_predictor = CausalPredictor(state_dim=4, action_dim=2)

# 学习因果图
states = torch.randn(32, 4)
actions = torch.randn(32, 2)
graph = causal_predictor.learn_causal_graph(states, actions)
print(f"因果图形状: {graph.shape}")

# 因果干预
intervention_mask = torch.tensor([0, 1, 0, 0])  # 干预第二个维度
intervened_pred = causal_predictor.causal_intervention(states, actions, intervention_mask)
print(f"干预后预测形状: {intervened_pred.shape}")

# 反事实预测
alternative_action = torch.randn(32, 2)
cf_result = causal_predictor.counterfactual_prediction(states, actions, alternative_action)
print(f"实际预测形状: {cf_result['actual'].shape}")
print(f"反事实预测形状: {cf_result['counterfactual'].shape}")
```

---

## 10. 实践练习

### 练习1：实现长期预测器

```python
class LongHorizonPredictor:
    def __init__(self, dynamics_model):
        self.dynamics = dynamics_model
    
    def predict_rollout(self, initial_state, actions, horizon, use_feedback=False):
        """
        长期预测
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            actions: 动作序列 [horizon, batch, action_dim]
            horizon: 预测步数
            use_feedback: 是否使用反馈
        
        返回:
            预测序列 [horizon+1, batch, state_dim]
        """
        predictions = [initial_state]
        current_state = initial_state
        
        for t in range(horizon):
            action = actions[t] if t < len(actions) else torch.zeros_like(actions[0])
            
            if use_feedback:
                # 闭环：使用预测的状态
                next_state = self.dynamics(current_state, action)
            else:
                # 开环：使用初始状态
                next_state = self.dynamics(initial_state, action)
            
            predictions.append(next_state)
            current_state = next_state
        
        return torch.stack(predictions)
    
    def compute_prediction_error(self, predicted_states, true_states):
        """
        计算预测误差
        
        参数:
            predicted_states: 预测状态序列 [horizon+1, batch, state_dim]
            true_states: 真实状态序列 [horizon+1, batch, state_dim]
        
        返回:
            误差统计
        """
        mse = nn.functional.mse_loss(predicted_states, true_states, reduction='none')
        mse_per_step = mse.mean(dim=-1).mean(dim=-1)  # 平均到每个时间步
        
        return {
            'total_mse': mse.mean().item(),
            'per_step_mse': mse_per_step.tolist(),
            'final_mse': mse_per_step[-1].item(),
            'average_mse': mse_per_step.mean().item()
        }
    
    def evaluate_long_horizon(self, initial_state, actions, true_states):
        """
        评估长期预测性能
        
        参数:
            initial_state: 初始状态
            actions: 动作序列
            true_states: 真实状态序列
        
        返回:
            评估结果
        """
        horizon = len(actions)
        
        # 开环预测
        open_loop_preds = self.predict_rollout(initial_state, actions, horizon, use_feedback=False)
        open_loop_errors = self.compute_prediction_error(open_loop_preds, true_states)
        
        # 闭环预测
        closed_loop_preds = self.predict_rollout(initial_state, actions, horizon, use_feedback=True)
        closed_loop_errors = self.compute_prediction_error(closed_loop_preds, true_states)
        
        return {
            'open_loop': open_loop_errors,
            'closed_loop': closed_loop_errors,
            'horizon': horizon
        }

# 测试
class SimpleDynamics(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
    
    def forward(self, state, action):
        combined = torch.cat([state, action], dim=-1)
        return self.net(combined)

dynamics = SimpleDynamics(state_dim=4, action_dim=2)
predictor = LongHorizonPredictor(dynamics)

initial_state = torch.randn(32, 4)
horizon = 10
actions = [torch.randn(32, 2) for _ in range(horizon)]
true_states = torch.stack([initial_state] + [torch.randn(32, 4) for _ in range(horizon)])

results = predictor.evaluate_long_horizon(initial_state, actions, true_states)
print(f"开环预测最终MSE: {results['open_loop']['final_mse']:.4f}")
print(f"闭环预测最终MSE: {results['closed_loop']['final_mse']:.4f}")
```

### 练习2：实现不确定性感知预测

```python
class UncertaintyAwarePredictor:
    def __init__(self, ensemble_models):
        self.models = ensemble_models
    
    def predict_with_uncertainty(self, state, action, num_samples=10):
        """
        带不确定性的预测
        
        参数:
            state: 当前状态 [batch, state_dim]
            action: 动作 [batch, action_dim]
            num_samples: 采样次数
        
        返回:
            预测均值、标准差、样本
        """
        samples = []
        
        for model in self.models[:num_samples]:
            with torch.no_grad():
                pred = model(state, action)
                samples.append(pred)
        
        # 如果模型不够，进行多次采样
        if len(samples) < num_samples:
            for _ in range(num_samples - len(samples)):
                noise = torch.randn_like(state) * 0.1
                with torch.no_grad():
                    pred = self.models[0](state + noise, action)
                    samples.append(pred)
        
        samples = torch.stack(samples)
        
        # 计算统计量
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        var = samples.var(dim=0)
        
        # 计算置信区间
        lower_bound = mean - 1.96 * std
        upper_bound = mean + 1.96 * std
        
        return {
            'mean': mean,
            'std': std,
            'var': var,
            'samples': samples,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    def langevin_prediction(self, model, state, action, noise_scale=0.1, num_steps=20):
        """
        使用朗之万动力学的预测
        
        参数:
            model: 动态模型
            state: 当前状态
            action: 动作
            noise_scale: 噪声规模
            num_steps: 步数
        
        返回:
            多次采样的均值和标准差
        """
        samples = []
        current_state = state.clone()
        
        for _ in range(num_steps):
            # 添加噪声
            noise = torch.randn_like(current_state) * noise_scale
            noisy_state = current_state + noise
            
            # 预测
            with torch.no_grad():
                next_state = model(noisy_state, action)
            
            samples.append(next_state)
            current_state = next_state
        
        samples = torch.stack(samples)
        return samples.mean(dim=0), samples.std(dim=0)
    
    def bayesian_predict(self, state, action, num_particles=50):
        """
        贝叶斯预测
        
        参数:
            state: 当前状态
            action: 动作
            num_particles: 粒子数量
        
        返回:
            预测分布
        """
        # 初始化粒子
        particles = [state + torch.randn_like(state) * 0.01 for _ in range(num_particles)]
        
        # 粒子滤波更新
        for t in range(5):
            new_particles = []
            for particle in particles:
                # 预测
                with torch.no_grad():
                    pred = self.models[0](particle, action)
                
                # 添加噪声
                pred = pred + torch.randn_like(pred) * 0.05
                new_particles.append(pred)
            
            particles = new_particles
        
        particles = torch.stack(particles)
        
        return {
            'mean': particles.mean(dim=0),
            'std': particles.std(dim=0),
            'particles': particles
        }

# 测试
ensemble = [SimpleDynamics(4, 2) for _ in range(5)]
predictor = UncertaintyAwarePredictor(ensemble)

state = torch.randn(32, 4)
action = torch.randn(32, 2)

# 带不确定性预测
result = predictor.predict_with_uncertainty(state, action, num_samples=5)
print(f"预测均值形状: {result['mean'].shape}")
print(f"预测标准差形状: {result['std'].shape}")

# 朗之万预测
mean, std = predictor.langevin_prediction(ensemble[0], state, action)
print(f"朗之万均值形状: {mean.shape}")

# 贝叶斯预测
bayesian_result = predictor.bayesian_predict(state, action)
print(f"贝叶斯预测均值形状: {bayesian_result['mean'].shape}")
```

### 练习3：实现分层长期预测器

```python
class HierarchicalLongHorizonPredictor:
    def __init__(self, models):
        """
        分层长期预测器
        
        参数:
            models: 不同时间尺度的模型列表
        """
        self.models = models
        self.num_levels = len(models)
    
    def predict_at_scale(self, state, action, level):
        """
        在指定时间尺度上预测
        
        参数:
            state: 当前状态
            action: 动作
            level: 时间尺度级别（0=细粒度，越高越粗粒度）
        
        返回:
            预测的状态
        """
        model = self.models[level]
        return model(state, action)
    
    def multi_scale_prediction(self, initial_state, horizon, actions=None):
        """
        多尺度长期预测
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            horizon: 预测步数
            actions: 动作序列（可选）
        
        返回:
            预测状态序列 [horizon+1, batch, state_dim]
        """
        predictions = [initial_state]
        current_state = initial_state
        
        for t in range(horizon):
            # 根据当前步数选择时间尺度
            level = min(self.num_levels - 1, t // 5)
            
            # 获取动作
            if actions is not None and t < len(actions):
                action = actions[t]
            else:
                action = torch.zeros_like(initial_state[:, :2])  # 假设action_dim=2
            
            # 使用对应尺度的模型预测
            next_state = self.predict_at_scale(current_state, action, level)
            predictions.append(next_state)
            current_state = next_state
        
        return torch.stack(predictions)
    
    def coarse_to_fine(self, initial_state, horizon, coarse_steps=5):
        """
        粗到细预测
        
        参数:
            initial_state: 初始状态
            horizon: 总预测步数
            coarse_steps: 粗粒度步数
        
        返回:
            细粒度预测序列
        """
        # 第一步：粗粒度预测
        coarse_states = [initial_state]
        current_state = initial_state
        
        # 使用最高层级模型
        for _ in range(coarse_steps):
            action = torch.zeros_like(initial_state[:, :2])
            next_state = self.predict_at_scale(current_state, action, self.num_levels - 1)
            coarse_states.append(next_state)
            current_state = next_state
        
        # 第二步：细粒度细化
        fine_states = [initial_state]
        current_state = initial_state
        
        for t in range(horizon):
            # 计算对应的粗粒度目标
            coarse_idx = min(t * coarse_steps // horizon, coarse_steps)
            target_coarse = coarse_states[coarse_idx]
            
            # 使用细粒度模型预测
            action = torch.zeros_like(initial_state[:, :2])
            next_state = self.predict_at_scale(current_state, action, 0)
            
            # 向粗粒度目标调整
            next_state = 0.8 * next_state + 0.2 * target_coarse
            
            fine_states.append(next_state)
            current_state = next_state
        
        return torch.stack(fine_states)

# 测试
models = [
    SimpleDynamics(4, 2),  # 细粒度模型
    SimpleDynamics(4, 2),  # 中等粒度模型
    SimpleDynamics(4, 2)   # 粗粒度模型
]

hierarchical_predictor = HierarchicalLongHorizonPredictor(models)

initial_state = torch.randn(32, 4)
horizon = 20

# 多尺度预测
multi_scale_preds = hierarchical_predictor.multi_scale_prediction(initial_state, horizon)
print(f"多尺度预测形状: {multi_scale_preds.shape}")

# 粗到细预测
cf_preds = hierarchical_predictor.coarse_to_fine(initial_state, horizon)
print(f"粗到细预测形状: {cf_preds.shape}")
```

---

**下一节**：[模型预测控制](05-model-predictive-control.md)

---

## 参考文献

1. Hafner, D., et al. (2021). Mastering Atari with Discrete World Models.
2. Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.
3. Micheli, V., et al. (2023). Model-Based Reinforcement Learning for Long-Horizon Visuomotor Control.
4. Chiappa, S., et al. (2020). Latent World Models for Planning.
5. Kaiser, L., et al. (2019). Model-Based Reinforcement Learning with Discrete World Models.