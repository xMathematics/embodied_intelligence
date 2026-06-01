# 6.4 长期预测

## 目录

- [1. 引言](#1-引言)
- [2. 长期预测概述](#2-长期预测概述)
- [3. 长期预测方法](#3-长期预测方法)
- [4. 模型架构](#4-模型架构)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 长期预测的重要性

**长期预测**是指对环境未来状态进行多步预测的能力。这对于长期规划和决策至关重要，也是评估世界模型质量的重要指标。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **机器人规划** | 预测长期动作效果 | 复杂任务规划 |
| **游戏AI** | 预测游戏未来状态 | 策略游戏 |
| **交通预测** | 预测交通流 | 智慧城市 |
| **天气预报** | 预测天气变化 | 气象预报 |

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

### 2.2 长期预测的挑战

| 挑战 | 描述 |
|------|------|
| **误差累积** | 单步预测误差会累积放大 |
| **状态混淆** | 不同状态可能变得相似 |
| **长程依赖** | 长期依赖难以捕捉 |
| **计算复杂度** | 多步预测计算量大 |

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
        initial_state: 初始状态
        actions: 动作序列
        horizon: 预测步数
    
    返回:
        预测的状态序列
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
model = StandardDynamicsModel(state_dim=4, action_dim=2)
initial_state = torch.randn(4)
actions = [torch.randn(2) for _ in range(10)]

predictions = open_loop_prediction(model, initial_state, actions, horizon=10)
print(f"预测序列形状: {predictions.shape}")  # [11, 4]
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
        initial_state: 初始状态
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
            aggregation: 聚合方式 ('mean', 'max', 'min')
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
        
        return mean, std
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 正则化网络：预测状态变化
        self.delta_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        """前向传播"""
        combined = torch.cat([state, action], dim=-1)
        
        # 标准预测
        delta = self.delta_net(combined)
        
        # 限制变化幅度
        delta = torch.clamp(delta, min=-1.0, max=1.0)
        
        next_state = state + delta
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
```

---

## 4. 模型架构

### 4.1 循环预测架构

**使用 RNN 进行序列预测**。

```python
class RecurrentDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        # 动态模型
        self.dynamics = nn.GRU(
            input_size=state_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, state_dim)
    
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
        combined = torch.cat([state_sequence, action_sequence], dim=-1)
        
        # RNN前向传播
        output, hidden = self.dynamics(combined)
        
        # 输出预测
        predictions = self.output_layer(output)
        
        return predictions
```

### 4.2 Transformer预测架构

**使用 Transformer 进行长期预测**。

```python
class TransformerDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.state_embedding = nn.Linear(state_dim, latent_dim)
        self.action_embedding = nn.Linear(action_dim, latent_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, latent_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_layer = nn.Linear(latent_dim, state_dim)
    
    def forward(self, state_sequence, action_sequence):
        """
        Transformer预测
        
        参数:
            state_sequence: 状态序列 [batch, seq_len, state_dim]
            action_sequence: 动作序列 [batch, seq_len, action_dim]
        
        返回:
            预测的下一状态序列
        """
        # 嵌入
        state_emb = self.state_embedding(state_sequence)
        action_emb = self.action_embedding(action_sequence)
        combined = state_emb + action_emb
        
        # 添加位置编码
        combined = combined + self.pos_embedding[:, :combined.size(1), :]
        
        # Transformer编码
        encoded = self.transformer(combined)
        
        # 输出预测
        predictions = self.output_layer(encoded)
        
        return predictions
```

### 4.3 图神经网络预测

**使用 GNN 建模状态关系**。

```python
class GraphDynamicsModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        # 节点更新
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 边更新
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 图卷积层
        self.graph_conv = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, node_dim)
    
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
        
        # 编码
        h = self.node_encoder(node_features)
        h_edge = self.edge_encoder(edge_features)
        
        # 图卷积
        for conv in self.graph_conv:
            # 消息传递
            messages = []
            for i, j in zip(edge_index[0], edge_index[1]):
                msg = torch.cat([h[:, i, :], h[:, j, :]], dim=-1)
                messages.append(msg)
            
            messages = torch.stack(messages, dim=2).permute(0, 2, 1, 3).squeeze(0)
            updated = conv(messages)
            
            # 聚合
            h_new = torch.zeros_like(h)
            for idx, (i, j) in enumerate(zip(edge_index[0], edge_index[1])):
                h_new[:, j, :] += updated[idx]
            
            h = h_new + h
        
        # 输出
        output = self.output_layer(h)
        return output
```

---

## 5. 实践练习

### 练习1：实现长期预测器

```python
class LongHorizonPredictor:
    def __init__(self, dynamics_model):
        self.dynamics = dynamics_model
    
    def predict_rollout(self, initial_state, actions, horizon, use_feedback=False):
        """
        长期预测
        
        参数:
            initial_state: 初始状态
            actions: 动作序列
            horizon: 预测步数
            use_feedback: 是否使用反馈
        
        返回:
            预测序列
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
            predicted_states: 预测状态序列
            true_states: 真实状态序列
        
        返回:
            误差统计
        """
        mse = nn.functional.mse_loss(predicted_states, true_states, reduction='none')
        mse_per_step = mse.mean(dim=-1)
        
        return {
            'total_mse': mse.mean().item(),
            'per_step_mse': mse_per_step.tolist(),
            'final_mse': mse_per_step[-1].item()
        }

# 测试
import torch
import torch.nn as nn

class SimpleDynamics(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
    
    def forward(self, state, action):
        combined = torch.cat([state, action], dim=-1)
        return self.net(combined)

dynamics = SimpleDynamics(state_dim=4, action_dim=2)
predictor = LongHorizonPredictor(dynamics)

initial_state = torch.randn(1, 4)
actions = [torch.randn(1, 2) for _ in range(10)]
true_states = torch.stack([initial_state] + [torch.randn(1, 4) for _ in range(10)])

predictions = predictor.predict_rollout(initial_state, actions, horizon=10)
errors = predictor.compute_prediction_error(predictions, true_states)

print(f"总MSE: {errors['total_mse']:.4f}")
print(f"最终MSE: {errors['final_mse']:.4f}")
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
            state: 当前状态
            action: 动作
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
        
        return mean, std, samples
    
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
            多次采样的均值
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

# 测试
ensemble = [SimpleDynamics(4, 2) for _ in range(5)]
predictor = UncertaintyAwarePredictor(ensemble)

state = torch.randn(1, 4)
action = torch.randn(1, 2)

mean, std, samples = predictor.predict_with_uncertainty(state, action, num_samples=5)
print(f"预测均值形状: {mean.shape}")  # [1, 4]
print(f"不确定性形状: {std.shape}")    # [1, 4]
```

### 练习3：实现自回归预测器

```python
class AutoregressivePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=128, num_layers=2):
        super().__init__()
        # 嵌入层
        self.state_embed = nn.Linear(state_dim, latent_dim)
        self.action_embed = nn.Linear(action_dim, latent_dim)
        
        # RNN进行序列建模
        self.rnn = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, state_dim)
        )
        
        # 动作预测头
        self.action_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, action_dim)
        )
    
    def forward(self, state_sequence, action_sequence):
        """
        前向传播
        
        参数:
            state_sequence: 状态序列 [batch, seq_len, state_dim]
            action_sequence: 动作序列 [batch, seq_len, action_dim]
        
        返回:
            预测的状态和动作
        """
        # 嵌入
        state_emb = self.state_embed(state_sequence)
        action_emb = self.action_embed(action_sequence)
        
        # 组合
        combined = state_emb + action_emb
        
        # RNN
        output, hidden = self.rnn(combined)
        
        # 预测
        state_pred = self.predictor(output)
        action_pred = self.action_predictor(output)
        
        return state_pred, action_pred
    
    def autoregressive_predict(self, initial_state, horizon):
        """
        自回归预测
        
        参数:
            initial_state: 初始状态
            horizon: 预测步数
        
        返回:
            预测序列
        """
        self.eval()
        predictions = [initial_state]
        current_state = initial_state.unsqueeze(1)  # [1, 1, state_dim]
        
        with torch.no_grad():
            for t in range(horizon):
                # 假设零动作
                action = torch.zeros_like(current_state[:, :, :2])  # 假设action_dim=2
                
                # 嵌入
                state_emb = self.state_embed(current_state)
                action_emb = self.action_embed(action)
                
                # 组合
                combined = state_emb + action_emb
                
                # RNN
                output, hidden = self.rnn(combined)
                
                # 预测下一状态
                next_state = self.predictor(output)
                
                predictions.append(next_state.squeeze(1))
                current_state = next_state
        
        return torch.stack(predictions)

# 测试
predictor = AutoregressivePredictor(state_dim=4, action_dim=2)
initial_state = torch.randn(1, 4)
predictions = predictor.autoregressive_predict(initial_state, horizon=10)
print(f"预测序列形状: {predictions.shape}")  # [11, 4]
```

---

**下一节**：[模型预测控制](05-model-predictive-control.md)

---

## 参考文献

1. Hafner, D., et al. (2021). Mastering Atari with Discrete World Models.
2. Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.
3. Micheli, V., et al. (2023). Model-Based Reinforcement Learning for Long-Horizon Visuomotor Control.
