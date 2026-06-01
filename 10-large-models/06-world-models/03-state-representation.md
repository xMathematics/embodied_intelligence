# 6.3 状态表示学习

## 目录

- [1. 引言](#1-引言)
- [2. 状态表示学习概述](#2-状态表示学习概述)
- [3. 表示学习方法](#3-表示学习方法)
- [4. 表示学习架构](#4-表示学习架构)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 状态表示学习的重要性

**状态表示学习**是指从高维观测（如图像）中学习低维状态表示的过程。这是世界模型的关键组件，能够提取环境的关键信息而忽略无关细节。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **机器人操控** | 从图像学习机器人状态 | 视觉反馈控制 |
| **自动驾驶** | 理解道路场景 | 车道线检测 |
| **游戏AI** | 从游戏画面学习 | Atari游戏 |
| **医学影像** | 分析医学图像 | CT/MRI分析 |

---

## 2. 状态表示学习概述

### 2.1 定义

**状态表示学习**：从原始观测中提取有意义的低维表示。

**形式化表达**：
```
z_t = Encoder(o_t; θ_e)
```

其中：
- $o_t$：时刻t的观测
- $z_t$：学习到的状态表示
- $Encoder$：编码器网络
- $θ_e$：编码器参数

### 2.2 好的状态表示的特点

| 特点 | 描述 |
|------|------|
| **完整性** | 包含做出决策所需的所有信息 |
| **紧凑性** | 低维且紧凑 |
| **可解释性** | 具有明确的语义含义 |
| **马尔可夫性** | 满足马尔可夫假设 |

---

## 3. 表示学习方法

### 3.1 自动编码器

**方法**：学习压缩和解压表示。

```python
class Autoencoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim=256):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def forward(self, obs):
        """前向传播"""
        z = self.encoder(obs)
        obs_recon = self.decoder(z)
        return obs_recon, z
    
    def encode(self, obs):
        """仅编码"""
        return self.encoder(obs)
    
    def decode(self, z):
        """仅解码"""
        return self.decoder(z)
```

### 3.2 变分自编码器 (VAE)

**方法**：学习潜在空间的分布。

```python
class VAE(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim=256):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # 均值和方差
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def reparameterize(self, mu, logvar):
        """重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, obs):
        """前向传播"""
        h = self.encoder(obs)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        obs_recon = self.decoder(z)
        return obs_recon, mu, logvar, z
    
    def kl_loss(self, mu, logvar):
        """KL散度损失"""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
```

### 3.3 对比表示学习

**方法**：通过对比学习获取表示。

```python
class ContrastiveRepresentation(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim=256):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, obs1, obs2):
        """
        前向传播
        
        参数:
            obs1, obs2: 两个观测（应该对应同一状态）
        
        返回:
            相似度
        """
        z1 = self.encoder(obs1)
        z2 = self.encoder(obs2)
        
        # 投影
        h1 = self.projector(z1)
        h2 = self.projector(z2)
        
        # 计算相似度
        sim = F.cosine_similarity(h1, h2, dim=-1)
        return sim
    
    def contrastive_loss(self, obs1, obs2, obs_neg):
        """
        对比损失
        
        参数:
            obs1, obs2: 正样本对
            obs_neg: 负样本
        """
        z1 = self.encoder(obs1)
        z2 = self.encoder(obs2)
        z_neg = self.encoder(obs_neg)
        
        h1 = self.projector(z1)
        h2 = self.projector(z2)
        h_neg = self.projector(z_neg)
        
        # 正样本相似度
        pos_sim = F.cosine_similarity(h1, h2, dim=-1)
        
        # 负样本相似度
        neg_sim = F.cosine_similarity(h1, h_neg, dim=-1)
        
        # InfoNCE损失
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        return loss.mean()
```

### 3.4 因果表示学习

**方法**：学习因果关系而非相关性。

```python
class CausalRepresentation(nn.Module):
    def __init__(self, obs_dim, state_dim, hidden_dim=256):
        super().__init__()
        # 观测到状态的映射
        self.observation_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        # 因果机制
        self.causal_mechanism = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + 1, hidden_dim),  # +1 for action
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(state_dim)
        ])
    
    def forward(self, obs, action):
        """前向传播"""
        state = self.observation_net(obs)
        
        # 应用因果机制
        next_state = []
        for i, mechanism in enumerate(self.causal_mechanism):
            input_i = torch.cat([state, action], dim=-1)
            delta_i = mechanism(input_i)
            next_state.append(state[:, i:i+1] + delta_i)
        
        return torch.cat(next_state, dim=-1)
```

---

## 4. 表示学习架构

### 4.1 卷积编码器

**适用于图像观测**。

```python
class ConvEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        """前向传播"""
        # x: [batch, channels, height, width]
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z
```

### 4.2 Transformer编码器

**适用于序列表示**。

```python
class TransformerEncoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, num_heads=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(obs_dim, latent_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, latent_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(latent_dim, latent_dim)
    
    def forward(self, obs_seq):
        """
        前向传播
        
        参数:
            obs_seq: 观测序列 [batch, seq_len, obs_dim]
        
        返回:
            状态表示 [batch, latent_dim]
        """
        # 嵌入
        h = self.embedding(obs_seq)
        h = h + self.pos_embedding[:, :h.size(1), :]
        
        # Transformer编码
        h = self.transformer(h)
        
        # 取最后一个时刻
        z = h[:, -1, :]
        z = self.fc(z)
        
        return z
```

### 4.3 GRU编码器

**适用于时序数据**。

```python
class GRUEncoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True
        )
    
    def forward(self, obs_seq):
        """
        前向传播
        
        参数:
            obs_seq: 观测序列 [batch, seq_len, obs_dim]
        
        返回:
            隐藏状态 [batch, latent_dim]
        """
        output, hidden = self.gru(obs_seq)
        # 取最后一层的隐藏状态
        z = hidden[-1]
        return z
```

---

## 5. 实践练习

### 练习1：实现状态表示学习器

```python
class StateRepresentationLearner:
    def __init__(self, obs_dim, latent_dim, hidden_dim=256, lr=1e-3):
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr
        )
    
    def train_step(self, obs):
        """单步训练"""
        # 编码
        z = self.encoder(obs)
        # 解码
        obs_recon = self.decoder(z)
        # 计算损失
        loss = nn.functional.mse_loss(obs_recon, obs)
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def encode(self, obs):
        """编码观测到状态"""
        with torch.no_grad():
            return self.encoder(obs)
    
    def decode(self, z):
        """从状态解码到观测"""
        with torch.no_grad():
            return self.decoder(z)

# 测试
learner = StateRepresentationLearner(obs_dim=100, latent_dim=16)
obs = torch.randn(32, 100)
loss = learner.train_step(obs)
print(f"训练损失: {loss:.4f}")
z = learner.encode(obs)
print(f"状态表示形状: {z.shape}")  # [32, 16]
```

### 练习2：实现对比状态表示学习

```python
class ContrastiveStateLearner:
    def __init__(self, obs_dim, latent_dim, hidden_dim=256, lr=1e-3):
        self.online_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # 冻结目标编码器
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.online_encoder.parameters(), lr=lr)
    
    def train_step(self, obs1, obs2, tau=0.99):
        """
        对比训练步骤
        
        参数:
            obs1, obs2: 对应的两个观测
            tau: 目标网络更新率
        """
        # 在线网络预测
        z_online = self.online_encoder(obs1)
        h_online = self.projector(z_online)
        
        # 目标网络预测
        with torch.no_grad():
            z_target = self.target_encoder(obs2)
            h_target = self.projector(z_target)
        
        # 归一化
        h_online = F.normalize(h_online, dim=-1)
        h_target = F.normalize(h_target, dim=-1)
        
        # 对比损失
        similarity = torch.mm(h_online, h_target.T)
        labels = torch.arange(similarity.size(0))
        loss = F.cross_entropy(similarity, labels)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 软更新目标网络
        self._update_target(tau)
        
        return loss.item()
    
    def _update_target(self, tau):
        """软更新目标网络"""
        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data

# 测试
learner = ContrastiveStateLearner(obs_dim=100, latent_dim=16)
obs1 = torch.randn(32, 100)
obs2 = torch.randn(32, 100)  # 同一状态的另一个视图
loss = learner.train_step(obs1, obs2)
print(f"对比损失: {loss:.4f}")
```

### 练习3：实现多尺度状态表示

```python
class MultiScaleStateRepresentation(nn.Module):
    def __init__(self, obs_dim, latent_dims=[8, 16, 32], hidden_dim=128):
        super().__init__()
        self.scales = nn.ModuleList()
        
        for latent_dim in latent_dims:
            self.scales.append(nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            ))
        
        self.fusion = nn.Sequential(
            nn.Linear(sum(latent_dims), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dims[-1])
        )
    
    def forward(self, obs):
        """
        前向传播
        
        参数:
            obs: 观测 [batch, obs_dim]
        
        返回:
            多尺度融合表示 [batch, latent_dims[-1]]
        """
        # 多尺度编码
        scales = []
        for encoder in self.scales:
            scales.append(encoder(obs))
        
        # 拼接
        combined = torch.cat(scales, dim=-1)
        
        # 融合
        fused = self.fusion(combined)
        
        return fused
    
    def get_scale(self, obs, scale_idx):
        """获取特定尺度的表示"""
        return self.scales[scale_idx](obs)

# 测试
model = MultiScaleStateRepresentation(obs_dim=100, latent_dims=[8, 16, 32])
obs = torch.randn(32, 100)
z = model(obs)
print(f"融合表示形状: {z.shape}")  # [32, 32]

z0 = model.get_scale(obs, 0)
z1 = model.get_scale(obs, 1)
z2 = model.get_scale(obs, 2)
print(f"尺度0表示形状: {z0.shape}")  # [32, 8]
print(f"尺度1表示形状: {z1.shape}")  # [32, 16]
print(f"尺度2表示形状: {z2.shape}")  # [32, 32]
```

---

**下一节**：[长期预测](04-long-horizon-prediction.md)

---

## 参考文献

1. Higgins, I., et al. (2017). d-VAE: Learning Variational Autoencoders.
2. Sahani, M., & Dayan, P. (2020). Doubly Sparse Gaussian Processes.
3. Greff, K., et al. (2019). Multi-Object Representation Learning.
