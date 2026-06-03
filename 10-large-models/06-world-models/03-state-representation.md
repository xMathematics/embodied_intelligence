# 6.3 状态表示学习

## 目录

- [1. 引言](#1-引言)
- [2. 状态表示学习概述](#2-状态表示学习概述)
- [3. 表示学习方法](#3-表示学习方法)
- [4. 表示学习架构](#4-表示学习架构)
- [5. 表示学习评估指标](#5-表示学习评估指标)
- [6. 表示学习优化策略](#6-表示学习优化策略)
- [7. 表示学习工程实践](#7-表示学习工程实践)
- [8. 理论基础与分析](#8-理论基础与分析)
- [9. 前沿研究方向](#9-前沿研究方向)
- [10. 实践练习](#10-实践练习)

---

## 1. 引言

### 1.1 状态表示学习的重要性

**状态表示学习**是指从高维观测（如图像）中学习低维状态表示的过程。这是世界模型的关键组件，能够提取环境的关键信息而忽略无关细节。

在强化学习和世界模型中，状态表示学习具有以下重要作用：
- **降维**：将高维观测压缩到低维空间
- **去噪**：去除观测中的噪声和无关信息
- **抽象**：提取环境的本质特征
- **泛化**：支持在不同任务间迁移

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **机器人操控** | 从图像学习机器人状态 | 视觉反馈控制 |
| **自动驾驶** | 理解道路场景 | 车道线检测、障碍物识别 |
| **游戏AI** | 从游戏画面学习 | Atari游戏、策略游戏 |
| **医学影像** | 分析医学图像 | CT/MRI分析、病变检测 |
| **视频理解** | 从视频序列学习 | 动作识别、异常检测 |

### 1.3 状态表示学习的挑战

| 挑战 | 描述 |
|------|------|
| **观测噪声** | 原始观测可能包含大量噪声 |
| **部分可观测性** | 观测可能无法完全反映真实状态 |
| **维度爆炸** | 高维观测（如图像）带来的计算挑战 |
| **表示质量** | 需要学习有意义、可解释的表示 |

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
| **可预测性** | 支持准确的动态预测 |
| **解耦性** | 不同因素相互独立 |

### 2.3 表示学习的分类体系

```
状态表示学习
├── 基于重构
│   ├── 自动编码器
│   ├── 变分自编码器
│   └── 生成对抗网络
├── 基于对比
│   ├── InfoNCE
│   ├── SimCLR
│   └── MoCo
├── 基于预测
│   ├── 动态预测损失
│   ├── 未来观测预测
│   └── 动作条件预测
└── 基于结构
    ├── 解耦表示
    ├── 对象级表示
    └── 因果表示
```

---

## 3. 表示学习方法

### 3.1 自动编码器

**方法**：学习压缩和解压表示，通过最小化重构误差来学习有用的特征。

```python
class Autoencoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        
        # 编码器
        encoder_layers = []
        input_dim = obs_dim
        
        for i in range(num_layers):
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 解码器
        decoder_layers = []
        input_dim = latent_dim
        
        for i in range(num_layers):
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dim, obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
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
    
    def compute_loss(self, obs):
        """计算重构损失"""
        obs_recon, z = self.forward(obs)
        recon_loss = nn.functional.mse_loss(obs_recon, obs)
        return recon_loss
```

### 3.2 变分自编码器 (VAE)

**方法**：学习潜在空间的概率分布，通过最大化证据下界（ELBO）来训练。

```python
class VAE(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        
        # 编码器
        encoder_layers = []
        input_dim = obs_dim
        
        for i in range(num_layers):
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        # 输出均值和方差
        self.encoder = nn.Sequential(*encoder_layers)
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        decoder_layers = []
        input_dim = latent_dim
        
        for i in range(num_layers):
            decoder_layers.append(nn.Linear(input_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dim, obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, obs):
        """前向传播"""
        h = self.encoder(obs)
        mu = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        obs_recon = self.decoder(z)
        return obs_recon, mu, logvar, z
    
    def kl_loss(self, mu, logvar):
        """KL散度损失"""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    def elbo_loss(self, obs, beta=1.0):
        """计算ELBO损失"""
        obs_recon, mu, logvar, z = self.forward(obs)
        
        # 重构损失
        recon_loss = nn.functional.mse_loss(obs_recon, obs)
        
        # KL散度
        kl = self.kl_loss(mu, logvar)
        
        return recon_loss + beta * kl
```

### 3.3 对比表示学习

**方法**：通过最大化正样本对的相似度、最小化负样本对的相似度来学习表示。

```python
class ContrastiveRepresentation(nn.Module):
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
    
    def info_nce_loss(self, obs_anchor, obs_positive, obs_negatives, temperature=0.1):
        """
        InfoNCE对比损失
        
        参数:
            obs_anchor: 锚点观测 [batch, obs_dim]
            obs_positive: 正样本观测 [batch, obs_dim]
            obs_negatives: 负样本观测 [batch, num_neg, obs_dim]
        """
        # 编码
        z_anchor = self.encoder(obs_anchor)
        z_positive = self.encoder(obs_positive)
        z_negatives = self.encoder(obs_negatives.view(-1, obs_negatives.size(-1)))
        z_negatives = z_negatives.view(obs_negatives.size(0), obs_negatives.size(1), -1)
        
        # 投影
        h_anchor = self.projector(z_anchor)
        h_positive = self.projector(z_positive)
        h_negatives = self.projector(z_negatives)
        
        # 归一化
        h_anchor = F.normalize(h_anchor, dim=-1)
        h_positive = F.normalize(h_positive, dim=-1)
        h_negatives = F.normalize(h_negatives, dim=-1)
        
        # 正样本相似度
        pos_sim = torch.sum(h_anchor * h_positive, dim=-1, keepdim=True) / temperature
        
        # 负样本相似度
        neg_sim = torch.bmm(h_anchor.unsqueeze(1), h_negatives.transpose(1, 2)).squeeze(1) / temperature
        
        # InfoNCE损失
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(logits.size(0), dtype=torch.long)
        loss = F.cross_entropy(logits, labels)
        
        return loss
```

### 3.4 因果表示学习

**方法**：学习因果关系而非相关性，支持干预和反事实推理。

```python
class CausalRepresentation(nn.Module):
    def __init__(self, obs_dim, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # 观测到状态的映射
        self.observation_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 因果机制（每个维度独立）
        self.causal_mechanisms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(state_dim)
        ])
    
    def forward(self, obs, action):
        """前向传播"""
        state = self.observation_net(obs)
        
        # 应用因果机制
        next_state = []
        for i, mechanism in enumerate(self.causal_mechanisms):
            input_i = torch.cat([state, action], dim=-1)
            delta_i = mechanism(input_i)
            next_state.append(state[:, i:i+1] + delta_i)
        
        return torch.cat(next_state, dim=-1)
    
    def intervene(self, obs, action, intervention_idx, intervention_value):
        """
        执行干预操作
        
        参数:
            obs: 观测
            action: 动作
            intervention_idx: 干预的状态维度索引
            intervention_value: 干预值
        """
        state = self.observation_net(obs)
        
        # 执行干预
        state = state.clone()
        state[:, intervention_idx] = intervention_value
        
        # 预测下一状态
        next_state = []
        for i, mechanism in enumerate(self.causal_mechanisms):
            input_i = torch.cat([state, action], dim=-1)
            delta_i = mechanism(input_i)
            next_state.append(state[:, i:i+1] + delta_i)
        
        return torch.cat(next_state, dim=-1)
```

---

## 4. 表示学习架构

### 4.1 卷积编码器

**适用于图像观测**，能够自动学习空间特征。

```python
class ConvEncoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_channels=[32, 64, 128, 256]):
        super().__init__()
        
        conv_layers = []
        current_channels = in_channels
        
        for channels in hidden_channels:
            conv_layers.append(nn.Conv2d(current_channels, channels, kernel_size=4, stride=2, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.BatchNorm2d(channels))
            current_channels = channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 计算输出尺寸
        self.fc_input_dim = current_channels * 4 * 4  # 假设输入为64x64
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: [batch, channels, height, width]
        """
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels, hidden_channels=[256, 128, 64, 32]):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_channels[0] * 4 * 4)
        )
        
        deconv_layers = []
        current_channels = hidden_channels[0]
        
        for i, channels in enumerate(hidden_channels[1:]):
            deconv_layers.append(nn.ConvTranspose2d(current_channels, channels, kernel_size=4, stride=2, padding=1))
            deconv_layers.append(nn.ReLU())
            deconv_layers.append(nn.BatchNorm2d(channels))
            current_channels = channels
        
        deconv_layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=4, stride=2, padding=1))
        deconv_layers.append(nn.Sigmoid())
        
        self.deconv_layers = nn.Sequential(*deconv_layers)
    
    def forward(self, z):
        """前向传播"""
        h = self.fc(z)
        h = h.view(h.size(0), -1, 4, 4)
        x_recon = self.deconv_layers(h)
        return x_recon
```

### 4.2 Transformer编码器

**适用于序列表示**，能够建模长程依赖关系。

```python
class TransformerEncoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, num_heads=8, num_layers=4, max_seq_len=100):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Linear(obs_dim, latent_dim)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, latent_dim))
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
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
        
        # 添加位置编码
        seq_len = obs_seq.size(1)
        h = h + self.pos_embedding[:, :seq_len, :]
        
        # Transformer编码
        h = self.transformer(h)
        
        # 取最后一个时刻的输出作为状态表示
        z = h[:, -1, :]
        z = self.fc(z)
        
        return z
    
    def get_sequence_representation(self, obs_seq):
        """获取整个序列的表示"""
        h = self.embedding(obs_seq)
        seq_len = obs_seq.size(1)
        h = h + self.pos_embedding[:, :seq_len, :]
        h = self.transformer(h)
        return h
```

### 4.3 GRU编码器

**适用于时序数据**，能够捕捉时间依赖关系。

```python
class GRUEncoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, num_layers=2, bidirectional=False):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # 如果是双向GRU，需要合并两个方向的输出
        self.fc = nn.Linear(latent_dim * (2 if bidirectional else 1), latent_dim)
    
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
        if self.gru.bidirectional:
            # 合并两个方向
            hidden = hidden[-2:]  # 最后两层（前向和后向）
            hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
        else:
            hidden = hidden[-1]
        
        # 线性变换
        z = self.fc(hidden)
        
        return z
    
    def get_hidden_states(self, obs_seq):
        """获取所有时刻的隐藏状态"""
        output, _ = self.gru(obs_seq)
        return output
```

### 4.4 混合编码器

**结合多种编码器的优势**，适用于复杂场景。

```python
class HybridEncoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim=256):
        super().__init__()
        
        # 时序编码器（GRU）
        self.temporal_encoder = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # 特征编码器（MLP）
        self.feature_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, obs_seq):
        """
        前向传播
        
        参数:
            obs_seq: 观测序列 [batch, seq_len, obs_dim]
        
        返回:
            融合状态表示 [batch, latent_dim]
        """
        # 时序编码
        _, temporal_hidden = self.temporal_encoder(obs_seq)
        temporal_z = temporal_hidden[-1]
        
        # 特征编码（取最后一个时刻的观测）
        last_obs = obs_seq[:, -1, :]
        feature_z = self.feature_encoder(last_obs)
        
        # 融合
        combined = torch.cat([temporal_z, feature_z], dim=-1)
        z = self.fusion(combined)
        
        return z
```

---

## 5. 表示学习评估指标

### 5.1 重构质量指标

```python
class RepresentationEvaluator:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
    
    def reconstruction_error(self, obs):
        """计算重构误差"""
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            z = self.encoder(obs)
            obs_recon = self.decoder(z)
            mse = nn.functional.mse_loss(obs_recon, obs).item()
            mae = nn.functional.l1_loss(obs_recon, obs).item()
        
        return {'mse': mse, 'mae': mae}
    
    def latent_diversity(self, obs):
        """计算潜在空间的多样性"""
        self.encoder.eval()
        
        with torch.no_grad():
            z = self.encoder(obs)
            # 计算协方差矩阵的秩
            cov_matrix = torch.cov(z.T)
            rank = torch.matrix_rank(cov_matrix).item()
            # 计算平均方差
            variance = torch.var(z, dim=0).mean().item()
        
        return {'rank': rank, 'mean_variance': variance}
    
    def linear_separability(self, obs, labels):
        """评估线性可分性"""
        self.encoder.eval()
        
        with torch.no_grad():
            z = self.encoder(obs)
        
        # 使用逻辑回归评估
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        z_np = z.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(z_np, labels_np, test_size=0.2)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        
        return accuracy
```

### 5.2 预测质量指标

```python
class PredictiveEvaluator:
    def __init__(self, encoder, dynamics_model):
        self.encoder = encoder
        self.dynamics = dynamics_model
    
    def prediction_error(self, obs_seq, actions):
        """
        评估预测误差
        
        参数:
            obs_seq: 观测序列 [batch, seq_len, obs_dim]
            actions: 动作序列 [batch, seq_len-1, action_dim]
        """
        self.encoder.eval()
        self.dynamics.eval()
        
        with torch.no_grad():
            # 编码初始观测
            z = self.encoder(obs_seq[:, 0])
            
            errors = []
            for t in range(actions.size(1)):
                # 预测下一状态
                z_pred = self.dynamics(z, actions[:, t])
                
                # 编码真实下一观测
                z_true = self.encoder(obs_seq[:, t+1])
                
                # 计算误差
                error = torch.norm(z_pred - z_true, dim=-1).mean().item()
                errors.append(error)
                
                z = z_pred
        
        return errors
    
    def long_term_prediction(self, initial_obs, actions, decoder):
        """
        评估长期预测性能
        
        参数:
            initial_obs: 初始观测 [batch, obs_dim]
            actions: 动作序列 [batch, seq_len, action_dim]
            decoder: 解码器
        """
        self.encoder.eval()
        self.dynamics.eval()
        decoder.eval()
        
        with torch.no_grad():
            # 编码初始观测
            z = self.encoder(initial_obs)
            
            predictions = [initial_obs]
            
            for t in range(actions.size(1)):
                # 预测下一状态
                z = self.dynamics(z, actions[:, t])
                # 解码为观测
                obs_pred = decoder(z)
                predictions.append(obs_pred)
        
        return torch.stack(predictions, dim=1)
```

---

## 6. 表示学习优化策略

### 6.1 正则化技术

```python
class RegularizedAutoencoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, obs_dim)
        )
    
    def forward(self, obs):
        z = self.encoder(obs)
        obs_recon = self.decoder(z)
        return obs_recon, z
    
    def compute_loss(self, obs, lambda_l2=1e-4, lambda_sparse=1e-3):
        """
        计算带正则化的损失
        
        参数:
            lambda_l2: L2正则化权重
            lambda_sparse: 稀疏正则化权重
        """
        obs_recon, z = self.forward(obs)
        
        # 重构损失
        recon_loss = nn.functional.mse_loss(obs_recon, obs)
        
        # L2正则化
        l2_loss = sum(p.pow(2).sum() for p in self.parameters())
        
        # 稀疏正则化（L1）
        sparse_loss = torch.abs(z).mean()
        
        return recon_loss + lambda_l2 * l2_loss + lambda_sparse * sparse_loss
```

### 6.2 数据增强

```python
class RepresentationDataAugmenter:
    def __init__(self, noise_std=0.01, crop_size=None, flip_prob=0.5):
        self.noise_std = noise_std
        self.crop_size = crop_size
        self.flip_prob = flip_prob
    
    def add_noise(self, obs):
        """添加高斯噪声"""
        noise = torch.randn_like(obs) * self.noise_std
        return obs + noise
    
    def random_crop(self, obs):
        """随机裁剪（适用于图像）"""
        if self.crop_size is None:
            return obs
        
        batch_size, channels, height, width = obs.shape
        crop_h, crop_w = self.crop_size
        
        top = torch.randint(0, height - crop_h + 1, (batch_size,))
        left = torch.randint(0, width - crop_w + 1, (batch_size,))
        
        cropped = []
        for i in range(batch_size):
            cropped.append(obs[i, :, top[i]:top[i]+crop_h, left[i]:left[i]+crop_w])
        
        return torch.stack(cropped)
    
    def random_flip(self, obs):
        """随机水平翻转（适用于图像）"""
        if torch.rand(1) < self.flip_prob:
            obs = torch.flip(obs, dims=[-1])
        return obs
    
    def augment(self, obs):
        """应用数据增强"""
        obs = self.add_noise(obs)
        
        if len(obs.shape) == 4:  # 图像数据
            obs = self.random_flip(obs)
            if self.crop_size:
                obs = self.random_crop(obs)
        
        return obs
```

### 6.3 自监督学习策略

```python
class SelfSupervisedRepresentationLearner:
    def __init__(self, encoder, predictor, lr=1e-3):
        self.encoder = encoder
        self.predictor = predictor
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=lr
        )
    
    def temporal_contrastive_loss(self, obs_seq):
        """
        时间对比损失
        
        参数:
            obs_seq: 观测序列 [batch, seq_len, obs_dim]
        """
        # 编码所有时刻
        z_seq = self.encoder(obs_seq)
        
        losses = []
        
        for t in range(obs_seq.size(1) - 1):
            # 当前时刻表示
            z_t = z_seq[:, t]
            
            # 下一时刻表示（正样本）
            z_t1 = z_seq[:, t+1]
            
            # 预测下一时刻
            z_pred = self.predictor(z_t)
            
            # 对比损失
            pos_sim = F.cosine_similarity(z_pred, z_t1, dim=-1)
            
            # 负样本：其他时刻的表示
            negatives = []
            for tt in range(obs_seq.size(1)):
                if tt != t + 1:
                    negatives.append(z_seq[:, tt])
            
            if negatives:
                z_neg = torch.stack(negatives, dim=1)
                neg_sim = F.cosine_similarity(z_pred.unsqueeze(1), z_neg, dim=-1)
                
                # InfoNCE
                logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1) / 0.1
                labels = torch.zeros(logits.size(0), dtype=torch.long)
                loss = F.cross_entropy(logits, labels)
                losses.append(loss)
        
        return sum(losses) / len(losses)
    
    def train_step(self, obs_seq):
        """训练步骤"""
        self.optimizer.zero_grad()
        loss = self.temporal_contrastive_loss(obs_seq)
        loss.backward()
        self.optimizer.step()
        return loss.item()
```

---

## 7. 表示学习工程实践

### 7.1 模型压缩与加速

```python
class RepresentationModelCompressor:
    def __init__(self, encoder):
        self.encoder = encoder
    
    def prune_encoder(self, sparsity=0.5):
        """剪枝编码器"""
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                threshold = torch.quantile(torch.abs(param.data), sparsity)
                param.data[torch.abs(param.data) < threshold] = 0.0
    
    def quantize_encoder(self, bits=8):
        """量化编码器"""
        scale = 2 ** bits - 1
        
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                max_val = torch.max(torch.abs(param.data))
                normalized = param.data / max_val
                quantized = torch.round(normalized * scale) / scale
                param.data = quantized * max_val
    
    def export_onnx(self, input_shape, output_path):
        """导出为ONNX格式"""
        dummy_input = torch.randn(*input_shape)
        
        torch.onnx.export(
            self.encoder,
            dummy_input,
            output_path,
            opset_version=11,
            input_names=['obs'],
            output_names=['z'],
            dynamic_axes={
                'obs': {0: 'batch_size'},
                'z': {0: 'batch_size'}
            }
        )
```

### 7.2 分布式训练

```python
class DistributedRepresentationTrainer:
    def __init__(self, model, lr=1e-3):
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        
        # 模型并行化
        self.model = torch.nn.parallel.DistributedDataParallel(model)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def train_step(self, obs):
        """分布式训练步骤"""
        obs = obs.to(f'cuda:{self.rank}')
        
        self.optimizer.zero_grad()
        obs_recon, z = self.model(obs)
        loss = nn.functional.mse_loss(obs_recon, obs)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### 7.3 在线学习与增量更新

```python
class OnlineRepresentationLearner:
    def __init__(self, encoder, decoder, lr=1e-4):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=lr
        )
        
        # EMA更新
        self.ema_alpha = 0.99
        self.encoder_shadow = copy.deepcopy(encoder)
        self.decoder_shadow = copy.deepcopy(decoder)
    
    def update_online(self, obs):
        """在线更新模型"""
        self.encoder.train()
        self.decoder.train()
        
        # 训练步骤
        self.optimizer.zero_grad()
        obs_recon, z = self.encoder(obs), self.decoder(self.encoder(obs))
        loss = nn.functional.mse_loss(obs_recon, obs)
        loss.backward()
        self.optimizer.step()
        
        # EMA更新影子网络
        self._update_shadow()
        
        return loss.item()
    
    def _update_shadow(self):
        """更新影子网络"""
        for online_param, shadow_param in zip(
            self.encoder.parameters(),
            self.encoder_shadow.parameters()
        ):
            shadow_param.data = self.ema_alpha * shadow_param.data + \
                               (1 - self.ema_alpha) * online_param.data
        
        for online_param, shadow_param in zip(
            self.decoder.parameters(),
            self.decoder_shadow.parameters()
        ):
            shadow_param.data = self.ema_alpha * shadow_param.data + \
                               (1 - self.ema_alpha) * online_param.data
```

---

## 8. 理论基础与分析

### 8.1 表示学习的理论保证

**信息瓶颈理论**：好的表示应该在保持预测能力的同时最小化信息容量。

$$\min I(X; Z) \quad \text{s.t.} \quad I(Y; Z) \geq I(Y; X)$$

其中：
- $X$：输入观测
- $Z$：学习到的表示
- $Y$：预测目标

### 8.2 表示质量的数学分析

```python
class RepresentationAnalyzer:
    def __init__(self, encoder):
        self.encoder = encoder
    
    def mutual_information_estimate(self, obs1, obs2):
        """
        估计表示之间的互信息
        
        参数:
            obs1, obs2: 两个相关观测
        """
        self.encoder.eval()
        
        with torch.no_grad():
            z1 = self.encoder(obs1)
            z2 = self.encoder(obs2)
        
        # 使用MINE估计互信息
        from torch.nn.functional import softplus
        
        # 联合分布采样
        joint = torch.cat([z1, z2], dim=-1)
        
        # 边缘分布采样（打乱配对）
        z2_shuffled = z2[torch.randperm(z2.size(0))]
        marginal = torch.cat([z1, z2_shuffled], dim=-1)
        
        # 判别器
        discriminator = nn.Sequential(
            nn.Linear(z1.size(-1) * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
        
        for _ in range(100):
            optimizer.zero_grad()
            
            # 联合分布得分
            joint_score = discriminator(joint)
            
            # 边缘分布得分
            marginal_score = discriminator(marginal)
            
            # MINE损失
            mine_loss = -torch.mean(joint_score) + \
                        torch.log(torch.mean(torch.exp(marginal_score)))
            
            mine_loss.backward()
            optimizer.step()
        
        # 互信息估计
        mi_estimate = torch.mean(joint_score) - \
                      torch.log(torch.mean(torch.exp(marginal_score)))
        
        return mi_estimate.item()
    
    def linearity_test(self, obs):
        """测试表示的线性可分性"""
        self.encoder.eval()
        
        with torch.no_grad():
            z = self.encoder(obs)
        
        # 计算协方差矩阵
        cov_matrix = torch.cov(z.T)
        
        # 特征值分解
        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        
        # 有效维度（贡献95%方差的维度数）
        total_variance = eigenvalues.sum()
        cumulative = 0
        effective_dim = 0
        
        for val in reversed(eigenvalues):
            cumulative += val
            effective_dim += 1
            if cumulative / total_variance >= 0.95:
                break
        
        return {
            'eigenvalues': eigenvalues.detach().cpu().numpy(),
            'effective_dimension': effective_dim
        }
```

### 8.3 表示学习的泛化能力

```python
class GeneralizationAnalyzer:
    def __init__(self, encoder):
        self.encoder = encoder
    
    def measure_generalization(self, train_obs, test_obs):
        """
        测量表示的泛化能力
        
        参数:
            train_obs: 训练集观测
            test_obs: 测试集观测
        """
        self.encoder.eval()
        
        with torch.no_grad():
            train_z = self.encoder(train_obs)
            test_z = self.encoder(test_obs)
        
        # 计算训练集和测试集表示的统计差异
        train_mean = train_z.mean(dim=0)
        test_mean = test_z.mean(dim=0)
        mean_diff = torch.norm(train_mean - test_mean)
        
        train_cov = torch.cov(train_z.T)
        test_cov = torch.cov(test_z.T)
        cov_diff = torch.norm(train_cov - test_cov)
        
        return {
            'mean_difference': mean_diff.item(),
            'covariance_difference': cov_diff.item()
        }
```

---

## 9. 前沿研究方向

### 9.1 对象级表示学习

```python
class ObjectCentricRepresentation(nn.Module):
    def __init__(self, obs_dim, num_objects, object_dim, hidden_dim=256):
        super().__init__()
        
        self.num_objects = num_objects
        self.object_dim = object_dim
        
        # 对象提取器
        self.object_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objects * object_dim)
        )
        
        # 对象注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=object_dim,
            num_heads=4,
            batch_first=True
        )
        
        # 对象间关系建模
        self.relation_net = nn.Sequential(
            nn.Linear(object_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, object_dim)
        )
    
    def forward(self, obs):
        """
        前向传播
        
        参数:
            obs: 观测 [batch, obs_dim]
        
        返回:
            对象级表示 [batch, num_objects, object_dim]
        """
        # 提取对象表示
        h = self.object_extractor(obs)
        objects = h.view(-1, self.num_objects, self.object_dim)
        
        # 对象自注意力
        attended, _ = self.attention(objects, objects, objects)
        
        # 对象间关系建模
        relations = []
        for i in range(self.num_objects):
            for j in range(self.num_objects):
                if i != j:
                    relation = self.relation_net(torch.cat([
                        attended[:, i], 
                        attended[:, j]
                    ], dim=-1))
                    relations.append(relation)
        
        # 聚合关系信息
        if relations:
            relations = torch.stack(relations, dim=1)
            relation_info = relations.mean(dim=1)
            objects = objects + relation_info.unsqueeze(1)
        
        return objects
    
    def get_object(self, obs, object_idx):
        """获取特定对象的表示"""
        objects = self.forward(obs)
        return objects[:, object_idx]
```

### 9.2 层次化表示学习

```python
class HierarchicalRepresentation(nn.Module):
    def __init__(self, obs_dim, layer_dims=[32, 64, 128], hidden_dim=256):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # 逐层编码器
        for i, dim in enumerate(layer_dims):
            if i == 0:
                input_dim = obs_dim
            else:
                input_dim = layer_dims[i-1]
            
            self.layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim)
            ))
        
        # 跨层连接
        self.skip_connections = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.skip_connections.append(
                nn.Linear(layer_dims[i], layer_dims[i+1])
            )
    
    def forward(self, obs, return_all_layers=False):
        """
        前向传播
        
        参数:
            obs: 观测 [batch, obs_dim]
            return_all_layers: 是否返回所有层级的表示
        
        返回:
            最高层级的表示（或所有层级）
        """
        representations = []
        current = obs
        
        for i, layer in enumerate(self.layers):
            current = layer(current)
            representations.append(current)
            
            # 跨层连接
            if i < len(self.skip_connections):
                skip_output = self.skip_connections[i](representations[0])
                current = current + skip_output
        
        if return_all_layers:
            return representations
        else:
            return representations[-1]
```

### 9.3 持续学习表示

```python
class ContinualRepresentationLearner:
    def __init__(self, obs_dim, latent_dim, hidden_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # EWC参数
        self.ewc_lambda = 1.0
        self.fisher_information = None
        self.old_params = None
    
    def compute_fisher(self, dataloader):
        """计算Fisher信息矩阵"""
        self.encoder.train()
        fisher = {}
        
        for param_name, param in self.encoder.named_parameters():
            fisher[param_name] = torch.zeros_like(param)
        
        for obs in dataloader:
            self.encoder.zero_grad()
            z = self.encoder(obs)
            # 使用表示的方差作为损失
            loss = z.var().mean()
            loss.backward()
            
            for param_name, param in self.encoder.named_parameters():
                if param.grad is not None:
                    fisher[param_name] += param.grad.pow(2)
        
        # 归一化
        num_batches = len(dataloader)
        for key in fisher:
            fisher[key] /= num_batches
        
        self.fisher_information = fisher
        self.old_params = {name: param.detach().clone() 
                          for name, param in self.encoder.named_parameters()}
    
    def ewc_loss(self, obs):
        """计算带EWC正则化的损失"""
        z = self.encoder(obs)
        
        # 重建损失（简化）
        recon_loss = z.norm()
        
        # EWC正则化
        ewc_loss = 0.0
        if self.fisher_information is not None and self.old_params is not None:
            for param_name, param in self.encoder.named_parameters():
                ewc_loss += (self.fisher_information[param_name] * 
                            (param - self.old_params[param_name]).pow(2)).sum()
        
        return recon_loss + self.ewc_lambda * ewc_loss
```

---

## 10. 实践练习

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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
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
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()
    
    def encode(self, obs):
        """编码观测到状态"""
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(obs)
    
    def decode(self, z):
        """从状态解码到观测"""
        self.decoder.eval()
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
        
        # 初始化目标编码器与在线编码器相同
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        
        # 冻结目标编码器
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.online_encoder.parameters()) + list(self.projector.parameters()),
            lr=lr
        )
    
    def train_step(self, obs1, obs2, tau=0.99, temperature=0.1):
        """
        对比训练步骤
        
        参数:
            obs1, obs2: 对应的两个观测（同一状态的不同视图）
            tau: 目标网络更新率
            temperature: 温度参数
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
        
        # 对比损失（SimCLR风格）
        similarity = torch.mm(h_online, h_target.T) / temperature
        
        # 正样本在对角线上
        batch_size = obs1.size(0)
        labels = torch.arange(batch_size)
        
        # 双向对比
        loss = (F.cross_entropy(similarity, labels) + 
                F.cross_entropy(similarity.T, labels)) / 2
        
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
        self.latent_dims = latent_dims
        
        for latent_dim in latent_dims:
            self.scales.append(nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            ))
        
        # 跨尺度注意力融合
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=latent_dims[-1],
            num_heads=4,
            batch_first=True
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(sum(latent_dims), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dims[-1])
        )
    
    def forward(self, obs, return_all_scales=False):
        """
        前向传播
        
        参数:
            obs: 观测 [batch, obs_dim]
            return_all_scales: 是否返回所有尺度
        
        返回:
            多尺度融合表示 [batch, latent_dims[-1]]
        """
        # 多尺度编码
        scale_outs = []
        for encoder in self.scales:
            scale_outs.append(encoder(obs))
        
        # 调整到相同维度用于注意力
        aligned_scales = []
        for i, scale in enumerate(scale_outs):
            if scale.size(-1) != self.latent_dims[-1]:
                aligned = nn.Linear(scale.size(-1), self.latent_dims[-1])(scale)
            else:
                aligned = scale
            aligned_scales.append(aligned.unsqueeze(1))
        
        # 跨尺度注意力
        scales_tensor = torch.cat(aligned_scales, dim=1)
        attended, _ = self.cross_scale_attention(scales_tensor, scales_tensor, scales_tensor)
        
        # 拼接所有尺度
        combined = torch.cat(scale_outs, dim=-1)
        
        # 融合
        fused = self.fusion(combined)
        
        if return_all_scales:
            return fused, scale_outs
        else:
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

### 练习4：实现基于Transformer的状态表示学习

```python
class TransformerStateEncoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, num_heads=4, num_layers=2, max_seq_len=50):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Linear(obs_dim, latent_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, latent_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 聚合层
        self.aggregator = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, obs_seq):
        """
        前向传播
        
        参数:
            obs_seq: 观测序列 [batch, seq_len, obs_dim]
        
        返回:
            聚合状态表示 [batch, latent_dim]
        """
        # 嵌入
        h = self.embedding(obs_seq)
        
        # 添加位置编码
        seq_len = obs_seq.size(1)
        h = h + self.pos_encoding[:, :seq_len, :]
        
        # Transformer编码
        h = self.transformer(h)
        
        # 时间注意力聚合
        weights = F.softmax(h @ h[:, -1, :].unsqueeze(-1), dim=1)
        aggregated = (h * weights).sum(dim=1)
        
        # 最终处理
        z = self.aggregator(aggregated)
        
        return z
    
    def get_sequence_embeddings(self, obs_seq):
        """获取序列中每个时刻的嵌入"""
        h = self.embedding(obs_seq)
        seq_len = obs_seq.size(1)
        h = h + self.pos_encoding[:, :seq_len, :]
        h = self.transformer(h)
        return h

# 测试
encoder = TransformerStateEncoder(obs_dim=64, latent_dim=32, num_heads=4, num_layers=2)
obs_seq = torch.randn(16, 10, 64)  # [batch, seq_len, obs_dim]
z = encoder(obs_seq)
print(f"聚合状态表示形状: {z.shape}")  # [16, 32]

seq_emb = encoder.get_sequence_embeddings(obs_seq)
print(f"序列嵌入形状: {seq_emb.shape}")  # [16, 10, 32]
```

---

**下一节**：[长期预测](04-long-horizon-prediction.md)

---

## 参考文献

1. Higgins, I., et al. (2017). d-VAE: Learning Variational Autoencoders.
2. Sahani, M., & Dayan, P. (2020). Doubly Sparse Gaussian Processes.
3. Greff, K., et al. (2019). Multi-Object Representation Learning.
4. Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations.
5. He, K., et al. (2020). Momentum Contrast for Unsupervised Visual Representation Learning.
6. Vaswani, A., et al. (2017). Attention is All You Need.
7. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
8. Rezende, D. J., et al. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models.
9. Bengio, Y., et al. (2013). Representation Learning: A Review and New Perspectives.
10. Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the Dimensionality of Data with Neural Networks.
