# 音频-语言模型

## 目录

- [1. 概述](#1-概述)
- [2. 音频特征提取](#2-音频特征提取)
- [3. 音频-语言模型架构](#3-音频-语言模型架构)
- [4. 代表性模型详解](#4-代表性模型详解)
- [5. 预训练策略](#5-预训练策略)
- [6. 进阶话题](#6-进阶话题)
- [7. 实战项目案例](#7-实战项目案例)
- [8. 模型优化与部署](#8-模型优化与部署)
- [9. 未来方向](#9-未来方向)

---

## 1. 概述

### 1.1 音频-语言模型的定义

**音频-语言模型**是指能够同时处理音频和文本两种模态的人工智能模型。这类模型可以：

- **语音识别**：将音频转换为文本
- **语音合成**：将文本转换为音频
- **语音理解**：理解语音中的语义信息
- **语音翻译**：将一种语言的语音翻译成另一种语言

### 1.2 应用场景

| 应用领域 | 具体场景 | 示例 |
|---------|---------|------|
| **智能助手** | 语音交互 | Siri、Alexa |
| **视频会议** | 实时字幕 | Zoom字幕功能 |
| **教育领域** | 语音学习 | 语言学习APP |
| **医疗领域** | 医疗记录 | 语音病历录入 |
| **安防领域** | 语音识别 | 语音命令识别 |

### 1.3 发展历程

**早期阶段**：
- 基于隐马尔可夫模型(HMM)的语音识别
- 基于规则的语音合成

**深度学习时代**：
- 基于RNN/CNN的语音识别
- WaveNet语音合成

**预训练时代**：
- Wav2Vec 2.0自监督学习
- Whisper大规模语音识别
- AudioLM语音生成

---

## 2. 音频特征提取

### 2.1 音频信号基础

**音频信号特性**：
- 连续时间信号
- 频率范围：20Hz - 20kHz（人类听觉范围）
- 采样率：常见为16kHz、22.05kHz、44.1kHz

**音频表示**：
```python
import librosa
import numpy as np

# 加载音频文件
audio, sr = librosa.load('audio.wav', sr=16000)

# 音频特征
print(f"音频长度: {len(audio) / sr:.2f}秒")
print(f"采样率: {sr} Hz")
print(f"音频数据类型: {type(audio)}")
print(f"音频形状: {audio.shape}")
```

### 2.2 梅尔频谱图

**梅尔频率倒谱系数(MFCC)**：
```python
def extract_mfcc(audio, sr=16000, n_mfcc=13):
    """提取MFCC特征"""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=512,
        hop_length=256
    )
    return mfcc

# 提取MFCC
mfcc = extract_mfcc(audio, sr=16000)
print(f"MFCC形状: {mfcc.shape}")  # [n_mfcc, time_steps]
```

**梅尔频谱图**：
```python
def extract_melspectrogram(audio, sr=16000, n_mels=80):
    """提取梅尔频谱图"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=512,
        hop_length=256,
        n_mels=n_mels
    )
    # 转换为对数刻度
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec

# 提取梅尔频谱图
mel_spec = extract_melspectrogram(audio, sr=16000)
print(f"梅尔频谱图形状: {mel_spec.shape}")  # [n_mels, time_steps]
```

### 2.3 波形特征

**波形到特征的转换**：
```python
class WaveformProcessor(nn.Module):
    """波形处理器"""
    
    def __init__(self, sample_rate=16000, n_mels=80, n_fft=512, hop_length=256):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # 梅尔滤波器组
        self.mel_filter = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    
    def forward(self, waveform):
        # waveform: [batch, time]
        
        # STFT
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )
        
        # 幅度谱
        magnitude = torch.abs(stft)  # [batch, freq_bins, time_steps]
        
        # 梅尔频谱
        mel_spec = torch.matmul(
            torch.tensor(self.mel_filter).to(waveform.device),
            magnitude
        )  # [batch, n_mels, time_steps]
        
        # 对数压缩
        log_mel = torch.log(mel_spec + 1e-10)
        
        return log_mel
```

### 2.4 音频特征对比

| 特征类型 | 维度 | 特点 | 适用场景 |
|---------|------|------|---------|
| 原始波形 | 1D序列 | 保留完整信息 | 端到端模型 |
| MFCC | 13-40维 | 人耳感知相关 | 传统语音识别 |
| 梅尔频谱图 | 80-128维 | 频率信息丰富 | 深度学习模型 |
| 频谱图 | 257-513维 | 完整频率信息 | 需要细粒度分析 |

---

## 3. 音频-语言模型架构

### 3.1 序列到序列模型

**编码器-解码器架构**：
```python
class Seq2SeqASR(nn.Module):
    """序列到序列语音识别模型"""
    
    def __init__(self, audio_dim=80, hidden_dim=512, vocab_size=10000):
        super().__init__()
        
        # 编码器
        self.encoder = nn.LSTM(
            audio_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=3
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            hidden_dim * 2 + vocab_size,
            hidden_dim,
            batch_first=True,
            num_layers=2
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, audio_feat, text_input):
        # audio_feat: [batch, time_steps, audio_dim]
        # text_input: [batch, seq_len]
        
        # 编码
        encoder_out, (hidden, cell) = self.encoder(audio_feat)
        
        # 解码器输入（teacher forcing）
        batch_size = audio_feat.shape[0]
        decoder_input = torch.zeros(batch_size, 1, self.hidden_dim * 2 + self.vocab_size)
        
        outputs = []
        for t in range(text_input.shape[1]):
            decoder_out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            logits = self.classifier(decoder_out)
            outputs.append(logits)
            
            # 下一个输入
            decoder_input = torch.cat([
                encoder_out[:, t:t+1, :],
                F.one_hot(text_input[:, t], num_classes=self.vocab_size).float().unsqueeze(1)
            ], dim=-1)
        
        return torch.cat(outputs, dim=1)
```

### 3.2 注意力机制

**带注意力的解码器**：
```python
class AttentionDecoder(nn.Module):
    """带注意力的解码器"""
    
    def __init__(self, hidden_dim=512, vocab_size=10000):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(hidden_dim, 8)
        self.decoder = nn.LSTMCell(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, encoder_out, decoder_hidden, decoder_cell, text_emb):
        # encoder_out: [time_steps, batch, hidden_dim]
        # decoder_hidden: [batch, hidden_dim]
        
        # 注意力
        query = decoder_hidden.unsqueeze(0)  # [1, batch, hidden_dim]
        context, weights = self.attention(query, encoder_out, encoder_out)
        context = context.squeeze(0)  # [batch, hidden_dim]
        
        # 解码器
        decoder_hidden, decoder_cell = self.decoder(text_emb, (decoder_hidden, decoder_cell))
        
        # 融合上下文
        fused = decoder_hidden + context
        
        # 预测
        logits = self.classifier(fused)
        
        return logits, decoder_hidden, decoder_cell, weights
```

### 3.3 端到端模型

**端到端语音识别**：
```python
class EndToEndASR(nn.Module):
    """端到端语音识别模型"""
    
    def __init__(self, vocab_size=10000, hidden_dim=512):
        super().__init__()
        
        # 卷积层提取局部特征
        self.conv_layers = nn.Sequential(
            nn.Conv1d(80, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # CTC分类器
        self.ctc_classifier = nn.Linear(256, vocab_size)
    
    def forward(self, audio_feat):
        # audio_feat: [batch, n_mels, time_steps]
        
        # 卷积提取特征
        conv_out = self.conv_layers(audio_feat)  # [batch, 256, time_steps/2]
        conv_out = conv_out.transpose(1, 2)  # [batch, time_steps/2, 256]
        
        # Transformer编码
        transformer_out = self.transformer(conv_out.transpose(0, 1)).transpose(0, 1)
        
        # CTC预测
        logits = self.ctc_classifier(transformer_out)  # [batch, time_steps/2, vocab_size]
        
        return logits
```

### 3.4 语音合成模型

**Tacotron 2风格模型**：
```python
class Tacotron2(nn.Module):
    """Tacotron 2语音合成模型"""
    
    def __init__(self, vocab_size=1000, hidden_dim=512):
        super().__init__()
        
        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim * 2, 8)
        
        # 解码器
        self.decoder = nn.LSTM(hidden_dim * 2 + 80, hidden_dim, batch_first=True)
        
        # 梅尔频谱预测
        self.mel_prediction = nn.Linear(hidden_dim, 80)
    
    def forward(self, text, mel_input):
        # text: [batch, seq_len]
        # mel_input: [batch, time_steps, n_mels]
        
        # 文本编码
        text_emb = self.text_encoder[0](text)  # [batch, seq_len, hidden_dim]
        text_enc, _ = self.text_encoder[1](text_emb)  # [batch, seq_len, hidden_dim * 2]
        
        # 注意力
        query = mel_input[:, -1:, :]  # [batch, 1, n_mels]
        context, weights = self.attention(
            query.transpose(0, 1),
            text_enc.transpose(0, 1),
            text_enc.transpose(0, 1)
        )
        context = context.transpose(0, 1)  # [batch, 1, hidden_dim * 2]
        
        # 解码器输入
        decoder_input = torch.cat([context, mel_input[:, -1:, :]], dim=-1)
        
        # 解码
        decoder_out, _ = self.decoder(decoder_input)
        
        # 预测梅尔频谱
        mel_pred = self.mel_prediction(decoder_out)
        
        return mel_pred, weights
```

---

## 4. 代表性模型详解

### 4.1 Whisper

**核心特点**：
- 大规模弱监督训练
- 多语言支持（99种语言）
- 端到端语音识别

**架构**：
```python
class WhisperModel(nn.Module):
    """Whisper模型简化版"""
    
    def __init__(self, audio_dim=80, hidden_dim=512, num_layers=6, num_heads=8, vocab_size=51864):
        super().__init__()
        
        # 音频编码器
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(audio_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.ReLU()
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 文本解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, audio, text):
        # audio: [batch, n_mels, time_steps]
        # text: [batch, seq_len]
        
        # 音频编码
        audio_enc = self.audio_encoder(audio)  # [batch, hidden_dim, time_steps/2]
        audio_enc = audio_enc.transpose(1, 2)  # [batch, time_steps/2, hidden_dim]
        encoder_out = self.encoder(audio_enc.transpose(0, 1)).transpose(0, 1)
        
        # 文本解码
        decoder_out = self.decoder(text.transpose(0, 1), encoder_out.transpose(0, 1))
        decoder_out = decoder_out.transpose(0, 1)
        
        # 预测
        logits = self.classifier(decoder_out)
        
        return logits
```

**训练策略**：
```python
def train_whisper(model, dataloader, optimizer, num_epochs=10):
    """训练Whisper模型"""
    
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            audio = batch['audio']  # [batch, n_mels, time_steps]
            text = batch['text']    # [batch, seq_len]
            labels = batch['labels']  # [batch, seq_len]
            
            # 前向传播
            logits = model(audio, text)
            
            # 计算损失
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
```

### 4.2 Wav2Vec 2.0

**核心特点**：
- 自监督学习
- 无需标注数据预训练
- 学习通用音频表示

**架构**：
```python
class Wav2Vec2(nn.Module):
    """Wav2Vec 2.0模型简化版"""
    
    def __init__(self, input_dim=80, hidden_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 768, kernel_size=2, stride=2, padding=0),
            nn.ReLU()
        )
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(10000, hidden_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, waveform):
        # waveform: [batch, time]
        
        # 特征提取
        features = self.feature_extractor(waveform.unsqueeze(1))  # [batch, hidden_dim, time_steps]
        features = features.transpose(1, 2)  # [batch, time_steps, hidden_dim]
        
        # 添加位置编码
        seq_len = features.shape[1]
        pos_embed = self.pos_encoding[:seq_len, :].unsqueeze(0).repeat(features.shape[0], 1, 1)
        features = features + pos_embed
        
        # Transformer编码
        output = self.transformer(features.transpose(0, 1)).transpose(0, 1)
        
        return output
```

**预训练目标**：
```python
class Wav2VecPretraining(nn.Module):
    """Wav2Vec 2.0预训练"""
    
    def __init__(self, model, mask_prob=0.65, mask_length=10):
        super().__init__()
        self.model = model
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        
        # 量化头
        self.quantizer = nn.Linear(768, 320)
    
    def create_mask(self, length):
        """创建掩码"""
        mask = torch.zeros(length)
        i = 0
        while i < length:
            if torch.rand(1) < self.mask_prob:
                mask[i:i+self.mask_length] = 1
                i += self.mask_length
            else:
                i += 1
        return mask.bool()
    
    def forward(self, waveform):
        # 提取特征
        features = self.model(waveform)  # [batch, time_steps, hidden_dim]
        
        # 创建掩码
        batch_size, seq_len, _ = features.shape
        mask = self.create_mask(seq_len).repeat(batch_size, 1).to(features.device)
        
        # 掩码特征
        masked_features = features.clone()
        masked_features[mask] = 0
        
        # 量化预测
        quantized = self.quantizer(masked_features)
        
        # 计算损失（对比损失）
        loss = self.contrastive_loss(quantized, features, mask)
        
        return loss
    
    def contrastive_loss(self, quantized, features, mask):
        """对比损失"""
        # 简单对比损失实现
        positive = features[mask]
        negative = quantized[mask]
        
        sim = F.cosine_similarity(positive, negative, dim=-1)
        loss = -sim.mean()
        
        return loss
```

### 4.3 AudioLM

**核心特点**：
- 语言建模方法生成音频
- 无条件音频生成
- 长序列音频生成

**架构**：
```python
class AudioLM(nn.Module):
    """AudioLM模型简化版"""
    
    def __init__(self, codebook_size=1024, hidden_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(codebook_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(10000, hidden_dim))
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, codebook_size)
    
    def forward(self, codes, memory):
        # codes: [batch, seq_len]
        # memory: [batch, memory_len, hidden_dim]
        
        # 嵌入
        embedded = self.embedding(codes)  # [batch, seq_len, hidden_dim]
        
        # 位置编码
        seq_len = codes.shape[1]
        pos_embed = self.pos_encoding[:seq_len, :].unsqueeze(0).repeat(codes.shape[0], 1, 1)
        embedded = embedded + pos_embed
        
        # 解码
        decoder_out = self.decoder(
            embedded.transpose(0, 1),
            memory.transpose(0, 1)
        )
        decoder_out = decoder_out.transpose(0, 1)
        
        # 预测
        logits = self.classifier(decoder_out)
        
        return logits
```

**音频生成**：
```python
def generate_audio(model, codes, max_length=1000):
    """生成音频序列"""
    
    model.eval()
    generated = codes
    
    with torch.no_grad():
        for _ in range(max_length):
            # 获取memory（前几个token）
            memory = generated[:, -100:]
            
            # 预测下一个token
            logits = model(generated, memory)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            
            # 添加到生成序列
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查结束条件
            if next_token.item() == 0:  # EOS token
                break
    
    return generated
```

---

## 5. 预训练策略

### 5.1 自监督学习

**对比学习**：
```python
class AudioContrastiveLearning(nn.Module):
    """音频对比学习"""
    
    def __init__(self, encoder, temperature=0.07):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        
        # 投影层
        self.projection = nn.Linear(768, 512)
    
    def forward(self, audio1, audio2, audio3):
        # 编码
        feat1 = self.encoder(audio1)[:, 0, :]  # [batch, hidden_dim]
        feat2 = self.encoder(audio2)[:, 0, :]
        feat3 = self.encoder(audio3)[:, 0, :]
        
        # 投影
        proj1 = F.normalize(self.projection(feat1), dim=-1)
        proj2 = F.normalize(self.projection(feat2), dim=-1)
        proj3 = F.normalize(self.projection(feat3), dim=-1)
        
        # 对比损失
        # audio1和audio2是正样本对，audio3是负样本
        sim_pos = F.cosine_similarity(proj1, proj2, dim=-1) / self.temperature
        sim_neg = F.cosine_similarity(proj1, proj3, dim=-1) / self.temperature
        
        # NCE损失
        loss = -torch.log(torch.exp(sim_pos) / (torch.exp(sim_pos) + torch.exp(sim_neg)))
        loss = loss.mean()
        
        return loss
```

**掩码建模**：
```python
class AudioMaskedModeling(nn.Module):
    """音频掩码建模"""
    
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, audio, mask):
        # 掩码音频
        masked_audio = audio * (1 - mask)
        
        # 编码
        encoded = self.encoder(masked_audio)
        
        # 解码预测
        predicted = self.decoder(encoded)
        
        # 计算损失
        loss = F.mse_loss(predicted[mask.bool()], audio[mask.bool()])
        
        return loss
```

### 5.2 监督学习

**CTC损失**：
```python
import torch.nn.functional as F

def ctc_loss(logits, labels, input_lengths, label_lengths):
    """CTC损失"""
    # logits: [batch, time_steps, vocab_size]
    # labels: [batch, seq_len]
    
    # 转置为CTCLoss期望的格式
    logits = logits.transpose(0, 1).log_softmax(2)
    
    # 计算CTC损失
    loss = F.ctc_loss(
        logits,
        labels,
        input_lengths,
        label_lengths,
        blank=0,
        reduction='mean'
    )
    
    return loss
```

**注意力损失**：
```python
def attention_loss(logits, labels):
    """注意力解码损失"""
    # logits: [batch, seq_len, vocab_size]
    # labels: [batch, seq_len]
    
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-1
    )
    
    return loss
```

### 5.3 混合预训练

**多任务学习**：
```python
class MultiTaskAudioModel(nn.Module):
    """多任务音频模型"""
    
    def __init__(self, encoder, asr_head, tts_head, speaker_head):
        super().__init__()
        self.encoder = encoder
        self.asr_head = asr_head
        self.tts_head = tts_head
        self.speaker_head = speaker_head
        
        # 任务权重
        self.weights = nn.Parameter(torch.ones(3))
    
    def forward(self, audio, text, speaker_labels):
        # 编码
        encoded = self.encoder(audio)
        
        # ASR任务
        asr_logits = self.asr_head(encoded)
        asr_loss = F.cross_entropy(asr_logits.reshape(-1, asr_logits.shape[-1]), text.reshape(-1))
        
        # TTS任务
        tts_logits = self.tts_head(encoded)
        tts_loss = F.mse_loss(tts_logits, audio)
        
        # 说话人识别任务
        speaker_logits = self.speaker_head(encoded[:, 0, :])
        speaker_loss = F.cross_entropy(speaker_logits, speaker_labels)
        
        # 加权总损失
        weights = F.softmax(self.weights, dim=0)
        total_loss = (
            weights[0] * asr_loss +
            weights[1] * tts_loss +
            weights[2] * speaker_loss
        )
        
        return total_loss
```

---

## 6. 进阶话题

### 6.1 多语言支持

**多语言语音识别**：
```python
class MultilingualASR(nn.Module):
    """多语言语音识别模型"""
    
    def __init__(self, num_languages=99, hidden_dim=512, vocab_size=51864):
        super().__init__()
        
        # 语言嵌入
        self.language_embedding = nn.Embedding(num_languages, hidden_dim)
        
        # 音频编码器
        self.audio_encoder = nn.LSTM(80, hidden_dim, bidirectional=True, batch_first=True)
        
        # 解码器
        self.decoder = nn.LSTM(
            hidden_dim * 3,  # 语言嵌入 + 编码器输出
            hidden_dim,
            batch_first=True
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, audio, text, language_id):
        # audio: [batch, time_steps, 80]
        # text: [batch, seq_len]
        # language_id: [batch]
        
        # 语言嵌入
        lang_emb = self.language_embedding(language_id).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # 音频编码
        audio_enc, _ = self.audio_encoder(audio)  # [batch, time_steps, hidden_dim * 2]
        
        # 解码器输入
        decoder_input = torch.cat([lang_emb.repeat(1, text.shape[1], 1), audio_enc[:, :text.shape[1], :]], dim=-1)
        
        # 解码
        decoder_out, _ = self.decoder(decoder_input)
        
        # 预测
        logits = self.classifier(decoder_out)
        
        return logits
```

### 6.2 语音情感识别

**情感特征提取**：
```python
class SpeechEmotionRecognition(nn.Module):
    """语音情感识别"""
    
    def __init__(self, num_emotions=6, hidden_dim=512):
        super().__init__()
        
        # 卷积特征提取
        self.conv_layers = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM时序建模
        self.lstm = nn.LSTM(256, hidden_dim, bidirectional=True, batch_first=True)
        
        # 情感分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_emotions)
        )
    
    def forward(self, mel_spec):
        # mel_spec: [batch, n_mels, time_steps]
        
        # 卷积提取特征
        conv_out = self.conv_layers(mel_spec)  # [batch, 256, time_steps/4]
        conv_out = conv_out.transpose(1, 2)  # [batch, time_steps/4, 256]
        
        # LSTM编码
        lstm_out, (hidden, _) = self.lstm(conv_out)
        
        # 双向隐藏状态
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1)  # [batch, hidden_dim * 2]
        
        # 分类
        logits = self.classifier(hidden)
        
        return logits
```

### 6.3 语音增强

**语音增强模型**：
```python
class SpeechEnhancement(nn.Module):
    """语音增强"""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim * 2, 8)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, noisy_audio):
        # noisy_audio: [batch, time]
        
        # 编码
        encoded = self.encoder(noisy_audio.unsqueeze(1))  # [batch, hidden_dim * 2, time]
        
        # 注意力
        attn_out, _ = self.attention(
            encoded.transpose(1, 2).transpose(0, 1),
            encoded.transpose(1, 2).transpose(0, 1),
            encoded.transpose(1, 2).transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1).transpose(1, 2)  # [batch, hidden_dim * 2, time]
        
        # 解码
        enhanced = self.decoder(encoded + attn_out)  # [batch, 1, time]
        
        return enhanced.squeeze(1)
```

---

## 7. 实战项目案例

### 7.1 语音识别系统

**完整语音识别流程**：
```python
class SpeechRecognitionSystem(nn.Module):
    """语音识别系统"""
    
    def __init__(self, vocab_size=10000, hidden_dim=512):
        super().__init__()
        
        # 特征提取
        self.feature_extractor = WaveformProcessor()
        
        # 编码器
        self.encoder = nn.LSTM(80, hidden_dim, bidirectional=True, batch_first=True)
        
        # 解码器
        self.decoder = nn.LSTM(
            hidden_dim * 2 + vocab_size,
            hidden_dim,
            batch_first=True
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, waveform, text_input):
        # 特征提取
        mel_spec = self.feature_extractor(waveform)  # [batch, n_mels, time_steps]
        mel_spec = mel_spec.transpose(1, 2)  # [batch, time_steps, n_mels]
        
        # 编码
        encoder_out, _ = self.encoder(mel_spec)  # [batch, time_steps, hidden_dim * 2]
        
        # 解码器输入
        batch_size = waveform.shape[0]
        decoder_input = torch.zeros(batch_size, text_input.shape[1], hidden_dim * 2 + vocab_size)
        
        outputs = []
        for t in range(text_input.shape[1]):
            # 注意力加权
            attn_weights = F.softmax(encoder_out @ decoder_input[:, t:t+1, :hidden_dim * 2].transpose(1, 2), dim=1)
            context = (attn_weights @ encoder_out).squeeze(1)
            
            # 更新解码器输入
            decoder_input[:, t, :hidden_dim * 2] = context
            decoder_input[:, t, hidden_dim * 2:] = F.one_hot(text_input[:, t], num_classes=vocab_size).float()
            
            # 解码
            decoder_out, _ = self.decoder(decoder_input[:, :t+1, :])
            logits = self.classifier(decoder_out[:, -1, :])
            outputs.append(logits)
        
        return torch.stack(outputs, dim=1)
```

### 7.2 语音合成系统

**完整语音合成流程**：
```python
class SpeechSynthesisSystem(nn.Module):
    """语音合成系统"""
    
    def __init__(self, vocab_size=1000, hidden_dim=512):
        super().__init__()
        
        # 文本编码器
        self.text_encoder = nn.Sequential(
            nn.Embedding(vocab_size, hidden_dim),
            nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        )
        
        # 解码器
        self.decoder = nn.LSTM(
            hidden_dim * 2 + 80,
            hidden_dim,
            batch_first=True
        )
        
        # 梅尔频谱预测
        self.mel_head = nn.Linear(hidden_dim, 80)
        
        # 声码器
        self.vocoder = nn.ConvTranspose1d(80, 1, kernel_size=10, stride=5)
    
    def forward(self, text, mel_input):
        # 文本编码
        text_emb = self.text_encoder[0](text)  # [batch, seq_len, hidden_dim]
        text_enc, _ = self.text_encoder[1](text_emb)  # [batch, seq_len, hidden_dim * 2]
        
        # 解码器
        decoder_input = torch.cat([text_enc[:, :mel_input.shape[1], :], mel_input], dim=-1)
        decoder_out, _ = self.decoder(decoder_input)
        
        # 预测梅尔频谱
        mel_pred = self.mel_head(decoder_out)
        
        # 声码器生成波形
        waveform = self.vocoder(mel_pred.transpose(1, 2))
        
        return mel_pred, waveform
```

### 7.3 语音翻译系统

**语音翻译系统**：
```python
class SpeechTranslationSystem(nn.Module):
    """语音翻译系统"""
    
    def __init__(self, src_vocab_size=10000, tgt_vocab_size=10000, hidden_dim=512):
        super().__init__()
        
        # 语音编码器
        self.speech_encoder = nn.LSTM(80, hidden_dim, bidirectional=True, batch_first=True)
        
        # 语言模型解码器
        self.decoder = nn.LSTM(
            hidden_dim * 2 + tgt_vocab_size,
            hidden_dim,
            batch_first=True
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_dim, tgt_vocab_size)
    
    def forward(self, audio, tgt_text):
        # 语音编码
        audio_enc, _ = self.speech_encoder(audio)  # [batch, time_steps, hidden_dim * 2]
        
        # 解码器
        batch_size = audio.shape[0]
        decoder_input = torch.zeros(batch_size, tgt_text.shape[1], hidden_dim * 2 + tgt_vocab_size)
        
        outputs = []
        for t in range(tgt_text.shape[1]):
            # 注意力
            attn_weights = F.softmax(audio_enc @ decoder_input[:, t:t+1, :hidden_dim * 2].transpose(1, 2), dim=1)
            context = (attn_weights @ audio_enc).squeeze(1)
            
            decoder_input[:, t, :hidden_dim * 2] = context
            decoder_input[:, t, hidden_dim * 2:] = F.one_hot(tgt_text[:, t], num_classes=tgt_vocab_size).float()
            
            decoder_out, _ = self.decoder(decoder_input[:, :t+1, :])
            logits = self.classifier(decoder_out[:, -1, :])
            outputs.append(logits)
        
        return torch.stack(outputs, dim=1)
```

---

## 8. 模型优化与部署

### 8.1 模型压缩

**量化**：
```python
def quantize_audio_model(model):
    """量化音频模型"""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.LSTM, nn.Linear, nn.Conv1d},
        dtype=torch.qint8
    )
    return quantized_model
```

**剪枝**：
```python
def prune_audio_model(model, amount=0.5):
    """剪枝音频模型"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model
```

### 8.2 ONNX导出

```python
def export_audio_model_to_onnx(model, output_path):
    """导出音频模型到ONNX"""
    
    dummy_audio = torch.randn(1, 16000)  # 1秒音频
    dummy_text = torch.randint(0, 10000, (1, 20))
    
    torch.onnx.export(
        model,
        (dummy_audio, dummy_text),
        output_path,
        opset_version=13,
        input_names=['audio', 'text'],
        output_names=['logits'],
        dynamic_axes={
            'audio': {0: 'batch_size', 1: 'time'},
            'text': {0: 'batch_size', 1: 'seq_len'},
            'logits': {0: 'batch_size', 1: 'seq_len'}
        }
    )
```

### 8.3 实时推理

**流式语音识别**：
```python
class StreamingASR(nn.Module):
    """流式语音识别"""
    
    def __init__(self, model, chunk_size=1000, overlap=200):
        super().__init__()
        self.model = model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.buffer = []
    
    def process_chunk(self, audio_chunk):
        """处理音频块"""
        # 添加到缓冲区
        self.buffer.append(audio_chunk)
        
        # 如果缓冲区足够大
        if len(self.buffer) * self.chunk_size - self.overlap >= self.chunk_size:
            # 拼接音频
            full_audio = torch.cat(self.buffer, dim=-1)
            
            # 推理
            logits = self.model(full_audio, torch.zeros(1, 1))
            
            # 更新缓冲区（保留重叠部分）
            self.buffer = [full_audio[:, -self.overlap:]]
            
            return logits
        
        return None
```

---

## 9. 未来方向

### 9.1 研究趋势

| 方向 | 描述 | 关键技术 |
|------|------|---------|
| 端到端模型 | 直接从波形到文本 | 无需特征工程 |
| 自监督学习 | 无需标注数据 | Wav2Vec 2.0风格 |
| 多模态融合 | 结合视觉等其他模态 | AV-HuBERT |
| 低资源语言 | 支持更多语言 | 迁移学习 |
| 实时处理 | 低延迟推理 | 流式模型 |

### 9.2 挑战与机遇

**挑战**：
- 音频数据标注成本高
- 长音频处理困难
- 噪声鲁棒性
- 多语言支持

**机遇**：
- 大规模弱监督数据
- 自监督学习进展
- 边缘设备部署
- 多模态融合

### 9.3 推荐阅读

1. Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision.
2. Baevski, A., et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.
3. Chen, S., et al. (2022). AudioLM: a Language Modeling Approach to Audio Generation.

---

## 附录：音频处理工具

### A.1 librosa使用指南

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

# 加载音频
audio, sr = librosa.load('audio.wav', sr=16000)

# 波形可视化
plt.figure(figsize=(14, 5))
librosa.display.waveshow(audio, sr=sr)
plt.title('Audio Waveform')
plt.show()

# 梅尔频谱图
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
log_mel = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(14, 5))
librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()

# MFCC特征
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
plt.figure(figsize=(14, 5))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.show()
```

### A.2 PyTorch音频处理

```python
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, MFCC

# 加载音频
waveform, sample_rate = torchaudio.load('audio.wav')

# 梅尔频谱图变换
mel_spectrogram = MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=512,
    hop_length=256,
    n_mels=80
)
mel_spec = mel_spectrogram(waveform)

# MFCC变换
mfcc_transform = MFCC(
    sample_rate=sample_rate,
    n_mfcc=13,
    melkwargs={'n_fft': 512, 'hop_length': 256, 'n_mels': 80}
)
mfcc = mfcc_transform(waveform)

# 音频增强
from torchaudio.transforms import TimeStretch, PitchShift

# 时间拉伸
time_stretch = TimeStretch()
stretched = time_stretch(waveform, rate=1.5)

# 音调偏移
pitch_shift = PitchShift(sample_rate=sample_rate, n_steps=2)
shifted = pitch_shift(waveform)
```

### A.3 音频数据增强

```python
class AudioAugmentation:
    """音频数据增强"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def add_noise(self, audio, noise_level=0.005):
        """添加高斯噪声"""
        noise = torch.randn_like(audio) * noise_level
        return audio + noise
    
    def time_shift(self, audio, shift_max=0.1):
        """时间偏移"""
        shift = int(torch.rand(1) * shift_max * self.sample_rate)
        if shift > 0:
            audio = torch.cat([audio[:, shift:], torch.zeros(audio.shape[0], shift)], dim=-1)
        return audio
    
    def pitch_shift(self, audio, n_steps=2):
        """音调偏移"""
        return librosa.effects.pitch_shift(audio.numpy(), sr=self.sample_rate, n_steps=n_steps)
    
    def time_stretch(self, audio, rate=1.2):
        """时间拉伸"""
        return librosa.effects.time_stretch(audio.numpy(), rate=rate)
    
    def apply(self, audio):
        """随机应用增强"""
        if torch.rand(1) < 0.5:
            audio = self.add_noise(audio)
        if torch.rand(1) < 0.5:
            audio = self.time_shift(audio)
        if torch.rand(1) < 0.3:
            audio = torch.tensor(self.pitch_shift(audio))
        if torch.rand(1) < 0.3:
            audio = torch.tensor(self.time_stretch(audio))
        return audio
```

### A.4 语音识别评估指标

```python
def calculate_wer(hypothesis, reference):
    """计算字错误率(WER)"""
    # 简化实现
    hypothesis = hypothesis.split()
    reference = reference.split()
    
    # 动态规划计算编辑距离
    dp = [[0] * (len(reference) + 1) for _ in range(len(hypothesis) + 1)]
    
    for i in range(len(hypothesis) + 1):
        dp[i][0] = i
    for j in range(len(reference) + 1):
        dp[0][j] = j
    
    for i in range(1, len(hypothesis) + 1):
        for j in range(1, len(reference) + 1):
            if hypothesis[i-1] == reference[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[len(hypothesis)][len(reference)] / len(reference)

def calculate_cer(hypothesis, reference):
    """计算字符错误率(CER)"""
    # 简化实现
    hypothesis = list(hypothesis)
    reference = list(reference)
    
    dp = [[0] * (len(reference) + 1) for _ in range(len(hypothesis) + 1)]
    
    for i in range(len(hypothesis) + 1):
        dp[i][0] = i
    for j in range(len(reference) + 1):
        dp[0][j] = j
    
    for i in range(1, len(hypothesis) + 1):
        for j in range(1, len(reference) + 1):
            if hypothesis[i-1] == reference[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[len(hypothesis)][len(reference)] / len(reference)
```

### A.5 音频数据集处理

```python
class AudioDataset(Dataset):
    """音频数据集"""
    
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(data_path, '*.wav')))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # 加载音频
        file_path = self.files[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        
        # 确保单通道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # 应用变换
        if self.transform:
            waveform = self.transform(waveform)
        
        # 获取标签（从文件名提取）
        label = os.path.basename(file_path).split('_')[0]
        
        return waveform, label

# 数据加载器
dataset = AudioDataset('data/audio', transform=AudioAugmentation())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

---

## 10. 音频-语言模型评估与优化

### 10.1 评估指标详解

**语音识别评估指标**：

| 指标 | 计算方式 | 用途 |
|------|---------|------|
| **WER** | 词错误率 | 衡量识别准确性 |
| **CER** | 字符错误率 | 衡量字符级别准确性 |
| **BLEU** | 双语评估替换 | 衡量生成质量 |
| **PER** | 音素错误率 | 衡量音素级别准确性 |

```python
def calculate_metrics(references, hypotheses):
    """计算语音识别指标"""
    results = {
        'wer': [],
        'cer': [],
        'bleu': []
    }
    
    for ref, hyp in zip(references, hypotheses):
        # WER
        wer = word_error_rate(ref, hyp)
        results['wer'].append(wer)
        
        # CER
        cer = character_error_rate(ref, hyp)
        results['cer'].append(cer)
        
        # BLEU（简化计算）
        bleu = simple_bleu(ref, hyp)
        results['bleu'].append(bleu)
    
    # 计算平均值
    avg_results = {
        'wer': sum(results['wer']) / len(results['wer']),
        'cer': sum(results['cer']) / len(results['cer']),
        'bleu': sum(results['bleu']) / len(results['bleu'])
    }
    
    return avg_results
```

### 10.2 模型优化策略

**量化优化**：

```python
def optimize_for_edge(model):
    """优化模型用于边缘部署"""
    # 1. 量化
    model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv1d},
        dtype=torch.qint8
    )
    
    # 2. 剪枝
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.3)
            prune.remove(module, 'weight')
    
    # 3. 融合算子
    model = torch.jit.script(model)
    
    return model
```

**推理加速**：

```python
class FastInferenceEngine:
    """快速推理引擎"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # 预热
        self._warmup()
    
    def _warmup(self):
        """预热模型"""
        dummy_input = torch.randn(1, 80, 300)
        with torch.no_grad():
            for _ in range(5):
                self.model(dummy_input)
    
    @torch.no_grad()
    def infer(self, audio_features):
        """快速推理"""
        # 批处理
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        
        # 推理
        output = self.model(audio_features)
        
        # 后处理
        predictions = self._postprocess(output)
        
        return predictions
    
    def _postprocess(self, output):
        """后处理"""
        # 获取最可能的token序列
        tokens = torch.argmax(output, dim=-1)
        
        # 转换为文本
        text = self._tokens_to_text(tokens)
        
        return text
```

---

## 11. 音频-语言模型实战进阶

### 11.1 语音情感识别系统

```python
class SpeechEmotionRecognizer(nn.Module):
    """语音情感识别系统"""
    
    def __init__(self, num_classes=7):
        super().__init__()
        
        # 音频编码器
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(80, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 情感分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # 情感标签
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    def forward(self, mel_spec):
        """前向传播"""
        # mel_spec: [batch, n_mels, time_steps]
        
        # 编码
        features = self.audio_encoder(mel_spec)  # [batch, 512, 1]
        features = features.flatten(1)  # [batch, 512]
        
        # 分类
        logits = self.classifier(features)
        
        return logits
    
    def predict_emotion(self, audio):
        """预测情感"""
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(audio)
            probabilities = F.softmax(logits, dim=-1)
            predicted_idx = torch.argmax(probabilities, dim=-1)
            emotion = self.emotion_labels[predicted_idx.item()]
        
        return emotion, probabilities
```

### 11.2 语音翻译系统

```python
class SpeechTranslationSystem(nn.Module):
    """语音翻译系统"""
    
    def __init__(self, source_lang='en', target_lang='zh'):
        super().__init__()
        
        # 语音识别模块
        self.speech_recognizer = WhisperModel()
        
        # 机器翻译模块
        self.translator = TranslationModel(
            src_lang=source_lang,
            tgt_lang=target_lang
        )
        
        # 语音合成模块
        self.text_to_speech = TextToSpeech()
    
    def forward(self, audio):
        """端到端语音翻译"""
        # 1. 语音识别
        text = self.speech_recognizer.recognize(audio)
        
        # 2. 机器翻译
        translated_text = self.translator.translate(text)
        
        # 3. 语音合成（可选）
        translated_audio = self.text_to_speech.synthesize(translated_text)
        
        return {
            'recognized_text': text,
            'translated_text': translated_text,
            'translated_audio': translated_audio
        }
    
    def batch_translate(self, audios):
        """批量翻译"""
        results = []
        
        for audio in audios:
            result = self.forward(audio)
            results.append(result)
        
        return results
```

### 11.3 语音增强系统

```python
class SpeechEnhancementSystem(nn.Module):
    """语音增强系统"""
    
    def __init__(self):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noisy_audio):
        """增强语音"""
        # noisy_audio: [batch, 1, time_steps]
        
        # 编码
        encoded = self.encoder(noisy_audio)
        
        # 解码
        enhanced = self.decoder(encoded)
        
        return enhanced
    
    def enhance(self, audio_path, output_path):
        """增强音频文件"""
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        
        # 增强
        enhanced = self.forward(waveform.unsqueeze(0))
        
        # 保存
        torchaudio.save(output_path, enhanced.squeeze(0), sr)
```

---

## 12. 音频-语言模型未来方向

### 12.1 研究趋势

| 方向 | 描述 | 关键技术 |
|------|------|---------|
| **低资源语音识别** | 在数据稀缺场景下工作 | 自监督学习、迁移学习 |
| **多说话人分离** | 从混合音频中分离说话人 | 语音分离、说话人识别 |
| **情感感知语音** | 理解语音中的情感 | 情感特征提取、情感分类 |
| **语音生成增强** | 更高质量的语音合成 | 扩散模型、神经声码器 |
| **实时语音处理** | 低延迟语音交互 | 流式模型、高效推理 |

### 12.2 挑战与机遇

**挑战**：
- 噪声环境下的鲁棒性
- 多说话人场景处理
- 低资源语言支持
- 实时推理要求

**机遇**：
- 智能语音助手普及
- 语音医疗诊断应用
- 语音教育技术发展
- 语音无障碍技术

---

**下一节**：[视频-语言模型](03-video-language.md)