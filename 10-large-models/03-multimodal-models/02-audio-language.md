# 3.2 音频-语言模型

## 目录

- [1. 引言](#1-引言)
- [2. 音频-语言学习概述](#2-音频-语言学习概述)
- [3. 音频特征提取](#3-音频特征提取)
- [4. 音频-语言模型架构](#4-音频-语言模型架构)
- [5. 代表性模型](#5-代表性模型)
- [6. 音频-语言任务](#6-音频-语言任务)
- [7. 实践练习](#7-实践练习)

---

## 1. 引言

### 1.1 音频-语言模型的重要性

**音频-语言模型**是能够同时处理音频和语言信息的AI模型，在语音识别、语音合成、语音理解等领域有广泛应用。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **语音识别** | 将语音转换为文本 | 语音转文字 |
| **语音合成** | 将文本转换为语音 | 文字转语音 |
| **语音理解** | 理解语音内容 | 语音助手 |
| **说话人识别** | 识别说话人身份 | 声纹识别 |
| **情感分析** | 分析语音中的情感 | 语音情感识别 |

---

## 2. 音频-语言学习概述

### 2.1 音频数据特点

| 特点 | 描述 |
|------|------|
| **连续性** | 音频是连续的时间序列 |
| **频率信息** | 包含丰富的频率成分 |
| **动态范围** | 幅度变化范围大 |
| **时序性** | 顺序敏感 |

### 2.2 音频-语言任务类型

| 任务类型 | 描述 | 方向 |
|---------|------|------|
| **语音识别** | 语音→文本 | 音频到语言 |
| **语音合成** | 文本→语音 | 语言到音频 |
| **语音翻译** | 语音→另一种语言的文本 | 跨语言 |
| **语音问答** | 根据语音回答问题 | 理解任务 |

---

## 3. 音频特征提取

### 3.1 原始音频表示

**波形表示**：
```python
import torchaudio

# 加载音频
waveform, sample_rate = torchaudio.load("audio.wav")
print(f"波形形状: {waveform.shape}")  # [channels, time]
print(f"采样率: {sample_rate}")
```

### 3.2 频谱特征

**梅尔频谱图**：
```python
import torchaudio.transforms as T

# 计算梅尔频谱图
mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=1024,
    n_mels=128
)
mel_spec = mel_spectrogram(waveform)
print(f"梅尔频谱图形状: {mel_spec.shape}")  # [channels, n_mels, time]
```

**MFCC特征**：
```python
# 计算MFCC
mfcc_transform = T.MFCC(
    sample_rate=sample_rate,
    n_mfcc=40,
    log_mels=True
)
mfcc = mfcc_transform(waveform)
print(f"MFCC形状: {mfcc.shape}")  # [channels, n_mfcc, time]
```

### 3.3 音频编码器

| 编码器类型 | 描述 | 代表模型 |
|-----------|------|---------|
| **CNN** | 处理频谱图 | 2D CNN |
| **RNN/Transformer** | 处理时序 | LSTM、Transformer |
| **专门模型** | 针对音频优化 | Whisper、AudioLM |

---

## 4. 音频-语言模型架构

### 4.1 序列到序列架构

**语音识别架构**：
```
音频 → 编码器 → 隐藏状态 → 解码器 → 文本
```

**代码示例**：
```python
import torch
import torch.nn as nn

class AudioToTextModel(nn.Module):
    def __init__(self, audio_dim=80, hidden_dim=512, vocab_size=1000):
        super().__init__()
        self.encoder = nn.LSTM(audio_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, audio_feat, text_input):
        # 编码音频
        encoder_out, _ = self.encoder(audio_feat)
        
        # 解码文本
        decoder_out, _ = self.decoder(text_input)
        logits = self.classifier(decoder_out)
        
        return logits
```

### 4.2 注意力机制

**编码器-解码器注意力**：
```python
class AudioAttentionModel(nn.Module):
    def __init__(self, audio_dim=80, text_dim=512, hidden_dim=512):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
    
    def forward(self, audio_feat, text_feat):
        # 投影
        audio_proj = self.audio_proj(audio_feat).transpose(0, 1)  # [seq_len, batch, dim]
        text_proj = self.text_proj(text_feat).transpose(0, 1)
        
        # 注意力
        output, weights = self.attention(text_proj, audio_proj, audio_proj)
        return output.transpose(0, 1), weights
```

---

## 5. 代表性模型

### 5.1 Whisper

**论文**：Robust Speech Recognition via Large-Scale Weak Supervision (Radford et al., 2022)

**核心特点**：
- 大规模弱监督训练
- 支持99种语言
- 统一的模型架构
- 多种任务支持（识别、翻译）

**架构**：
```
音频 → 梅尔频谱图 → Encoder → Decoder → 文本
```

**使用示例**：
```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.wav")
print(result["text"])
```

### 5.2 AudioLM

**论文**：AudioLM: a Language Modeling Approach to Audio Generation (Borsos et al., 2022)

**核心特点**：
- 使用语言建模方法生成音频
- 支持多种音频类型
- 长序列生成能力

**架构**：
```
音频 → 离散token → Transformer LM → 音频生成
```

### 5.3 Wav2Vec 2.0

**论文**：wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)

**核心特点**：
- 自监督学习
- 预训练+微调范式
- 无需大量标注数据

**架构**：
```
音频 → CNN提取特征 → Transformer → 特征表示
```

---

## 6. 音频-语言任务

### 6.1 语音识别

**定义**：将语音信号转换为文本

**评估指标**：
- **WER**（Word Error Rate）：词错误率
- **CER**（Character Error Rate）：字符错误率

**代码示例**：
```python
import whisper

# 加载模型
model = whisper.load_model("large")

# 语音识别
audio = whisper.load_audio("speech.wav")
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# 检测语言
_, probs = model.detect_language(mel)
print(f"检测到的语言: {max(probs, key=probs.get)}")

# 识别
result = model.decode(mel)
print(f"识别结果: {result.text}")
```

### 6.2 语音合成

**定义**：将文本转换为语音

**代表性模型**：
- **Tacotron 2**：端到端语音合成
- **WaveNet**：基于扩散的生成
- **VITS**：变分自编码器

**代码示例**：
```python
from transformers import pipeline

# 加载语音合成模型
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

# 生成语音
text = "你好，这是一个语音合成示例。"
speech = synthesiser(text)

# 保存音频
import scipy
scipy.io.wavfile.write("output.wav", rate=speech["sampling_rate"], data=speech["audio"])
```

### 6.3 语音情感识别

**定义**：识别语音中的情感状态

**情感类别**：
| 类别 | 描述 |
|------|------|
| **中性** | 无明显情感 |
| **高兴** | 积极情绪 |
| **悲伤** | 消极情绪 |
| **愤怒** | 愤怒情绪 |
| **惊讶** | 惊讶情绪 |

---

## 7. 实践练习

### 练习1：使用Whisper进行语音识别

```python
import whisper

# 加载不同规模的模型
model_sizes = ["tiny", "base", "small", "medium", "large"]

for size in model_sizes:
    print(f"\n=== {size} 模型 ===")
    model = whisper.load_model(size)
    
    # 加载并识别音频
    result = model.transcribe("example.wav")
    print(f"识别结果: {result['text']}")
    
    # 打印检测到的语言
    print(f"检测语言: {result['language']}")
```

### 练习2：音频特征可视化

```python
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

# 加载音频
waveform, sample_rate = torchaudio.load("example.wav")

# 计算梅尔频谱图
mel_spectrogram = T.MelSpectrogram(sample_rate=sample_rate)
mel_spec = mel_spectrogram(waveform)

# 可视化
plt.figure(figsize=(10, 4))
plt.imshow(mel_spec[0].numpy(), origin='lower', aspect='auto')
plt.title("梅尔频谱图")
plt.xlabel("时间")
plt.ylabel("梅尔频率")
plt.colorbar()
plt.show()

# 波形可视化
plt.figure(figsize=(10, 3))
plt.plot(waveform[0].numpy())
plt.title("音频波形")
plt.xlabel("样本")
plt.ylabel("幅度")
plt.show()
```

### 练习3：音频-文本匹配

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioTextMatcher(nn.Module):
    def __init__(self, audio_dim=80, text_dim=768, hidden_dim=512):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
    
    def forward(self, audio_feat, text_feat):
        # 平均池化音频特征
        audio_avg = audio_feat.mean(dim=1)
        audio_proj = F.normalize(self.audio_proj(audio_avg), dim=-1)
        
        # 使用CLS token
        text_proj = F.normalize(self.text_proj(text_feat[:, 0, :]), dim=-1)
        
        # 计算相似度
        similarity = (audio_proj * text_proj).sum(dim=-1)
        return similarity

# 测试
model = AudioTextMatcher()
audio_feat = torch.randn(8, 100, 80)  # [batch, time, dim]
text_feat = torch.randn(8, 10, 768)   # [batch, seq_len, dim]

similarity = model(audio_feat, text_feat)
print(f"音频-文本相似度: {similarity}")
```

---

**下一节**：[视频-语言模型](03-video-language.md)

---

## 参考文献

1. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). Robust speech recognition via large-scale weak supervision.
2. Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations.
3. Borsos, Z., Donahue, J., Dieleman, S., Zen, H., & Casagrande, N. (2022). AudioLM: a language modeling approach to audio generation.
