# 1.3 著名LLM模型

## 目录

- [1. 引言](#1-引言)
- [2. GPT系列](#2-gpt系列)
- [3. BERT系列](#3-bert系列)
- [4. LLaMA系列](#4-llama系列)
- [5. PaLM系列](#5-palm系列)
- [6. T5系列](#6-t5系列)
- [7. Claude系列](#7-claude系列)
- [8. Gemini系列](#8-gemini系列)
- [9. 其他重要模型](#9-其他重要模型)
- [10. 模型对比与选择建议](#10-模型对比与选择建议)
- [11. 模型发展趋势](#11-模型发展趋势)
- [12. 实践练习](#12-实践练习)

---

## 1. 引言

### 1.1 LLM发展历程

大型语言模型（LLM）的发展经历了几个重要阶段：

| 阶段 | 时间 | 代表模型 | 关键突破 |
|------|------|---------|---------|
| **起步阶段** | 2018-2019 | GPT-1, BERT | 预训练-微调范式确立 |
| **规模化阶段** | 2020-2021 | GPT-3, T5 | 千亿参数模型出现 |
| **能力爆发阶段** | 2022-至今 | GPT-4, LLaMA, Gemini | 涌现能力、多模态能力 |

### 1.2 模型分类

根据预训练目标和架构特点，LLM可以分为：

| 类型 | 预训练目标 | 代表模型 | 主要能力 |
|------|-----------|---------|---------|
| **自回归模型** | CLM | GPT系列, LLaMA | 强生成能力 |
| **自编码模型** | MLM | BERT系列 | 强理解能力 |
| **混合模型** | 多种目标 | T5, XLNet | 兼顾理解与生成 |
| **多模态模型** | 多模态目标 | GPT-4, Gemini | 多模态理解与生成 |

---

## 2. GPT系列

### 2.1 GPT-1

**论文**：Improving Language Understanding by Generative Pre-Training (Radford et al., 2018)

**核心特点**：
- 首次将Transformer解码器用于预训练
- 12层Transformer，117M参数
- 使用BooksCorpus（约7000本书）进行预训练
- 采用因果语言模型（CLM）目标

**架构细节**：
```
输入 → [嵌入层] → [Transformer解码器×12] → [线性层] → 输出
```

**创新点**：
1. 证明了生成式预训练的有效性
2. 展示了few-shot学习的潜力
3. 开创了GPT系列的先河

### 2.2 GPT-2

**论文**：Language Models are Unsupervised Multitask Learners (Radford et al., 2019)

**核心特点**：
- 更大规模：1.5B参数，48层Transformer
- 更多数据：WebText（约800GB网页文本）
- 无监督微调（UL2）策略
- 零样本学习能力

**关键改进**：
1. **更大的模型规模**：参数增加10倍以上
2. **更多的训练数据**：WebText包含更多样化的内容
3. **无监督微调**：不需要标注数据即可完成任务

**能力展示**：
- 文本生成质量显著提升
- 可以完成多种NLP任务（阅读理解、问答、翻译等）
- 展示了"更大=更好"的趋势

### 2.3 GPT-3

**论文**：Language Models are Few-Shot Learners (Brown et al., 2020)

**核心特点**：
- 超大规模：175B参数
- 海量数据：500GB文本数据（Common Crawl等）
- 上下文窗口：2048 token
- 强大的few-shot学习能力

**架构细节**：
| 配置 | GPT-3 Small | GPT-3 Medium | GPT-3 Large | GPT-3 XL | GPT-3 XXL |
|------|------------|-------------|------------|----------|-----------|
| 层数 | 12 | 24 | 24 | 48 | 96 |
| 头数 | 12 | 16 | 24 | 24 | 96 |
| 维度 | 768 | 1024 | 1536 | 2048 | 12288 |
| 参数 | 125M | 350M | 760M | 1.3B | 175B |

**关键发现**：
1. **涌现能力**：当模型规模足够大时，会出现新的能力
2. **Few-shot学习**：只需少量示例即可完成任务
3. **上下文学习**：通过prompt工程引导模型完成任务

**影响**：
- 开启了大模型时代
- 推动了prompt工程的发展
- 证明了大规模预训练的价值

### 2.4 GPT-4

**论文**：GPT-4 Technical Report (OpenAI, 2023)

**核心特点**：
- 多模态能力：支持文本和图像输入
- 更大的上下文窗口：8192 token（后续扩展到32768）
- 更强的推理能力：数学推理、逻辑推理
- 更好的安全性和对齐

**关键改进**：
1. **多模态理解**：可以理解图像内容
2. **长上下文理解**：支持更长的对话历史
3. **推理能力增强**：在数学、编程等任务上表现出色

**能力边界**：
- 复杂推理任务（如数学证明）
- 多模态理解（看图说话、图表分析）
- 创意内容生成（故事、代码、艺术）

---

## 3. BERT系列

### 3.1 BERT

**论文**：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2019)

**核心特点**：
- 双向Transformer编码器
- 掩码语言模型（MLM）预训练目标
- 句子级预测任务（NSP）
- 在11个NLP任务上达到SOTA

**架构细节**：
```
输入 → [嵌入层] → [Transformer编码器×12/24] → 输出
```

**预训练目标**：
1. **掩码语言模型（MLM）**：随机掩盖15%的token并预测
2. **下一句预测（NSP）**：判断两个句子是否连续

**创新点**：
1. 证明了双向预训练的有效性
2. 统一了多种NLP任务的处理方式
3. 开创了"预训练-微调"范式的先河

### 3.2 RoBERTa

**论文**：RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)

**核心改进**：
- 移除NSP任务（发现NSP对大多数任务有害）
- 使用更大的批次和更多的训练步骤
- 动态掩码（每次训练使用不同的掩码模式）
- 使用更多的数据（160GB vs BERT的16GB）

**效果**：
- 在多个任务上超过BERT
- 成为后续模型的基线

### 3.3 ALBERT

**论文**：ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (Lan et al., 2020)

**核心创新**：
- **参数共享**：跨层共享参数，减少参数量
- **句子顺序预测（SOP）**：替代NSP，更难的任务
- **因式分解嵌入**：减少嵌入层参数

**优势**：
- 参数量减少，但性能接近BERT
- 更适合资源受限环境

---

## 4. LLaMA系列

### 4.1 LLaMA

**论文**：LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)

**核心特点**：
- 开源模型，可商用
- 高效训练：使用更多token（1T vs GPT-3的500B）
- 多种规模：7B, 13B, 33B, 65B参数
- 性能接近GPT-3（175B）

**训练策略**：
- 使用Meta内部的高质量数据
- 优化的训练流程
- 注重效率和可部署性

**影响**：
- 推动了开源大模型的发展
- 降低了大模型研究的门槛
- 催生了大量基于LLaMA的微调模型

### 4.2 LLaMA-2

**论文**：LLaMA 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al., 2023)

**核心改进**：
- 更大的上下文窗口：4096 token
- 专门优化的聊天模型（LLaMA-2-Chat）
- 更多训练数据：2T token
- 支持商用

**关键特性**：
1. **对话优化**：针对对话场景进行微调
2. **安全性**：内置安全机制
3. **开源协议**：友好的商用许可

**模型规模**：
| 模型 | 参数 | 上下文窗口 |
|------|------|-----------|
| LLaMA-2-7B | 7B | 4096 |
| LLaMA-2-13B | 13B | 4096 |
| LLaMA-2-70B | 70B | 4096 |

### 4.3 LLaMA-3

**最新进展**：
- 更大规模：70B+参数
- 更长上下文：8192 token
- 更强的推理能力
- 多语言支持

---

## 5. PaLM系列

### 5.1 PaLM

**论文**：PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)

**核心特点**：
- 540B参数（当时最大的公开模型）
- Pathways架构：支持多任务学习
- 在28个NLP任务上达到SOTA
- 强大的推理能力

**关键技术**：
1. **Pathways**：Google的多任务训练框架
2. **稀疏激活**：条件计算，提高效率
3. **大规模训练**：6144 TPU v4芯片

**能力展示**：
- 复杂推理（数学、逻辑）
- 代码生成
- 多语言理解

### 5.2 PaLM 2

**论文**：PaLM 2 Technical Report (Google, 2023)

**核心改进**：
- 更高效的架构
- 更强的多语言能力（100+语言）
- 更好的推理和代码能力
- 支持多种型号（Gecko, Otter, Bison, Unicorn）

**模型系列**：
| 型号 | 特点 | 适用场景 |
|------|------|---------|
| Gecko | 最小，最快 | 移动端、实时应用 |
| Otter | 平衡性能与速度 | 通用场景 |
| Bison | 高性能 | 复杂任务 |
| Unicorn | 最大，最强 | 研究、复杂推理 |

---

## 6. T5系列

### 6.1 T5

**论文**：Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al., 2020)

**核心思想**：
- **统一框架**：将所有NLP任务转换为文本到文本任务
- **Span Corruption**：替代MLM的预训练目标
- **多种规模**：从Small（60M）到XXL（11B）

**预训练目标**：
- **Span Corruption**：随机选择连续的token span，用特殊符号替换
- 更接近真实的文本生成任务

**任务统一**：
```
输入: "translate English to German: Hello World"
输出: "Hallo Welt"

输入: "summarize: This is a long document..."
输出: "Summary of the document..."
```

**优势**：
- 单一模型处理多种任务
- 灵活的任务定义
- 易于扩展

### 6.2 T5-v1.1

**改进**：
- 移除NSP任务
- 调整层归一化位置
- 更好的初始化策略
- 整体性能提升

---

## 7. Claude系列

### 7.1 Claude

**开发者**：Anthropic

**核心特点**：
- 注重安全性和对齐
- 长上下文窗口：100K token
- 基于Transformer架构
- 强调AI安全研究

**安全特性**：
- Constitutional AI：基于宪法原则训练
- 减少有害输出
- 可解释性增强

### 7.2 Claude 2

**改进**：
- 更长上下文：200K token
- 更强的推理能力
- 更好的代码生成
- 支持工具调用

---

## 8. Gemini系列

### 8.1 Gemini

**论文**：Gemini: A Family of Highly Capable Multimodal Models (Google, 2023)

**核心特点**：
- 原生多模态：文本、图像、音频、视频
- 三个型号：Ultra, Pro, Nano
- 强大的推理能力
- 支持多种语言

**模型系列**：
| 型号 | 特点 | 适用场景 |
|------|------|---------|
| Gemini Ultra | 最强能力 | 复杂推理、研究 |
| Gemini Pro | 平衡性能 | 通用场景、API |
| Gemini Nano | 轻量级 | 移动端、边缘设备 |

**能力亮点**：
1. **多模态理解**：无缝处理文本、图像、音频
2. **推理能力**：数学、逻辑、常识推理
3. **代码能力**：代码生成、调试
4. **安全性**：内置安全机制

---

## 9. 其他重要模型

### 9.1 XLNet

**论文**：XLNet: Generalized Autoregressive Pretraining for Language Understanding (Yang et al., 2019)

**核心创新**：
- 排列语言建模（PLM）
- 结合AR和AE的优点
- 使用Transformer-XL处理长序列

### 9.2 ELECTRA

**论文**：ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (Clark et al., 2020)

**核心思想**：
- 替换检测任务（RTD）
- 比MLM更高效
- 训练速度更快

### 9.3 Mistral

**特点**：
- 开源模型
- 高效架构
- 支持8K上下文
- 优秀的性价比

### 9.4 Qwen（通义千问）

**开发者**：阿里巴巴

**特点**：
- 多语言支持
- 长上下文理解
- 支持工具调用
- 开源版本可用

---

## 10. 模型对比与选择建议

### 10.1 模型对比

| 模型 | 参数 | 上下文窗口 | 开源 | 多模态 | 特点 |
|------|------|-----------|------|--------|------|
| GPT-4 | ~1T | 8K/32K | 否 | 是 | 最强综合能力 |
| LLaMA-2 | 7B/13B/70B | 4K | 是 | 否 | 开源首选 |
| PaLM 2 | 多种 | 8K | 否 | 是 | 多语言强 |
| Claude 2 | 未知 | 200K | 否 | 否 | 长上下文 |
| Gemini | 多种 | 8K+ | 否 | 是 | 原生多模态 |

### 10.2 选择建议

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **研究/实验** | LLaMA-2, Mistral | 开源、可修改 |
| **生产部署** | GPT-4, Claude 2 | API稳定、性能强 |
| **多模态任务** | GPT-4, Gemini | 原生多模态支持 |
| **长文本处理** | Claude 2 | 200K上下文 |
| **代码生成** | GPT-4, CodeLlama | 代码能力强 |
| **多语言** | PaLM 2, Gemini | 多语言支持好 |

---

## 11. 模型发展趋势

### 11.1 当前趋势

| 趋势 | 说明 | 代表工作 |
|------|------|---------|
| **更大规模** | 模型参数持续增加 | GPT-4 (~1T) |
| **多模态** | 融合文本、图像、音频 | GPT-4, Gemini |
| **长上下文** | 支持更长的输入 | Claude 2 (200K) |
| **开源化** | 更多模型开源 | LLaMA, Mistral |
| **效率优化** | 提高训练和推理效率 | FlashAttention |
| **安全性** | 注重AI安全和对齐 | Claude, Gemini |

### 11.2 未来方向

| 方向 | 说明 |
|------|------|
| **通用智能** | 更接近人类的通用推理能力 |
| **高效推理** | 降低推理成本 |
| **个性化** | 适应不同用户需求 |
| **实时学习** | 持续学习新知识 |
| **可解释性** | 理解模型决策过程 |

---

## 12. 实践练习

### 练习1：理解模型规模与性能的关系

```python
import matplotlib.pyplot as plt

# 模型规模与性能示例数据
models = ["GPT-1", "GPT-2", "GPT-3", "GPT-4"]
parameters = [0.117, 1.5, 175, 1000]  # 单位：B
performance = [60, 75, 85, 95]  # 模拟性能分数

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('模型')
ax1.set_ylabel('参数 (B)', color=color)
ax1.bar(models, parameters, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('性能', color=color)
ax2.plot(models, performance, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('模型规模与性能关系')
plt.show()
```

### 练习2：设计一个简单的模型选择器

```python
def select_model(task_type, requirements):
    """
    根据任务类型和需求选择合适的模型
    
    参数:
        task_type: 任务类型（'text-generation', 'qa', 'summarization', 'multimodal', 'code'）
        requirements: 需求字典，包含 'open_source', 'context_length', 'performance'
    
    返回:
        推荐的模型列表
    """
    models = []
    
    if requirements.get('open_source', False):
        if task_type == 'multimodal':
            models.append("LLaVA (基于LLaMA的多模态)")
        else:
            models.extend(["LLaMA-2", "Mistral", "Qwen"])
    else:
        if task_type == 'multimodal':
            models.extend(["GPT-4", "Gemini Pro"])
        elif task_type == 'code':
            models.extend(["GPT-4", "CodeLlama"])
        elif requirements.get('context_length', 0) > 100000:
            models.append("Claude 2")
        else:
            models.extend(["GPT-4", "Claude 2", "Gemini Pro"])
    
    return models

# 测试
print("开源文本生成任务:", select_model('text-generation', {'open_source': True}))
print("多模态任务:", select_model('multimodal', {'open_source': False}))
print("长上下文任务:", select_model('qa', {'open_source': False, 'context_length': 200000}))
```

### 练习3：理解不同预训练目标

```python
# 不同预训练目标的特点对比
pretraining_objectives = {
    "MLM (BERT)": {
        "description": "掩码语言模型，随机掩盖token并预测",
        "advantages": ["双向上下文", "适合理解任务"],
        "disadvantages": ["预训练-微调差异", "[MASK]符号问题"]
    },
    "CLM (GPT)": {
        "description": "因果语言模型，预测下一个token",
        "advantages": ["预训练-微调一致", "适合生成任务"],
        "disadvantages": ["单向上下文", "推理速度慢"]
    },
    "PLM (XLNet)": {
        "description": "排列语言模型，结合AR和AE",
        "advantages": ["双向上下文", "预训练-微调一致"],
        "disadvantages": ["计算复杂度高", "训练速度慢"]
    },
    "Span Corruption (T5)": {
        "description": "随机替换连续token span",
        "advantages": ["统一框架", "接近生成任务"],
        "disadvantages": ["复杂度较高"]
    }
}

for obj, info in pretraining_objectives.items():
    print(f"【{obj}】")
    print(f"  描述: {info['description']}")
    print(f"  优点: {', '.join(info['advantages'])}")
    print(f"  缺点: {', '.join(info['disadvantages'])}")
    print()
```

---

**下一节**：[LLM能力分析](04-capabilities.md)

---

## 参考文献

1. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding.
3. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners.
4. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M. A., Lacroix, T., ... & Jegou, H. (2023). Llama: Open and efficient foundation language models.
5. Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., ... & Chen, P. (2022). Palm: Scaling language modeling with pathways.
6. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer.

---

## 13. 模型架构深度解析

### 13.1 GPT架构实现

**代码实现：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    """GPT配置"""
    def __init__(self, vocab_size=50257, max_len=1024, d_model=768, 
                 num_heads=12, num_layers=12, dim_feedforward=3072, 
                 dropout=0.1, activation='gelu'):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation


class CausalSelfAttention(nn.Module):
    """
    因果自注意力
    
    使用下三角掩码确保自回归性质。
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        
        # Q, K, V投影
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        
        # 输出投影
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 因果掩码
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(config.max_len, config.max_len))
            .view(1, 1, config.max_len, config.max_len)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V投影
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        # 重塑为多头
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        
        # 应用因果掩码
        attn = attn.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        
        # Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # 加权求和
        y = attn @ v
        
        # 合并多头
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class GPTBlock(nn.Module):
    """
    GPT Transformer块
    
    包含因果自注意力和前馈网络。
    """
    
    def __init__(self, config):
        super().__init__()
        
        # 层归一化
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)
        
        # 自注意力
        self.attn = CausalSelfAttention(config)
        
        # 前馈网络
        if config.activation == 'gelu':
            activation = nn.GELU()
        elif config.activation == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.GELU()
        
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            activation,
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # 自注意力 + 残差连接
        x = x + self.attn(self.ln_1(x))
        
        # 前馈网络 + 残差连接
        x = x + self.mlp(self.ln_2(x))
        
        return x


class GPT(nn.Module):
    """
    GPT模型
    
    论文核心思想（Radford et al., 2018）：
    使用Transformer解码器进行自回归语言建模。
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer块
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.num_layers)])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.wte.weight
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        token_embeds = self.wte(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        x = self.drop(token_embeds + position_embeds)
        
        # Transformer块
        for block in self.blocks:
            x = block(x)
        
        # 最终归一化
        x = self.ln_f(x)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        生成文本
        
        Args:
            input_ids: (batch_size, seq_len)
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_k: top-k采样
        
        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # 前向传播
            logits = self.forward(generated_ids)
            
            # 取最后一个token的logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k采样
            if top_k is not None:
                values, indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, indices, values)
            
            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到生成序列
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        return generated_ids


# GPT模型配置
gpt2_small_config = GPTConfig(
    vocab_size=50257,
    max_len=1024,
    d_model=768,
    num_heads=12,
    num_layers=12,
    dim_feedforward=3072,
    dropout=0.1,
    activation='gelu'
)

gpt2_medium_config = GPTConfig(
    vocab_size=50257,
    max_len=1024,
    d_model=1024,
    num_heads=16,
    num_layers=24,
    dim_feedforward=4096,
    dropout=0.1,
    activation='gelu'
)

gpt2_large_config = GPTConfig(
    vocab_size=50257,
    max_len=1024,
    d_model=1280,
    num_heads=20,
    num_layers=36,
    dim_feedforward=5120,
    dropout=0.1,
    activation='gelu'
)

gpt2_xl_config = GPTConfig(
    vocab_size=50257,
    max_len=1024,
    d_model=1600,
    num_heads=25,
    num_layers=48,
    dim_feedforward=6400,
    dropout=0.1,
    activation='gelu'
)
```

### 13.2 BERT架构实现

**代码实现：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BERTConfig:
    """BERT配置"""
    def __init__(self, vocab_size=30522, max_len=512, d_model=768, 
                 num_heads=12, num_layers=12, dim_feedforward=3072, 
                 dropout=0.1, activation='gelu'):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation


class BERTEmbedding(nn.Module):
    """
    BERT嵌入层
    
    组成：
    - Token嵌入
    - 位置嵌入
    - 段嵌入
    """
    
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_len, config.d_model)
        self.segment_embedding = nn.Embedding(2, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.d_model = config.d_model
    
    def forward(self, input_ids, segment_ids):
        """
        Args:
            input_ids: (batch_size, seq_len)
            segment_ids: (batch_size, seq_len)
        
        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token嵌入
        token_embeds = self.token_embedding(input_ids)
        
        # 位置嵌入
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(position_ids)
        
        # 段嵌入
        segment_embeds = self.segment_embedding(segment_ids)
        
        # 组合并缩放
        embeddings = token_embeds + position_embeds + segment_embeds
        embeddings = embeddings * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BERTSelfAttention(nn.Module):
    """
    BERT自注意力
    
    多头自注意力机制。
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        
        # Q, K, V投影
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Q, K, V投影
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # 重塑为多头
        query_layer = mixed_query_layer.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_layer = mixed_key_layer.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_layer = mixed_value_layer.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 加权求和
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # 合并多头
        context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return context_layer


class BERTSelfOutput(nn.Module):
    """
    BERT自注意力输出
    
    包含线性变换和残差连接。
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states, input_tensor):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            input_tensor: (batch_size, seq_len, d_model)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class BERTAttention(nn.Module):
    """
    BERT注意力层
    
    组合自注意力和输出层。
    """
    
    def __init__(self, config):
        super().__init__()
        self.self_attn = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)
        self.pruned_heads = set()
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        self_outputs = self.self_attn(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class BERTIntermediate(nn.Module):
    """
    BERT中间层
    
    前馈网络。
    """
    
    def __init__(self, config):
        super().__init__()
        if config.activation == 'gelu':
            activation = nn.GELU()
        elif config.activation == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.GELU()
        
        self.dense = nn.Linear(config.d_model, config.dim_feedforward)
        self.intermediate_act_fn = activation
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
        
        Returns:
            output: (batch_size, seq_len, dim_feedforward)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    """
    BERT输出层
    
    包含线性变换和残差连接。
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.dim_feedforward, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states, input_tensor):
        """
        Args:
            hidden_states: (batch_size, seq_len, dim_feedforward)
            input_tensor: (batch_size, seq_len, d_model)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class BERTLayer(nn.Module):
    """
    BERT层
    
    包含注意力层和前馈网络。
    """
    
    def __init__(self, config):
        super().__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # 自注意力
        attention_output = self.attention(hidden_states, attention_mask)
        
        # 前馈网络
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output


class BERT(nn.Module):
    """
    BERT模型
    
    论文核心思想（Devlin et al., 2018）：
    使用双向Transformer编码器进行预训练。
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embeddings = BERTEmbedding(config)
        
        # Transformer编码器层
        self.encoder = nn.ModuleList([BERTLayer(config) for _ in range(config.num_layers)])
        
        # MLM头
        self.mlm_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.vocab_size, bias=False)
        )
        self.mlm_head[3].weight = self.embeddings.token_embedding.weight
        
        # NSP头
        self.nsp_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Tanh(),
            nn.Linear(config.d_model, 2)
        )
    
    def forward(self, input_ids, segment_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            segment_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            mlm_logits: (batch_size, seq_len, vocab_size)
            nsp_logits: (batch_size, 2)
            hidden_states: (batch_size, seq_len, d_model)
        """
        # 嵌入
        embedding_output = self.embeddings(input_ids, segment_ids)
        
        # 编码器
        hidden_states = embedding_output
        for layer in self.encoder:
            hidden_states = layer(hidden_states, attention_mask)
        
        # MLM预测
        mlm_logits = self.mlm_head(hidden_states)
        
        # NSP预测（使用[CLS] token）
        cls_hidden = hidden_states[:, 0, :]
        nsp_logits = self.nsp_head(cls_hidden)
        
        return mlm_logits, nsp_logits, hidden_states


# BERT模型配置
bert_base_config = BERTConfig(
    vocab_size=30522,
    max_len=512,
    d_model=768,
    num_heads=12,
    num_layers=12,
    dim_feedforward=3072,
    dropout=0.1,
    activation='gelu'
)

bert_large_config = BERTConfig(
    vocab_size=30522,
    max_len=512,
    d_model=1024,
    num_heads=16,
    num_layers=24,
    dim_feedforward=4096,
    dropout=0.1,
    activation='gelu'
)
```

### 13.3 LLaMA架构实现

**代码实现：**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLaMAConfig:
    """LLaMA配置"""
    def __init__(self, vocab_size=32000, max_len=2048, d_model=4096, 
                 num_heads=32, num_layers=32, dim_feedforward=11008, 
                 num_kv_heads=None, dropout=0.1, activation='swish'):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.dropout = dropout
        self.activation = activation


class RMSNorm(nn.Module):
    """
    RMSNorm
    
    Root Mean Square Layer Normalization。
    """
    
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码（RoPE）
    
    论文核心思想（Su et al., 2021）：
    通过旋转矩阵编码位置信息。
    """
    
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.d_model = d_model
        
        # 计算频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 生成位置索引
        self.max_len = max_len
    
    def forward(self, seq_len, device):
        """
        Args:
            seq_len: 序列长度
            device: 设备
        
        Returns:
            cos: (seq_len, d_model // 2)
            sin: (seq_len, d_model // 2)
        """
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # 生成cos和sin
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin


def rotate_half(x):
    """
    旋转一半维度
    
    Args:
        x: (..., d_model)
    
    Returns:
        rotated: (..., d_model)
    """
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    应用旋转位置编码
    
    Args:
        q: (batch_size, num_heads, seq_len, head_dim)
        k: (batch_size, num_heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)
    
    Returns:
        q_rotated: (batch_size, num_heads, seq_len, head_dim)
        k_rotated: (batch_size, num_heads, seq_len, head_dim)
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LLaMASelfAttention(nn.Module):
    """
    LLaMA自注意力
    
    使用分组查询注意力（GQA）和旋转位置编码。
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.num_heads == 0
        
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.d_model // config.num_heads
        self.num_kv_groups = config.num_heads // config.num_kv_heads
        
        # Q投影
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # K, V投影（使用较少的头）
        self.k_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.num_kv_heads * self.head_dim, bias=False)
        
        # 输出投影
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # 旋转位置编码
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, config.max_len)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Q投影
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K, V投影
        k = self.k_proj(hidden_states)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        v = self.v_proj(hidden_states)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 应用旋转位置编码
        cos, sin = self.rotary_emb(seq_len, hidden_states.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # 扩展K, V以匹配Q的头数
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 应用注意力掩码
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        output = torch.matmul(attn_weights, v)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.o_proj(output)
        
        return output


class LLaMABlock(nn.Module):
    """
    LLaMA Transformer块
    
    包含自注意力和前馈网络。
    """
    
    def __init__(self, config):
        super().__init__()
        
        # RMSNorm
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        
        # 自注意力
        self.self_attn = LLaMASelfAttention(config)
        
        # 前馈网络
        if config.activation == 'swish':
            activation = nn.SiLU()
        elif config.activation == 'gelu':
            activation = nn.GELU()
        else:
            activation = nn.SiLU()
        
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            activation,
            nn.Linear(config.dim_feedforward, config.d_model)
        )
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # 自注意力 + 残差连接
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # 前馈网络 + 残差连接
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class LLaMA(nn.Module):
    """
    LLaMA模型
    
    论文核心思想（Touvron et al., 2023）：
    使用高效的Transformer架构进行预训练。
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer块
        self.layers = nn.ModuleList([LLaMABlock(config) for _ in range(config.num_layers)])
        
        # 最终归一化
        self.norm = RMSNorm(config.d_model)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.embed_tokens.weight
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # 嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # Transformer块
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        # 语言模型头
        logits = self.lm_head(hidden_states)
        
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        生成文本
        
        Args:
            input_ids: (batch_size, seq_len)
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_k: top-k采样
        
        Returns:
            generated_ids: (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        generated_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # 前向传播
            logits = self.forward(generated_ids)
            
            # 取最后一个token的logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-k采样
            if top_k is not None:
                values, indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, indices, values)
            
            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到生成序列
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        return generated_ids


# LLaMA模型配置
llama_7b_config = LLaMAConfig(
    vocab_size=32000,
    max_len=2048,
    d_model=4096,
    num_heads=32,
    num_layers=32,
    dim_feedforward=11008,
    num_kv_heads=32,
    dropout=0.1,
    activation='swish'
)

llama_13b_config = LLaMAConfig(
    vocab_size=32000,
    max_len=2048,
    d_model=5120,
    num_heads=40,
    num_layers=40,
    dim_feedforward=13696,
    num_kv_heads=40,
    dropout=0.1,
    activation='swish'
)

llama_33b_config = LLaMAConfig(
    vocab_size=32000,
    max_len=2048,
    d_model=6656,
    num_heads=52,
    num_layers=60,
    dim_feedforward=17920,
    num_kv_heads=52,
    dropout=0.1,
    activation='swish'
)

llama_65b_config = LLaMAConfig(
    vocab_size=32000,
    max_len=2048,
    d_model=8192,
    num_heads=64,
    num_layers=80,
    dim_feedforward=22016,
    num_kv_heads=8,
    dropout=0.1,
    activation='swish'
)
```

---

## 14. 模型训练细节

### 14.1 训练数据

| 模型 | 数据量 | 数据来源 | 训练token数 |
|------|-------|---------|-----------|
| GPT-2 | 40GB | WebText | ~10B |
| GPT-3 | 500GB | Common Crawl, WebText, Books | ~300B |
| BERT | 16GB | Wikipedia + BookCorpus | ~3.3B |
| LLaMA | 2T tokens | Common Crawl, Wikipedia, Books | 2T |
| PaLM | 780B tokens | 多样化数据集 | 780B |

### 14.2 训练配置

| 模型 | 批次大小 | 学习率 | 训练步数 | 训练时间 |
|------|---------|--------|---------|---------|
| GPT-2 | 512 | 2.5e-4 | 1M | 数周 |
| GPT-3 | 3.2M | 6e-5 | 300B | 数月 |
| BERT | 256 | 1e-4 | 1M | 4天 |
| LLaMA | 4M | 3e-4 | 2T | 21天 |
| PaLM | 4M | 1e-5 | 780B | 数月 |

### 14.3 训练技巧

**代码实现：**

```python
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

class LLMTrainer:
    """
    LLM训练器
    
    包含混合精度训练、梯度累积、学习率调度等技巧。
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练
        self.scaler = GradScaler()
        
        # 梯度累积
        self.accumulation_steps = config.accumulation_steps
        self.current_step = 0
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            progress = float(step - self.config.warmup_steps) / float(max(1, self.config.total_steps - self.config.warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch):
        """
        训练步骤
        
        Args:
            batch: 训练批次
        
        Returns:
            loss: 损失值
        """
        self.model.train()
        
        # 混合精度训练
        with autocast():
            outputs = self.model(**batch)
            loss = outputs.loss / self.accumulation_steps
        
        # 缩放梯度
        self.scaler.scale(loss).backward()
        
        self.current_step += 1
        
        # 梯度累积
        if self.current_step % self.accumulation_steps == 0:
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            # 更新参数
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # 更新学习率
            self.scheduler.step()
        
        return loss.item() * self.accumulation_steps
```

---

## 15. 模型性能对比

### 15.1 基准测试

| 模型 | MMLU | HellaSwag | PIQA | WinoGrande | ARC |
|------|------|-----------|------|-----------|-----|
| GPT-3 | 43.9 | 78.9 | 81.0 | 70.1 | 69.9 |
| GPT-4 | 86.4 | 95.3 | 93.0 | 87.5 | 96.3 |
| LLaMA-2-70B | 68.9 | 81.2 | 82.3 | 76.6 | 78.5 |
| PaLM 2 | 78.3 | 86.8 | 88.5 | 82.1 | 85.2 |
| Claude 2 | 76.5 | 85.2 | 87.0 | 80.5 | 83.8 |

### 15.2 代码生成能力

| 模型 | HumanEval | MBPP | CodeContests |
|------|-----------|------|--------------|
| GPT-4 | 67.0 | 72.3 | 10.2 |
| Claude 2 | 71.2 | 75.6 | 12.5 |
| CodeLlama-34B | 48.8 | 53.7 | 4.5 |
| StarCoder | 37.0 | 45.9 | 3.2 |

---

## 16. 总结

著名LLM模型的发展展示了AI技术的快速进步。从GPT-1到GPT-4，从BERT到LLaMA，模型规模和能力不断提升。

**关键要点：**

1. **规模定律**：更大的模型通常表现更好
2. **数据质量**：高质量数据比大量数据更重要
3. **架构创新**：Transformer及其变体是核心
4. **训练技巧**：混合精度、梯度累积等提升效率
5. **开源趋势**：更多模型开源，降低研究门槛

**未来方向：**

- 更高效的训练方法
- 更强的推理能力
- 更好的多模态融合
- 更高的安全性

---

## 参考文献

1. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training". OpenAI.
2. Devlin, J., et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers". NAACL.
3. Brown, T. B., et al. (2020). "Language Models are Few-Shot Learners". NeurIPS.
4. Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models".
5. Chowdhery, A., et al. (2022). "PaLM: Scaling Language Modeling with Pathways". arXiv.
6. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". JMLR.
7. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding". arXiv.
