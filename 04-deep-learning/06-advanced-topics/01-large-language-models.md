# 4.2 大语言模型

## 目录

- [1. 大语言模型概述](#1-大语言模型概述)
- [2. GPT系列](#2-gpt系列)
- [3. 开源大模型](#3-开源大模型)
- [4. 大模型训练与微调](#4-大模型训练与微调)
- [5. 提示工程](#5-提示工程)
- [6. 重要论文详解](#6-重要论文详解)
- [7. 实践练习](#7-实践练习)

---

## 1. 大语言模型概述

### 1.1 什么是大语言模型？

**大语言模型（LLM）**：具有大量参数的Transformer架构语言模型。

**核心特点**：
- 大规模参数（数十亿到万亿）
- 海量文本预训练
- 涌现能力（少样本学习、推理等）

### 1.2 LLM发展历程

| 阶段 | 模型 | 特点 |
|------|------|------|
| 早期 | ELMo, GPT-1 | 小规模，单向/双向 |
| 中期 | BERT, GPT-2 | 中等规模，预训练+微调 |
| 现代 | GPT-3, PaLM | 大规模，上下文学习 |
| 当前 | GPT-4, Gemini | 多模态，超强能力 |

---

## 2. GPT系列

### 2.1 GPT-1 (2018)

**参数**：117M
**特点**：
- 12层Transformer decoder
- 768维隐藏层
- 12个注意力头
- 首次展示预训练+微调范式

### 2.2 GPT-2 (2019)

**参数**：1.5B
**特点**：
- 零样本学习
- 更大的上下文窗口
- 改进的训练策略

### 2.3 GPT-3 (2020)

**参数**：175B
**特点**：
- 少样本学习
- 上下文学习
- 强大的泛化能力

### 2.4 GPT-4 (2023)

**参数**：未知
**特点**：
- 多模态能力
- 更强的推理能力
- 更长的上下文

---

## 3. 开源大模型

### 3.1 LLaMA系列

**Meta开源模型**：
- LLaMA: 7B, 13B, 33B, 65B
- LLaMA-2: 7B, 13B, 70B
- 高质量开源模型

### 3.2 Mistral

**特点**：
- 高效架构
- 8K上下文窗口
- 快速推理

### 3.3 Falcon

**特点**：
- 180B参数
- 开源商用友好
- 高性能

### 3.4 Qwen（通义千问）

**阿里巴巴开源模型**：
- 支持多种语言
- 多模态能力
- 开源友好

---

## 4. 大模型训练与微调

### 4.1 预训练

**数据准备**：
- 大规模文本语料
- 数据清洗和预处理
- 分词和编码

**训练策略**：
- 因果语言建模（CLM）
- 批次大小优化
- 混合精度训练

### 4.2 微调方法

**LoRA（Low-Rank Adaptation）**：
- 冻结主模型参数
- 只训练低秩矩阵
- 高效微调

**QLoRA**：
- 量化+LoRA
- 更低显存占用
- 保持性能

### 4.3 实践示例

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 配置LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 添加LoRA层
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 训练
# 准备数据...
# trainer.train()
```

---

## 5. 提示工程

### 5.1 什么是提示工程？

**提示工程**：设计有效的提示词以获得最佳模型输出。

### 5.2 提示技巧

**指令清晰**：
```
请总结以下文章，限制在3句话内。
文章：...
```

**提供示例**：
```
Q: 苹果是什么颜色？
A: 红色或绿色。

Q: 天空是什么颜色？
A: 
```

**链式思考（CoT）**：
```
我有5个苹果，吃了2个，又买了3个，现在有几个？
让我想想：5-2=3，3+3=6。答案是6。
```

### 5.3 提示框架

**Few-shot**：
```
示例1：...
示例2：...
示例3：...
问题：...
```

**思维链**：
```
问题：...
让我一步步思考：
1. ...
2. ...
3. ...
因此，答案是：...
```

---

## 6. 重要论文详解

### 6.1 GPT-3: Language Models are Few-Shot Learners (2020)

**作者**：Tom B. Brown et al.

**发表会议**：NeurIPS

**详细分析**：[查看论文详解](../papers/gpt3.md)

**核心贡献**：
- 175B参数模型
- 少样本学习能力
- 上下文学习范式

### 6.2 LLaMA: Open and Efficient Foundation Language Models (2023)

**作者**：Hugo Touvron et al.

**发表会议**：arXiv

**详细分析**：[查看论文详解](../papers/llama.md)

**核心贡献**：
- 高质量开源模型
- 高效训练
- 良好的性能

### 6.3 LoRA: Low-Rank Adaptation of Large Language Models (2021)

**作者**：Edward J. Hu et al.

**发表会议**：ICLR

**详细分析**：[查看论文详解](../papers/lora.md)

**核心贡献**：
- 参数高效微调
- 低秩矩阵更新
- 保持模型性能

---

## 7. 实践练习

### 练习1：使用LLaMA-2生成文本

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 提示词
prompt = """
你是一个乐于助人的助手。请用中文回答以下问题：
什么是大语言模型？
"""

# 生成
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True
)

# 输出
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 练习2：使用LangChain构建问答系统

```python
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline

# 加载文档
loader = TextLoader("docs/knowledge.txt")
documents = loader.load()

# 创建向量库
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# 创建QA链
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 问答
query = "什么是机器学习？"
result = qa.run(query)
print(result)
```

### 练习3：使用OpenAI API

```python
import openai

# 设置API密钥
openai.api_key = "your-api-key"

# 调用GPT-4
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "你是一个专业的机器学习助手。"},
        {"role": "user", "content": "解释什么是Transformer架构？"}
    ],
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

**下一步**：[强化学习](../05-reinforcement-learning/01-rl-fundamentals.md)