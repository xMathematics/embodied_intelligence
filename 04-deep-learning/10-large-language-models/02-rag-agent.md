# 10.2 RAG与Agent

## 1. 为什么需要RAG

**问题**：LLM知识有截止日期，且无法获取私有数据。RAG（检索增强生成）让LLM检索外部知识库来回答问题。

**流程**：索引文档 → 检索相关段落 → 注入LLM上下文 → 生成回答。

## 2. 为什么需要Agent

**问题**：LLM只能生成文本，无法自主行动。Agent让LLM**使用工具、执行动作、观察结果**。

**ReAct模式**：
```
Thought: 我需要知道天气
Action: 调用 get_weather(location="北京")
Observation: 北京气温25°C
Thought: 我现在知道了天气...
```

## 3. Agent框架

| 框架 | 特点 |
|------|------|
| AutoGPT | 自动任务分解 |
| LangChain | 模块化Agent |
| CrewAI | 多Agent协作 |
| OpenAI Function Calling | 原生函数调用 |

## 4. 在具身智能中的应用

- **机器人任务规划**：Agent将自然语言任务分解为机器人技能序列
- **多机器人协作**：多个LLM Agent协调机器人行动
- **人机交互**：Agent理解用户意图并执行操作

## 5. 参考文献

1. Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS*.
2. Yao, S., et al. (2022). ReAct: Synergizing reasoning and acting in language models. *ICLR*.
3. Nakajima, Y. (2023). AutoGPT. *GitHub*.
