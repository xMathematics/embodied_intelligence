# 11.1 模型训练与推理优化

## 1. 为什么需要训练优化

**问题**：大模型（>1B参数）单GPU无法训练，需要分布式策略。

## 2. 分布式训练

### 2.1 数据并行（Data Parallel）

每个GPU完整模型，不同数据分片 → All-Reduce梯度。

### 2.2 模型并行

**张量并行**：拆分矩阵乘法到多个GPU。
**流水线并行**：按层拆分到不同GPU。

### 2.3 ZeRO优化（Zero Redundancy Optimizer）

| 阶段 | 优化状态 | 梯度 | 参数 | 内存节省 |
|------|----------|------|------|----------|
| ZeRO-1 | 分区 | 复制 | 复制 | 4x |
| ZeRO-2 | 分区 | 分区 | 复制 | 8x |
| ZeRO-3 | 分区 | 分区 | 分区 | 内存=参数量 |

## 3. 推理优化

| 技术 | 原理 | 加速 |
|------|------|------|
| FlashAttention | IO感知分块 | 2-4x |
| PagedAttention | KV Cache管理 | 2-4x |
| Continuous Batching | 动态批处理 | 2-10x |
| 量化(INT8/INT4) | 降低精度 | 2-4x |
| Speculative Decoding | 小模型+大模型 | 1.5-3x |

## 4. 在具身智能中的应用

- **边缘部署**：模型量化使LLM能在Jetson上运行
- **实时控制**：推理优化降低机器人策略延迟
- **能效优化**：低精度推理适合电池供电机器人

## 5. 参考文献

1. Rajbhandari, S., et al. (2020). ZeRO: Memory optimizations toward training trillion parameter models. *SC*.
2. Kwon, W., et al. (2023). Efficient memory management for large language model serving with PagedAttention. *SOSP*.
3. Dao, T., et al. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *NeurIPS*.
