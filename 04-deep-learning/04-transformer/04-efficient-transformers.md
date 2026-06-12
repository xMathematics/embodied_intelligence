# 4.4 高效Transformer

## 1. 为什么需要高效Transformer

### 1.1 问题：$O(n^2)$ 复杂度瓶颈

标准自注意力的计算量随着序列长度平方增长。对于8K序列，注意力操作占推理时间的80%以上。

### 1.2 需要高效Transformer的场景

| 场景 | 序列长度 | 挑战 |
|------|---------|------|
| 长文档理解 | 32K-200K | 内存爆炸 |
| 书籍/代码 | 100K-1M | 无法处理 |
| 视频理解 | 100K+帧 | 帧间注意力 |
| 机器人历史 | 长轨迹 | 实时性要求 |

## 2. FlashAttention

**论文**：Dao et al., 2022 — NeurIPS

### 2.1 核心洞察

标准注意力的问题是：频繁读写GPU HBM（高带宽内存）非常慢。

**解决方案**：**分块计算**（tiling），在SRAM上完成注意力计算：

1. 将Q/K/V分块加载到SRAM
2. 在SRAM上计算注意力
3. 写回HBM

**加速比**：2-4x（FlashAttention），3-6x（FlashAttention-2）。

## 3. 稀疏注意力

| 方法 | 模式 | 复杂度 | 代表 |
|------|------|--------|------|
| Sliding Window | 局部窗口 | $O(nw)$ | Longformer |
| Dilated Window | 空洞窗口 | $O(nw)$ | BigBird |
| Global+Local | 全局token+局部 | $O(nw)$ | Longformer |
| Block Sparse | 块状掩码 | 灵活 | Sparse Transformer |

## 4. 线性注意力

**Linear Transformer**（Katharopoulos et al., 2020）：

$$\text{Attention} = \phi(\mathbf{Q})(\phi(\mathbf{K})^T \mathbf{V})$$

其中 $\phi$ 是核函数（如ELU+1）。

**优势**：$O(n)$ 复杂度。
**局限**：在精度敏感任务上不如标准注意力。

## 5. 注意力优化综合对比

| 方法 | 复杂度 | 精度损失 | 加速比 | 适用 |
|------|--------|----------|--------|------|
| FlashAttention | $O(n^2)$(IO优化) | 无 | 2-4x | 通用 |
| 稀疏注意力 | $O(nw)$ | 小 | 4-8x | 长序列 |
| 线性注意力 | $O(n)$ | 中 | 10x+ | 极长序列 |
| MQA/GQA | $O(n^2)$(KV减小) | 微小 | 2x | 推理加速 |
| Speculative Decoding | - | 无 | 1.5-3x | 自回归 |
| KV Cache优化 | - | 无 | 变化 | 推理 |

## 6. 在具身智能中的应用

- **边缘部署**：FlashAttention使Transformer能在Jetson上运行
- **实时控制**：高效注意力使机器人策略推理更快
- **长程任务**：稀疏注意力处理长时间尺度的任务历史
- **多模态融合**：高效处理多相机/多传感器的注意力

## 7. 参考文献

1. Dao, T., et al. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *NeurIPS*.
2. Beltagy, I., Peters, M. E., & Cohan, A. (2020). Longformer: The long-document transformer. *arXiv:2004.05150*.
3. Katharopoulos, A., et al. (2020). Transformers are RNNs: Fast autoregressive transformers with linear attention. *ICML*.
4. Shazeer, N. (2019). Fast transformer decoding: One write-head is all you need. *arXiv:1911.02150*.
5. Leviathan, Y., et al. (2023). Fast inference from transformers via speculative decoding. *ICML*.
