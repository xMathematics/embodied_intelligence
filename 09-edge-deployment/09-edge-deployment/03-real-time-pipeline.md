# 9.3 实时推理管线设计

## 1. 延迟预算 (轮式避障, 目标<50ms)

| 阶段 | 目标 | 实现方式 |
|------|------|---------|
| 传感器采集 | <5ms | DMA+零拷贝 |
| 图像预处理 | <3ms | CUDA Kernel (GPU) |
| 模型推理 | <20ms | TensorRT FP16 |
| 后处理 | <5ms | CPU (pinned memory) |
| 路径规划 | <10ms | 快速搜索 |
| 控制输出 | <5ms | CAN FD/EtherCAT |
| **总计** | **<48ms** | ✅ |

## 2. 管线设计原则

```
双缓冲: 一帧推理时, 下一帧已开始预处理
异步: 传感器采集与推理重叠
核心隔离: 控制循环独占核心
零拷贝: DMA直接将传感器数据写入GPU内存
```

## 3. 延迟抖动控制

```cpp
// CUDA Graph消除Kernel launch开销
cudaGraph_t graph;
cudaStreamBeginCapture(stream);
inference_kernel<<<grid, block, 0, stream>>>(input, output);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph);

// 每次执行都是确定性延迟
cudaGraphLaunch(instance, stream);  // ~0μs启动开销
```

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Real-Time Sensor Fusion on Edge GPU" | RSS 2022 | 实时传感器融合 |
| "Latency-Aware Pipeline Design" | IEEE TRO 2023 | 延迟感知管线 |
