# 6.5 CUDA Stream与异步操作

## 1. 多Stream并行

```
时间 ──────────────────────────────────────────────→
Stream 1: [H→D] [Kernel A]        [D→H]
Stream 2:         [H→D] [Kernel B]        [D→H]
Stream 3:                 [H→D] [Kernel C]
         ← 传输与计算重叠 →
```

## 2. 多传感器流水线

```cpp
// 三个传感器独立Stream并行处理
cudaStream_t rgb_stream, depth_stream, lidar_stream;
cudaStreamCreate(&rgb_stream);
cudaStreamCreate(&depth_stream);
cudaStreamCreate(&lidar_stream);

while (running) {
    // 三个传感器并行
    cudaMemcpyAsync(d_rgb, h_rgb, size, H2D, rgb_stream);
    cudaMemcpyAsync(d_depth, h_depth, size, H2D, depth_stream);
    cudaMemcpyAsync(d_lidar, h_lidar, size, H2D, lidar_stream);
    
    // 并行推理
    rgb_kernel<<<grid, block, 0, rgb_stream>>>(d_rgb);
    depth_kernel<<<grid, block, 0, depth_stream>>>(d_depth);
    lidar_kernel<<<grid, block, 0, lidar_stream>>>(d_lidar);
    
    // 同步后融合
    cudaDeviceSynchronize();
    fusion_kernel<<<grid, block>>>(outputs);
}
```

## 3. CUDA Graph

```cpp
// 固定推理管线的极致优化
cudaGraph_t graph;
cudaStreamBeginCapture(stream);

// 记录固定操作序列
kernel1<<<grid, block, 0, stream>>>(input, tmp);
kernel2<<<grid, block, 0, stream>>>(tmp, output);
cudaMemcpyAsync(h_out, d_out, size, D2H, stream);

cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph);

// 重复执行 (无需Kernel launch开销)
while (running) {
    cudaGraphLaunch(instance, stream);
}
```

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "CUDA Streams for Real-Time Robotics" | ICRA 2022 | CUDA Stream机器人应用 |
| "CUDA Graph Optimization" | NVIDIA GTC 2021 | CUDA Graph |
