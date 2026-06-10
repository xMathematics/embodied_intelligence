# 6.2 CUDA内存管理与传输优化

## 1. 内存类型

| 分配函数 | 位置 | 速度 | 作用域 |
|---------|------|------|-------|
| `cudaMalloc` | 全局内存 | 慢 (~400 cycle) | 所有线程 |
| `cudaMallocManaged` | 统一内存 | 自动迁移 | 所有 |
| `cudaHostAlloc` | 固定内存 | 快 (H2D) | 主机 |
| `cudaMallocHost` | 固定内存 | 快 (H2D) | 主机 |

## 2. 关键技巧

### 固定内存 (Pinned Memory)
```cpp
// 固定内存加速传输 2-3×
float* h_pinned;
cudaHostAlloc(&h_pinned, N * sizeof(float), cudaHostAllocDefault);

// 传感器数据流使用固定内存
// RealSense相机 → 固定内存 → 异步传输到GPU
```

### 统一内存 (Unified Memory)
```cpp
// 简化编程 (但可能牺牲性能)
float* u_data;
cudaMallocManaged(&u_data, N * sizeof(float));

// CPU访问
for (int i = 0; i < N; i++) u_data[i] = i;   // page fault
// GPU访问
kernel<<<grid, block>>>(u_data, N);             // 自动迁移
```

## 3. 机器人数据传输

```cpp
// 相机帧到GPU的高效传输
void frame_to_gpu(Frame& frame, float* d_buffer) {
    // 1. 使用固定内存作为中间缓冲区
    static float* h_buffer;
    if (!h_buffer) cudaHostAlloc(&h_buffer, FRAME_SIZE, ...);
    
    // 2. DMA将相机数据写入固定内存
    frame.dma_to(h_buffer);
    
    // 3. 异步传输到GPU (非阻塞)
    cudaMemcpyAsync(d_buffer, h_buffer, FRAME_SIZE,
                    cudaMemcpyHostToDevice, stream);
}
```

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Unified Memory for Heterogeneous Computing" | NVIDIA 2016 | 统一内存 |
| "Streaming Data Processing for Robot Perception" | ICRA 2021 | 机器人流式数据处理 |
