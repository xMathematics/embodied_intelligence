# 5.5 性能度量与基准测试

## 1. 关键指标

| 指标 | 定义 | 机器人意义 |
|------|------|-----------|
| FLOPS | 每秒浮点运算 | 模型训练速度 |
| TFLOPS利用率 | 实际/峰值 | GPU利用效率 |
| 吞吐量 | samples/s | 训练/推理效率 |
| 延迟 | ms | 实时性 |
| 每瓦性能 | FLOPS/W | 边缘部署能效 |

## 2. 机器人相关基准

| 基准 | 测试内容 | 指标 |
|------|---------|------|
| MLPerf | 训练/推理 | 吞吐量 |
| MLPerf Tiny | 边缘推理 | 延迟+功耗 |
| Jetson Benchmarks | 边缘AI | FPS, 功耗 |
| ROS2 Performance | 通信延迟 | 延迟分布 |

## 3. 在具身智能中的应用

```bash
# Jetson性能测试
sudo jetson_clocks                    # 最大性能模式
sudo nvpmodel -m 8                    # MAXN模式
# 运行推理测试
trtexec --loadEngine=yolov8.engine \
        --iterations=100 \
        --warmUp=10
# 输出: Throughput, Latency (p50/p90/p99)
```

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "MLPerf Industry Standard" | Mattson 2020 | MLPerf基准 |
| "Edge AI Benchmark" | IROS 2021 | 边缘AI基准测试 |
