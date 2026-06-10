# 8.2 TensorRT深度解析

## 1. 核心特性

| 特性 | 说明 | 机器人价值 |
|------|------|-----------|
| 图优化 | 自动消除冗余层, 融合算子 | 加速2-3× |
| 精度校准 | INT8/FP16校准 | 边缘实时推理 |
| 动态形状 | 支持可变输入尺寸 | 适配不同相机 |
| Tensor Core | 自动利用混合精度 | 加速5-10× |
| CUDA Graph | 固定管线极致优化 | 消除launch开销 |
| DLA | Jetson专用加速器 | 低功耗推理 |

## 2. 部署流程

```python
import tensorrt as trt

# 1. 构建Engine
builder = trt.Builder(logger)
network = builder.create_network(1 << int(
    trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
parser.parse(model_file.read())  # 从ONNX导入

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
config.set_flag(trt.BuilderFlag.FP16)

# 2. 构建 (耗时, 但只需一次)
engine = builder.build_serialized_network(network, config)

# 3. 推理
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()
context.execute_v2(buffers)
```

## 3. 在Jetson上的效果

| 模型 | FP32 | TensorRT FP16 | TensorRT INT8 |
|------|------|--------------|--------------|
| YOLOv8m | 30ms | 8ms | **5ms** |
| ResNet-50 | 15ms | 4ms | 3ms |
| EfficientNet-B0 | 8ms | 2ms | 1.5ms |

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "TensorRT Inference Optimizer" | NVIDIA 2022 | 官方白皮书 |
| "Optimizing ViT on Edge GPUs with TensorRT" | ICRA 2023 | ViT优化 |
| "Real-Time Multi-Model Inference on Jetson" | IROS 2022 | 多模型推理 |
