# 8.3 ONNX Runtime

## 1. 核心特性

| 特性 | 说明 | 机器人价值 |
|------|------|-----------|
| 跨平台 | Windows/Linux/Android/iOS | PC↔边缘无缝迁移 |
| 多后端 | CUDA, TensorRT, OpenVINO | 适配各种硬件 |
| 图优化 | 常量折叠, 节点消除 | 通用加速 |
| 量化支持 | INT8/FP16动态量化 | 模型压缩 |
| 自定义算子 | C/C++/CUDA | 机器人特定预处理 |

## 2. ONNX作为模型交换格式

```
PyTorch → ONNX → TensorRT Engine
PyTorch → ONNX → ONNX Runtime
Keras   → ONNX → OpenVINO
```

## 3. 机器人中的应用

```python
# ONNX Runtime + TensorRT后端
import onnxruntime as ort

providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_int8_enable': True,
        'trt_fp16_enable': True,
    }),
    'CUDAExecutionProvider',
]

session = ort.InferenceSession('model.onnx', providers=providers)
result = session.run(None, {'input': image})
```

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "ONNX Format" | Microsoft/Facebook 2017 | ONNX规范 |
| "ONNX Runtime" | Microsoft 2021 | ORT框架 |
