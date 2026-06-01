# 9.3 推理优化

## 目录

- [1. 引言](#1-引言)
- [2. 推理优化概述](#2-推理优化概述)
- [3. 优化方法](#3-优化方法)
- [4. 推理引擎](#4-推理引擎)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 推理优化的重要性

**推理优化**是通过各种技术手段提高模型推理效率的过程。这对于实现低延迟、高吞吐量的AI服务至关重要。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **实时推理** | 需要毫秒级响应 | 自动驾驶、实时翻译 |
| **大规模部署** | 高吞吐量需求 | 云端API服务 |
| **边缘计算** | 资源受限环境 | 手机、IoT设备 |
| **成本优化** | 降低服务器成本 | 大规模推理集群 |

---

## 2. 推理优化概述

### 2.1 定义

**推理优化**：通过算法优化、硬件加速等手段提高模型推理效率。

**形式化表达**：
```
Optimize(Model) → OptimizedModel
```

### 2.2 优化目标

| 目标 | 描述 | 衡量指标 |
|------|------|---------|
| **低延迟** | 减少单次推理时间 | 平均延迟、P99延迟 |
| **高吞吐量** | 增加单位时间推理次数 | 请求/秒（QPS） |
| **内存效率** | 减少内存占用 | 峰值内存、显存使用 |
| **能耗优化** | 降低推理能耗 | 推理能耗 |

---

## 3. 优化方法

### 3.1 图优化

**定义**：通过优化计算图来减少冗余操作。

```python
class GraphOptimizer:
    def __init__(self):
        pass
    
    def optimize(self, model):
        """
        优化计算图
        
        参数:
            model: 原始模型
        
        返回:
            优化后的模型
        """
        # 1. 算子融合
        model = self._fuse_operators(model)
        
        # 2. 常量折叠
        model = self._constant_folding(model)
        
        # 3. 死代码消除
        model = self._dead_code_elimination(model)
        
        return model
    
    def _fuse_operators(self, model):
        """融合相邻算子"""
        fused_model = []
        
        i = 0
        while i < len(model):
            # 尝试融合
            if i + 1 < len(model):
                op1 = model[i]
                op2 = model[i+1]
                
                # 如果是连续的Conv + BN + ReLU，融合
                if isinstance(op1, nn.Conv2d) and isinstance(op2, nn.BatchNorm2d):
                    fused = self._fuse_conv_bn(op1, op2)
                    fused_model.append(fused)
                    i += 2
                    continue
            
            fused_model.append(model[i])
            i += 1
        
        return nn.Sequential(*fused_model)
    
    def _fuse_conv_bn(self, conv, bn):
        """融合Conv和BN"""
        # 计算融合后的权重和偏置
        weight = conv.weight.data
        bias = conv.bias.data if conv.bias is not None else torch.zeros(conv.out_channels)
        
        # BN参数
        gamma = bn.weight.data
        beta = bn.bias.data
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        
        # 融合
        std = torch.sqrt(var + eps)
        fused_weight = weight * (gamma / std).view(-1, 1, 1, 1)
        fused_bias = (bias - mean) * (gamma / std) + beta
        
        # 创建融合后的卷积层
        fused_conv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True
        )
        fused_conv.weight.data = fused_weight
        fused_conv.bias.data = fused_bias
        
        return fused_conv
    
    def _constant_folding(self, model):
        """常量折叠"""
        # 简化实现：移除常量节点
        return model
    
    def _dead_code_elimination(self, model):
        """死代码消除"""
        # 简化实现：移除无用节点
        return model

# 测试
model = nn.Sequential(
    nn.Conv2d(3, 16, 3),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3),
    nn.BatchNorm2d(32),
    nn.ReLU()
)

optimizer = GraphOptimizer()
optimized_model = optimizer.optimize(model)

print("优化前:")
for i, module in enumerate(model):
    print(f"{i}: {type(module).__name__}")

print("\n优化后:")
for i, module in enumerate(optimized_model):
    print(f"{i}: {type(module).__name__}")
```

### 3.2 算子优化

**定义**：优化单个算子的实现。

```python
class OperatorOptimizer:
    def __init__(self):
        pass
    
    def optimize_matmul(self, a, b):
        """
        优化矩阵乘法
        
        参数:
            a: 矩阵A
            b: 矩阵B
        
        返回:
            结果矩阵
        """
        # 使用Strassen算法加速
        if a.size(0) > 128 and b.size(1) > 128:
            return self._strassen_matmul(a, b)
        
        # 普通矩阵乘法
        return torch.matmul(a, b)
    
    def _strassen_matmul(self, a, b):
        """Strassen矩阵乘法"""
        n = a.size(0)
        
        if n <= 64:
            return torch.matmul(a, b)
        
        # 分块
        half = n // 2
        a11 = a[:half, :half]
        a12 = a[:half, half:]
        a21 = a[half:, :half]
        a22 = a[half:, half:]
        
        b11 = b[:half, :half]
        b12 = b[:half, half:]
        b21 = b[half:, :half]
        b22 = b[half:, half:]
        
        # Strassen公式
        m1 = self._strassen_matmul(a11 + a22, b11 + b22)
        m2 = self._strassen_matmul(a21 + a22, b11)
        m3 = self._strassen_matmul(a11, b12 - b22)
        m4 = self._strassen_matmul(a22, b21 - b11)
        m5 = self._strassen_matmul(a11 + a12, b22)
        m6 = self._strassen_matmul(a21 - a11, b11 + b12)
        m7 = self._strassen_matmul(a12 - a22, b21 + b22)
        
        # 合并结果
        result = torch.zeros(n, n, device=a.device)
        result[:half, :half] = m1 + m4 - m5 + m7
        result[:half, half:] = m3 + m5
        result[half:, :half] = m2 + m4
        result[half:, half:] = m1 - m2 + m3 + m6
        
        return result
    
    def optimize_conv(self, input, weight, bias=None, stride=1, padding=0):
        """
        优化卷积运算
        
        参数:
            input: 输入
            weight: 权重
            bias: 偏置
            stride: 步长
            padding: 填充
        
        返回:
            卷积结果
        """
        # 使用Winograd算法加速
        if weight.size(2) == 3 and stride == 1:
            return self._winograd_conv(input, weight, bias, padding)
        
        # 普通卷积
        return torch.nn.functional.conv2d(input, weight, bias, stride, padding)
    
    def _winograd_conv(self, input, weight, bias, padding):
        """Winograd卷积"""
        # 简化实现
        return torch.nn.functional.conv2d(input, weight, bias, padding=padding)

# 测试
optimizer = OperatorOptimizer()

# 测试矩阵乘法
a = torch.randn(256, 256)
b = torch.randn(256, 256)
result = optimizer.optimize_matmul(a, b)
print(f"矩阵乘法结果形状: {result.shape}")

# 测试卷积
input = torch.randn(1, 3, 64, 64)
weight = torch.randn(16, 3, 3, 3)
result = optimizer.optimize_conv(input, weight, padding=1)
print(f"卷积结果形状: {result.shape}")
```

### 3.3 内存优化

**定义**：优化内存访问模式。

```python
class MemoryOptimizer:
    def __init__(self):
        pass
    
    def optimize_memory(self, model, batch_size=1):
        """
        优化模型内存使用
        
        参数:
            model: 模型
            batch_size: 批次大小
        
        返回:
            优化后的模型
        """
        # 1. 激活函数重计算
        model = self._enable_recompute(model)
        
        # 2. 梯度检查点
        model = self._enable_checkpointing(model)
        
        # 3. 内存高效层
        model = self._replace_with_memory_efficient_layers(model)
        
        return model
    
    def _enable_recompute(self, model):
        """启用激活函数重计算"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU)):
                module.inplace = True
        
        return model
    
    def _enable_checkpointing(self, model):
        """启用梯度检查点"""
        def checkpoint_forward(module, *args):
            """检查点前向传播"""
            return torch.utils.checkpoint.checkpoint(module, *args)
        
        # 对大层启用检查点
        for name, module in model.named_modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                module.forward = checkpoint_forward
        
        return model
    
    def _replace_with_memory_efficient_layers(self, model):
        """替换为内存高效层"""
        # 简化实现
        return model
    
    def estimate_memory_usage(self, model, input_shape):
        """
        估算内存使用
        
        参数:
            model: 模型
            input_shape: 输入形状
        
        返回:
            内存使用量（MB）
        """
        input = torch.randn(*input_shape)
        model.eval()
        
        # 运行一次前向传播
        with torch.no_grad():
            output = model(input)
        
        # 计算内存使用
        total_memory = 0
        for param in model.parameters():
            total_memory += param.numel() * param.element_size()
        
        # 加上激活值
        for buf in model.buffers():
            total_memory += buf.numel() * buf.element_size()
        
        return total_memory / (1024 * 1024)  # MB

# 测试
model = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.GELU(),
    nn.Linear(256, 10)
)

optimizer = MemoryOptimizer()
optimized_model = optimizer.optimize_memory(model)

memory_usage = optimizer.estimate_memory_usage(optimized_model, (32, 1024))
print(f"内存使用估计: {memory_usage:.2f} MB")
```

---

## 4. 推理引擎

### 4.1 ONNX Runtime

```python
class ONNXEngine:
    def __init__(self):
        self.session = None
    
    def load_model(self, model_path):
        """
        加载ONNX模型
        
        参数:
            model_path: ONNX模型路径
        """
        import onnxruntime as ort
        
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(model_path, providers=providers)
    
    def infer(self, input_data):
        """
        执行推理
        
        参数:
            input_data: 输入数据
        
        返回:
            推理结果
        """
        if self.session is None:
            raise ValueError("模型未加载")
        
        # 获取输入名称
        input_name = self.session.get_inputs()[0].name
        
        # 执行推理
        outputs = self.session.run(None, {input_name: input_data.numpy()})
        
        return torch.tensor(outputs[0])
    
    def benchmark(self, input_data, num_runs=100):
        """
        基准测试
        
        参数:
            input_data: 输入数据
            num_runs: 运行次数
        
        返回:
            平均延迟（毫秒）
        """
        import time
        
        # 预热
        for _ in range(10):
            self.infer(input_data)
        
        # 正式测试
        start = time.time()
        for _ in range(num_runs):
            self.infer(input_data)
        
        avg_time = (time.time() - start) / num_runs * 1000
        return avg_time

# 测试
engine = ONNXEngine()
# engine.load_model('model.onnx')  # 需要实际的ONNX模型文件

# 模拟推理
# input_data = torch.randn(1, 3, 224, 224)
# result = engine.infer(input_data)
# print(f"推理结果形状: {result.shape}")
```

### 4.2 TensorRT

```python
class TensorRTEngine:
    def __init__(self):
        self.engine = None
        self.context = None
    
    def build_engine(self, model, input_shape, precision='fp16'):
        """
        构建TensorRT引擎
        
        参数:
            model: PyTorch模型
            input_shape: 输入形状
            precision: 精度（fp32/fp16/int8）
        """
        import tensorrt as trt
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with trt.Builder(TRT_LOGGER) as builder:
            builder.max_workspace_size = 1 << 30  # 1GB
            
            # 设置精度
            if precision == 'fp16':
                builder.fp16_mode = True
            elif precision == 'int8':
                builder.int8_mode = True
            
            # 创建网络定义
            network = builder.create_network()
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # 导出为ONNX
            dummy_input = torch.randn(*input_shape)
            torch.onnx.export(model, dummy_input, 'temp.onnx')
            
            # 解析ONNX
            with open('temp.onnx', 'rb') as f:
                parser.parse(f.read())
            
            # 构建引擎
            self.engine = builder.build_cuda_engine(network)
            
            # 创建执行上下文
            self.context = self.engine.create_execution_context()
    
    def infer(self, input_data):
        """
        执行推理
        
        参数:
            input_data: 输入数据
        
        返回:
            推理结果
        """
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # 分配GPU内存
        input_buffer = cuda.mem_alloc(input_data.numpy().nbytes)
        output_buffer = cuda.mem_alloc(10 * 4)  # 假设输出大小
        
        # 复制输入到GPU
        cuda.memcpy_htod(input_buffer, input_data.numpy())
        
        # 设置输入输出
        bindings = [int(input_buffer), int(output_buffer)]
        
        # 执行推理
        self.context.execute_v2(bindings)
        
        # 复制输出到CPU
        output = np.empty(10, dtype=np.float32)
        cuda.memcpy_dtoh(output, output_buffer)
        
        return torch.tensor(output)

# 测试（需要安装TensorRT）
# engine = TensorRTEngine()
# engine.build_engine(model, (1, 3, 224, 224), precision='fp16')
# result = engine.infer(torch.randn(1, 3, 224, 224))
```

### 4.3 vLLM

```python
class VLLMEngine:
    def __init__(self):
        self.llm = None
    
    def load_model(self, model_name, device='auto'):
        """
        加载模型
        
        参数:
            model_name: 模型名称
            device: 设备
        """
        from vllm import LLM
        
        self.llm = LLM(model=model_name, device=device)
    
    def generate(self, prompts, max_tokens=100):
        """
        生成文本
        
        参数:
            prompts: 提示列表
            max_tokens: 最大生成token数
        
        返回:
            生成结果
        """
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(max_tokens=max_tokens)
        outputs = self.llm.generate(prompts, sampling_params)
        
        return [output.outputs[0].text for output in outputs]
    
    def benchmark(self, prompt, num_runs=10):
        """
        基准测试
        
        参数:
            prompt: 测试提示
            num_runs: 运行次数
        
        返回:
            平均延迟（毫秒）
        """
        import time
        
        # 预热
        self.generate([prompt])
        
        # 测试
        start = time.time()
        for _ in range(num_runs):
            self.generate([prompt])
        
        avg_time = (time.time() - start) / num_runs * 1000
        return avg_time

# 测试（需要安装vLLM）
# engine = VLLMEngine()
# engine.load_model('lmsys/vicuna-7b-v1.5')
# result = engine.generate(["Hello, my name is"])
# print(f"生成结果: {result[0]}")
```

---

## 5. 实践练习

### 练习1：实现推理优化器

```python
class InferenceOptimizer:
    def __init__(self, model):
        self.model = model
        self.optimizations = []
    
    def add_optimization(self, name, func):
        """
        添加优化
        
        参数:
            name: 优化名称
            func: 优化函数
        """
        self.optimizations.append({'name': name, 'func': func})
    
    def apply_optimizations(self):
        """应用所有优化"""
        for opt in self.optimizations:
            print(f"应用优化: {opt['name']}")
            self.model = opt['func'](self.model)
        
        return self.model
    
    def benchmark(self, input_shape, num_runs=100):
        """
        基准测试
        
        参数:
            input_shape: 输入形状
            num_runs: 运行次数
        
        返回:
            基准测试结果
        """
        import time
        
        self.model.eval()
        input_data = torch.randn(*input_shape)
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                self.model(input_data)
        
        # 测试
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                self.model(input_data)
        
        avg_time = (time.time() - start) / num_runs * 1000
        
        return {
            'avg_latency_ms': avg_time,
            'throughput': num_runs / (time.time() - start)
        }

# 测试
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(128, 10)
)

optimizer = InferenceOptimizer(model)

# 添加优化
optimizer.add_optimization('算子融合', GraphOptimizer().optimize)
optimizer.add_optimization('内存优化', MemoryOptimizer().optimize_memory)

# 应用优化
optimized_model = optimizer.apply_optimizations()

# 基准测试
results = optimizer.benchmark((32, 3, 224, 224))
print(f"平均延迟: {results['avg_latency_ms']:.2f} ms")
print(f"吞吐量: {results['throughput']:.2f} qps")
```

### 练习2：实现批处理优化

```python
class BatchOptimizer:
    def __init__(self, model):
        self.model = model
    
    def optimize_batching(self, inputs, max_batch_size=64):
        """
        优化批处理
        
        参数:
            inputs: 输入列表
            max_batch_size: 最大批次大小
        
        返回:
            处理结果
        """
        results = []
        
        # 将输入分组
        for i in range(0, len(inputs), max_batch_size):
            batch = inputs[i:i+max_batch_size]
            
            # 填充到相同长度
            batch = self._pad_batch(batch)
            
            # 批量推理
            with torch.no_grad():
                outputs = self.model(batch)
            
            results.extend(outputs)
        
        return results
    
    def _pad_batch(self, inputs):
        """填充批次到相同长度"""
        # 找到最大长度
        max_len = max(input.size(1) for input in inputs)
        
        # 填充
        padded = []
        for input in inputs:
            if input.size(1) < max_len:
                padding = torch.zeros(input.size(0), max_len - input.size(1))
                padded.append(torch.cat([input, padding], dim=1))
            else:
                padded.append(input)
        
        return torch.stack(padded)
    
    def dynamic_batching(self, requests, timeout=100):
        """
        动态批处理
        
        参数:
            requests: 请求队列
            timeout: 超时时间（毫秒）
        
        返回:
            处理结果
        """
        import time
        
        batch = []
        results = {}
        request_ids = []
        
        start_time = time.time()
        
        while requests or batch:
            # 收集请求
            while requests and len(batch) < 64:
                req_id, req = requests.pop(0)
                batch.append(req)
                request_ids.append(req_id)
            
            # 检查超时或批次满
            elapsed = (time.time() - start_time) * 1000
            if batch and (len(batch) >= 64 or elapsed >= timeout):
                # 处理批次
                padded_batch = self._pad_batch(batch)
                
                with torch.no_grad():
                    outputs = self.model(padded_batch)
                
                # 分发结果
                for req_id, output in zip(request_ids, outputs):
                    results[req_id] = output
                
                # 重置
                batch = []
                request_ids = []
                start_time = time.time()
            
            time.sleep(0.001)
        
        return results

# 测试
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

batch_optimizer = BatchOptimizer(model)

# 测试批量优化
inputs = [torch.randn(1, 100) for _ in range(10)]
results = batch_optimizer.optimize_batching(inputs, max_batch_size=5)
print(f"处理结果数量: {len(results)}")

# 测试动态批处理
requests = [(i, torch.randn(1, 100)) for i in range(10)]
results = batch_optimizer.dynamic_batching(requests, timeout=50)
print(f"动态批处理结果数量: {len(results)}")
```

### 练习3：实现推理服务框架

```python
class InferenceServer:
    def __init__(self, model, port=8080):
        self.model = model
        self.port = port
        self.server = None
    
    def start(self):
        """启动服务"""
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        
        @app.route('/infer', methods=['POST'])
        def infer():
            data = request.get_json()
            inputs = torch.tensor(data['inputs'])
            
            with torch.no_grad():
                outputs = self.model(inputs)
            
            return jsonify({
                'outputs': outputs.tolist()
            })
        
        @app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy'})
        
        self.server = app.run(port=self.port, threaded=True)
    
    def stop(self):
        """停止服务"""
        if self.server:
            self.server.shutdown()
    
    def benchmark(self, num_requests=100, concurrency=10):
        """
        基准测试服务
        
        参数:
            num_requests: 请求数量
            concurrency: 并发数
        
        返回:
            基准测试结果
        """
        import requests
        import threading
        import time
        
        results = []
        
        def send_requests(count):
            for _ in range(count):
                data = {'inputs': torch.randn(1, 100).tolist()}
                
                start = time.time()
                response = requests.post(
                    f'http://localhost:{self.port}/infer',
                    json=data
                )
                elapsed = (time.time() - start) * 1000
                
                if response.status_code == 200:
                    results.append(elapsed)
        
        # 创建线程
        threads = []
        requests_per_thread = num_requests // concurrency
        
        for _ in range(concurrency):
            t = threading.Thread(target=send_requests, args=(requests_per_thread,))
            threads.append(t)
        
        # 启动线程
        start = time.time()
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # 计算结果
        avg_latency = sum(results) / len(results)
        throughput = num_requests / (time.time() - start)
        
        return {
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': sorted(results)[int(len(results) * 0.95)],
            'throughput': throughput
        }

# 测试
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

server = InferenceServer(model, port=8080)

# 启动服务（在单独线程中）
import threading
server_thread = threading.Thread(target=server.start)
server_thread.daemon = True
server_thread.start()

# 等待服务启动
import time
time.sleep(2)

# 基准测试
results = server.benchmark(num_requests=100, concurrency=10)
print("服务基准测试:")
print(f"平均延迟: {results['avg_latency_ms']:.2f} ms")
print(f"P95延迟: {results['p95_latency_ms']:.2f} ms")
print(f"吞吐量: {results['throughput']:.2f} qps")
```

---

**下一节**：[分布式推理](04-distributed-inference.md)

---

## 参考文献

1. Chen, T., et al. (2015). MXNet: A Flexible and Efficient Machine Learning Library.
2. NVIDIA (2023). TensorRT Documentation.
3. vLLM Team (2023). vLLM: Easy, Fast, and Cheap LLM Serving.