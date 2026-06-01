# 9.4 分布式推理

## 目录

- [1. 引言](#1-引言)
- [2. 分布式推理概述](#2-分布式推理概述)
- [3. 分布式策略](#3-分布式策略)
- [4. 实现方法](#4-实现方法)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 分布式推理的重要性

**分布式推理**是将模型推理任务分布到多个设备或机器上执行的技术。这对于处理超大型模型和高吞吐量需求至关重要。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **超大型模型** | 模型太大无法放在单个设备上 | 万亿参数模型 |
| **高吞吐量** | 需要处理大量并发请求 | 大规模API服务 |
| **容错性** | 需要高可用性 | 生产环境部署 |
| **地理分布式** | 需要全球低延迟访问 | CDN推理 |

---

## 2. 分布式推理概述

### 2.1 定义

**分布式推理**：将模型推理任务分布到多个计算节点上执行。

**形式化表达**：
```
DistributedInfer(Model, Nodes) → Results
```

### 2.2 分布式类型

| 类型 | 描述 | 适用场景 |
|------|------|---------|
| **模型并行** | 将模型分成多个部分 | 超大型模型 |
| **数据并行** | 并行处理多个批次 | 高吞吐量 |
| **流水线并行** | 将计算分成阶段 | 长序列处理 |
| **混合并行** | 组合多种策略 | 复杂场景 |

---

## 3. 分布式策略

### 3.1 模型并行

**定义**：将模型的不同层分布到不同设备上。

```python
class ModelParallelModel(nn.Module):
    def __init__(self, layers, devices):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.devices = devices
        
        # 将层分布到不同设备
        for i, layer in enumerate(self.layers):
            self.layers[i] = layer.to(self.devices[i % len(self.devices)])
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入
        
        返回:
            输出
        """
        for i, layer in enumerate(self.layers):
            # 将输入移动到当前层的设备
            x = x.to(self.devices[i % len(self.devices)])
            
            # 执行前向传播
            x = layer(x)
        
        return x

# 测试
layers = [
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 250),
    nn.ReLU(),
    nn.Linear(250, 10)
]

devices = ['cuda:0', 'cuda:1'] if torch.cuda.device_count() >= 2 else ['cpu', 'cpu']

model = ModelParallelModel(layers, devices)
inputs = torch.randn(32, 1000)
outputs = model(inputs)
print(f"输出形状: {outputs.shape}")
```

### 3.2 数据并行

**定义**：将批次数据分成多个子批次并行处理。

```python
class DataParallelModel(nn.Module):
    def __init__(self, model, devices):
        super().__init__()
        self.model = model
        self.devices = devices
        
        # 在每个设备上复制模型
        self.models = nn.ModuleList([
            model.to(device) for device in devices
        ])
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入 [batch, ...]
        
        返回:
            输出
        """
        # 分割批次
        batch_size = x.size(0)
        splits = torch.split(x, batch_size // len(self.devices), dim=0)
        
        # 并行处理
        outputs = []
        for i, split in enumerate(splits):
            split = split.to(self.devices[i])
            output = self.models[i](split)
            outputs.append(output.cpu())
        
        # 合并结果
        return torch.cat(outputs, dim=0)

# 测试
base_model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

devices = ['cuda:0', 'cuda:1'] if torch.cuda.device_count() >= 2 else ['cpu', 'cpu']

model = DataParallelModel(base_model, devices)
inputs = torch.randn(64, 100)
outputs = model(inputs)
print(f"输出形状: {outputs.shape}")
```

### 3.3 流水线并行

**定义**：将计算分成多个阶段，按流水线方式执行。

```python
class PipelineParallelModel(nn.Module):
    def __init__(self, stages, devices):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.devices = devices
        
        # 将阶段分布到不同设备
        for i, stage in enumerate(self.stages):
            self.stages[i] = stage.to(self.devices[i % len(self.devices)])
    
    def forward(self, x, micro_batch_size=8):
        """
        前向传播（流水线）
        
        参数:
            x: 输入 [batch, ...]
            micro_batch_size: 微批次大小
        
        返回:
            输出
        """
        batch_size = x.size(0)
        num_micro_batches = batch_size // micro_batch_size
        
        # 分割成微批次
        micro_batches = torch.split(x, micro_batch_size, dim=0)
        
        # 执行流水线
        outputs = []
        
        for i, micro_batch in enumerate(micro_batches):
            # 前向传播当前微批次
            current = micro_batch
            
            for j, stage in enumerate(self.stages):
                # 移动到当前阶段的设备
                current = current.to(self.devices[j % len(self.devices)])
                
                # 执行阶段
                current = stage(current)
            
            outputs.append(current.cpu())
        
        # 合并结果
        return torch.cat(outputs, dim=0)

# 测试
stages = [
    nn.Sequential(nn.Linear(100, 50), nn.ReLU()),
    nn.Sequential(nn.Linear(50, 30), nn.ReLU()),
    nn.Sequential(nn.Linear(30, 10))
]

devices = ['cuda:0', 'cuda:1', 'cuda:0'] if torch.cuda.device_count() >= 2 else ['cpu', 'cpu', 'cpu']

model = PipelineParallelModel(stages, devices)
inputs = torch.randn(32, 100)
outputs = model(inputs, micro_batch_size=8)
print(f"输出形状: {outputs.shape}")
```

---

## 4. 实现方法

### 4.1 使用PyTorch分布式

```python
class PyTorchDistributedInference:
    def __init__(self, model):
        self.model = model
    
    def setup(self, backend='nccl'):
        """
        设置分布式环境
        
        参数:
            backend: 后端（nccl/gloo）
        """
        import torch.distributed as dist
        
        # 初始化进程组
        dist.init_process_group(backend=backend)
        
        # 获取本地排名
        local_rank = dist.get_rank()
        
        # 将模型移动到对应设备
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        # 使用分布式数据并行
        self.model = nn.parallel.DistributedDataParallel(self.model)
    
    def infer(self, inputs):
        """
        执行推理
        
        参数:
            inputs: 输入数据
        
        返回:
            推理结果
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(inputs)
        
        return outputs
    
    def cleanup(self):
        """清理分布式环境"""
        import torch.distributed as dist
        dist.destroy_process_group()

# 测试
model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

dist_infer = PyTorchDistributedInference(model)
# dist_infer.setup()  # 需要在多进程环境中运行

# inputs = torch.randn(32, 100)
# outputs = dist_infer.infer(inputs)
# print(f"输出形状: {outputs.shape}")
```

### 4.2 使用Ray分布式

```python
class RayDistributedInference:
    def __init__(self, model):
        self.model = model
        self.actors = []
    
    def setup(self, num_workers=4):
        """
        设置Ray分布式
        
        参数:
            num_workers: 工作器数量
        """
        import ray
        
        # 初始化Ray
        ray.init()
        
        # 创建工作器
        @ray.remote
        class InferenceWorker:
            def __init__(self, model):
                self.model = model
                self.model.eval()
            
            def infer(self, inputs):
                with torch.no_grad():
                    return self.model(inputs)
        
        # 创建多个工作器
        self.actors = [InferenceWorker.remote(self.model) for _ in range(num_workers)]
    
    def infer(self, inputs, batch_size=8):
        """
        执行推理
        
        参数:
            inputs: 输入数据
            batch_size: 批次大小
        
        返回:
            推理结果
        """
        # 分割输入
        batches = torch.split(inputs, batch_size, dim=0)
        
        # 并行推理
        results = ray.get([
            self.actors[i % len(self.actors)].infer.remote(batch)
            for i, batch in enumerate(batches)
        ])
        
        # 合并结果
        return torch.cat(results, dim=0)
    
    def cleanup(self):
        """清理Ray环境"""
        import ray
        ray.shutdown()

# 测试（需要安装Ray）
# model = nn.Sequential(
#     nn.Linear(100, 50),
#     nn.ReLU(),
#     nn.Linear(50, 10)
# )

# ray_infer = RayDistributedInference(model)
# ray_infer.setup(num_workers=2)

# inputs = torch.randn(32, 100)
# outputs = ray_infer.infer(inputs)
# print(f"输出形状: {outputs.shape}")
```

### 4.3 使用gRPC分布式

```python
class GRPCDistributedInference:
    def __init__(self):
        self.stubs = []
    
    def add_worker(self, address):
        """
        添加工作器
        
        参数:
            address: 工作器地址
        """
        import grpc
        from inference_pb2 import InferRequest, InferResponse
        from inference_pb2_grpc import InferenceServiceStub
        
        channel = grpc.insecure_channel(address)
        stub = InferenceServiceStub(channel)
        self.stubs.append(stub)
    
    def infer(self, inputs):
        """
        执行推理
        
        参数:
            inputs: 输入数据
        
        返回:
            推理结果
        """
        import grpc
        from inference_pb2 import InferRequest
        
        # 分割输入
        batch_size = inputs.size(0) // len(self.stubs)
        batches = torch.split(inputs, batch_size, dim=0)
        
        # 并行推理
        results = []
        
        for i, (stub, batch) in enumerate(zip(self.stubs, batches)):
            request = InferRequest(
                inputs=batch.numpy().flatten().tolist(),
                shape=list(batch.shape)
            )
            
            response = stub.Infer(request)
            result = torch.tensor(response.outputs).view(response.shape)
            results.append(result)
        
        # 合并结果
        return torch.cat(results, dim=0)

# 测试（需要gRPC服务端）
# grpc_infer = GRPCDistributedInference()
# grpc_infer.add_worker('localhost:50051')
# grpc_infer.add_worker('localhost:50052')

# inputs = torch.randn(32, 100)
# outputs = grpc_infer.infer(inputs)
# print(f"输出形状: {outputs.shape}")
```

---

## 5. 实践练习

### 练习1：实现模型并行推理

```python
class AdvancedModelParallel:
    def __init__(self, model, devices):
        self.devices = devices
        self.model_parts = []
        
        # 分割模型
        self._split_model(model)
    
    def _split_model(self, model):
        """分割模型到多个设备"""
        layers = list(model.children())
        num_layers = len(layers)
        layers_per_device = num_layers // len(self.devices)
        
        for i, device in enumerate(self.devices):
            start = i * layers_per_device
            end = start + layers_per_device if i < len(self.devices) - 1 else num_layers
            
            part = nn.Sequential(*layers[start:end])
            self.model_parts.append(part.to(device))
    
    def infer(self, inputs):
        """
        执行推理
        
        参数:
            inputs: 输入数据
        
        返回:
            推理结果
        """
        x = inputs
        
        for i, part in enumerate(self.model_parts):
            # 移动到当前设备
            x = x.to(self.devices[i])
            
            # 执行前向传播
            x = part(x)
        
        return x.cpu()
    
    def benchmark(self, input_shape, num_runs=100):
        """
        基准测试
        
        参数:
            input_shape: 输入形状
            num_runs: 运行次数
        
        返回:
            平均延迟（毫秒）
        """
        import time
        
        inputs = torch.randn(*input_shape)
        
        # 预热
        for _ in range(10):
            self.infer(inputs)
        
        # 测试
        start = time.time()
        for _ in range(num_runs):
            self.infer(inputs)
        
        avg_time = (time.time() - start) / num_runs * 1000
        return avg_time

# 测试
model = nn.Sequential(
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

devices = ['cuda:0', 'cuda:1'] if torch.cuda.device_count() >= 2 else ['cpu', 'cpu']
parallel_model = AdvancedModelParallel(model, devices)

# 测试推理
inputs = torch.randn(32, 1024)
outputs = parallel_model.infer(inputs)
print(f"输出形状: {outputs.shape}")

# 基准测试
latency = parallel_model.benchmark((32, 1024), num_runs=50)
print(f"平均延迟: {latency:.2f} ms")
```

### 练习2：实现数据并行推理

```python
class AdvancedDataParallel:
    def __init__(self, model, devices):
        self.devices = devices
        
        # 在每个设备上复制模型
        self.models = [model.to(device) for device in devices]
        
        # 设置模型为评估模式
        for m in self.models:
            m.eval()
    
    def infer(self, inputs):
        """
        执行推理
        
        参数:
            inputs: 输入数据
        
        返回:
            推理结果
        """
        batch_size = inputs.size(0)
        num_devices = len(self.devices)
        
        # 分割批次
        splits = torch.split(inputs, batch_size // num_devices, dim=0)
        
        # 并行处理
        outputs = []
        
        for i, (device, split) in enumerate(zip(self.devices, splits)):
            # 移动到设备
            split = split.to(device)
            
            # 推理
            with torch.no_grad():
                output = self.models[i](split)
            
            outputs.append(output.cpu())
        
        # 合并结果
        return torch.cat(outputs, dim=0)
    
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
        
        inputs = torch.randn(*input_shape)
        
        # 预热
        for _ in range(10):
            self.infer(inputs)
        
        # 测试
        start = time.time()
        for _ in range(num_runs):
            self.infer(inputs)
        
        total_time = time.time() - start
        
        return {
            'avg_latency_ms': total_time / num_runs * 1000,
            'throughput': num_runs / total_time
        }

# 测试
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128 * 56 * 56, 10)
)

devices = ['cuda:0', 'cuda:1'] if torch.cuda.device_count() >= 2 else ['cpu', 'cpu']
parallel_model = AdvancedDataParallel(model, devices)

# 测试推理
inputs = torch.randn(16, 3, 224, 224)
outputs = parallel_model.infer(inputs)
print(f"输出形状: {outputs.shape}")

# 基准测试
results = parallel_model.benchmark((16, 3, 224, 224), num_runs=50)
print(f"平均延迟: {results['avg_latency_ms']:.2f} ms")
print(f"吞吐量: {results['throughput']:.2f} qps")
```

### 练习3：实现分布式推理调度器

```python
class DistributedScheduler:
    def __init__(self, workers):
        self.workers = workers
        self.task_queue = []
        self.results = {}
    
    def submit_task(self, task_id, inputs):
        """
        提交任务
        
        参数:
            task_id: 任务ID
            inputs: 输入数据
        """
        self.task_queue.append({'task_id': task_id, 'inputs': inputs})
    
    def schedule(self, strategy='round_robin'):
        """
        调度任务
        
        参数:
            strategy: 调度策略（round_robin/load_balanced）
        
        返回:
            任务结果
        """
        while self.task_queue:
            task = self.task_queue.pop(0)
            
            # 选择工作器
            if strategy == 'round_robin':
                worker = self.workers[task['task_id'] % len(self.workers)]
            elif strategy == 'load_balanced':
                worker = self._select_least_loaded()
            else:
                worker = self.workers[0]
            
            # 执行任务
            result = worker.infer(task['inputs'])
            self.results[task['task_id']] = result
        
        return self.results
    
    def _select_least_loaded(self):
        """选择负载最低的工作器"""
        # 简化实现：随机选择
        import random
        return random.choice(self.workers)
    
    def get_results(self):
        """获取所有结果"""
        return self.results

# 测试
class MockWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
    
    def infer(self, inputs):
        """模拟推理"""
        with torch.no_grad():
            return self.model(inputs)

# 创建工作器
workers = [MockWorker(i) for i in range(3)]

# 创建调度器
scheduler = DistributedScheduler(workers)

# 提交任务
for i in range(10):
    scheduler.submit_task(i, torch.randn(8, 100))

# 调度任务
results = scheduler.schedule(strategy='round_robin')
print(f"完成任务数量: {len(results)}")

# 获取结果
all_results = scheduler.get_results()
print(f"获取结果数量: {len(all_results)}")
```

---

**下一节**：[服务部署](05-service-deployment.md)

---

## 参考文献

1. Rajbhandari, S., et al. (2021). ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning.
2. Shoeybi, M., et al. (2019). Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.
3. Ray Team (2023). Ray Documentation.