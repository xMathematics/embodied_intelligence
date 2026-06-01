# 9.2 量化技术

## 目录

- [1. 引言](#1-引言)
- [2. 量化技术概述](#2-量化技术概述)
- [3. 量化方法](#3-量化方法)
- [4. 量化工具](#4-量化工具)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 量化技术的重要性

**量化**是通过减少参数的精度来压缩模型的技术。这是实现高效推理的关键方法，能够显著减少模型大小和内存占用。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **边缘推理** | 在资源受限设备上运行 | 手机、嵌入式设备 |
| **云端部署** | 提高吞吐量和降低延迟 | 大规模API服务 |
| **模型分发** | 减少下载体积 | 移动端应用 |
| **成本优化** | 降低服务器成本 | 大规模推理集群 |

---

## 2. 量化技术概述

### 2.1 定义

**量化**：将浮点数参数转换为低精度整数表示的过程。

**形式化表达**：
```
Quantize(weights, bits) → QuantizedWeights
```

### 2.2 量化类型

| 类型 | 描述 | 精度 |
|------|------|------|
| **FP32** | 单精度浮点 | 32位 |
| **FP16** | 半精度浮点 | 16位 |
| **BF16** | 脑浮点 | 16位 |
| **INT8** | 8位整数 | 8位 |
| **INT4** | 4位整数 | 4位 |

---

## 3. 量化方法

### 3.1 后训练量化（PTQ）

**定义**：在训练完成后对模型进行量化。

```python
class PostTrainingQuantization:
    def __init__(self, model):
        self.model = model
        self.quantized_model = None
    
    def quantize(self, calibration_data, bits=8):
        """
        后训练量化
        
        参数:
            calibration_data: 校准数据
            bits: 量化位数
        
        返回:
            量化后的模型
        """
        # 设置量化配置
        self.bits = bits
        
        # 收集统计信息
        self._collect_statistics(calibration_data)
        
        # 量化权重
        self._quantize_weights()
        
        # 创建量化模型
        self._create_quantized_model()
        
        return self.quantized_model
    
    def _collect_statistics(self, data):
        """收集激活值统计信息"""
        self.model.eval()
        self.activation_stats = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                self.activation_stats[name] = {
                    'min': float('inf'),
                    'max': float('-inf')
                }
        
        with torch.no_grad():
            for inputs in data:
                outputs = self.model(inputs)
        
        return self.activation_stats
    
    def _quantize_weights(self):
        """量化权重"""
        self.quantized_weights = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # 量化权重
                min_val = param.data.min().item()
                max_val = param.data.max().item()
                
                scale = (max_val - min_val) / (2**self.bits - 1)
                zero_point = round(-min_val / scale)
                
                quantized = torch.round(param.data / scale + zero_point)
                quantized = torch.clamp(quantized, 0, 2**self.bits - 1)
                
                self.quantized_weights[name] = {
                    'quantized': quantized,
                    'scale': scale,
                    'zero_point': zero_point
                }
    
    def _create_quantized_model(self):
        """创建量化模型"""
        self.quantized_model = self.model.copy()
        
        for name, param in self.quantized_model.named_parameters():
            if name in self.quantized_weights:
                qw = self.quantized_weights[name]
                param.data = qw['quantized']
                param.scale = qw['scale']
                param.zero_point = qw['zero_point']

# 测试
model = SimpleModel()
calibration_data = [torch.randn(32, 100) for _ in range(10)]

ptq = PostTrainingQuantization(model)
quantized_model = ptq.quantize(calibration_data, bits=8)
print("后训练量化完成")
```

### 3.2 量化感知训练（QAT）

**定义**：在训练过程中模拟量化效果。

```python
class QuantizationAwareTraining:
    def __init__(self, model, bits=8):
        self.model = model
        self.bits = bits
        self._prepare_model()
    
    def _prepare_model(self):
        """准备模型进行量化感知训练"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 添加量化钩子
                module.register_forward_pre_hook(self._quantize_hook)
    
    def _quantize_hook(self, module, input):
        """量化钩子"""
        if hasattr(module, 'weight'):
            # 量化权重
            weight = module.weight.data
            min_val = weight.min().item()
            max_val = weight.max().item()
            
            scale = (max_val - min_val) / (2**self.bits - 1)
            zero_point = round(-min_val / scale)
            
            # 量化并反量化（模拟量化误差）
            quantized = torch.round(weight / scale + zero_point)
            quantized = torch.clamp(quantized, 0, 2**self.bits - 1)
            dequantized = (quantized - zero_point) * scale
            
            # 替换权重
            module.weight.data = dequantized
        
        return input
    
    def train(self, dataloader, epochs=1):
        """训练模型"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for inputs, labels in dataloader:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"QAT Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 测试
model = SimpleModel()
qat = QuantizationAwareTraining(model, bits=8)

dataloader = [(torch.randn(32, 100), torch.randint(0, 10, (32,))) for _ in range(10)]
qat.train(dataloader, epochs=2)
```

### 3.3 动态量化

**定义**：只量化权重，激活值保持浮点。

```python
class DynamicQuantization:
    def __init__(self, model):
        self.model = model
    
    def quantize(self, bits=8):
        """
        动态量化
        
        参数:
            bits: 量化位数
        
        返回:
            量化后的模型
        """
        self.bits = bits
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # 量化权重
                min_val = param.data.min().item()
                max_val = param.data.max().item()
                
                scale = (max_val - min_val) / (2**bits - 1)
                zero_point = round(-min_val / scale)
                
                quantized = torch.round(param.data / scale + zero_point)
                quantized = torch.clamp(quantized, 0, 2**bits - 1)
                
                # 保存量化参数
                param.data = quantized.to(torch.int8)
                param.scale = scale
                param.zero_point = zero_point
        
        return self.model
    
    def forward(self, x):
        """
        前向传播（动态量化）
        
        参数:
            x: 输入
        
        返回:
            输出
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 反量化权重
                weight = (module.weight.data.float() - module.zero_point) * module.scale
                output = torch.matmul(x, weight.t())
                if module.bias is not None:
                    output += module.bias
                x = output
        
        return x

# 测试
model = SimpleModel()
dq = DynamicQuantization(model)
quantized_model = dq.quantize(bits=8)

inputs = torch.randn(32, 100)
outputs = dq.forward(inputs)
print(f"输出形状: {outputs.shape}")
```

---

## 4. 量化工具

### 4.1 GPTQ

**定义**：基于 Hessian 的量化方法。

```python
class GPTQQuantizer:
    def __init__(self, model):
        self.model = model
    
    def quantize(self, calibration_data, bits=4):
        """
        GPTQ量化
        
        参数:
            calibration_data: 校准数据
            bits: 量化位数
        
        返回:
            量化后的模型
        """
        self.bits = bits
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # GPTQ量化
                weight = module.weight.data
                
                # 计算重要性
                importance = self._compute_importance(module, calibration_data)
                
                # 按重要性排序
                sorted_indices = torch.argsort(importance, descending=True)
                
                # 逐步量化
                for i in range(0, weight.size(0), 64):
                    end = min(i + 64, weight.size(0))
                    block_indices = sorted_indices[i:end]
                    
                    # 量化这个块
                    block = weight[block_indices]
                    quantized_block = self._quantize_block(block, bits)
                    
                    # 更新权重
                    weight[block_indices] = quantized_block
                
                module.weight.data = weight
        
        return self.model
    
    def _compute_importance(self, module, data):
        """计算权重重要性"""
        # 简化实现：使用权重的幅度
        return torch.norm(module.weight.data, p=2, dim=1)
    
    def _quantize_block(self, block, bits):
        """量化一个块"""
        min_val = block.min().item()
        max_val = block.max().item()
        
        scale = (max_val - min_val) / (2**bits - 1)
        zero_point = round(-min_val / scale)
        
        quantized = torch.round(block / scale + zero_point)
        quantized = torch.clamp(quantized, 0, 2**bits - 1)
        
        return (quantized - zero_point) * scale

# 测试
model = SimpleModel()
calibration_data = [torch.randn(32, 100) for _ in range(10)]

gptq = GPTQQuantizer(model)
quantized_model = gptq.quantize(calibration_data, bits=4)
print("GPTQ量化完成")
```

### 4.2 AWQ

**定义**：Activation-aware Weight Quantization。

```python
class AWQQuantizer:
    def __init__(self, model):
        self.model = model
    
    def quantize(self, calibration_data, bits=4):
        """
        AWQ量化
        
        参数:
            calibration_data: 校准数据
            bits: 量化位数
        
        返回:
            量化后的模型
        """
        # 收集激活值统计
        activation_max = self._collect_activation_stats(calibration_data)
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # AWQ量化
                weight = module.weight.data
                
                # 基于激活值调整量化
                if name in activation_max:
                    act_max = activation_max[name]
                    weight = weight / act_max
                
                # 量化
                min_val = weight.min().item()
                max_val = weight.max().item()
                
                scale = (max_val - min_val) / (2**bits - 1)
                zero_point = round(-min_val / scale)
                
                quantized = torch.round(weight / scale + zero_point)
                quantized = torch.clamp(quantized, 0, 2**bits - 1)
                
                module.weight.data = (quantized - zero_point) * scale * act_max
        
        return self.model
    
    def _collect_activation_stats(self, data):
        """收集激活值统计"""
        activation_max = {}
        
        def hook(module, input, output):
            name = None
            for n, m in self.model.named_modules():
                if m is module:
                    name = n
                    break
            
            if name:
                activation_max[name] = output.abs().max().item()
        
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook))
        
        self.model.eval()
        with torch.no_grad():
            for inputs in data:
                self.model(inputs)
        
        for h in hooks:
            h.remove()
        
        return activation_max

# 测试
model = SimpleModel()
calibration_data = [torch.randn(32, 100) for _ in range(10)]

awq = AWQQuantizer(model)
quantized_model = awq.quantize(calibration_data, bits=4)
print("AWQ量化完成")
```

---

## 5. 实践练习

### 练习1：实现INT8量化

```python
class INT8Quantizer:
    def __init__(self):
        self.scale = {}
        self.zero_point = {}
    
    def quantize_tensor(self, tensor):
        """
        量化张量
        
        参数:
            tensor: 浮点数张量
        
        返回:
            量化后的整数张量
        """
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # 计算缩放因子和零点
        scale = (max_val - min_val) / 255.0
        zero_point = round(-min_val / scale)
        
        # 量化
        quantized = torch.round(tensor / scale + zero_point)
        quantized = torch.clamp(quantized, 0, 255).to(torch.uint8)
        
        return quantized, scale, zero_point
    
    def dequantize_tensor(self, quantized_tensor, scale, zero_point):
        """
        反量化张量
        
        参数:
            quantized_tensor: 量化后的张量
            scale: 缩放因子
            zero_point: 零点
        
        返回:
            反量化后的浮点数张量
        """
        return (quantized_tensor.float() - zero_point) * scale
    
    def quantize_model(self, model):
        """
        量化整个模型
        
        参数:
            model: PyTorch模型
        
        返回:
            量化后的模型
        """
        for name, param in model.named_parameters():
            if 'weight' in name:
                quantized, scale, zero_point = self.quantize_tensor(param.data)
                
                # 保存量化参数
                self.scale[name] = scale
                self.zero_point[name] = zero_point
                
                # 替换为量化后的值
                param.data = quantized
        
        return model
    
    def dequantize_model(self, model):
        """
        反量化模型
        
        参数:
            model: 量化后的模型
        
        返回:
            反量化后的模型
        """
        for name, param in model.named_parameters():
            if name in self.scale:
                dequantized = self.dequantize_tensor(
                    param.data,
                    self.scale[name],
                    self.zero_point[name]
                )
                param.data = dequantized
        
        return model

# 测试
model = SimpleModel()
quantizer = INT8Quantizer()

# 量化模型
quantized_model = quantizer.quantize_model(model)

# 检查量化效果
for name, param in quantized_model.named_parameters():
    if 'weight' in name:
        print(f"{name}: {param.dtype}, 范围: [{param.min()}, {param.max()}]")

# 反量化模型
dequantized_model = quantizer.dequantize_model(quantized_model)
print("\n反量化后:")
for name, param in dequantized_model.named_parameters():
    if 'weight' in name:
        print(f"{name}: {param.dtype}")
```

### 练习2：实现量化推理引擎

```python
class QuantizedInferenceEngine:
    def __init__(self, model, scale, zero_point):
        self.model = model
        self.scale = scale
        self.zero_point = zero_point
    
    def quantized_matmul(self, input_float, weight_int8, scale_w, zp_w):
        """
        量化矩阵乘法
        
        参数:
            input_float: 浮点输入
            weight_int8: 量化后的权重
            scale_w: 权重缩放因子
            zp_w: 权重零点
        
        返回:
            浮点输出
        """
        # 量化输入
        input_min = input_float.min().item()
        input_max = input_float.max().item()
        scale_x = (input_max - input_min) / 255.0
        zp_x = round(-input_min / scale_x)
        
        input_int8 = torch.round(input_float / scale_x + zp_x)
        input_int8 = torch.clamp(input_int8, 0, 255).to(torch.int8)
        
        # 整数矩阵乘法
        output_int32 = torch.matmul(input_int8.to(torch.int32), weight_int8.to(torch.int32).t())
        
        # 反量化
        output_float = (output_int32 - zp_x * weight_int8.size(0)) * scale_x * scale_w
        
        return output_float
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入
        
        返回:
            输出
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 获取量化参数
                scale_w = self.scale[name + '.weight']
                zp_w = self.zero_point[name + '.weight']
                
                # 量化矩阵乘法
                x = self.quantized_matmul(x, module.weight.data, scale_w, zp_w)
                
                if module.bias is not None:
                    x += module.bias.data
                
                # ReLU
                if isinstance(module, nn.Sequential) or 'relu' in name.lower():
                    x = torch.relu(x)
        
        return x

# 测试
model = SimpleModel()

# 量化模型
quantizer = INT8Quantizer()
quantized_model = quantizer.quantize_model(model)

# 创建推理引擎
engine = QuantizedInferenceEngine(
    quantized_model,
    quantizer.scale,
    quantizer.zero_point
)

# 推理
inputs = torch.randn(32, 100)
outputs = engine.forward(inputs)
print(f"推理输出形状: {outputs.shape}")
```

### 练习3：实现量化评估工具

```python
class QuantizationEvaluator:
    def __init__(self):
        pass
    
    def evaluate(self, original_model, quantized_model, test_data):
        """
        评估量化效果
        
        参数:
            original_model: 原始模型
            quantized_model: 量化后的模型
            test_data: 测试数据
        
        返回:
            评估报告
        """
        report = {}
        
        # 准确率评估
        original_acc = self._evaluate_accuracy(original_model, test_data)
        quantized_acc = self._evaluate_accuracy(quantized_model, test_data)
        
        report['original_accuracy'] = original_acc
        report['quantized_accuracy'] = quantized_acc
        report['accuracy_drop'] = original_acc - quantized_acc
        
        # 模型大小
        original_size = self._get_model_size(original_model)
        quantized_size = self._get_model_size(quantized_model)
        
        report['original_size_mb'] = original_size
        report['quantized_size_mb'] = quantized_size
        report['compression_ratio'] = original_size / quantized_size
        
        # 推理速度
        original_time = self._measure_inference_time(original_model, test_data)
        quantized_time = self._measure_inference_time(quantized_model, test_data)
        
        report['original_time_ms'] = original_time * 1000
        report['quantized_time_ms'] = quantized_time * 1000
        report['speedup_ratio'] = original_time / quantized_time
        
        return report
    
    def _evaluate_accuracy(self, model, test_data):
        """评估准确率"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_data:
                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return correct / total
    
    def _get_model_size(self, model):
        """获取模型大小（MB）"""
        import sys
        total_bytes = 0
        for param in model.parameters():
            total_bytes += sys.getsizeof(param.storage())
        return total_bytes / (1024 * 1024)
    
    def _measure_inference_time(self, model, test_data):
        """测量推理时间（秒）"""
        import time
        
        model.eval()
        start = time.time()
        
        with torch.no_grad():
            for inputs, _ in test_data:
                model(inputs)
        
        return time.time() - start

# 测试
evaluator = QuantizationEvaluator()

# 模拟测试数据
test_data = [(torch.randn(32, 100), torch.randint(0, 10, (32,))) for _ in range(50)]

# 原始模型
original_model = SimpleModel()

# 量化模型
quantizer = INT8Quantizer()
quantized_model = quantizer.quantize_model(original_model)

# 评估
report = evaluator.evaluate(original_model, quantized_model, test_data)

print("量化评估报告:")
print(f"原始准确率: {report['original_accuracy']:.2%}")
print(f"量化准确率: {report['quantized_accuracy']:.2%}")
print(f"准确率下降: {report['accuracy_drop']:.2%}")
print(f"\n原始大小: {report['original_size_mb']:.2f} MB")
print(f"量化大小: {report['quantized_size_mb']:.2f} MB")
print(f"压缩比: {report['compression_ratio']:.2f}x")
print(f"\n原始推理时间: {report['original_time_ms']:.2f} ms")
print(f"量化推理时间: {report['quantized_time_ms']:.2f} ms")
print(f"加速比: {report['speedup_ratio']:.2f}x")
```

---

**下一节**：[推理优化](03-inference-optimization.md)

---

## 参考文献

1. Jacob, B., et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.
2. Frantar, E., et al. (2022). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.
3. Lin, J., et al. (2023). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.