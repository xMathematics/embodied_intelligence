# 9.1 模型压缩

## 目录

- [1. 引言](#1-引言)
- [2. 模型压缩概述](#2-模型压缩概述)
- [3. 压缩方法](#3-压缩方法)
- [4. 压缩评估](#4-压缩评估)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 模型压缩的重要性

**模型压缩**是通过减少模型的大小和复杂度来提高推理效率的技术。这对于将大型模型部署到资源受限的设备上至关重要。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **边缘设备** | 在手机、嵌入式设备上运行 | 手机AI助手 |
| **实时推理** | 需要低延迟响应 | 自动驾驶 |
| **云端部署** | 减少内存和计算资源消耗 | 大规模API服务 |
| **模型分发** | 减少下载时间和带宽占用 | 移动端AI应用 |

---

## 2. 模型压缩概述

### 2.1 定义

**模型压缩**：在保持模型性能的前提下，减少模型的存储大小和计算复杂度。

**形式化表达**：
```
Compress(Model, TargetSize) → CompressedModel
```

### 2.2 压缩目标

| 目标 | 描述 | 衡量指标 |
|------|------|---------|
| **存储压缩** | 减少模型文件大小 | 压缩率、文件大小 |
| **计算加速** | 减少推理时间 | 延迟、吞吐量 |
| **内存优化** | 减少内存占用 | 峰值内存、显存使用 |
| **能耗降低** | 减少能源消耗 | 推理能耗 |

---

## 3. 压缩方法

### 3.1 模型剪枝

**定义**：移除模型中不重要的参数或结构。

```python
import torch
import torch.nn as nn

class PruningModule(nn.Module):
    def __init__(self, original_model, pruning_rate=0.5):
        super().__init__()
        self.original_model = original_model
        self.pruning_rate = pruning_rate
        self.masks = {}
    
    def prune(self):
        """执行剪枝"""
        for name, param in self.original_model.named_parameters():
            if 'weight' in name:
                # 计算参数重要性（使用绝对值）
                importance = torch.abs(param.data)
                
                # 确定剪枝阈值
                threshold = torch.quantile(importance, self.pruning_rate)
                
                # 创建掩码
                mask = importance > threshold
                self.masks[name] = mask
                
                # 应用剪枝
                param.data *= mask.float()
    
    def apply_masks(self):
        """应用剪枝掩码"""
        for name, param in self.original_model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name].float()
    
    def forward(self, x):
        """前向传播"""
        self.apply_masks()
        return self.original_model(x)

# 测试
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = SimpleModel()
pruner = PruningModule(model, pruning_rate=0.3)
pruner.prune()

# 检查剪枝效果
for name, param in model.named_parameters():
    if 'weight' in name:
        zeros = (param.data == 0).sum().item()
        total = param.data.numel()
        print(f"{name}: 稀疏率 = {zeros/total:.2%}")
```

### 3.2 知识蒸馏

**定义**：将大模型（教师）的知识转移到小模型（学生）。

```python
class KnowledgeDistillationTrainer:
    def __init__(self, teacher_model, student_model, temperature=2.0):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
        self.criterion = nn.KLDivLoss()
    
    def train_step(self, inputs):
        """
        蒸馏训练步骤
        
        参数:
            inputs: 输入数据
        
        返回:
            损失
        """
        # 教师模型推理（不更新）
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
            teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)
        
        # 学生模型推理
        student_logits = self.student(inputs)
        student_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        
        # 计算蒸馏损失
        distillation_loss = self.criterion(student_probs, teacher_probs)
        
        # 计算学生预测损失（可选）
        labels = torch.argmax(teacher_logits, dim=-1)
        student_loss = nn.CrossEntropyLoss()(student_logits, labels)
        
        # 总损失
        loss = distillation_loss + 0.1 * student_loss
        
        # 更新学生模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# 测试
teacher = SimpleModel()
student = nn.Sequential(
    nn.Linear(100, 30),
    nn.Linear(30, 10)
)

trainer = KnowledgeDistillationTrainer(teacher, student)
inputs = torch.randn(32, 100)

loss = trainer.train_step(inputs)
print(f"蒸馏损失: {loss:.4f}")
```

### 3.3 权重共享

**定义**：让多个参数共享相同的值。

```python
class WeightSharingModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        # 共享的权重
        self.shared_weight = nn.Parameter(torch.randn(input_dim, hidden_dim))
        
        # 输出层权重
        self.output_weight = nn.Parameter(torch.randn(hidden_dim, output_dim))
        
        self.num_layers = num_layers
    
    def forward(self, x):
        """
        前向传播（多层共享权重）
        
        参数:
            x: 输入 [batch, input_dim]
        
        返回:
            输出 [batch, output_dim]
        """
        for _ in range(self.num_layers):
            x = torch.matmul(x, self.shared_weight)
            x = torch.relu(x)
        
        x = torch.matmul(x, self.output_weight)
        return x

# 测试
shared_model = WeightSharingModule(100, 50, 10, num_layers=3)

# 统计参数数量
total_params = sum(p.numel() for p in shared_model.parameters())
print(f"总参数数量: {total_params}")

# 普通模型对比
normal_model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
normal_params = sum(p.numel() for p in normal_model.parameters())
print(f"普通模型参数数量: {normal_params}")
print(f"参数减少比例: {(1 - total_params/normal_params):.2%}")
```

---

## 4. 压缩评估

### 4.1 评估指标

| 指标 | 描述 | 计算方法 |
|------|------|---------|
| **压缩率** | 模型大小减少比例 | (原始大小 - 压缩后大小) / 原始大小 |
| **准确率保持率** | 压缩后准确率与原模型的比例 | 压缩后准确率 / 原准确率 |
| **推理加速比** | 推理速度提升比例 | 原推理时间 / 压缩后推理时间 |
| **内存节省率** | 内存使用减少比例 | (原内存 - 压缩后内存) / 原内存 |

### 4.2 评估框架

```python
class CompressionEvaluator:
    def __init__(self):
        pass
    
    def evaluate(self, original_model, compressed_model, test_data):
        """
        评估压缩效果
        
        参数:
            original_model: 原始模型
            compressed_model: 压缩后的模型
            test_data: 测试数据
        
        返回:
            评估报告
        """
        report = {}
        
        # 模型大小
        original_size = self._get_model_size(original_model)
        compressed_size = self._get_model_size(compressed_model)
        report['compression_rate'] = (original_size - compressed_size) / original_size
        
        # 准确率
        original_acc = self._evaluate_accuracy(original_model, test_data)
        compressed_acc = self._evaluate_accuracy(compressed_model, test_data)
        report['accuracy_retention'] = compressed_acc / original_acc
        report['original_accuracy'] = original_acc
        report['compressed_accuracy'] = compressed_acc
        
        # 推理速度
        original_time = self._measure_inference_time(original_model, test_data)
        compressed_time = self._measure_inference_time(compressed_model, test_data)
        report['speedup_ratio'] = original_time / compressed_time
        report['original_time'] = original_time
        report['compressed_time'] = compressed_time
        
        return report
    
    def _get_model_size(self, model):
        """获取模型大小（字节）"""
        import sys
        return sys.getsizeof(model.state_dict())
    
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
    
    def _measure_inference_time(self, model, test_data):
        """测量推理时间"""
        import time
        
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, _ in test_data:
                model(inputs)
        
        return time.time() - start_time

# 测试
evaluator = CompressionEvaluator()

# 模拟测试数据
test_data = [(torch.randn(32, 100), torch.randint(0, 10, (32,))) for _ in range(10)]

original_model = SimpleModel()
compressed_model = WeightSharingModule(100, 50, 10)

report = evaluator.evaluate(original_model, compressed_model, test_data)
print("压缩评估报告:")
print(f"压缩率: {report['compression_rate']:.2%}")
print(f"准确率保持率: {report['accuracy_retention']:.2%}")
print(f"推理加速比: {report['speedup_ratio']:.2f}x")
```

---

## 5. 实践练习

### 练习1：实现结构化剪枝

```python
class StructuredPruning:
    def __init__(self, model):
        self.model = model
    
    def prune_channels(self, layer_name, keep_ratio=0.5):
        """
        结构化剪枝（按通道）
        
        参数:
            layer_name: 层名称
            keep_ratio: 保留比例
        
        返回:
            剪枝后的模型
        """
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                # 获取权重
                weights = module.weight.data  # [out_channels, in_channels, kernel, kernel]
                
                # 计算通道重要性（L1范数）
                importance = torch.norm(weights, p=1, dim=[1, 2, 3])
                
                # 确定要保留的通道
                num_keep = int(len(importance) * keep_ratio)
                _, indices = torch.topk(importance, num_keep)
                
                # 创建新的卷积层
                new_conv = nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=num_keep,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding
                )
                
                # 复制保留的权重
                new_conv.weight.data = weights[indices]
                if module.bias is not None:
                    new_conv.bias.data = module.bias.data[indices]
                
                # 替换原层
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                
                if parent_name == '':
                    setattr(self.model, attr_name, new_conv)
                else:
                    parent = self.model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, attr_name, new_conv)
                
                break
        
        return self.model
    
    def prune_filters(self, layer_name, keep_ratio=0.5):
        """
        结构化剪枝（按滤波器）
        
        参数:
            layer_name: 层名称
            keep_ratio: 保留比例
        """
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Conv2d):
                weights = module.weight.data
                
                # 计算滤波器重要性
                importance = torch.norm(weights, p=1, dim=[0, 2, 3])
                
                num_keep = int(len(importance) * keep_ratio)
                _, indices = torch.topk(importance, num_keep)
                
                # 创建新层
                new_conv = nn.Conv2d(
                    in_channels=num_keep,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding
                )
                
                # 复制保留的权重
                new_conv.weight.data = weights[:, indices]
                if module.bias is not None:
                    new_conv.bias.data = module.bias.data
                
                # 更新后续层的输入通道数
                self._update_subsequent_layers(name, num_keep)
                
                # 替换原层
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                if parent_name == '':
                    setattr(self.model, attr_name, new_conv)
                else:
                    parent = self.model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, attr_name, new_conv)
                
                break
        
        return self.model
    
    def _update_subsequent_layers(self, pruned_layer_name, new_channels):
        """更新后续层的输入通道数"""
        # 简化实现：假设后续层是下一个Conv2d
        layers = list(self.model.named_modules())
        pruned_idx = None
        
        for i, (name, module) in enumerate(layers):
            if name == pruned_layer_name:
                pruned_idx = i
                break
        
        if pruned_idx is not None and pruned_idx + 1 < len(layers):
            next_name, next_module = layers[pruned_idx + 1]
            if isinstance(next_module, nn.Conv2d):
                new_next_conv = nn.Conv2d(
                    in_channels=new_channels,
                    out_channels=next_module.out_channels,
                    kernel_size=next_module.kernel_size,
                    stride=next_module.stride,
                    padding=next_module.padding
                )
                new_next_conv.weight.data = next_module.weight.data[:, :new_channels]
                if next_module.bias is not None:
                    new_next_conv.bias.data = next_module.bias.data
                
                parent_name = '.'.join(next_name.split('.')[:-1])
                attr_name = next_name.split('.')[-1]
                if parent_name == '':
                    setattr(self.model, attr_name, new_next_conv)
                else:
                    parent = self.model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, attr_name, new_next_conv)

# 测试
conv_model = nn.Sequential(
    nn.Conv2d(3, 16, 3),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3)
)

pruner = StructuredPruning(conv_model)
pruned_model = pruner.prune_channels('0', keep_ratio=0.5)

print("剪枝前:")
for name, module in conv_model.named_modules():
    if isinstance(module, nn.Conv2d):
        print(f"{name}: {module.out_channels} -> {module.in_channels}")

print("\n剪枝后:")
for name, module in pruned_model.named_modules():
    if isinstance(module, nn.Conv2d):
        print(f"{name}: {module.out_channels} -> {module.in_channels}")
```

### 练习2：实现量化感知训练

```python
class QuantizationAwareTraining:
    def __init__(self, model, bits=8):
        self.model = model
        self.bits = bits
        self.quantize_layers()
    
    def quantize_layers(self):
        """为模型添加量化节点"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 添加量化/反量化节点
                module.register_buffer('scale', torch.tensor(1.0))
                module.register_buffer('zero_point', torch.tensor(0))
    
    def quantize_weight(self, weight):
        """
        量化权重
        
        参数:
            weight: 原始权重
        
        返回:
            量化后的权重
        """
        # 计算缩放因子和零点
        min_val = weight.min().item()
        max_val = weight.max().item()
        
        scale = (max_val - min_val) / (2**self.bits - 1)
        zero_point = round(-min_val / scale)
        
        # 量化
        quantized = torch.round(weight / scale + zero_point)
        quantized = torch.clamp(quantized, 0, 2**self.bits - 1)
        
        return quantized, scale, zero_point
    
    def dequantize_weight(self, quantized_weight, scale, zero_point):
        """
        反量化权重
        
        参数:
            quantized_weight: 量化后的权重
            scale: 缩放因子
            zero_point: 零点
        
        返回:
            反量化后的权重
        """
        return (quantized_weight - zero_point) * scale
    
    def forward(self, x):
        """
        前向传播（模拟量化效果）
        
        参数:
            x: 输入
        
        返回:
            输出
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 量化权重
                quantized_weight, scale, zero_point = self.quantize_weight(module.weight.data)
                
                # 反量化（模拟推理时的计算）
                dequantized_weight = self.dequantize_weight(quantized_weight, scale, zero_point)
                
                # 使用量化后的权重进行计算
                original_weight = module.weight.data.clone()
                module.weight.data = dequantized_weight
                
                # 保存量化参数
                module.scale = torch.tensor(scale)
                module.zero_point = torch.tensor(zero_point)
        
        output = self.model(x)
        
        return output
    
    def train(self, dataloader, epochs=1):
        """
        量化感知训练
        
        参数:
            dataloader: 数据加载器
            epochs: 训练轮数
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for inputs, labels in dataloader:
                # 前向传播（带量化）
                outputs = self.forward(inputs)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 测试
model = SimpleModel()
qat_trainer = QuantizationAwareTraining(model, bits=8)

# 模拟数据
dataloader = [(torch.randn(32, 100), torch.randint(0, 10, (32,))) for _ in range(10)]
qat_trainer.train(dataloader, epochs=2)
```

### 练习3：实现模型压缩流水线

```python
class ModelCompressionPipeline:
    def __init__(self):
        self.steps = []
    
    def add_step(self, step_name, step_function):
        """
        添加压缩步骤
        
        参数:
            step_name: 步骤名称
            step_function: 步骤函数
        """
        self.steps.append({'name': step_name, 'function': step_function})
    
    def run(self, model, **kwargs):
        """
        运行压缩流水线
        
        参数:
            model: 原始模型
            kwargs: 额外参数
        
        返回:
            压缩后的模型, 压缩报告
        """
        current_model = model
        report = {
            'original_size': self._get_size(model),
            'steps': []
        }
        
        for step in self.steps:
            print(f"执行步骤: {step['name']}")
            
            # 记录步骤前的大小
            before_size = self._get_size(current_model)
            
            # 执行步骤
            current_model = step['function'](current_model, **kwargs)
            
            # 记录步骤后的大小
            after_size = self._get_size(current_model)
            
            # 更新报告
            report['steps'].append({
                'name': step['name'],
                'before_size': before_size,
                'after_size': after_size,
                'compression_rate': (before_size - after_size) / before_size
            })
        
        # 最终报告
        report['final_size'] = self._get_size(current_model)
        report['overall_compression_rate'] = (
            report['original_size'] - report['final_size']
        ) / report['original_size']
        
        return current_model, report
    
    def _get_size(self, model):
        """获取模型大小（MB）"""
        import sys
        state_dict = model.state_dict()
        total_bytes = sum(sys.getsizeof(v.storage()) for v in state_dict.values())
        return total_bytes / (1024 * 1024)  # MB

# 测试
pipeline = ModelCompressionPipeline()

# 添加压缩步骤
pipeline.add_step('剪枝', lambda model, **kwargs: PruningModule(model, 0.3).original_model)
pipeline.add_step('知识蒸馏', lambda model, **kwargs: train_distillation(model))
pipeline.add_step('权重共享', lambda model, **kwargs: WeightSharingModule(100, 50, 10))

def train_distillation(model):
    """简化的蒸馏函数"""
    return model

# 运行流水线
original_model = SimpleModel()
compressed_model, report = pipeline.run(original_model)

print("\n压缩报告:")
print(f"原始大小: {report['original_size']:.2f} MB")
print(f"最终大小: {report['final_size']:.2f} MB")
print(f"总压缩率: {report['overall_compression_rate']:.2%}")

print("\n各步骤详情:")
for step in report['steps']:
    print(f"{step['name']}: {step['before_size']:.2f} MB -> {step['after_size']:.2f} MB ({step['compression_rate']:.2%})")
```

---

**下一节**：[量化技术](02-quantization.md)

---

## 参考文献

1. Han, S., et al. (2015). Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.
2. Hinton, G., et al. (2015). Distilling the Knowledge in a Neural Network.
3. Li, H., et al. (2016). Pruning Filters for Efficient ConvNets.