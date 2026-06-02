# 1.5 上下文学习

## 目录

- [1. 引言](#1-引言)
- [2. 上下文学习的定义与原理](#2-上下文学习的定义与原理)
- [3. 上下文学习的类型](#3-上下文学习的类型)
- [4. 上下文学习的机制](#4-上下文学习的机制)
- [5. 影响上下文学习效果的因素](#5-影响上下文学习效果的因素)
- [6. 上下文学习的应用场景](#6-上下文学习的应用场景)
- [7. 上下文学习的挑战与局限性](#7-上下文学习的挑战与局限性)
- [8. 高级上下文学习技术](#8-高级上下文学习技术)
- [9. 实践练习](#9-实践练习)

---

## 1. 引言

### 1.1 什么是上下文学习

**上下文学习**（In-Context Learning, ICL）是大型语言模型的一种核心能力，指模型通过在prompt中提供少量示例，无需参数更新即可学会执行新任务。

**核心特点**：
- **无需微调**：不需要更新模型参数
- **快速适应**：可以快速学习新任务
- **依赖prompt**：性能高度依赖prompt设计

### 1.2 上下文学习的重要性

| 方面 | 说明 |
|------|------|
| **降低部署成本** | 无需为每个任务训练专用模型 |
| **快速原型开发** | 可以快速测试新想法 |
| **少样本场景** | 在标注数据稀缺时特别有用 |
| **模型复用** | 单个模型可以处理多种任务 |

---

## 2. 上下文学习的定义与原理

### 2.1 正式定义

上下文学习是指语言模型在推理时，通过在输入序列中包含任务示例来学习执行新任务的能力。

**形式化表达**：
```
输入: [示例1: x₁ → y₁, 示例2: x₂ → y₂, ..., 示例k: x_k → y_k, 新问题: x_new]
输出: y_new
```

### 2.2 工作原理

**直观理解**：
- 模型在训练期间学习了大量的语言模式和任务模式
- 当看到示例时，模型识别出任务模式
- 然后将该模式应用于新问题

**类比**：
就像人类通过例子学习一样，如果你看到几个数学题及其解答，你就能学会解决类似的问题，而不需要重新学习整个数学体系。

### 2.3 与传统机器学习的对比

| 维度 | 传统机器学习 | 上下文学习 |
|------|-------------|-----------|
| **训练方式** | 需要大量标注数据训练 | 只需少量示例 |
| **参数更新** | 需要更新模型参数 | 无需参数更新 |
| **任务适应** | 需要重新训练 | 即时适应 |
| **数据效率** | 低（需要大量数据） | 高（少量示例即可） |
| **部署成本** | 高（每个任务一个模型） | 低（一个模型处理多任务） |

---

## 3. 上下文学习的类型

### 3.1 Zero-shot Learning（零样本学习）

**定义**：不提供任何示例，直接让模型完成任务。

**示例**：
```
翻译中文到英文：你好
```

**适用场景**：
- 模型已经在训练数据中学过的任务
- 简单的常识性任务
- 语言理解任务

**优缺点**：
| 优点 | 缺点 |
|------|------|
| 无需示例，使用方便 | 性能可能不如有示例的情况 |
| 适用于广泛的任务 | 对prompt设计要求高 |

### 3.2 One-shot Learning（单样本学习）

**定义**：提供一个示例，让模型完成类似任务。

**示例**：
```
示例：苹果 → apple
翻译中文到英文：香蕉
```

**适用场景**：
- 任务相对简单但需要明确格式
- 需要指定输出格式时
- 模型对任务不太熟悉时

### 3.3 Few-shot Learning（少样本学习）

**定义**：提供少量示例（通常2-10个），让模型学习任务模式。

**示例**：
```
示例1：猫 → 动物
示例2：狗 → 动物
示例3：桌子 → 家具
示例4：椅子 → 家具
问题：汽车 → ?
```

**适用场景**：
- 复杂任务需要多个示例来展示模式
- 需要覆盖不同情况时
- 模型需要理解任务的多样性时

**示例数量的影响**：
| 示例数量 | 效果 | 说明 |
|---------|------|------|
| 1-3 | 基本理解任务 | 适合简单任务 |
| 4-6 | 较好的泛化 | 适合大多数任务 |
| 7-10 | 进一步提升 | 适合复杂任务 |
| >10 | 边际收益递减 | 受上下文窗口限制 |

### 3.4 Chain-of-Thought Learning（思维链学习）

**定义**：提供包含推理过程的示例，引导模型进行多步推理。

**示例**：
```
问题：小明有5个苹果，给了小红2个，又买了3个，现在有几个？
答案：5 - 2 + 3 = 6个

问题：篮子里有10个鸡蛋，拿走3个，又放进去5个，现在有几个？
```

**关键特点**：
- 示例不仅包含输入输出，还包含推理过程
- 特别适合需要多步推理的任务
- 显著提升复杂推理任务的性能

### 3.5 Self-Consistency（自一致性）

**定义**：生成多个推理路径，选择最一致的答案。

**示例**：
```
问题：计算 23 × 17

推理路径1：20×17=340，3×17=51，340+51=391
推理路径2：23×10=230，23×7=161，230+161=391
推理路径3：23×(20-3)=460-69=391

答案：391（三个路径一致）
```

**适用场景**：
- 需要精确推理的数学问题
- 逻辑推理任务
- 需要验证答案正确性的场景

---

## 4. 上下文学习的机制

### 4.1 理论解释

目前关于上下文学习的机制有几种理论：

| 理论 | 描述 | 论文支持 |
|------|------|---------|
| **模式匹配** | 模型在示例中识别模式，并应用到新问题 | Brown et al., 2020 |
| **隐式微调** | 示例作为"软提示"，隐式调整模型行为 | Wei et al., 2023 |
| **任务识别** | 模型识别任务类型，并调用相应的内部"子程序" | Min et al., 2022 |
| **记忆检索** | 示例帮助模型检索相关的训练数据 | Xie et al., 2022 |

**论文核心思想**：

1. **Brown et al. (2020)** 在《Language Models are Few-Shot Learners》中首次系统性地研究了LLM的few-shot学习能力。该论文提出了以下关键观点：

   - **In-Context Learning (ICL)**：模型可以通过上下文中的示例学习新任务，无需参数更新
   - **Meta-Learning**：模型在预训练过程中学习了如何从示例中学习
   - **Task Distribution**：模型在多样化的任务分布上预训练，获得了泛化能力

2. **Min et al. (2022)** 在《Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?》中研究了ICL的工作机制。论文发现：

   - **Label Space**：示例的标签空间比示例本身更重要
   - **Input-Label Mapping**：模型学习的是输入到标签的映射关系
   - **Format**：示例的格式对性能有重要影响

3. **Wei et al. (2023)** 在《Larger Language Models Do In-Context Learning Differently》中研究了模型规模对ICL的影响。论文发现：

   - **Scale Matters**：大模型的ICL行为与小模型不同
   - **Emergent ICL**：某些ICL能力只在足够大的模型上出现
   - **Robustness**：大模型对示例选择更鲁棒

4. **Xie et al. (2022)** 在《An Explanation of In-Context Learning as Implicit Bayesian Inference》中提出了贝叶斯推断的解释。论文认为：

   - **Bayesian Inference**：ICL可以看作是贝叶斯推断
   - **Posterior Distribution**：示例帮助模型更新对任务的后验分布
   - **Prior Knowledge**：模型利用预训练中学到的先验知识

**问题如何提出**：
传统机器学习需要大量标注数据和模型训练。研究者们思考：能否让模型通过几个示例就学会新任务，无需训练？

**如何解决**：
论文通过实验发现，大模型确实具备few-shot学习能力。模型通过预训练获得了元学习能力，可以从少量示例中快速学习新任务。

**优缺点**：
- **优点**：无需训练，快速适应新任务，数据效率高
- **缺点**：性能依赖示例选择，受上下文窗口限制

**未解决的问题**：
1. ICL的内在机制是什么？
2. 如何选择最优的示例？
3. 能否在较小规模的模型上实现ICL？
4. ICL与传统学习的关系是什么？

**未来方向**：
- 研究ICL的内在机制
- 开发自动示例选择方法
- 研究更高效的ICL方法
- 探索ICL与传统学习的结合

### 4.2 工作流程

```
输入prompt → [解析示例] → [识别任务模式] → [检索相关知识] → [生成输出]
```

**详细步骤**：

1. **解析示例**：
   - 模型读取prompt中的示例
   - 提取输入输出对
   - 识别任务格式

2. **识别任务模式**：
   - 分析示例之间的关系
   - 识别任务类型（分类、翻译、问答等）
   - 学习输入到输出的映射

3. **检索相关知识**：
   - 从预训练知识中检索相关信息
   - 激活相关的内部表示
   - 整合多个知识源

4. **生成输出**：
   - 应用学习到的模式
   - 生成符合格式的输出
   - 确保与示例一致

### 4.3 关键因素

| 因素 | 说明 | 影响 |
|------|------|------|
| **示例质量** | 示例必须准确、清晰 | 高质量示例显著提升性能 |
| **示例数量** | 足够但不过多 | 过少导致学习不足，过多导致混乱 |
| **示例多样性** | 覆盖不同情况 | 提高泛化能力 |
| **格式一致性** | 输入输出格式一致 | 减少模型困惑 |
| **示例顺序** | 示例的排列顺序 | 影响模型的学习效率 |

**代码示例：上下文学习机制分析**

```python
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter

@dataclass
class Example:
    """示例"""
    input: str
    output: str
    features: Dict[str, float] = None

class ICLMechanismAnalyzer:
    """上下文学习机制分析器"""
    
    def __init__(self):
        self.examples = []
        self.patterns = {}
    
    def add_examples(self, examples: List[Example]):
        """
        添加示例
        
        参数:
            examples: 示例列表
        """
        self.examples.extend(examples)
    
    def analyze_pattern(self) -> Dict:
        """
        分析任务模式
        
        返回:
            模式分析结果
        """
        if not self.examples:
            return {}
        
        # 分析输入长度分布
        input_lengths = [len(ex.input) for ex in self.examples]
        
        # 分析输出长度分布
        output_lengths = [len(ex.output) for ex in self.examples]
        
        # 分析输入输出关系
        io_ratios = [len(ex.output) / len(ex.input) if len(ex.input) > 0 else 0 
                     for ex in self.examples]
        
        # 分析输出类型
        output_types = self._classify_outputs()
        
        return {
            'num_examples': len(self.examples),
            'input_length_stats': {
                'mean': np.mean(input_lengths),
                'std': np.std(input_lengths),
                'min': np.min(input_lengths),
                'max': np.max(input_lengths)
            },
            'output_length_stats': {
                'mean': np.mean(output_lengths),
                'std': np.std(output_lengths),
                'min': np.min(output_lengths),
                'max': np.max(output_lengths)
            },
            'io_ratio_stats': {
                'mean': np.mean(io_ratios),
                'std': np.std(io_ratios)
            },
            'output_types': output_types,
            'task_type': self._infer_task_type(output_types)
        }
    
    def _classify_outputs(self) -> Dict:
        """
        分类输出类型
        
        返回:
            输出类型统计
        """
        output_types = {
            'short': 0,  # 短文本（<10字符）
            'medium': 0,  # 中等文本（10-50字符）
            'long': 0,  # 长文本（>50字符）
            'numeric': 0,  # 数字
            'categorical': 0  # 分类标签
        }
        
        for ex in self.examples:
            output = ex.output.strip()
            
            # 检查是否为数字
            if output.replace('.', '').replace('-', '').isdigit():
                output_types['numeric'] += 1
            # 检查长度
            elif len(output) < 10:
                output_types['short'] += 1
            elif len(output) < 50:
                output_types['medium'] += 1
            else:
                output_types['long'] += 1
            
            # 检查是否为分类标签
            if len(output) <= 10 and not output.replace('.', '').replace('-', '').isdigit():
                output_types['categorical'] += 1
        
        return output_types
    
    def _infer_task_type(self, output_types: Dict) -> str:
        """
        推断任务类型
        
        参数:
            output_types: 输出类型统计
        
        返回:
            任务类型
        """
        if output_types['categorical'] > len(self.examples) * 0.8:
            return 'classification'
        elif output_types['numeric'] > len(self.examples) * 0.8:
            return 'regression'
        elif output_types['long'] > len(self.examples) * 0.5:
            return 'generation'
        else:
            return 'unknown'
    
    def compute_pattern_similarity(self, input1: str, input2: str) -> float:
        """
        计算输入相似度
        
        参数:
            input1: 输入1
            input2: 输入2
        
        返回:
            相似度
        """
        # 简单实现：基于词重叠的相似度
        words1 = set(input1.split())
        words2 = set(input2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def analyze_example_diversity(self) -> Dict:
        """
        分析示例多样性
        
        返回:
            多样性分析结果
        """
        if len(self.examples) < 2:
            return {'message': '需要至少2个示例来分析多样性'}
        
        # 计算所有示例对的相似度
        similarities = []
        for i in range(len(self.examples)):
            for j in range(i + 1, len(self.examples)):
                sim = self.compute_pattern_similarity(
                    self.examples[i].input,
                    self.examples[j].input
                )
                similarities.append(sim)
        
        # 统计
        return {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'diversity_score': 1 - np.mean(similarities)  # 相似度越低，多样性越高
        }
    
    def recommend_examples(self, query: str, num_examples: int = 3) -> List[Example]:
        """
        推荐示例
        
        参数:
            query: 查询
            num_examples: 示例数量
        
        返回:
            推荐的示例
        """
        # 计算查询与每个示例的相似度
        similarities = []
        for ex in self.examples:
            sim = self.compute_pattern_similarity(query, ex.input)
            similarities.append((ex, sim))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 选择最相似的示例
        recommended = [item[0] for item in similarities[:num_examples]]
        
        return recommended

# 示例使用
analyzer = ICLMechanismAnalyzer()

# 添加示例
examples = [
    Example(input="猫", output="Cat"),
    Example(input="狗", output="Dog"),
    Example(input="鸟", output="Bird"),
    Example(input="鱼", output="Fish"),
    Example(input="马", output="Horse")
]
analyzer.add_examples(examples)

# 分析任务模式
pattern = analyzer.analyze_pattern()
print("任务模式分析:")
print(f"  示例数量: {pattern['num_examples']}")
print(f"  输入长度统计: {pattern['input_length_stats']}")
print(f"  输出长度统计: {pattern['output_length_stats']}")
print(f"  输出类型: {pattern['output_types']}")
print(f"  任务类型: {pattern['task_type']}")

# 分析示例多样性
diversity = analyzer.analyze_example_diversity()
print("\n示例多样性分析:")
print(f"  平均相似度: {diversity['mean_similarity']:.2f}")
print(f"  多样性评分: {diversity['diversity_score']:.2f}")

# 推荐示例
query = "兔子"
recommended = analyzer.recommend_examples(query, num_examples=3)
print(f"\n推荐的示例（查询: {query}）:")
for i, ex in enumerate(recommended, 1):
    print(f"  {i}. {ex.input} -> {ex.output}")
```

### 4.4 Meta-Learning视角

**Meta-Learning（元学习）**：
- 模型在预训练过程中学习了如何从示例中学习
- 预训练数据包含多样化的任务
- 模型学会了快速适应新任务

**代码示例：Meta-Learning模型**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple

class MetaLearningModel(nn.Module):
    """元学习模型"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        初始化元学习模型
        
        参数:
            input_dim: 输入维度
            hidden_dim: 隐藏维度
            output_dim: 输出维度
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入
        
        返回:
            输出
        """
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output
    
    def meta_train(self, tasks: List[Dict], num_epochs: int = 100):
        """
        元训练
        
        参数:
            tasks: 任务列表
            num_epochs: 训练轮数
        """
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for task in tasks:
                # 支持集
                support_x = task['support_x']
                support_y = task['support_y']
                
                # 查询集
                query_x = task['query_x']
                query_y = task['query_y']
                
                # 在支持集上快速适应
                adapted_params = self._adapt_to_task(support_x, support_y)
                
                # 在查询集上评估
                query_output = self._forward_with_params(query_x, adapted_params)
                loss = nn.functional.mse_loss(query_output, query_y)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / len(tasks):.4f}")
    
    def _adapt_to_task(self, support_x: torch.Tensor, support_y: torch.Tensor) -> Dict:
        """
        适应到特定任务
        
        参数:
            support_x: 支持集输入
            support_y: 支持集输出
        
        返回:
            适应后的参数
        """
        # 简单实现：计算梯度并更新
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        
        for _ in range(5):  # 快速适应
            output = self.forward(support_x)
            loss = nn.functional.mse_loss(output, support_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 返回当前参数
        return {name: param.clone() for name, param in self.named_parameters()}
    
    def _forward_with_params(self, x: torch.Tensor, params: Dict) -> torch.Tensor:
        """
        使用特定参数前向传播
        
        参数:
            x: 输入
            params: 参数
        
        返回:
            输出
        """
        # 简单实现：使用当前参数
        return self.forward(x)
    
    def few_shot_inference(self, support_x: torch.Tensor, support_y: torch.Tensor,
                          query_x: torch.Tensor) -> torch.Tensor:
        """
        Few-shot推理
        
        参数:
            support_x: 支持集输入
            support_y: 支持集输出
            query_x: 查询输入
        
        返回:
            查询输出
        """
        # 适应到任务
        self._adapt_to_task(support_x, support_y)
        
        # 推理
        output = self.forward(query_x)
        
        return output

# 示例使用
print("元学习模型示例:")
model = MetaLearningModel(input_dim=10, hidden_dim=32, output_dim=5)

# 创建任务
tasks = []
for _ in range(10):
    # 每个任务有不同的输入输出映射
    task = {
        'support_x': torch.randn(5, 10),
        'support_y': torch.randn(5, 5),
        'query_x': torch.randn(3, 10),
        'query_y': torch.randn(3, 5)
    }
    tasks.append(task)

# 元训练
print("开始元训练...")
model.meta_train(tasks, num_epochs=50)

# Few-shot推理
print("\nFew-shot推理:")
support_x = torch.randn(3, 10)
support_y = torch.randn(3, 5)
query_x = torch.randn(2, 10)

output = model.few_shot_inference(support_x, support_y, query_x)
print(f"  查询输出形状: {output.shape}")
```

### 4.5 Bayesian Learning视角

**Bayesian Learning（贝叶斯学习）**：
- ICL可以看作是贝叶斯推断
- 模型根据示例更新对任务的理解
- 输出是对任务的后验推断

**代码示例：Bayesian ICL模型**

```python
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import beta

class BayesianICLModel:
    """贝叶斯ICL模型"""
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        """
        初始化贝叶斯ICL模型
        
        参数:
            prior_alpha: 先验Alpha参数
            prior_beta: 先验Beta参数
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.posteriors = {}
    
    def update_posterior(self, task_name: str, examples: List[Tuple[str, str]]):
        """
        更新后验分布
        
        参数:
            task_name: 任务名称
            examples: 示例列表
        """
        # 简单实现：使用Beta分布建模
        # 统计正确和错误的数量
        correct = 0
        total = len(examples)
        
        for input_text, output_text in examples:
            # 这里简化处理，实际应用中需要更复杂的逻辑
            if output_text == "正确":
                correct += 1
        
        # 更新后验
        posterior_alpha = self.prior_alpha + correct
        posterior_beta = self.prior_beta + (total - correct)
        
        self.posteriors[task_name] = {
            'alpha': posterior_alpha,
            'beta': posterior_beta,
            'correct': correct,
            'total': total
        }
    
    def predict(self, task_name: str, input_text: str) -> Tuple[str, float]:
        """
        预测
        
        参数:
            task_name: 任务名称
            input_text: 输入
        
        返回:
            (预测输出, 置信度)
        """
        if task_name not in self.posteriors:
            # 使用先验
            alpha = self.prior_alpha
            beta = self.prior_beta
        else:
            posterior = self.posteriors[task_name]
            alpha = posterior['alpha']
            beta = posterior['beta']
        
        # 计算期望
        expected_value = alpha / (alpha + beta)
        
        # 简单实现：基于期望值预测
        if expected_value > 0.5:
            prediction = "正确"
            confidence = expected_value
        else:
            prediction = "错误"
            confidence = 1 - expected_value
        
        return prediction, confidence
    
    def sample_from_posterior(self, task_name: str, num_samples: int = 1000) -> np.ndarray:
        """
        从后验分布采样
        
        参数:
            task_name: 任务名称
            num_samples: 采样数量
        
        返回:
            采样结果
        """
        if task_name not in self.posteriors:
            alpha = self.prior_alpha
            beta = self.prior_beta
        else:
            posterior = self.posteriors[task_name]
            alpha = posterior['alpha']
            beta = posterior['beta']
        
        samples = beta.rvs(alpha, beta, size=num_samples)
        
        return samples
    
    def compute_uncertainty(self, task_name: str) -> float:
        """
        计算不确定性
        
        参数:
            task_name: 任务名称
        
        返回:
            不确定性
        """
        samples = self.sample_from_posterior(task_name)
        uncertainty = np.std(samples)
        
        return uncertainty

# 示例使用
print("贝叶斯ICL模型示例:")
bayesian_model = BayesianICLModel(prior_alpha=1.0, prior_beta=1.0)

# 更新后验
examples = [
    ("输入1", "正确"),
    ("输入2", "正确"),
    ("输入3", "错误"),
    ("输入4", "正确")
]
bayesian_model.update_posterior("任务1", examples)

# 预测
prediction, confidence = bayesian_model.predict("任务1", "新输入")
print(f"  预测: {prediction}")
print(f"  置信度: {confidence:.2f}")

# 计算不确定性
uncertainty = bayesian_model.compute_uncertainty("任务1")
print(f"  不确定性: {uncertainty:.2f}")

# 采样
samples = bayesian_model.sample_from_posterior("任务1", num_samples=1000)
print(f"  采样均值: {np.mean(samples):.2f}")
print(f"  采样标准差: {np.std(samples):.2f}")
```

---

## 5. 影响上下文学习效果的因素

### 5.1 示例选择

**示例的重要性**：
- 示例是模型学习的"教材"
- 示例的质量直接影响模型的表现

**选择示例的原则**：
| 原则 | 说明 |
|------|------|
| **代表性** | 示例应代表任务的典型情况 |
| **多样性** | 覆盖不同的输入输出模式 |
| **正确性** | 示例必须完全正确 |
| **简洁性** | 示例应简洁明了 |

### 5.2 示例顺序

**研究发现**：
- 示例的顺序会影响模型的表现
- 通常将简单示例放在前面
- 相似的示例放在一起可能有帮助

**示例顺序策略**：
1. **由易到难**：先简单后复杂
2. **多样性优先**：先展示不同类型的示例
3. **对比展示**：将相似但不同的示例放在一起

### 5.3 输入输出格式

**格式一致性**：
- 所有示例的格式必须一致
- 新问题的格式应与示例相同

**格式设计建议**：
| 任务类型 | 推荐格式 |
|---------|---------|
| 分类 | "输入: ... 分类: ..." |
| 问答 | "问题: ... 答案: ..." |
| 翻译 | "原文: ... 翻译: ..." |
| 摘要 | "文本: ... 摘要: ..." |

### 5.4 Prompt工程

**Prompt设计原则**：
1. **清晰的指令**：明确告诉模型要做什么
2. **合适的示例**：提供高质量的示例
3. **适当的格式**：使用清晰的格式分隔
4. **语言自然**：使用自然语言指令

**示例Prompt**：
```
请将以下中文句子翻译成英文：

示例1：
中文：今天天气很好
英文：The weather is nice today

示例2：
中文：我喜欢吃苹果
英文：I like to eat apples

现在翻译：
中文：明天我们去公园吧
英文：
```

---

## 6. 上下文学习的应用场景

### 6.1 自然语言处理

| 任务 | 应用示例 |
|------|---------|
| **文本分类** | 情感分析、主题分类 |
| **问答系统** | 常识问答、领域问答 |
| **机器翻译** | 多语言翻译 |
| **文本摘要** | 文档摘要生成 |
| **命名实体识别** | 识别实体类型 |

### 6.2 代码生成

| 任务 | 应用示例 |
|------|---------|
| **代码生成** | 从自然语言生成代码 |
| **代码解释** | 解释代码功能 |
| **代码调试** | 发现并修复代码错误 |
| **代码翻译** | 在不同语言间转换代码 |

### 6.3 数据分析

| 任务 | 应用示例 |
|------|---------|
| **数据清洗** | 格式化、标准化数据 |
| **数据提取** | 从文本中提取信息 |
| **报告生成** | 自动生成数据分析报告 |

### 6.4 教育领域

| 任务 | 应用示例 |
|------|---------|
| **辅导学习** | 提供个性化学习指导 |
| **作业批改** | 自动批改作业 |
| **知识问答** | 解答学生问题 |

---

## 7. 上下文学习的挑战与局限性

### 7.1 主要挑战

| 挑战 | 描述 |
|------|------|
| **Prompt敏感性** | 微小的prompt变化可能导致结果差异很大 |
| **示例选择困难** | 选择合适的示例需要专业知识 |
| **上下文窗口限制** | 示例数量受模型上下文长度限制 |
| **任务边界** | 无法处理与训练数据差异太大的任务 |
| **性能不稳定** | 相同prompt可能产生不同结果 |

### 7.2 常见问题

**问题1：模型忽略示例**
- 可能原因：示例不够清晰或数量不足
- 解决方案：增加示例数量，改进示例质量

**问题2：输出格式不正确**
- 可能原因：示例格式不一致或不清晰
- 解决方案：统一示例格式，明确输出格式要求

**问题3：产生幻觉**
- 可能原因：模型生成了不正确的内容
- 解决方案：增加事实核查步骤，使用检索增强

### 7.3 局限性

| 局限性 | 说明 |
|--------|------|
| **无法学习全新概念** | 只能学习训练数据中已有的知识 |
| **对复杂任务能力有限** | 复杂推理任务仍有困难 |
| **缺乏解释性** | 无法解释为什么做出某个预测 |

---

## 8. 高级上下文学习技术

### 8.1 检索增强生成（RAG）

**核心思想**：结合外部知识库，在生成前先检索相关信息。

**流程**：
```
用户问题 → [检索知识库] → [获取相关文档] → [生成答案]
```

**优势**：
- 减少幻觉
- 提供最新信息
- 可解释性更强

### 8.2 自我反思（Self-Reflection）

**核心思想**：模型先生成初步答案，然后反思并改进。

**流程**：
```
问题 → [生成初步答案] → [检查错误] → [修正答案]
```

**示例**：
```
问题：计算 100 - 23 × 3

初步答案：100 - 23 = 77，77 × 3 = 231

反思：运算顺序错误，应该先算乘法再算减法

修正答案：23 × 3 = 69，100 - 69 = 31
```

### 8.3 工具使用

**核心思想**：模型可以调用外部工具来完成任务。

**工具类型**：
| 工具类型 | 示例 |
|---------|------|
| **计算器** | 数学计算 |
| **搜索引擎** | 获取最新信息 |
| **数据库** | 查询数据 |
| **代码执行器** | 运行代码 |

---

## 9. 实践练习

### 练习1：设计有效的Few-shot Prompt

```python
def create_few_shot_prompt(task_description, examples, test_input):
    """
    创建有效的Few-shot Prompt
    
    参数:
        task_description: 任务描述
        examples: 示例列表，每个示例包含input和output
        test_input: 测试输入
    
    返回:
        完整的prompt
    """
    prompt = f"{task_description}\n\n"
    
    for i, example in enumerate(examples, 1):
        prompt += f"示例{i}：\n"
        prompt += f"输入：{example['input']}\n"
        prompt += f"输出：{example['output']}\n\n"
    
    prompt += f"现在处理：\n"
    prompt += f"输入：{test_input}\n"
    prompt += f"输出："
    
    return prompt

# 测试：创建一个情感分析的prompt
task_desc = "请判断以下句子的情感倾向，输出'正面'、'负面'或'中性'。"
examples = [
    {"input": "这部电影非常精彩，我非常喜欢！", "output": "正面"},
    {"input": "今天天气不好，心情很差。", "output": "负面"},
    {"input": "会议将于下午3点举行。", "output": "中性"}
]
test_input = "这家餐厅的服务很棒，食物也很美味！"

prompt = create_few_shot_prompt(task_desc, examples, test_input)
print("生成的Prompt：")
print(prompt)
```

### 练习2：实现Chain-of-Thought推理

```python
def create_cot_prompt(question, examples):
    """
    创建Chain-of-Thought Prompt
    
    参数:
        question: 问题
        examples: 包含推理过程的示例
    
    返回:
        完整的prompt
    """
    prompt = "请按照以下示例的格式回答问题，包括思考过程：\n\n"
    
    for i, example in enumerate(examples, 1):
        prompt += f"示例{i}：\n"
        prompt += f"问题：{example['question']}\n"
        prompt += f"思考：{example['thought']}\n"
        prompt += f"答案：{example['answer']}\n\n"
    
    prompt += f"现在回答：\n"
    prompt += f"问题：{question}\n"
    prompt += f"思考："
    
    return prompt

# 测试：创建一个数学题的CoT prompt
examples = [
    {
        "question": "小明有5个苹果，给了小红2个，又买了3个，现在有几个？",
        "thought": "1. 小明原来有5个苹果\n2. 给了小红2个，剩下5-2=3个\n3. 又买了3个，现在有3+3=6个",
        "answer": "6个"
    }
]
question = "篮子里有10个鸡蛋，拿走3个，又放进去5个，现在有几个？"

prompt = create_cot_prompt(question, examples)
print("生成的Chain-of-Thought Prompt：")
print(prompt)
```

### 练习3：评估上下文学习效果

```python
def evaluate_icl_performance(predictions, ground_truth):
    """
    评估上下文学习的性能
    
    参数:
        predictions: 模型预测结果列表
        ground_truth: 真实标签列表
    
    返回:
        评估指标
    """
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p.strip() == g.strip())
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    
    return {
        "total_samples": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "feedback": f"准确率：{accuracy*100:.2f}%，共{total}个样本，正确{correct}个"
    }

# 测试
predictions = ["正面", "负面", "中性", "正面"]
ground_truth = ["正面", "负面", "正面", "正面"]

result = evaluate_icl_performance(predictions, ground_truth)
print("上下文学习评估结果：")
print(f"  总样本数：{result['total_samples']}")
print(f"  正确预测：{result['correct_predictions']}")
print(f"  准确率：{result['accuracy']*100:.2f}%")
print(f"  反馈：{result['feedback']}")
```

---

**返回**：[预训练策略](02-pretraining.md)

---

## 参考文献

1. Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners.
2. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Chi, E. H., Le, Q. V., & Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models.
3. Liu, A. W., Tam, D., Mu, S., Liu, Y., & Chung, H. W. (2022). Understanding chain-of-thought prompting: An empirical study of what matters.
