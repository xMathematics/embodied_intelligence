# 8.2 指令微调

## 目录

- [1. 引言](#1-引言)
- [2. 指令微调概述](#2-指令微调概述)
- [3. 指令微调方法](#3-指令微调方法)
- [4. 数据集构建](#4-数据集构建)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 指令微调的重要性

**指令微调**是通过在指令-响应对数据上微调预训练模型，使其能够理解并遵循人类指令的过程。这是实现模型对齐的关键技术。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **对话系统** | 遵循用户指令进行对话 | "帮我写一封邮件" |
| **任务执行** | 执行特定任务 | "总结这篇文章" |
| **多任务学习** | 执行多种不同任务 | 问答、摘要、翻译 |
| **少样本学习** | 仅需少量示例 | 学习新任务 |

---

## 2. 指令微调概述

### 2.1 定义

**指令微调**：在包含指令-响应对的数据集上对预训练模型进行微调，使模型学会理解和执行指令。

**形式化表达**：
```
FineTune(Model, InstructionDataset) → InstructionFollower
```

### 2.2 指令微调的优势

| 优势 | 描述 |
|------|------|
| **通用性** | 一个模型可以处理多种任务 |
| **自然语言接口** | 用户可以用自然语言交互 |
| **少样本能力** | 只需少量示例即可学习新任务 |
| **对齐能力** | 更容易与人类意图对齐 |

---

## 3. 指令微调方法

### 3.1 监督指令微调

**定义**：使用人工标注的指令-响应对进行监督学习。

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class InstructionDataset(Dataset):
    def __init__(self, instructions, responses, tokenizer):
        self.instructions = instructions
        self.responses = responses
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.instructions)
    
    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        response = self.responses[idx]
        
        # 构建输入：指令 + 响应
        input_text = f"指令: {instruction}\n响应: {response}"
        encoding = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # 自回归目标
        }

# 示例数据
instructions = [
    "解释什么是人工智能",
    "写一首关于春天的诗",
    "总结文章内容"
]
responses = [
    "人工智能是研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。",
    "春风吹绿江南岸，细雨滋润万物生。",
    "本文介绍了机器学习的基本概念和应用。"
]

# 假设我们有一个tokenizer
class MockTokenizer:
    def __call__(self, text, **kwargs):
        return {
            'input_ids': torch.randint(0, 1000, (1, 512)),
            'attention_mask': torch.ones(1, 512)
        }

tokenizer = MockTokenizer()
dataset = InstructionDataset(instructions, responses, tokenizer)
dataloader = DataLoader(dataset, batch_size=2)

print(f"数据集大小: {len(dataset)}")
print(f"批次大小: {len(dataloader)}")
```

### 3.2 强化学习微调

**定义**：使用强化学习优化模型，奖励信号来自人类反馈或其他评估指标。

```python
class RLHFModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_model = None
    
    def set_reward_model(self, reward_model):
        """设置奖励模型"""
        self.reward_model = reward_model
    
    def forward(self, input_ids, attention_mask):
        """前向传播"""
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs
    
    def generate_with_reward(self, prompt, max_length=512):
        """
        生成并计算奖励
        
        参数:
            prompt: 提示
            max_length: 最大长度
        
        返回:
            生成的文本, 奖励
        """
        # 生成响应
        output = self.base_model.generate(
            input_ids=prompt,
            max_length=max_length,
            num_return_sequences=3  # 生成多个候选
        )
        
        # 计算奖励
        rewards = []
        for candidate in output:
            reward = self.reward_model(candidate)
            rewards.append(reward)
        
        # 选择奖励最高的
        best_idx = torch.argmax(torch.tensor(rewards))
        best_output = output[best_idx]
        
        return best_output, rewards[best_idx]

# 测试
class MockBaseModel:
    def generate(self, input_ids, **kwargs):
        return [torch.randint(0, 1000, (50,)) for _ in range(3)]

class MockRewardModel:
    def __call__(self, candidate):
        return torch.rand(1).item()

base_model = MockBaseModel()
reward_model = MockRewardModel()

rlhf_model = RLHFModel(base_model)
rlhf_model.set_reward_model(reward_model)

prompt = torch.randint(0, 1000, (1, 10))
output, reward = rlhf_model.generate_with_reward(prompt)
print(f"生成长度: {len(output)}")
print(f"奖励: {reward:.4f}")
```

### 3.3 混合方法

**定义**：结合监督学习和强化学习的方法。

```python
class HybridInstructionTuner:
    def __init__(self, model):
        self.model = model
        self.supervised_optimizer = None
        self.rl_optimizer = None
    
    def supervised_fine_tune(self, dataloader, epochs=3):
        """
        监督微调
        
        参数:
            dataloader: 数据加载器
            epochs: 训练轮数
        """
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for batch in dataloader:
                input_ids = batch['input_ids']
                labels = batch['labels']
                
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits
                
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                self.supervised_optimizer.zero_grad()
                loss.backward()
                self.supervised_optimizer.step()
            
            print(f"监督微调 Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    def rl_fine_tune(self, prompts, reward_fn, iterations=100):
        """
        强化学习微调
        
        参数:
            prompts: 提示列表
            reward_fn: 奖励函数
            iterations: 迭代次数
        """
        for iteration in range(iterations):
            total_reward = 0
            
            for prompt in prompts:
                # 生成响应
                response = self.model.generate(input_ids=prompt)
                
                # 计算奖励
                reward = reward_fn(response)
                total_reward += reward
                
                # 策略梯度更新（简化版）
                loss = -reward  # 最大化奖励
                self.rl_optimizer.zero_grad()
                loss.backward()
                self.rl_optimizer.step()
            
            avg_reward = total_reward / len(prompts)
            print(f"RL微调 Iteration {iteration+1}, Avg Reward: {avg_reward:.4f}")
```

---

## 4. 数据集构建

### 4.1 数据集类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **自然指令** | 自然语言指令 | "写一个故事" |
| **结构化指令** | 格式化指令 | {"task": "summarize", "input": "..."} |
| **多任务指令** | 包含多种任务 | 问答、摘要、翻译混合 |
| **对话指令** | 对话形式 | 用户问，模型答 |

### 4.2 数据收集方法

```python
class InstructionDataCollector:
    def __init__(self):
        self.data = []
    
    def collect_from_human(self, instructions, max_responses=3):
        """
        从人类收集响应
        
        参数:
            instructions: 指令列表
            max_responses: 每个指令的最大响应数
        """
        for instruction in instructions:
            print(f"指令: {instruction}")
            responses = []
            
            for i in range(max_responses):
                response = input(f"响应 {i+1}: ")
                responses.append(response)
            
            self.data.append({
                'instruction': instruction,
                'responses': responses
            })
    
    def augment_with_gpt(self, instructions, gpt_model):
        """
        使用GPT生成响应
        
        参数:
            instructions: 指令列表
            gpt_model: GPT模型
        """
        for instruction in instructions:
            response = gpt_model.generate(f"完成以下指令: {instruction}")
            self.data.append({
                'instruction': instruction,
                'responses': [response]
            })
    
    def save_to_json(self, filename):
        """保存到JSON文件"""
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

# 测试
collector = InstructionDataCollector()
collector.data = [
    {
        'instruction': '解释什么是机器学习',
        'responses': ['机器学习是人工智能的一个分支...']
    }
]
collector.save_to_json('instructions.json')
print("数据已保存")
```

### 4.3 数据质量评估

```python
class DataQualityEvaluator:
    def __init__(self):
        pass
    
    def evaluate(self, dataset):
        """
        评估数据集质量
        
        参数:
            dataset: 数据集
        
        返回:
            质量报告
        """
        report = {}
        
        # 基本统计
        report['total_samples'] = len(dataset)
        
        # 平均长度
        avg_instruction_len = sum(len(item['instruction']) for item in dataset) / len(dataset)
        avg_response_len = sum(sum(len(r) for r in item['responses']) / len(item['responses']) for item in dataset) / len(dataset)
        report['avg_instruction_length'] = avg_instruction_len
        report['avg_response_length'] = avg_response_len
        
        # 多样性评估
        unique_instructions = len(set(item['instruction'] for item in dataset))
        report['instruction_diversity'] = unique_instructions / len(dataset)
        
        # 质量评分（简化）
        quality_score = min(1, avg_response_len / 50) * 0.5 + report['instruction_diversity'] * 0.5
        report['quality_score'] = quality_score
        
        return report

# 测试
dataset = [
    {'instruction': '什么是AI', 'responses': ['AI是人工智能']},
    {'instruction': '写一首诗', 'responses': ['春眠不觉晓...']},
    {'instruction': '总结文章', 'responses': ['本文介绍了...']}
]

evaluator = DataQualityEvaluator()
report = evaluator.evaluate(dataset)
print(f"质量报告: {report}")
```

---

## 5. 实践练习

### 练习1：实现指令微调训练循环

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class InstructionTuningTrainer:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    def prepare_dataset(self, instructions, responses):
        """准备数据集"""
        dataset = []
        
        for inst, resp in zip(instructions, responses):
            # 构建输入格式
            input_text = f"### 指令:\n{inst}\n\n### 响应:\n{resp}"
            
            encoding = self.tokenizer(
                input_text,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 标签：只预测响应部分
            labels = encoding['input_ids'].clone()
            labels[0, :len(self.tokenizer.encode(f"### 指令:\n{inst}\n\n### 响应:\n"))] = -100
            
            dataset.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': labels.flatten()
            })
        
        return dataset
    
    def train(self, dataset, batch_size=4, epochs=3):
        """训练模型"""
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    def generate_response(self, instruction, max_length=200):
        """生成响应"""
        prompt = f"### 指令:\n{instruction}\n\n### 响应:\n"
        encoding = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **encoding,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response.replace(prompt, '')

# 测试（使用mock模型）
class MockLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Transformer(d_model=512, nhead=8)
        self.lm_head = nn.Linear(512, 1000)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids.unsqueeze(1), input_ids.unsqueeze(1))
        logits = self.lm_head(outputs[0].squeeze(1))
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, 1000), labels.view(-1))
        
        return type('obj', (object,), {'loss': loss, 'logits': logits})
    
    def generate(self, **kwargs):
        return torch.randint(0, 1000, (1, 100))

model = MockLMModel()
trainer = InstructionTuningTrainer(model, MockTokenizer())

# 准备数据
instructions = ["解释什么是人工智能", "写一首关于秋天的诗"]
responses = ["人工智能是...", "秋高气爽..."]
dataset = trainer.prepare_dataset(instructions, responses)

# 训练
trainer.train(dataset, batch_size=1, epochs=2)
```

### 练习2：实现奖励模型

```python
class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_dim=256):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播
        
        参数:
            input_ids: 输入ID
            attention_mask: 注意力掩码
        
        返回:
            奖励分数
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用CLS token的输出
        cls_output = outputs.last_hidden_state[:, 0, :]
        reward = self.reward_head(cls_output)
        
        return reward
    
    def rank_responses(self, prompt, responses):
        """
        对响应进行排序
        
        参数:
            prompt: 提示
            responses: 响应列表
        
        返回:
            排序后的响应和分数
        """
        scores = []
        
        for response in responses:
            full_text = f"{prompt}\n{response}"
            encoding = self.tokenizer(full_text, return_tensors='pt')
            
            with torch.no_grad():
                score = self.forward(encoding['input_ids'], encoding['attention_mask'])
            
            scores.append((response, score.item()))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores

# 测试
class MockBaseModelConfig:
    hidden_size = 768

class MockRewardBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockBaseModelConfig()
    
    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.size(0)
        return type('obj', (object,), {
            'last_hidden_state': torch.randn(batch_size, 10, 768)
        })

reward_model = RewardModel(MockRewardBaseModel())
reward_model.tokenizer = MockTokenizer()

prompt = "解释什么是机器学习"
responses = [
    "机器学习是AI的分支",
    "机器学习就是让计算机从数据中学习",
    "机器学习是一种数据分析方法"
]

ranked = reward_model.rank_responses(prompt, responses)
print("响应排名:")
for response, score in ranked:
    print(f"{score:.4f}: {response}")
```

### 练习3：实现PPO微调

```python
import torch
import torch.nn.functional as F

class PPOTrainer:
    def __init__(self, model, reward_model, clip_param=0.2, gamma=0.99):
        self.model = model
        self.reward_model = reward_model
        self.clip_param = clip_param
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    
    def compute_advantages(self, rewards, values):
        """
        计算优势函数
        
        参数:
            rewards: 奖励列表
            values: 值函数列表
        
        返回:
            优势值
        """
        advantages = []
        last_advantage = 0
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * (values[i+1] if i+1 < len(values) else 0) - values[i]
            last_advantage = delta + self.gamma * 0.95 * last_advantage
            advantages.insert(0, last_advantage)
        
        # 标准化
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def train_step(self, prompts, batch_size=4):
        """
        PPO训练步骤
        
        参数:
            prompts: 提示列表
            batch_size: 批次大小
        
        返回:
            损失
        """
        total_loss = 0
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            # 生成响应
            responses = []
            old_log_probs = []
            
            for prompt in batch_prompts:
                output = self.model.generate(
                    **self.tokenizer(prompt, return_tensors='pt'),
                    output_scores=True,
                    return_dict_in_generate=True
                )
                
                response = self.tokenizer.decode(output.sequences[0])
                responses.append(response)
                
                # 计算旧策略的对数概率
                scores = output.scores
                log_probs = sum([torch.log_softmax(s, dim=-1)[:, :, output.sequences[0][i+1]].mean() 
                                for i, s in enumerate(scores)])
                old_log_probs.append(log_probs)
            
            # 计算奖励
            rewards = []
            for prompt, response in zip(batch_prompts, responses):
                full_text = f"{prompt}\n{response}"
                reward = self.reward_model(
                    **self.tokenizer(full_text, return_tensors='pt')
                )
                rewards.append(reward.item())
            
            # 计算优势
            values = [0.5] * len(rewards)  # 简化：假设值函数为常数
            advantages = self.compute_advantages(rewards, values)
            
            # PPO更新
            for j, (prompt, response, old_log_prob, advantage) in enumerate(zip(
                    batch_prompts, responses, old_log_probs, advantages)):
                
                # 重新计算当前策略的对数概率
                full_text = f"{prompt}\n{response}"
                encoding = self.tokenizer(full_text, return_tensors='pt')
                outputs = self.model(**encoding)
                
                new_log_prob = torch.log_softmax(outputs.logits, dim=-1)
                new_log_prob = new_log_prob[0, :-1, :].gather(
                    dim=-1, 
                    index=encoding['input_ids'][0, 1:].unsqueeze(-1)
                ).mean()
                
                # PPO损失
                ratio = torch.exp(new_log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                
                loss = -torch.min(surr1, surr2).mean()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
        
        return total_loss / (len(prompts) // batch_size)

# 测试
ppo_trainer = PPOTrainer(MockLMModel(), reward_model)
ppo_trainer.tokenizer = MockTokenizer()

prompts = ["解释什么是AI", "写一首诗"]
loss = ppo_trainer.train_step(prompts)
print(f"PPO训练损失: {loss:.4f}")
```

---

**下一节**：[人类反馈强化学习](03-rlhf.md)

---

## 参考文献

1. Wei, J., et al. (2022). Chain of Thought Prompting Elicits Reasoning in Large Language Models.
2. Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback.
3. Stiennon, N., et al. (2020). Learning to Summarize with Human Feedback.