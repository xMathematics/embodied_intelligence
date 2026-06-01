# 8.3 人类反馈强化学习（RLHF）

## 目录

- [1. 引言](#1-引言)
- [2. RLHF概述](#2-rlhf概述)
- [3. RLHF方法](#3-rlhf方法)
- [4. 实施步骤](#4-实施步骤)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 RLHF的重要性

**人类反馈强化学习（RLHF）**是一种将人类偏好纳入训练过程的方法，通过人类反馈来指导模型学习，实现更好的对齐效果。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **对话系统** | 使对话更符合人类偏好 | 更礼貌、更有帮助 |
| **内容生成** | 生成更优质的内容 | 更有趣、更准确 |
| **决策系统** | 做出更符合人类价值观的决策 | 更公平、更安全 |
| **创意写作** | 生成更符合人类审美的作品 | 更好的故事、诗歌 |

---

## 2. RLHF概述

### 2.1 定义

**RLHF**：通过人类反馈来训练强化学习的奖励模型，然后使用该奖励模型优化策略的方法。

**形式化表达**：
```
RLHF(Model, HumanFeedback) → AlignedModel
```

### 2.2 RLHF流程

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. 收集人类反馈  │ --> │  2. 训练奖励模型  │ --> │  3. 强化学习优化  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 3. RLHF方法

### 3.1 收集人类反馈

```python
class HumanFeedbackCollector:
    def __init__(self):
        self.feedback_data = []
    
    def collect_comparative_feedback(self, prompt, responses):
        """
        收集比较性反馈
        
        参数:
            prompt: 提示
            responses: 响应列表
        
        返回:
            排名结果
        """
        print(f"提示: {prompt}")
        print("请为以下响应排名（1=最佳）:")
        
        for i, response in enumerate(responses):
            print(f"{i+1}. {response}")
        
        # 获取排名
        ranks = []
        for i in range(len(responses)):
            rank = int(input(f"响应 {i+1} 的排名: "))
            ranks.append((response, rank))
        
        # 按排名排序
        ranks.sort(key=lambda x: x[1])
        
        self.feedback_data.append({
            'prompt': prompt,
            'responses': responses,
            'ranking': [r[0] for r in ranks]
        })
        
        return ranks
    
    def collect_rating_feedback(self, prompt, response):
        """
        收集评分反馈
        
        参数:
            prompt: 提示
            response: 响应
        
        返回:
            评分（1-5）
        """
        print(f"提示: {prompt}")
        print(f"响应: {response}")
        rating = int(input("请评分（1-5）: "))
        
        self.feedback_data.append({
            'prompt': prompt,
            'response': response,
            'rating': rating
        })
        
        return rating
    
    def save_feedback(self, filename):
        """保存反馈数据"""
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)

# 测试
collector = HumanFeedbackCollector()
prompt = "解释什么是人工智能"
responses = [
    "人工智能是研究智能的科学",
    "AI就是让计算机像人一样思考",
    "人工智能是模拟人类智能的技术"
]

ranking = collector.collect_comparative_feedback(prompt, responses)
print(f"排名结果: {ranking}")
```

### 3.2 训练奖励模型

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
    
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
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 使用最后一层的[CLS] token表示
        cls_token = outputs.hidden_states[-1][:, 0, :]
        reward = self.reward_head(cls_token)
        
        return reward
    
    def predict_reward(self, text, tokenizer):
        """
        预测奖励
        
        参数:
            text: 文本
            tokenizer: 分词器
        
        返回:
            奖励分数
        """
        encoding = tokenizer(text, return_tensors='pt')
        
        with torch.no_grad():
            reward = self.forward(encoding['input_ids'], encoding['attention_mask'])
        
        return reward.item()

# 训练奖励模型
class RewardModelTrainer:
    def __init__(self, reward_model, tokenizer, device='cuda'):
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-5)
        self.criterion = nn.MSELoss()
    
    def train_on_pair(self, prompt, better_response, worse_response):
        """
        在配对数据上训练
        
        参数:
            prompt: 提示
            better_response: 更好的响应
            worse_response: 更差的响应
        
        返回:
            损失
        """
        # 构建完整文本
        better_text = f"{prompt}\n{better_response}"
        worse_text = f"{prompt}\n{worse_response}"
        
        # 编码
        better_encoding = self.tokenizer(better_text, return_tensors='pt').to(self.device)
        worse_encoding = self.tokenizer(worse_text, return_tensors='pt').to(self.device)
        
        # 预测奖励
        better_reward = self.reward_model(
            better_encoding['input_ids'],
            better_encoding['attention_mask']
        )
        worse_reward = self.reward_model(
            worse_encoding['input_ids'],
            worse_encoding['attention_mask']
        )
        
        # 计算损失：让更好响应的奖励更高
        target = torch.tensor([1.0]).to(self.device)
        loss = self.criterion(better_reward - worse_reward, target)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# 测试
class MockBaseModelConfig:
    hidden_size = 768

class MockBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = MockBaseModelConfig()
    
    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        hidden_states = [torch.randn(batch_size, seq_len, 768) for _ in range(13)]
        
        return type('obj', (object,), {
            'hidden_states': hidden_states if output_hidden_states else None
        })

reward_model = RewardModel(MockBaseModel())
trainer = RewardModelTrainer(reward_model, MockTokenizer())

loss = trainer.train_on_pair(
    "什么是AI",
    "人工智能是研究智能的科学",
    "AI就是电脑"
)
print(f"训练损失: {loss:.4f}")
```

### 3.3 强化学习优化

```python
class RLHFTrainer:
    def __init__(self, policy_model, reward_model, tokenizer, device='cuda'):
        self.policy_model = policy_model.to(device)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.Adam(policy_model.parameters(), lr=3e-5)
    
    def compute_policy_gradient(self, prompt, response):
        """
        计算策略梯度
        
        参数:
            prompt: 提示
            response: 响应
        
        返回:
            损失
        """
        # 构建完整文本
        full_text = f"{prompt}\n{response}"
        encoding = self.tokenizer(full_text, return_tensors='pt').to(self.device)
        
        # 获取响应部分的token
        prompt_len = len(self.tokenizer.encode(prompt))
        
        # 计算奖励
        reward = self.reward_model(
            encoding['input_ids'],
            encoding['attention_mask']
        )
        
        # 计算对数概率
        outputs = self.policy_model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask'],
            labels=encoding['input_ids']
        )
        
        # 只考虑响应部分的对数概率
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        response_log_probs = log_probs[0, prompt_len-1:-1, :].gather(
            dim=-1,
            index=encoding['input_ids'][0, prompt_len:].unsqueeze(-1)
        )
        
        # 策略梯度：最大化奖励 * 对数概率
        loss = -(reward * response_log_probs.mean())
        
        return loss, reward.item()
    
    def train_step(self, prompts, num_episodes=10):
        """
        训练步骤
        
        参数:
            prompts: 提示列表
            num_episodes: 训练轮数
        
        返回:
            平均奖励
        """
        total_reward = 0
        
        for episode in range(num_episodes):
            for prompt in prompts:
                # 生成响应
                prompt_encoding = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                response_ids = self.policy_model.generate(
                    **prompt_encoding,
                    max_length=100
                )
                response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
                
                # 计算梯度
                loss, reward = self.compute_policy_gradient(prompt, response)
                
                # 更新策略
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_reward += reward
            
            avg_reward = total_reward / ((episode + 1) * len(prompts))
            print(f"Episode {episode+1}, Avg Reward: {avg_reward:.4f}")
        
        return avg_reward

# 测试
class MockPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(768, 1000)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = torch.randn(input_ids.size(0), input_ids.size(1), 1000)
        loss = torch.tensor(0.5) if labels is not None else None
        return type('obj', (object,), {'logits': logits, 'loss': loss})
    
    def generate(self, **kwargs):
        return torch.randint(0, 1000, (1, 50))

policy_model = MockPolicyModel()
rlhf_trainer = RLHFTrainer(policy_model, reward_model, MockTokenizer())

prompts = ["什么是AI", "写一首诗"]
avg_reward = rlhf_trainer.train_step(prompts, num_episodes=2)
print(f"最终平均奖励: {avg_reward:.4f}")
```

---

## 4. 实施步骤

### 4.1 阶段一：监督预训练

```python
class SupervisedPretrainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    def train(self, instruction_response_pairs, epochs=3):
        """
        监督预训练
        
        参数:
            instruction_response_pairs: 指令-响应对列表
            epochs: 训练轮数
        """
        for epoch in range(epochs):
            total_loss = 0
            
            for instruction, response in instruction_response_pairs:
                # 构建输入
                input_text = f"指令: {instruction}\n响应: {response}"
                encoding = self.tokenizer(input_text, return_tensors='pt')
                
                # 前向传播
                outputs = self.model(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask'],
                    labels=encoding['input_ids']
                )
                
                loss = outputs.loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(instruction_response_pairs)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

### 4.2 阶段二：奖励模型训练

```python
class RewardModelTrainer:
    def __init__(self, reward_model, tokenizer):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-5)
    
    def train(self, comparison_data, epochs=5):
        """
        训练奖励模型
        
        参数:
            comparison_data: 比较数据列表
            epochs: 训练轮数
        """
        for epoch in range(epochs):
            total_loss = 0
            
            for prompt, better_response, worse_response in comparison_data:
                # 编码
                better_encoding = self.tokenizer(
                    f"{prompt}\n{better_response}", 
                    return_tensors='pt'
                )
                worse_encoding = self.tokenizer(
                    f"{prompt}\n{worse_response}",
                    return_tensors='pt'
                )
                
                # 预测奖励
                better_reward = self.reward_model(**better_encoding)
                worse_reward = self.reward_model(**worse_encoding)
                
                # 损失：让更好响应的奖励更高
                loss = -torch.log(torch.sigmoid(better_reward - worse_reward))
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(comparison_data)
            print(f"Reward Model Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

### 4.3 阶段三：强化学习微调

```python
class PPOTrainer:
    def __init__(self, policy_model, reward_model, tokenizer):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.Adam(policy_model.parameters(), lr=3e-5)
    
    def train(self, prompts, iterations=100):
        """
        PPO训练
        
        参数:
            prompts: 提示列表
            iterations: 迭代次数
        """
        for iteration in range(iterations):
            total_reward = 0
            
            for prompt in prompts:
                # 生成响应
                prompt_encoding = self.tokenizer(prompt, return_tensors='pt')
                response_ids = self.policy_model.generate(**prompt_encoding)
                response = self.tokenizer.decode(response_ids[0])
                
                # 计算奖励
                full_encoding = self.tokenizer(f"{prompt}\n{response}", return_tensors='pt')
                reward = self.reward_model(**full_encoding).item()
                
                # PPO更新（简化版）
                loss = -reward  # 最大化奖励
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_reward += reward
            
            avg_reward = total_reward / len(prompts)
            print(f"PPO Iteration {iteration+1}, Avg Reward: {avg_reward:.4f}")
```

---

## 5. 实践练习

### 练习1：实现反馈收集系统

```python
class AdvancedFeedbackCollector:
    def __init__(self):
        self.feedback_data = []
    
    def collect_comparative(self, prompt, responses):
        """
        收集比较性反馈
        
        参数:
            prompt: 提示
            responses: 响应列表
        
        返回:
            排序后的响应
        """
        print(f"\n提示: {prompt}")
        print("-" * 50)
        
        # 随机打乱顺序
        import random
        shuffled = list(enumerate(responses))
        random.shuffle(shuffled)
        
        for i, (orig_idx, response) in enumerate(shuffled):
            print(f"{i+1}. {response}")
        
        # 获取排名
        while True:
            try:
                ranking = input("请输入排名（如：1 3 2）: ").strip()
                ranks = list(map(int, ranking.split()))
                
                if len(ranks) != len(responses):
                    print(f"请输入{len(responses)}个数字")
                    continue
                
                if set(ranks) != set(range(1, len(responses)+1)):
                    print("请输入1到{}的排列".format(len(responses)))
                    continue
                
                break
            except ValueError:
                print("请输入有效数字")
        
        # 重建原始顺序的排名
        original_ranking = [0] * len(responses)
        for new_rank, (orig_idx, _) in enumerate(shuffled):
            original_ranking[orig_idx] = ranks[new_rank]
        
        # 按排名排序
        ranked_responses = sorted(
            [(r, original_ranking[i]) for i, r in enumerate(responses)],
            key=lambda x: x[1]
        )
        
        self.feedback_data.append({
            'prompt': prompt,
            'responses': responses,
            'ranking': [r[0] for r in ranked_responses]
        })
        
        return [r[0] for r in ranked_responses]
    
    def collect_scalar(self, prompt, response):
        """
        收集标量反馈
        
        参数:
            prompt: 提示
            response: 响应
        
        返回:
            评分
        """
        print(f"\n提示: {prompt}")
        print(f"响应: {response}")
        
        while True:
            try:
                rating = int(input("评分（1-5）: "))
                if 1 <= rating <= 5:
                    break
                print("请输入1-5之间的数字")
            except ValueError:
                print("请输入有效数字")
        
        self.feedback_data.append({
            'prompt': prompt,
            'response': response,
            'rating': rating
        })
        
        return rating
    
    def analyze_feedback(self):
        """分析反馈数据"""
        if not self.feedback_data:
            return {'error': '没有反馈数据'}
        
        total_ratings = sum(1 for d in self.feedback_data if 'rating' in d)
        avg_rating = sum(d.get('rating', 0) for d in self.feedback_data) / max(total_ratings, 1)
        
        return {
            'total_samples': len(self.feedback_data),
            'avg_rating': avg_rating,
            'comparative_count': sum(1 for d in self.feedback_data if 'ranking' in d),
            'scalar_count': total_ratings
        }

# 测试
collector = AdvancedFeedbackCollector()

# 收集比较性反馈
responses = [
    "人工智能是研究智能的科学",
    "AI就是让计算机像人一样思考",
    "人工智能是模拟人类智能的技术"
]
collector.collect_comparative("什么是AI", responses)

# 收集标量反馈
collector.collect_scalar("写一首诗", "春眠不觉晓，处处闻啼鸟。")

# 分析反馈
analysis = collector.analyze_feedback()
print(f"\n反馈分析: {analysis}")
```

### 练习2：实现奖励模型训练

```python
class RewardModelTrainingPipeline:
    def __init__(self, base_model_name='gpt2'):
        self.base_model = None
        self.reward_model = None
        self.tokenizer = None
    
    def load_models(self):
        """加载模型"""
        # 简化：使用mock模型
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('obj', (object,), {'hidden_size': 768})
                self.embed = nn.Embedding(1000, 768)
            
            def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
                embeddings = self.embed(input_ids)
                return type('obj', (object,), {
                    'hidden_states': [embeddings] if output_hidden_states else None
                })
        
        self.base_model = MockModel()
        self.tokenizer = MockTokenizer()
        
        # 创建奖励模型
        self.reward_model = RewardModel(self.base_model)
    
    def prepare_training_data(self, feedback_data):
        """准备训练数据"""
        training_data = []
        
        for item in feedback_data:
            if 'ranking' in item:
                # 比较性数据
                prompt = item['prompt']
                responses = item['ranking']
                
                # 创建配对
                for i in range(len(responses)):
                    for j in range(i+1, len(responses)):
                        training_data.append({
                            'prompt': prompt,
                            'better': responses[i],
                            'worse': responses[j]
                        })
        
        return training_data
    
    def train(self, feedback_data, epochs=5):
        """训练奖励模型"""
        self.load_models()
        
        training_data = self.prepare_training_data(feedback_data)
        
        optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=1e-5)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for item in training_data:
                # 编码
                better_text = f"{item['prompt']}\n{item['better']}"
                worse_text = f"{item['prompt']}\n{item['worse']}"
                
                better_encoding = self.tokenizer(better_text, return_tensors='pt')
                worse_encoding = self.tokenizer(worse_text, return_tensors='pt')
                
                # 预测奖励
                better_reward = self.reward_model(
                    better_encoding['input_ids'],
                    better_encoding['attention_mask']
                )
                worse_reward = self.reward_model(
                    worse_encoding['input_ids'],
                    worse_encoding['attention_mask']
                )
                
                # 损失
                loss = -torch.log(torch.sigmoid(better_reward - worse_reward))
                
                # 更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        return self.reward_model

# 测试
pipeline = RewardModelTrainingPipeline()

feedback_data = [
    {
        'prompt': '什么是AI',
        'responses': ['AI是人工智能', 'AI是计算机科学'],
        'ranking': ['AI是人工智能', 'AI是计算机科学']
    }
]

reward_model = pipeline.train(feedback_data, epochs=2)
print("奖励模型训练完成")
```

### 练习3：实现完整RLHF流程

```python
class RLHFWorkflow:
    def __init__(self):
        self.collector = AdvancedFeedbackCollector()
        self.reward_model = None
        self.policy_model = None
    
    def step1_collect_feedback(self, prompts, responses_per_prompt=3):
        """步骤1：收集人类反馈"""
        print("步骤1：收集人类反馈")
        
        for prompt in prompts:
            # 生成多个响应（简化）
            responses = [
                f"响应A: {prompt}",
                f"响应B: {prompt}",
                f"响应C: {prompt}"
            ][:responses_per_prompt]
            
            self.collector.collect_comparative(prompt, responses)
        
        print(f"已收集 {len(self.collector.feedback_data)} 条反馈")
    
    def step2_train_reward_model(self):
        """步骤2：训练奖励模型"""
        print("\n步骤2：训练奖励模型")
        
        pipeline = RewardModelTrainingPipeline()
        self.reward_model = pipeline.train(
            self.collector.feedback_data,
            epochs=3
        )
        print("奖励模型训练完成")
    
    def step3_rl_fine_tune(self, prompts, iterations=10):
        """步骤3：强化学习微调"""
        print("\n步骤3：强化学习微调")
        
        # 创建策略模型
        class MockPolicy(nn.Module):
            def __init__(self):
                super().__init__()
            
            def generate(self, **kwargs):
                return torch.randint(0, 1000, (1, 50))
        
        self.policy_model = MockPolicy()
        
        # 简化的RL训练
        for i in range(iterations):
            avg_reward = 0.5 + i * 0.05  # 模拟奖励递增
            print(f"Iteration {i+1}, Avg Reward: {avg_reward:.4f}")
        
        print("强化学习微调完成")
    
    def run(self, prompts):
        """运行完整流程"""
        self.step1_collect_feedback(prompts)
        self.step2_train_reward_model()
        self.step3_rl_fine_tune(prompts)
        
        print("\nRLHF流程完成！")

# 测试
workflow = RLHFWorkflow()
workflow.run(["什么是人工智能", "写一首关于春天的诗"])
```

---

**下一节**：[价值对齐](04-value-alignment.md)

---

## 参考文献

1. Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback.
2. Stiennon, N., et al. (2020). Learning to Summarize with Human Feedback.
3. Christiano, P., et al. (2017). Deep Reinforcement Learning from Human Preferences.