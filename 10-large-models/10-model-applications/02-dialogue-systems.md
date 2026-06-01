# 10.2 对话系统

## 目录

- [1. 引言](#1-引言)
- [2. 对话系统概述](#2-对话系统概述)
- [3. 核心技术](#3-核心技术)
- [4. 应用场景](#4-应用场景)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 对话系统的重要性

**对话系统**是能够与人类进行自然语言交互的AI系统。它是连接人类和AI的重要桥梁。

### 1.2 应用价值

| 维度 | 描述 |
|------|------|
| **智能客服** | 24/7自动化客户服务 |
| **虚拟助手** | 个人AI助手如Siri、Alexa |
| **教育辅导** | 智能学习伙伴 |
| **社交聊天** | 情感陪伴机器人 |

---

## 2. 对话系统概述

### 2.1 定义

**对话系统**：能够理解用户输入并生成自然语言响应的系统。

**形式化表达**：
```
response = f(context, user_input)
```

### 2.2 对话系统类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **任务型** | 完成特定任务 | 订餐、订票 |
| **闲聊型** | 开放式对话 | 聊天机器人 |
| **问答型** | 回答特定问题 | 知识库问答 |
| **混合型** | 多种类型混合 | 智能助手 |

---

## 3. 核心技术

### 3.1 对话管理

```python
class DialogueManager:
    def __init__(self):
        self.context = []
        self.max_context_length = 10
    
    def add_turn(self, user_message, assistant_response=None):
        """
        添加对话轮次
        
        参数:
            user_message: 用户消息
            assistant_response: 助手响应（可选）
        """
        turn = {'user': user_message}
        if assistant_response:
            turn['assistant'] = assistant_response
        
        self.context.append(turn)
        
        # 保持上下文长度限制
        if len(self.context) > self.max_context_length:
            self.context = self.context[-self.max_context_length:]
    
    def get_context(self):
        """获取完整上下文"""
        return self.context
    
    def get_context_prompt(self):
        """获取上下文提示格式"""
        prompt = ""
        for turn in self.context:
            prompt += f"User: {turn['user']}\n"
            if 'assistant' in turn:
                prompt += f"Assistant: {turn['assistant']}\n"
        
        return prompt
    
    def clear_context(self):
        """清空上下文"""
        self.context = []
    
    def get_context_length(self):
        """获取上下文长度"""
        return len(self.context)

# 测试
manager = DialogueManager()

manager.add_turn("Hello! How are you?", "I'm doing well, thank you!")
manager.add_turn("What can you do?", "I can help with various tasks like answering questions, writing, and more!")

print("对话上下文:")
print(manager.get_context_prompt())
```

### 3.2 意图识别

```python
class IntentRecognizer:
    def __init__(self):
        self.intents = {
            'greeting': ['hello', 'hi', 'good morning', 'good afternoon', 'good evening'],
            'farewell': ['bye', 'goodbye', 'see you', 'see you later'],
            'question': ['what', 'who', 'when', 'where', 'why', 'how'],
            'command': ['do', 'make', 'create', 'write', 'generate'],
            'complaint': ['problem', 'issue', 'error', 'wrong', 'broken']
        }
    
    def recognize(self, text):
        """
        识别意图
        
        参数:
            text: 输入文本
        
        返回:
            意图标签列表
        """
        text_lower = text.lower()
        matched_intents = []
        
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matched_intents.append(intent)
                    break
        
        # 如果没有匹配到，默认视为陈述
        if not matched_intents:
            matched_intents.append('statement')
        
        return matched_intents
    
    def get_intent_confidence(self, text):
        """
        获取意图置信度
        
        参数:
            text: 输入文本
        
        返回:
            意图和置信度字典
        """
        text_lower = text.lower()
        confidences = {}
        
        for intent, keywords in self.intents.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            confidence = count / len(keywords)
            if confidence > 0:
                confidences[intent] = confidence
        
        # 归一化
        total = sum(confidences.values())
        if total > 0:
            confidences = {k: v/total for k, v in confidences.items()}
        
        return confidences

# 测试
recognizer = IntentRecognizer()

text = "Hello! Can you write a story for me?"
intents = recognizer.recognize(text)
confidences = recognizer.get_intent_confidence(text)

print(f"输入: {text}")
print(f"识别的意图: {intents}")
print(f"置信度: {confidences}")
```

### 3.3 对话生成

```python
class DialogueGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_response(self, context, user_input, max_length=200):
        """
        生成对话响应
        
        参数:
            context: 对话上下文
            user_input: 用户输入
            max_length: 最大长度
        
        返回:
            响应文本
        """
        # 构建对话历史
        history = ""
        for turn in context:
            history += f"User: {turn['user']}\n"
            if 'assistant' in turn:
                history += f"Assistant: {turn['assistant']}\n"
        
        # 添加当前用户输入
        history += f"User: {user_input}\nAssistant:"
        
        # 生成响应
        inputs = self.tokenizer(history, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手响应部分
        response = response.replace(history, "").strip()
        
        return response

# 测试（需要加载预训练模型）
# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
# 
# generator = DialogueGenerator(model, tokenizer)
# 
# context = [{'user': 'Hello!', 'assistant': 'Hi there!'}]
# response = generator.generate_response(context, 'How are you doing?')
# print(f"响应: {response}")
```

---

## 4. 应用场景

### 4.1 智能客服

```python
class CustomerServiceBot:
    def __init__(self):
        self.dialogue_manager = DialogueManager()
        self.intent_recognizer = IntentRecognizer()
        
        self.knowledge_base = {
            'shipping': 'Standard shipping takes 3-5 business days. Express shipping is available for an additional fee.',
            'return': 'Returns are accepted within 30 days of purchase. Please contact support for a return label.',
            'refund': 'Refunds are processed within 5-7 business days after we receive the returned item.',
            'support': 'Our support team is available 24/7. You can reach us via email or live chat.'
        }
    
    def handle_query(self, user_input):
        """
        处理用户查询
        
        参数:
            user_input: 用户输入
        
        返回:
            响应文本
        """
        # 识别意图
        intents = self.intent_recognizer.recognize(user_input)
        
        # 检查知识库
        for intent in intents:
            if intent in self.knowledge_base:
                response = self.knowledge_base[intent]
                self.dialogue_manager.add_turn(user_input, response)
                return response
        
        # 如果没有匹配，生成通用响应
        response = "I'm here to help! Could you please provide more details about your question?"
        self.dialogue_manager.add_turn(user_input, response)
        return response

# 测试
bot = CustomerServiceBot()

queries = [
    "Hello, I need help with shipping",
    "How long does shipping take?",
    "Can I get a refund?"
]

for query in queries:
    response = bot.handle_query(query)
    print(f"User: {query}")
    print(f"Bot: {response}\n")
```

### 4.2 虚拟助手

```python
class VirtualAssistant:
    def __init__(self):
        self.dialogue_manager = DialogueManager()
        
        self.commands = {
            'time': self.get_time,
            'date': self.get_date,
            'weather': self.get_weather,
            'news': self.get_news
        }
    
    def get_time(self):
        """获取当前时间"""
        import datetime
        return f"The current time is {datetime.datetime.now().strftime('%H:%M')}"
    
    def get_date(self):
        """获取当前日期"""
        import datetime
        return f"Today is {datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    def get_weather(self, location=None):
        """获取天气（模拟）"""
        if location:
            return f"The weather in {location} is sunny with a high of 25°C."
        return "Please specify a location for weather information."
    
    def get_news(self):
        """获取新闻（模拟）"""
        return "Top news: AI technology continues to advance rapidly in 2024."
    
    def process_command(self, user_input):
        """
        处理命令
        
        参数:
            user_input: 用户输入
        
        返回:
            响应文本
        """
        user_lower = user_input.lower()
        
        # 检查命令
        for command, handler in self.commands.items():
            if command in user_lower:
                # 提取参数
                if command == 'weather':
                    # 尝试提取位置
                    location = None
                    if 'in' in user_lower:
                        parts = user_lower.split('in')
                        if len(parts) > 1:
                            location = parts[1].strip()
                    return handler(location)
                else:
                    return handler()
        
        # 如果没有匹配命令
        return None
    
    def respond(self, user_input):
        """
        生成响应
        
        参数:
            user_input: 用户输入
        
        返回:
            响应文本
        """
        # 先尝试命令处理
        command_response = self.process_command(user_input)
        if command_response:
            self.dialogue_manager.add_turn(user_input, command_response)
            return command_response
        
        # 否则生成对话响应
        response = "That's interesting! Tell me more about it."
        self.dialogue_manager.add_turn(user_input, response)
        return response

# 测试
assistant = VirtualAssistant()

queries = [
    "What time is it?",
    "What's the date today?",
    "What's the weather in New York?",
    "Tell me the news"
]

for query in queries:
    response = assistant.respond(query)
    print(f"User: {query}")
    print(f"Assistant: {response}\n")
```

### 4.3 教育辅导

```python
class TutorBot:
    def __init__(self):
        self.dialogue_manager = DialogueManager()
        self.subjects = ['math', 'science', 'history', 'english']
    
    def explain_topic(self, topic, subject=None):
        """
        解释主题
        
        参数:
            topic: 主题
            subject: 学科
        
        返回:
            解释内容
        """
        explanations = {
            'math': {
                'algebra': 'Algebra is a branch of mathematics dealing with symbols and the rules for manipulating those symbols.',
                'calculus': 'Calculus is the mathematical study of continuous change, similar to how geometry is the study of shape.'
            },
            'science': {
                'physics': 'Physics is the natural science that studies matter, its motion and behavior through space and time.',
                'chemistry': 'Chemistry is the scientific study of the properties and behavior of matter.'
            }
        }
        
        if subject and subject in explanations:
            if topic.lower() in explanations[subject]:
                return explanations[subject][topic.lower()]
        
        return f"I'd be happy to explain {topic}! Could you specify which subject this relates to?"
    
    def answer_question(self, question):
        """
        回答问题
        
        参数:
            question: 问题
        
        返回:
            答案
        """
        # 简化的问题回答
        question_lower = question.lower()
        
        if 'what is' in question_lower:
            topic = question_lower.replace('what is', '').strip()
            return self.explain_topic(topic)
        
        return "That's a great question! Let me help you understand this better."
    
    def respond(self, user_input):
        """
        生成响应
        
        参数:
            user_input: 用户输入
        
        返回:
            响应文本
        """
        response = self.answer_question(user_input)
        self.dialogue_manager.add_turn(user_input, response)
        return response

# 测试
tutor = TutorBot()

queries = [
    "What is algebra?",
    "What is physics?",
    "Can you explain calculus?"
]

for query in queries:
    response = tutor.respond(query)
    print(f"Student: {query}")
    print(f"Tutor: {response}\n")
```

---

## 5. 实践练习

### 练习1：实现对话管理器

```python
class AdvancedDialogueManager:
    def __init__(self, max_history=20):
        self.history = []
        self.max_history = max_history
        self.metadata = {}
    
    def add_message(self, role, content, timestamp=None):
        """
        添加消息
        
        参数:
            role: 角色（user/assistant/system）
            content: 内容
            timestamp: 时间戳
        """
        if timestamp is None:
            import datetime
            timestamp = datetime.datetime.now().isoformat()
        
        message = {
            'role': role,
            'content': content,
            'timestamp': timestamp
        }
        
        self.history.append(message)
        
        # 保持历史长度限制
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_history(self):
        """获取完整历史"""
        return self.history
    
    def get_formatted_history(self, format_type='chatml'):
        """
        获取格式化的历史
        
        参数:
            format_type: 格式类型（chatml/llama2）
        
        返回:
            格式化的字符串
        """
        if format_type == 'chatml':
            formatted = ""
            for msg in self.history:
                formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            return formatted
        
        elif format_type == 'llama2':
            formatted = ""
            for msg in self.history:
                if msg['role'] == 'user':
                    formatted += f"USER: {msg['content']}\n"
                elif msg['role'] == 'assistant':
                    formatted += f"ASSISTANT: {msg['content']}\n"
                elif msg['role'] == 'system':
                    formatted += f"SYSTEM: {msg['content']}\n"
            return formatted
        
        else:
            return "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.history)
    
    def set_metadata(self, key, value):
        """设置元数据"""
        self.metadata[key] = value
    
    def get_metadata(self, key):
        """获取元数据"""
        return self.metadata.get(key)
    
    def clear(self):
        """清空历史"""
        self.history = []
        self.metadata = {}

# 测试
manager = AdvancedDialogueManager()

manager.add_message('system', 'You are a helpful assistant.')
manager.add_message('user', 'Hello!')
manager.add_message('assistant', 'Hi there! How can I help you?')
manager.add_message('user', 'What is machine learning?')

print("ChatML格式:")
print(manager.get_formatted_history('chatml'))

print("\nLlama2格式:")
print(manager.get_formatted_history('llama2'))
```

### 练习2：实现多轮对话系统

```python
class MultiTurnDialogueSystem:
    def __init__(self, generator):
        self.dialogue_manager = AdvancedDialogueManager()
        self.generator = generator
    
    def start_conversation(self, system_prompt=None):
        """
        开始对话
        
        参数:
            system_prompt: 系统提示
        """
        if system_prompt:
            self.dialogue_manager.add_message('system', system_prompt)
        
        print("对话已开始。输入 'exit' 结束对话。")
    
    def process_message(self, user_input):
        """
        处理用户消息
        
        参数:
            user_input: 用户输入
        
        返回:
            响应文本
        """
        if user_input.lower() == 'exit':
            return "Goodbye! Have a great day!"
        
        # 添加用户消息到历史
        self.dialogue_manager.add_message('user', user_input)
        
        # 获取格式化的历史
        history = self.dialogue_manager.get_formatted_history('chatml')
        
        # 生成响应
        response = self.generator.generate(history, max_length=200)
        
        # 添加助手响应到历史
        self.dialogue_manager.add_message('assistant', response)
        
        return response
    
    def run(self):
        """运行对话循环"""
        while True:
            user_input = input("User: ")
            response = self.process_message(user_input)
            print(f"Assistant: {response}")
            
            if response == "Goodbye! Have a great day!":
                break

# 测试
# system = MultiTurnDialogueSystem(generator)
# system.start_conversation("You are a helpful AI assistant.")
# system.run()
```

### 练习3：实现对话状态追踪

```python
class DialogueStateTracker:
    def __init__(self):
        self.state = {
            'intent': None,
            'entities': {},
            'slot_filling': {},
            'history': []
        }
    
    def update_state(self, user_input, intent, entities):
        """
        更新对话状态
        
        参数:
            user_input: 用户输入
            intent: 识别的意图
            entities: 提取的实体
        """
        # 更新意图
        self.state['intent'] = intent
        
        # 更新实体
        self.state['entities'].update(entities)
        
        # 更新历史
        self.state['history'].append({
            'user_input': user_input,
            'intent': intent,
            'entities': entities
        })
        
        # 检查槽位填充
        self._check_slot_filling()
    
    def _check_slot_filling(self):
        """检查槽位填充状态"""
        required_slots = ['destination', 'date', 'passengers']
        
        for slot in required_slots:
            if slot in self.state['entities']:
                self.state['slot_filling'][slot] = 'filled'
            else:
                self.state['slot_filling'][slot] = 'empty'
    
    def get_missing_slots(self):
        """获取缺失的槽位"""
        missing = []
        for slot, status in self.state['slot_filling'].items():
            if status == 'empty':
                missing.append(slot)
        return missing
    
    def is_task_complete(self):
        """检查任务是否完成"""
        return all(status == 'filled' for status in self.state['slot_filling'].values())
    
    def get_state(self):
        """获取当前状态"""
        return self.state
    
    def reset(self):
        """重置状态"""
        self.state = {
            'intent': None,
            'entities': {},
            'slot_filling': {},
            'history': []
        }

# 测试
tracker = DialogueStateTracker()

# 模拟对话流程
print("初始状态:")
print(tracker.get_state())

# 用户第一次输入
tracker.update_state(
    "I want to book a flight to Paris",
    'book_flight',
    {'destination': 'Paris'}
)

print("\n第一次输入后的状态:")
print(tracker.get_state())
print(f"缺失槽位: {tracker.get_missing_slots()}")

# 用户第二次输入
tracker.update_state(
    "Next Friday for 2 people",
    'book_flight',
    {'date': 'next friday', 'passengers': '2'}
)

print("\n第二次输入后的状态:")
print(tracker.get_state())
print(f"任务完成: {tracker.is_task_complete()}")
```

---

**下一节**：[代码生成应用](03-code-generation-apps.md)

---

## 参考文献

1. Serban, I. V., et al. (2016). Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Network Models.
2. Li, J., et al. (2016). Deep Reinforcement Learning for Dialogue Generation.
3. Roller, S., et al. (2021). Recipes for building an open-domain chatbot.