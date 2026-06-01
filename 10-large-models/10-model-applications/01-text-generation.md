# 10.1 文本生成应用

## 目录

- [1. 引言](#1-引言)
- [2. 文本生成概述](#2-文本生成概述)
- [3. 核心技术](#3-核心技术)
- [4. 应用场景](#4-应用场景)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 文本生成的重要性

**文本生成**是大型语言模型最核心的应用之一。它能够根据输入的提示自动生成连贯、有意义的文本内容。

### 1.2 应用价值

| 维度 | 描述 |
|------|------|
| **内容创作** | 自动生成文章、故事、诗歌等 |
| **自动化写作** | 新闻稿、报告、邮件自动生成 |
| **创意辅助** | 帮助作家、设计师激发灵感 |
| **个性化内容** | 根据用户偏好生成定制内容 |

---

## 2. 文本生成概述

### 2.1 定义

**文本生成**：根据给定的上下文或提示，自动生成符合语法和语义规则的文本序列。

**形式化表达**：
```
P(text | prompt) = argmax ∏ P(token_i | token_1, ..., token_{i-1}, prompt)
```

### 2.2 生成模式

| 模式 | 描述 | 示例 |
|------|------|------|
| **续写** | 给定开头，继续生成 | 故事续写、代码补全 |
| **问答** | 根据问题生成答案 | 知识库问答、客服对话 |
| **摘要** | 生成文本摘要 | 文档摘要、新闻摘要 |
| **创作** | 从零开始创作 | 写诗、写故事 |

---

## 3. 核心技术

### 3.1 解码策略

```python
class TextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def generate_greedy(self, prompt, max_length=100):
        """
        贪心解码
        
        参数:
            prompt: 输入提示
            max_length: 最大长度
        
        返回:
            生成的文本
        """
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        for _ in range(max_length):
            outputs = self.model(**inputs)
            next_token = outputs.logits.argmax(dim=-1)[:, -1]
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token.unsqueeze(0)], dim=-1)
        
        return self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    
    def generate_beam_search(self, prompt, max_length=100, num_beams=5):
        """
        束搜索解码
        
        参数:
            prompt: 输入提示
            max_length: 最大长度
            num_beams: 束宽
        
        返回:
            生成的文本
        """
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_sampling(self, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        """
        采样解码
        
        参数:
            prompt: 输入提示
            max_length: 最大长度
            temperature: 温度参数
            top_k: Top-K采样
            top_p: Top-P采样
        
        返回:
            生成的文本
        """
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试（需要加载预训练模型）
# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("gpt2")
# generator = TextGenerator(model, tokenizer)
# 
# prompt = "In the year 2050, artificial intelligence has become"
# print(generator.generate_greedy(prompt))
```

### 3.2 提示工程

```python
class PromptEngineer:
    def __init__(self):
        self.templates = {
            'creative_writing': {
                'prompt': """Write a creative story about {topic}.
The story should be engaging and have a surprising ending.

Story:""",
                'examples': []
            },
            'technical_writing': {
                'prompt': """Explain {topic} in simple terms.
Provide a clear explanation suitable for someone with no technical background.

Explanation:""",
                'examples': []
            },
            'summarization': {
                'prompt': """Summarize the following text in 3-5 sentences:

{text}

Summary:""",
                'examples': []
            }
        }
    
    def create_prompt(self, template_name, **kwargs):
        """
        创建提示
        
        参数:
            template_name: 模板名称
            kwargs: 模板参数
        
        返回:
            完整提示
        """
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"模板 {template_name} 不存在")
        
        prompt = template['prompt'].format(**kwargs)
        
        # 添加示例（如果有）
        if template['examples']:
            examples = "\n\n".join(template['examples'])
            prompt = examples + "\n\n" + prompt
        
        return prompt
    
    def add_example(self, template_name, example):
        """
        添加示例
        
        参数:
            template_name: 模板名称
            example: 示例内容
        """
        if template_name in self.templates:
            self.templates[template_name]['examples'].append(example)

# 测试
engineer = PromptEngineer()

# 创建创意写作提示
prompt = engineer.create_prompt(
    'creative_writing',
    topic='a time traveler who visits their own past'
)
print("创意写作提示:")
print(prompt)

# 创建技术解释提示
prompt = engineer.create_prompt(
    'technical_writing',
    topic='quantum computing'
)
print("\n技术解释提示:")
print(prompt)
```

### 3.3 控制生成

```python
class GenerationController:
    def __init__(self, generator):
        self.generator = generator
    
    def generate_with_constraints(self, prompt, constraints):
        """
        带约束的生成
        
        参数:
            prompt: 输入提示
            constraints: 约束条件
        
        返回:
            生成的文本
        """
        # 应用约束
        if 'max_tokens' in constraints:
            max_length = constraints['max_tokens']
        else:
            max_length = 100
        
        if 'style' in constraints:
            prompt = f"Write in a {constraints['style']} style: {prompt}"
        
        if 'language' in constraints:
            prompt = f"Write in {constraints['language']}: {prompt}"
        
        if 'keywords' in constraints:
            keywords = ", ".join(constraints['keywords'])
            prompt = f"{prompt}\n\nInclude these keywords: {keywords}"
        
        return self.generator.generate_sampling(prompt, max_length=max_length)
    
    def generate_multiple(self, prompt, num_variants=3, **kwargs):
        """
        生成多个变体
        
        参数:
            prompt: 输入提示
            num_variants: 变体数量
        
        返回:
            变体列表
        """
        variants = []
        
        for i in range(num_variants):
            temperature = 0.7 + i * 0.1  # 逐渐增加随机性
            variant = self.generator.generate_sampling(prompt, temperature=temperature, **kwargs)
            variants.append(variant)
        
        return variants

# 测试
# controller = GenerationController(generator)
# 
# constraints = {
#     'max_tokens': 150,
#     'style': 'formal',
#     'language': 'English',
#     'keywords': ['innovation', 'technology', 'future']
# }
# 
# result = controller.generate_with_constraints(
#     'Write an article about AI',
#     constraints
# )
# print(result)
```

---

## 4. 应用场景

### 4.1 内容创作

```python
class ContentGenerator:
    def __init__(self, generator):
        self.generator = generator
    
    def generate_blog_post(self, topic, style='informative'):
        """
        生成博客文章
        
        参数:
            topic: 主题
            style: 风格
        
        返回:
            博客文章
        """
        prompt = f"""Write a {style} blog post about {topic}.
Include an introduction, 3-5 main points with explanations, and a conclusion.

Blog Post:"""
        
        return self.generator.generate_sampling(prompt, max_length=500)
    
    def generate_poem(self, theme, style='free verse'):
        """
        生成诗歌
        
        参数:
            theme: 主题
            style: 风格
        
        返回:
            诗歌
        """
        prompt = f"""Write a {style} poem about {theme}.
Use vivid imagery and emotional language.

Poem:"""
        
        return self.generator.generate_sampling(prompt, max_length=200)
    
    def generate_story(self, genre, characters=None):
        """
        生成故事
        
        参数:
            genre: 类型
            characters: 角色列表
        
        返回:
            故事
        """
        if characters:
            char_str = ", ".join(characters)
            prompt = f"""Write a {genre} story featuring: {char_str}.
The story should have a beginning, middle, and ending with character development.

Story:"""
        else:
            prompt = f"""Write a {genre} story.
Include interesting characters and an engaging plot.

Story:"""
        
        return self.generator.generate_sampling(prompt, max_length=800)

# 测试
# content_gen = ContentGenerator(generator)
# poem = content_gen.generate_poem('autumn leaves', 'haiku')
# print("生成的诗歌:")
# print(poem)
```

### 4.2 文档生成

```python
class DocumentGenerator:
    def __init__(self, generator):
        self.generator = generator
    
    def generate_email(self, recipient, purpose, tone='professional'):
        """
        生成邮件
        
        参数:
            recipient: 收件人
            purpose: 目的
            tone: 语气
        
        返回:
            邮件内容
        """
        prompt = f"""Write a {tone} email to {recipient} about {purpose}.
Include subject line, greeting, body with clear message, and closing.

Email:"""
        
        return self.generator.generate_sampling(prompt, max_length=300)
    
    def generate_report(self, title, sections):
        """
        生成报告
        
        参数:
            title: 标题
            sections: 章节列表
        
        返回:
            报告内容
        """
        sections_str = "\n".join(f"- {section}" for section in sections)
        
        prompt = f"""Write a professional report titled '{title}'.
Include the following sections:
{sections_str}

Report:"""
        
        return self.generator.generate_sampling(prompt, max_length=1000)
    
    def generate_resume(self, name, experience, skills):
        """
        生成简历
        
        参数:
            name: 姓名
            experience: 工作经历
            skills: 技能列表
        
        返回:
            简历内容
        """
        skills_str = ", ".join(skills)
        
        prompt = f"""Write a professional resume for {name}.
Work experience: {experience}
Key skills: {skills_str}

Resume:"""
        
        return self.generator.generate_sampling(prompt, max_length=500)

# 测试
# doc_gen = DocumentGenerator(generator)
# email = doc_gen.generate_email(
#     'John Smith',
#     'project update',
#     'friendly'
# )
# print("生成的邮件:")
# print(email)
```

### 4.3 教育应用

```python
class EducationalGenerator:
    def __init__(self, generator):
        self.generator = generator
    
    def generate_explanation(self, topic, level='beginner'):
        """
        生成解释
        
        参数:
            topic: 主题
            level: 难度级别
        
        返回:
            解释内容
        """
        prompt = f"""Explain {topic} at a {level} level.
Use simple language and provide examples.

Explanation:"""
        
        return self.generator.generate_sampling(prompt, max_length=400)
    
    def generate_questions(self, topic, num_questions=5, difficulty='medium'):
        """
        生成问题
        
        参数:
            topic: 主题
            num_questions: 问题数量
            difficulty: 难度
        
        返回:
            问题列表
        """
        prompt = f"""Generate {num_questions} {difficulty} level questions about {topic}.
Include multiple choice questions with options.

Questions:"""
        
        return self.generator.generate_sampling(prompt, max_length=500)
    
    def generate_quiz(self, topic, num_questions=5):
        """
        生成测验
        
        参数:
            topic: 主题
            num_questions: 问题数量
        
        返回:
            测验内容
        """
        prompt = f"""Create a quiz with {num_questions} questions about {topic}.
Include questions, options, and answers.

Quiz:"""
        
        return self.generator.generate_sampling(prompt, max_length=600)

# 测试
# edu_gen = EducationalGenerator(generator)
# explanation = edu_gen.generate_explanation('machine learning', 'beginner')
# print("生成的解释:")
# print(explanation)
```

---

## 5. 实践练习

### 练习1：实现文本生成器

```python
class AdvancedTextGenerator:
    def __init__(self, model_name='gpt2'):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 添加pad token（如果不存在）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt, **kwargs):
        """
        生成文本
        
        参数:
            prompt: 输入提示
            kwargs: 生成参数
        
        返回:
            生成的文本
        """
        default_kwargs = {
            'max_length': 200,
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.9,
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id
        }
        
        default_kwargs.update(kwargs)
        
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(**inputs, **default_kwargs)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_batch(self, prompts, **kwargs):
        """
        批量生成
        
        参数:
            prompts: 提示列表
            kwargs: 生成参数
        
        返回:
            生成的文本列表
        """
        inputs = self.tokenizer(prompts, padding=True, return_tensors='pt')
        
        default_kwargs = {
            'max_length': 200,
            'temperature': 0.7,
            'pad_token_id': self.tokenizer.pad_token_id
        }
        default_kwargs.update(kwargs)
        
        outputs = self.model.generate(**inputs, **default_kwargs)
        
        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

# 测试
# generator = AdvancedTextGenerator()
# 
# prompt = "The future of artificial intelligence is"
# result = generator.generate(prompt, max_length=150)
# print("生成结果:")
# print(result)
```

### 练习2：实现提示模板系统

```python
class PromptTemplateSystem:
    def __init__(self):
        self.templates = {}
    
    def register_template(self, name, template, examples=None):
        """
        注册模板
        
        参数:
            name: 模板名称
            template: 模板字符串
            examples: 示例列表
        """
        self.templates[name] = {
            'template': template,
            'examples': examples or []
        }
    
    def generate_prompt(self, template_name, **variables):
        """
        生成提示
        
        参数:
            template_name: 模板名称
            variables: 变量
        
        返回:
            完整提示
        """
        if template_name not in self.templates:
            raise ValueError(f"模板 {template_name} 未找到")
        
        template = self.templates[template_name]
        prompt = template['template'].format(**variables)
        
        # 添加示例
        if template['examples']:
            examples_str = "\n\n---\n\n".join(template['examples'])
            prompt = f"{examples_str}\n\n---\n\n{prompt}"
        
        return prompt
    
    def list_templates(self):
        """列出所有模板"""
        return list(self.templates.keys())

# 测试
system = PromptTemplateSystem()

# 注册模板
system.register_template(
    'creative_story',
    """Write a {genre} story about {protagonist} who {goal}.
The story should have a surprising twist at the end.

Story:""",
    examples=[
        """Write a fantasy story about a young wizard who wants to find their missing mentor.
The story should have a surprising twist at the end.

Story:
Eldric had searched for years, following every clue about his mentor's disappearance. 
On the night he finally found the ancient tower, he discovered a mirror showing his mentor 
was trapped inside - as the tower's guardian, bound by a spell Eldric himself had cast 
as a child in a moment of anger.""",
        """Write a sci-fi story about a space explorer who discovers an alien artifact.
The story should have a surprising twist at the end.

Story:
Captain Maro's hands trembled as she touched the glowing orb. 
The alien script swirled before her eyes, revealing its purpose - 
it was a communication device, and the message was clear: 
they'd been waiting for her. She realized with a shock that 
the artifact wasn't found by her - it had been guiding her here all along."""
    ]
)

# 生成提示
prompt = system.generate_prompt(
    'creative_story',
    genre='mystery',
    protagonist='detective',
    goal='solves a murder case'
)

print("生成的提示:")
print(prompt)
```

### 练习3：实现内容生成流水线

```python
class ContentPipeline:
    def __init__(self, generator):
        self.generator = generator
        self.steps = []
    
    def add_step(self, name, func):
        """
        添加处理步骤
        
        参数:
            name: 步骤名称
            func: 处理函数
        """
        self.steps.append({'name': name, 'func': func})
    
    def run(self, input_data):
        """
        运行流水线
        
        参数:
            input_data: 输入数据
        
        返回:
            最终结果
        """
        result = input_data
        
        for step in self.steps:
            print(f"执行步骤: {step['name']}")
            result = step['func'](result, self.generator)
        
        return result

# 定义流水线步骤
def step_generate_title(topic, generator):
    """生成标题"""
    prompt = f"Generate 5 catchy titles for an article about {topic}:"
    titles = generator.generate(prompt, max_length=100)
    return {'topic': topic, 'titles': titles.split('\n')[:5]}

def step_select_title(data, generator):
    """选择最佳标题"""
    title = data['titles'][0]  # 简化：选择第一个
    return {'topic': data['topic'], 'title': title}

def step_generate_content(data, generator):
    """生成内容"""
    prompt = f"""Write a detailed article titled '{data['title']}'.
Include introduction, main points, and conclusion.

Article:"""
    content = generator.generate(prompt, max_length=800)
    return {'title': data['title'], 'content': content}

def step_improve_content(data, generator):
    """优化内容"""
    prompt = f"""Improve the following article for better flow and readability:

{data['content']}

Improved article:"""
    improved = generator.generate(prompt, max_length=1000)
    return {'title': data['title'], 'content': improved}

# 测试
# pipeline = ContentPipeline(generator)
# 
# pipeline.add_step('生成标题', step_generate_title)
# pipeline.add_step('选择标题', step_select_title)
# pipeline.add_step('生成内容', step_generate_content)
# pipeline.add_step('优化内容', step_improve_content)
# 
# result = pipeline.run('the future of AI')
# print(f"\n最终结果:")
# print(f"标题: {result['title']}")
# print(f"内容:\n{result['content']}")
```

---

**下一节**：[对话系统](02-dialogue-systems.md)

---

## 参考文献

1. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners.
2. Brown, T. B., et al. (2020). Language Models are Few-Shot Learners.
3. Holtzman, A., et al. (2020). The Curious Case of Neural Text Degeneration.