# 10.4 多模态应用

## 目录

- [1. 引言](#1-引言)
- [2. 多模态应用概述](#2-多模态应用概述)
- [3. 核心技术](#3-核心技术)
- [4. 应用场景](#4-应用场景)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 多模态应用的重要性

**多模态应用**结合了文本、图像、音频、视频等多种模态，为用户提供更丰富的交互体验。

### 1.2 应用价值

| 维度 | 描述 |
|------|------|
| **丰富表达** | 结合视觉和语言信息 |
| **增强理解** | 多维度理解用户意图 |
| **创意生成** | 跨模态内容创作 |
| **无障碍支持** | 支持不同能力用户 |

---

## 2. 多模态应用概述

### 2.1 定义

**多模态应用**：同时处理和生成多种模态数据的应用。

**形式化表达**：
```
output = f(text, image, audio, video)
```

### 2.2 多模态类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **图像-文本** | 图像和文本交互 | 图像描述、视觉问答 |
| **音频-文本** | 语音和文本交互 | 语音识别、语音合成 |
| **视频-文本** | 视频和文本交互 | 视频描述、视频问答 |
| **混合模态** | 多种模态融合 | 多模态对话系统 |

---

## 3. 核心技术

### 3.1 多模态理解

```python
class Multimodal理解:
    def __init__(self, vision_model, text_model, fusion_model):
        self.vision_model = vision_model
        self.text_model = text_model
        self.fusion_model = fusion_model
    
    def process_image(self, image):
        """处理图像"""
        features = self.vision_model(image)
        return features
    
    def process_text(self, text):
        """处理文本"""
        features = self.text_model(text)
        return features
    
    def fuse_features(self, image_features, text_features):
        """融合特征"""
        fused = self.fusion_model(image_features, text_features)
        return fused
    
    def predict(self, image, text):
        """预测"""
        img_feat = self.process_image(image)
        txt_feat = self.process_text(text)
        fused = self.fuse_features(img_feat, txt_feat)
        return fused

# 测试（需要加载预训练模型）
# from transformers import CLIPModel, CLIPProcessor
# 
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# 
# class SimpleMultimodal理解:
#     def __init__(self, model, processor):
#         self.model = model
#         self.processor = processor
#     
#     def predict(self, image, text):
#         inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True)
#         outputs = self.model(**inputs)
#         logits_per_image = outputs.logits_per_image
#         return logits_per_image.softmax(dim=1)
```

### 3.2 多模态生成

```python
class MultimodalGenerator:
    def __init__(self, text_to_image_model, image_to_text_model):
        self.text_to_image = text_to_image_model
        self.image_to_text = image_to_text_model
    
    def generate_image(self, prompt):
        """根据文本生成图像"""
        image = self.text_to_image(prompt)
        return image
    
    def generate_caption(self, image):
        """根据图像生成描述"""
        caption = self.image_to_text(image)
        return caption
    
    def generate_story(self, image, max_length=300):
        """根据图像生成故事"""
        caption = self.generate_caption(image)
        story_prompt = f"Write a story based on this scene: {caption}"
        
        # 使用文本生成模型扩展成故事
        story = self._generate_text(story_prompt, max_length)
        return story
    
    def _generate_text(self, prompt, max_length):
        """生成文本（简化实现）"""
        return f"A story about: {prompt}"

# 测试（需要加载图像生成模型）
# from diffusers import StableDiffusionPipeline
# 
# pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
# 
# class ImageGenerator:
#     def __init__(self, pipe):
#         self.pipe = pipe
#     
#     def generate(self, prompt):
#         image = self.pipe(prompt).images[0]
#         return image
```

### 3.3 多模态对话

```python
class MultimodalDialogueSystem:
    def __init__(self, text_model, vision_model):
        self.text_model = text_model
        self.vision_model = vision_model
        self.context = []
    
    def add_message(self, message_type, content):
        """
        添加消息
        
        参数:
            message_type: 消息类型（text/image/audio）
            content: 内容
        """
        self.context.append({
            'type': message_type,
            'content': content
        })
    
    def generate_response(self):
        """生成响应"""
        # 构建上下文提示
        prompt = ""
        for msg in self.context:
            if msg['type'] == 'text':
                prompt += f"User: {msg['content']}\n"
            elif msg['type'] == 'image':
                prompt += f"[Image provided]\n"
            elif msg['type'] == 'audio':
                prompt += f"[Audio provided]\n"
        
        prompt += "Assistant:"
        
        # 生成响应
        response = self.text_model.generate(prompt, max_length=200)
        return response
    
    def process_vision_query(self, image, question):
        """
        处理视觉问答
        
        参数:
            image: 图像
            question: 问题
        
        返回:
            答案
        """
        # 使用视觉问答模型
        answer = self.vision_model.answer(image, question)
        return answer

# 测试（需要加载VQA模型）
# from transformers import ViltProcessor, ViltForQuestionAnswering
# 
# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# 
# class VQASystem:
#     def __init__(self, processor, model):
#         self.processor = processor
#         self.model = model
#     
#     def answer(self, image, question):
#         encoding = self.processor(image, question, return_tensors="pt")
#         outputs = self.model(**encoding)
#         logits = outputs.logits
#         idx = logits.argmax(-1).item()
#         return self.model.config.id2label[idx]
```

---

## 4. 应用场景

### 4.1 图像描述

```python
class ImageCaptioningSystem:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def generate_caption(self, image, max_length=50):
        """
        生成图像描述
        
        参数:
            image: 图像
            max_length: 最大长度
        
        返回:
            描述文本
        """
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    
    def generate_multiple_captions(self, image, num_captions=3):
        """
        生成多个描述
        
        参数:
            image: 图像
            num_captions: 数量
        
        返回:
            描述列表
        """
        captions = []
        for _ in range(num_captions):
            caption = self.generate_caption(image)
            captions.append(caption)
        return captions

# 测试（需要加载图像字幕模型）
# from transformers import BlipProcessor, BlipForConditionalGeneration
# 
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# 
# captioner = ImageCaptioningSystem(model, processor)
# # image = Image.open("example.jpg")
# # caption = captioner.generate_caption(image)
# # print(caption)
```

### 4.2 视觉问答

```python
class VisualQuestionAnswering:
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model
    
    def answer(self, image, question):
        """
        回答视觉问题
        
        参数:
            image: 图像
            question: 问题
        
        返回:
            答案
        """
        encoding = self.processor(image, question, return_tensors="pt")
        outputs = self.model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return self.model.config.id2label[idx]
    
    def answer_batch(self, images, questions):
        """
        批量回答问题
        
        参数:
            images: 图像列表
            questions: 问题列表
        
        返回:
            答案列表
        """
        answers = []
        for image, question in zip(images, questions):
            answer = self.answer(image, question)
            answers.append(answer)
        return answers

# 测试（需要加载VQA模型）
# processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# 
# vqa = VisualQuestionAnswering(processor, model)
# # answer = vqa.answer(image, "What is in the image?")
# # print(answer)
```

### 4.3 语音助手

```python
class VoiceAssistant:
    def __init__(self, asr_model, tts_model, llm_model):
        self.asr_model = asr_model  # 语音识别
        self.tts_model = tts_model  # 语音合成
        self.llm_model = llm_model  # 语言模型
    
    def speech_to_text(self, audio):
        """语音转文本"""
        text = self.asr_model.transcribe(audio)
        return text
    
    def text_to_speech(self, text):
        """文本转语音"""
        audio = self.tts_model.synthesize(text)
        return audio
    
    def process_voice_command(self, audio):
        """
        处理语音命令
        
        参数:
            audio: 语音输入
        
        返回:
            语音响应
        """
        # 语音转文本
        text = self.speech_to_text(audio)
        
        # 生成响应
        response = self.llm_model.generate(text, max_length=100)
        
        # 文本转语音
        audio_response = self.text_to_speech(response)
        
        return audio_response, response

# 测试（需要加载语音模型）
# from transformers import pipeline
# 
# asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
# tts = pipeline("text-to-speech", model="espnet/kan-bayashi_ljspeech_vits")
# 
# class SimpleVoiceAssistant:
#     def __init__(self, asr, tts, llm):
#         self.asr = asr
#         self.tts = tts
#         self.llm = llm
#     
#     def process(self, audio):
#         text = self.asr(audio)['text']
#         response = self.llm.generate(text)[0]['generated_text']
#         audio_out = self.tts(response)
#         return audio_out, response
```

---

## 5. 实践练习

### 练习1：实现多模态内容生成器

```python
class MultimodalContentGenerator:
    def __init__(self):
        self.generators = {}
    
    def register_generator(self, modality, generator):
        """
        注册生成器
        
        参数:
            modality: 模态类型
            generator: 生成器函数
        """
        self.generators[modality] = generator
    
    def generate(self, prompt, output_modality='text', **kwargs):
        """
        生成内容
        
        参数:
            prompt: 提示
            output_modality: 输出模态
        
        返回:
            生成的内容
        """
        if output_modality not in self.generators:
            raise ValueError(f"不支持的输出模态: {output_modality}")
        
        return self.generators[output_modality](prompt, **kwargs)
    
    def cross_modal_generate(self, input_content, input_modality, output_modality):
        """
        跨模态生成
        
        参数:
            input_content: 输入内容
            input_modality: 输入模态
            output_modality: 输出模态
        
        返回:
            生成的内容
        """
        # 首先转换为文本
        if input_modality != 'text':
            text = self._modal_to_text(input_content, input_modality)
        else:
            text = input_content
        
        # 然后生成目标模态
        return self.generate(text, output_modality)
    
    def _modal_to_text(self, content, modality):
        """将其他模态转换为文本"""
        # 简化实现
        return f"描述内容来自{modality}模态"

# 测试
generator = MultimodalContentGenerator()

# 注册生成器
generator.register_generator('text', lambda p: f"生成的文本: {p}")
generator.register_generator('image', lambda p: f"生成图像提示: {p}")
generator.register_generator('audio', lambda p: f"生成音频提示: {p}")

# 生成文本
text = generator.generate("写一首诗", 'text')
print(text)

# 跨模态生成
result = generator.cross_modal_generate("一幅画", 'image', 'text')
print(result)
```

### 练习2：实现多模态问答系统

```python
class MultimodalQA:
    def __init__(self):
        self.qa_models = {}
    
    def register_model(self, modality, model):
        """
        注册模型
        
        参数:
            modality: 模态类型
            model: 模型
        """
        self.qa_models[modality] = model
    
    def answer(self, question, context=None, context_modality='text'):
        """
        回答问题
        
        参数:
            question: 问题
            context: 上下文（可选）
            context_modality: 上下文模态
        
        返回:
            答案
        """
        if context_modality not in self.qa_models:
            raise ValueError(f"不支持的上下文模态: {context_modality}")
        
        model = self.qa_models[context_modality]
        return model.answer(question, context)

# 测试
class TextQAModel:
    def answer(self, question, context=None):
        return f"基于文本回答: {question}"

class ImageQAModel:
    def answer(self, question, context=None):
        return f"基于图像回答: {question}"

qa = MultimodalQA()
qa.register_model('text', TextQAModel())
qa.register_model('image', ImageQAModel())

# 文本问答
answer = qa.answer("什么是AI?", context="AI是人工智能", context_modality='text')
print(answer)

# 图像问答
answer = qa.answer("图中有什么?", context="image_data", context_modality='image')
print(answer)
```

### 练习3：实现多模态故事生成器

```python
class MultimodalStoryGenerator:
    def __init__(self, text_generator, image_generator):
        self.text_generator = text_generator
        self.image_generator = image_generator
    
    def generate_story_with_images(self, prompt, num_images=3):
        """
        生成带图像的故事
        
        参数:
            prompt: 故事提示
            num_images: 图像数量
        
        返回:
            故事内容和图像描述列表
        """
        # 生成故事大纲
        outline = self.text_generator.generate(
            f"Create a story outline with {num_images} scenes based on: {prompt}",
            max_length=300
        )
        
        # 提取场景描述
        scenes = self._extract_scenes(outline, num_images)
        
        # 为每个场景生成图像描述和详细内容
        story_parts = []
        image_descriptions = []
        
        for i, scene in enumerate(scenes):
            # 生成图像描述
            image_desc = self.text_generator.generate(
                f"Describe a detailed scene for an illustration: {scene}",
                max_length=100
            )
            image_descriptions.append(image_desc)
            
            # 生成场景内容
            scene_content = self.text_generator.generate(
                f"Write a vivid scene description: {scene}",
                max_length=200
            )
            story_parts.append(scene_content)
        
        # 组合成完整故事
        full_story = "\n\n".join(story_parts)
        
        return {
            'story': full_story,
            'scenes': scenes,
            'image_descriptions': image_descriptions
        }
    
    def _extract_scenes(self, outline, num_scenes):
        """从大纲中提取场景"""
        # 简化实现
        lines = outline.split('\n')
        scenes = []
        for line in lines:
            if line.strip() and len(scenes) < num_scenes:
                scenes.append(line.strip())
        
        # 如果不够，生成额外场景
        while len(scenes) < num_scenes:
            scenes.append(f"Scene {len(scenes) + 1}")
        
        return scenes

# 测试
class SimpleTextGenerator:
    def generate(self, prompt, max_length=200):
        return f"Generated text for: {prompt}"

class SimpleImageGenerator:
    def generate(self, prompt):
        return f"Generated image for: {prompt}"

story_gen = MultimodalStoryGenerator(SimpleTextGenerator(), SimpleImageGenerator())
result = story_gen.generate_story_with_images("一个关于太空探险的故事", num_images=3)

print("故事内容:")
print(result['story'])
print("\n图像描述:")
for i, desc in enumerate(result['image_descriptions']):
    print(f"{i+1}. {desc}")
```

---

**下一节**：[综合应用](05-comprehensive-applications.md)

---

## 参考文献

1. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision.
2. Li, X., et al. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.
3. OpenAI (2023). GPT-4V(ision) Technical Report.