# 2.5 图文生成

## 目录

- [1. 引言](#1-引言)
- [2. 图文生成概述](#2-图文生成概述)
- [3. 图像描述生成](#3-图像描述生成)
- [4. 文本到图像生成](#4-文本到图像生成)
- [5. 图像条件文本生成](#5-图像条件文本生成)
- [6. 图文生成模型](#6-图文生成模型)
- [7. 评估指标](#7-评估指标)
- [8. 实践练习](#8-实践练习)

---

## 1. 引言

### 1.1 什么是图文生成

**图文生成**是指根据图像生成文本描述，或根据文本描述生成图像的任务。这是视觉-语言模型的重要应用方向。

### 1.2 任务类型

| 任务 | 描述 | 示例 |
|------|------|------|
| **图像描述** | 根据图像生成文字描述 | 输入图像，输出"一只猫坐在沙发上" |
| **文本到图像** | 根据文本生成图像 | 输入"一只猫坐在沙发上"，输出图像 |
| **图像条件生成** | 根据图像和文本生成新内容 | 图像+指令→编辑后的图像 |

---

## 2. 图文生成概述

### 2.1 生成任务的特点

| 特点 | 描述 |
|------|------|
| **创造性** | 需要生成新颖的内容 |
| **多样性** | 同一输入可能有多种合理输出 |
| **质量评估** | 生成质量难以量化评估 |
| **可控性** | 需要控制生成内容的属性 |

### 2.2 生成模型分类

| 类型 | 描述 | 代表模型 |
|------|------|---------|
| **基于Transformer** | 使用Transformer架构 | BLIP、Flamingo |
| **基于扩散模型** | 使用扩散过程生成 | Stable Diffusion、DALL-E |
| **基于GAN** | 使用生成对抗网络 | StyleGAN |

---

## 3. 图像描述生成

### 3.1 任务定义

给定图像I，生成文本描述T：
```
Image Captioning(I) → T
```

### 3.2 经典模型

| 模型 | 特点 |
|------|------|
| **Show and Tell** | 首个端到端图像描述模型 |
| **Neural Talk** | 使用LSTM进行生成 |
| **Show, Attend and Tell** | 引入注意力机制 |
| **BLIP** | 统一理解和生成框架 |

### 3.3 图像描述架构

```
图像 → CNN/ViT → 图像特征
图像特征 → Transformer解码器 → 文本序列
```

### 3.4 代码示例：BLIP图像描述

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 加载模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 加载图像
image = Image.open("example.jpg").convert("RGB")

# 生成描述
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print(f"图像描述: {caption}")
```

### 3.5 不同生成策略

| 策略 | 描述 | 参数 |
|------|------|------|
| **贪婪搜索** | 每次选择概率最高的token | greedy decoding |
| **束搜索** | 保留多个候选序列 | beam search |
| **采样** | 随机采样token | sampling |
| **Top-k采样** | 从概率最高的k个token中采样 | top-k |
| **Top-p采样** | 从累积概率达到p的token中采样 | top-p (nucleus) |

---

## 4. 文本到图像生成

### 4.1 任务定义

给定文本描述T，生成图像I：
```
Text-to-Image(T) → I
```

### 4.2 发展历程

| 阶段 | 模型 | 特点 |
|------|------|------|
| **早期** | GAN-based | 分辨率低，质量差 |
| **中期** | DALL-E | 首次展示高质量生成 |
| **近期** | Stable Diffusion | 开源，高效 |
| **最新** | DALL-E 3 | 更高质量，更可控 |

### 4.3 扩散模型原理

**扩散过程**：
```
步骤1: 从高斯噪声开始
步骤2: 逐步去噪（T步）
步骤3: 最终得到清晰图像
```

**公式**：
```
x_0 → x_1 → ... → x_T  (加噪过程)
x_T → x_{T-1} → ... → x_0  (去噪过程)
```

### 4.4 代码示例：Stable Diffusion

```python
from diffusers import StableDiffusionPipeline
import torch

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# 生成图像
prompt = "a cat sitting on a couch in a cozy living room"
image = pipe(prompt).images[0]

# 保存图像
image.save("generated_image.png")
print("图像已生成并保存为 generated_image.png")
```

### 4.5 可控生成技术

| 技术 | 描述 | 示例 |
|------|------|------|
| **文本提示** | 详细的文本描述 | "一只可爱的橘猫" |
| **图像引导** | 使用参考图像 | 风格迁移 |
| **ControlNet** | 结构控制 | 深度图、边缘图 |
| **Inpainting** | 局部编辑 | 修改图像的一部分 |

---

## 5. 图像条件文本生成

### 5.1 任务类型

| 任务 | 描述 | 示例 |
|------|------|------|
| **图像描述** | 生成图像的文字描述 | 如前所述 |
| **视觉故事生成** | 根据图像生成故事 | 多图像→连贯故事 |
| **图像问答生成** | 根据图像生成问答对 | 自动生成训练数据 |
| **图像字幕** | 为图像添加字幕 | 短视频字幕 |

### 5.2 视觉故事生成

**挑战**：
- 保持故事连贯性
- 理解图像序列的时间关系
- 生成自然流畅的文本

### 5.3 代码示例：多图像描述

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 加载模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 加载多张图像
images = [
    Image.open("image1.jpg").convert("RGB"),
    Image.open("image2.jpg").convert("RGB"),
    Image.open("image3.jpg").convert("RGB")
]

# 为每张图像生成描述
captions = []
for i, image in enumerate(images):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    captions.append(caption)
    print(f"图像{i+1}描述: {caption}")

# 生成连贯的故事
story_prompt = f"根据以下图像描述生成一个连贯的故事:\n"
for i, caption in enumerate(captions):
    story_prompt += f"图像{i+1}: {caption}\n"
story_prompt += "故事:"

# 这里可以使用语言模型生成故事
print("\n故事提示:", story_prompt)
```

---

## 6. 图文生成模型

### 6.1 DALL-E系列

**DALL-E**：
- 基于GPT-3的变体
- 使用Transformer架构
- 首次实现高质量文本到图像生成

**DALL-E 2**：
- 使用扩散模型
- 更高分辨率（1024x1024）
- 更好的图像质量

**DALL-E 3**：
- 与GPT-4集成
- 更好的文本理解
- 更高的细节和一致性

### 6.2 Stable Diffusion

**特点**：
- 开源模型
- 高效推理
- 社区活跃
- 支持多种扩展（ControlNet、Inpainting等）

**架构**：
```
文本 → CLIP编码器 → 文本嵌入
文本嵌入 + 噪声 → UNet → 去噪图像
```

### 6.3 MidJourney

**特点**：
- 闭源模型
- 极高的图像质量
- 强大的风格理解
- 社区生成大量优秀作品

### 6.4 Flamingo

**特点**：
- 视觉语言模型
- 可以处理图像和文本序列
- 支持few-shot学习
- 可以进行图像条件生成

---

## 7. 评估指标

### 7.1 图像描述评估

| 指标 | 描述 |
|------|------|
| **BLEU** | 机器翻译评估指标，衡量n-gram匹配 |
| **ROUGE** | 文本摘要评估指标 |
| **METEOR** | 考虑词干化和同义词 |
| **CIDEr** | 基于CIDEr-D的改进，适合图像描述 |
| **SPICE** | 基于语义相似度 |

### 7.2 文本到图像评估

| 指标 | 描述 |
|------|------|
| **FID** | Fréchet Inception Distance，衡量生成图像与真实图像的分布差异 |
| **IS** | Inception Score，衡量多样性和质量 |
| **CLIP Score** | 使用CLIP模型计算生成图像与文本的相似度 |
| **人工评估** | 人类评估生成质量 |

### 7.3 评估挑战

| 挑战 | 描述 |
|------|------|
| **多样性** | 同一文本可能生成多种合理图像 |
| **主观性** | 图像质量评估主观 |
| **计算成本** | FID等指标计算成本高 |
| **数据集偏差** | 评估数据集可能存在偏差 |

---

## 8. 实践练习

### 练习1：图像描述生成

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 加载模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 加载图像
image = Image.open("example.jpg").convert("RGB")

# 不同生成策略
print("=== 贪婪搜索 ===")
out = model.generate(**processor(image, return_tensors="pt"))
print(processor.decode(out[0], skip_special_tokens=True))

print("\n=== 束搜索 ===")
out = model.generate(**processor(image, return_tensors="pt"), num_beams=5)
print(processor.decode(out[0], skip_special_tokens=True))

print("\n=== Top-p采样 ===")
out = model.generate(**processor(image, return_tensors="pt"), do_sample=True, top_p=0.9)
print(processor.decode(out[0], skip_special_tokens=True))
```

### 练习2：文本到图像生成

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

# 加载模型，使用更快的调度器
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# 生成图像
prompts = [
    "a beautiful sunset over the ocean with golden clouds",
    "a cute cat wearing a hat sitting on a windowsill",
    "a futuristic city with flying cars at night"
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt, num_inference_steps=20).images[0]
    image.save(f"generated_{i+1}.png")
    print(f"生成图像 {i+1}: {prompt}")
```

### 练习3：图像编辑（Inpainting）

```python
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

# 加载inpainting模型
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 加载图像和掩码
image = Image.open("original.jpg").convert("RGB")
mask = Image.open("mask.png").convert("L")  # 掩码图像，白色区域将被替换

# 生成新图像
prompt = "a beautiful flower in a vase"
image = pipe(prompt=prompt, image=image, mask_image=mask).images[0]

# 保存结果
image.save("edited_image.png")
print("图像编辑完成，保存为 edited_image.png")
```

### 练习4：评估图像描述质量

```python
import evaluate

# 加载评估指标
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

# 假设的预测和参考
predictions = [
    "a cat sitting on a couch",
    "two dogs playing in the park"
]
references = [
    ["a cat sits on the couch", "there is a cat on the couch"],
    ["two dogs are playing in the park", "two dogs play in the park"]
]

# 计算指标
bleu_result = bleu.compute(predictions=predictions, references=references)
meteor_result = meteor.compute(predictions=predictions, references=references)

print(f"BLEU: {bleu_result}")
print(f"METEOR: {meteor_result}")
```

---

**返回**：[VLM基础](01-vlm-basics.md)

---

## 参考文献

1. Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). Show and tell: A neural image caption generator.
2. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision.
3. Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., ... & Sutskever, I. (2021). Zero-shot text-to-image generation.
4. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models.
