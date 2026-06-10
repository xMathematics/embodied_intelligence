# 10.8 具身智能前沿: VLA与端到端

## 1. "RT-2: Vision-Language-Action Models"

| 项目 | 内容 |
|------|------|
| 作者 | Brohan et al. (Google) |
| 发表 | CoRL 2023 |
| 核心 | VLM→机器人策略, 55B参数 |
| 部署挑战 | 需要蒸馏+量化才能在边缘运行 |
| 参考价值 | VLA大模型的部署优化目标 |

## 2. "PaLM-E: An Embodied Multimodal Language Model"

| 项目 | 内容 |
|------|------|
| 作者 | Driess et al. (Google) |
| 发表 | ICML 2023 |
| 核心 | 传感器数据纳入LLM训练, 540B |
| 机器人价值 | 多模态(视觉+语言+状态)建模 |

## 3. "Mobile ALOHA: Low-Cost Bimanual Teleoperation"

| 项目 | 内容 |
|------|------|
| 作者 | Zhao et al. (Stanford) |
| 发表 | CoRL 2023 |
| 核心 | 低成本遥操作+行为克隆 |
| 部署要点 | ~10M参数, 可在Jetson上运行 |

## 4. "Octo: An Open-Source Generalist Robot Policy"

| 项目 | 内容 |
|------|------|
| 作者 | Octo Team |
| 发表 | RSS 2024 |
| 核心 | 开源通用策略, 多机器人形态 |
| 模型 | Octo-S (93M), Octo-B (1.3B) |
| 机器人价值 | Octo-S可在Jetson Orin上实时运行 |

## 5. 论文阅读路径

```
量化: Jacob(CVPR18) → LSQ(ICLR20) → GPTQ(ICLR23) → QLoRA(NeurIPS23)
剪枝: Han(NeurIPS15) → Li(ICLR17) → Lottery Ticket(ICLR19)
蒸馏: Hinton(NeurIPS15) → Survey(IJCV21) → VLM蒸馏(CoRL23)
部署: Survey(ACM22) → 实时融合(RSS22) → 四足VLM(CoRL23)
VLA:  RT-2(CoRL23) → PaLM-E(ICML23) → Mobile ALOHA(CoRL23) → Octo(RSS24)
```
