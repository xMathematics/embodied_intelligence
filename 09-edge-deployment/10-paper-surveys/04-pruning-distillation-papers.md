# 10.4 剪枝与蒸馏论文

## 1. "Learning both Weights and Connections"

| 项目 | 内容 |
|------|------|
| 作者 | Han et al. |
| 发表 | NeurIPS 2015 |
| 核心 | 训练→剪枝→微调三阶段 |
| 机器人价值 | 压缩感知模型90%+参数 |

## 2. "Pruning Filters for Efficient ConvNets"

| 项目 | 内容 |
|------|------|
| 作者 | Li et al. |
| 发表 | ICLR 2017 |
| 核心 | 基于L1范数的通道剪枝 |
| 机器人价值 | 结构化剪枝YOLO卷积层 |

## 3. "The Lottery Ticket Hypothesis"

| 项目 | 内容 |
|------|------|
| 作者 | Frankle & Carbin |
| 发表 | ICLR 2019 |
| 核心 | 稀疏子网络可达到全网络精度 |
| 引用 | >2000次 |
| 机器人价值 | 寻找策略网络的"中奖彩票" |

## 4. "Distilling the Knowledge in a Neural Network"

| 项目 | 内容 |
|------|------|
| 作者 | Hinton et al. |
| 发表 | NeurIPS 2015 |
| 核心 | 温度参数+KL散度蒸馏 |
| 引用 | >15000次 |
| 机器人价值 | VLA大模型蒸馏为轻量策略 |

## 5. "Distilling VLM for Embodied Tasks"

| 项目 | 内容 |
|------|------|
| 发表 | CoRL 2023 |
| 核心 | 行为克隆蒸馏+特征对齐 |
| 机器人价值 | 在Jetson上运行蒸馏后的VLA |
