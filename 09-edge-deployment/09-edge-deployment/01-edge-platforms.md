# 9.1 边缘平台深度对比与选型

## 1. 主流平台对比

| 平台 | AI算力 | 功耗 | 价格 | AI加速器 | 适用机器人 |
|------|-------|------|------|---------|-----------|
| **Orin AGX 64GB** | 275 TOPS | 15-60W | ~$2000 | Ampere GPU+DLA | 人形机器人 |
| **Orin NX 16GB** | 70 TOPS | 10-25W | ~$800 | Ampere GPU+DLA | 四足/轮式 |
| **Xavier NX** | 21 TOPS | 10-20W | ~$400 | Volta GPU | 服务机器人 |
| **Jetson Nano** | 0.47 TFLOPS | 5-10W | ~$100 | Maxwell GPU | 教育 |
| **RK3588** | 6 TOPS NPU | 5-15W | ~$200 | Mali+三核NPU | 国产机器人 |
| **Intel NUC 13** | 2-5 TFLOPS | 15-28W | ~$500 | Iris Xe | 工业机器人 |
| **树莓派5** | 0.5 TFLOPS | 5-10W | ~$100 | VideoCore | 教育/轻量 |

## 2. Jetson系列深度对比

| 特性 | Orin AGX | Orin NX | Xavier NX | Nano |
|------|---------|---------|-----------|------|
| GPU | 2048 Ampere | 1024 Ampere | 384 Volta | 128 Maxwell |
| Tensor Core | 64 (Gen3) | 32 (Gen3) | 48 (Gen2) | 无 |
| DLA | 2× | 1× | 无 | 无 |
| CPU | 12核A78AE | 8核A78AE | 6核Carmel | 4核A57 |
| 内存 | 64GB | 16GB | 8GB | 4GB |
| 算力 | 275 TOPS | 70 TOPS | 21 TOPS | 0.47 TFLOPS |
| 功耗 | 30-60W | 10-25W | 10-20W | 5-10W |

## 3. 选型建议

| 需求 | 推荐 | 理由 |
|------|------|------|
| 多模型VLA推理 | Orin AGX | 最强算力 |
| 平衡算力重量 | Orin NX | 四足/无人车 |
| 入门学习 | Jetson Nano | 最低成本 |
| 国产替代 | RK3588 | 成本低 |
| 工业级 | Intel NUC | 可靠性高 |
