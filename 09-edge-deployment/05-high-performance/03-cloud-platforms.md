# 5.3 云计算平台与云边协同

## 1. GPU云实例对比

| 云平台 | 实例 | GPU | GPU显存 | 互联 | 费用/h | 机器人用途 |
|-------|------|-----|--------|------|-------|-----------|
| AWS | p4d.24xlarge | 8×A100 | 40GB×8 | 600GB/s | ~$32 | VLA训练 |
| AWS | g5.xlarge | 1×A10G | 24GB | — | ~$1 | 小模型 |
| GCP | a2-highgpu-8g | 8×A100 | 80GB×8 | 3D torus | ~$40 | 极限训练 |
| Azure | ND96amsr | 8×A100 | 80GB×8 | NVSwitch | ~$36 | 大模型 |
| Lambda Labs | 8×H100 | 8×H100 | 80GB×8 | NVLink | ~$28 | 性价比 |

## 2. 云-边协同架构

```
云端训练中心                             边缘设备(Jetson)
┌──────────────────┐               ┌──────────────────┐
│ 大模型训练 (H100) │  模型权重OTA   │ 轻量推理模型     │
│ 数据存储与分析   │ ────────────→ │ 实时控制逻辑     │
│ 仿真验证 (Isaac) │   ←─────────  │ 边缘数据采集     │
│ 模型管理/版本    │   数据回传     │ 异常日志上传     │
└──────────────────┘               └──────────────────┘
```

## 3. 容器化

```dockerfile
# 机器人训练环境Docker
FROM nvidia/cuda:12.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    ros2-humble-base

RUN pip install torch torchvision tensorrt

COPY train.py /workspace/
CMD ["python", "/workspace/train.py"]
```

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Cloud-Edge Robotics" | IEEE TMC 2022 | 云边协同机器人 |
| "Kubernetes Robot Fleet" | RSS 2022 | K8s机器人集群 |
