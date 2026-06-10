# 5.1 HPC集群架构

## 1. HPC集群组成

```
                 登录/管理节点
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼───┐ ┌────▼───┐ ┌────▼───┐
    │计算节点1│ │计算节点2│ │计算节点N│
    │8×A100  │ │8×A100  │ │8×A100  │
    │512GB   │ │512GB   │ │512GB   │
    │80核CPU │ │80核CPU │ │80核CPU │
    └────┬───┘ └────┬───┘ └────┬───┘
         │          │          │
    ┌────▼──────────▼──────────▼────┐
    │   InfiniBand (400Gbps)        │
    └────────────┬──────────────────┘
                 │
         ┌───────▼────────┐
         │  Lustre并行文件  │
         │   ~100 PB      │
         │   ~1 TB/s      │
         └────────────────┘
```

## 2. 在机器人训练中的应用

| 场景 | GPU需求 | 内存 | 时间(单卡) | 时间(集群) |
|------|---------|------|-----------|-----------|
| YOLOv8训练 | 1×A100 | 80GB | 7天 | — |
| ViT-L训练 | 8×A100 | 640GB | 30天 | 4天 |
| VLA 7B训练 | 64×A100 | 5TB | 60天 | 1天 |
| 大规模仿真 | 32×A100 | 2.5TB | — | 实时并行 |

## 3. Slurm作业调度

```bash
# Slurm作业脚本示例
#!/bin/bash
#SBATCH --job-name=robot_train
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00

srun python train_vla.py --config config.yaml
```

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "DGX Supercomputer Architecture" | NVIDIA 2020 | DGX集群 |
| "Slurm Resource Management" | Yoo et al. 2003 | Slurm调度 |
