# 9.4 功耗优化与管理

## 1. 功耗优化策略

| 策略 | 效果 | 实现 |
|------|------|------|
| DVFS | 降低20-40% | 动态频率电压缩放 |
| GPU频率限制 | 降低10-30% | nvpmodel |
| 核心关停 | 降低5-15% | 关停未用核心 |
| 推理调度 | 降低10-20% | 非高峰降低频率 |
| 模型切换 | 降低30-50% | 轻量模型 |

## 2. Jetson功耗配置

```bash
# 查看功耗模式
sudo nvpmodel -q
# 设置15W模式
sudo nvpmodel -m 8   # MAXN
# 设置7.5W节能模式
sudo nvpmodel -m 1

# 手动控制
sudo jetson_clocks --fan       # 最大性能
sudo jetson_clocks --show      # 当前状态
```

## 3. 机器人续航估算

| 机器人 | 电池 | 典型功耗 | 续航 |
|-------|------|---------|------|
| 四足机器人 | 500Wh | 30-80W | 6-16h |
| 无人机 | 100Wh | 50-100W | 1-2h |
| 轮式服务 | 1000Wh | 20-50W | 20-50h |

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Power-Aware Inference Scheduling" | ICRA 2022 | 功耗感知调度 |
| "Energy-Efficient Edge AI for Robots" | IEEE RAM 2023 | 能效边缘AI综述 |
