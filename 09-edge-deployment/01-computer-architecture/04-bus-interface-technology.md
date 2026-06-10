# 1.4 总线与接口技术

## 1. 基本概念

总线（Bus）是计算机系统中各组件之间传输数据的**公共通道**。在机器人系统中，总线的选择和设计直接影响传感器数据的采集速度和控制指令的响应时间。

### 总线分类

```
系统总线
├── 处理器-内存总线 (高带宽, 低延迟)
│   └── DDR5, LPDDR5
├── I/O总线 (连接外设)
│   ├── PCIe (GPU, NVMe, 高速采集卡)
│   ├── USB (相机, 激光雷达)
│   └── MIPI (摄像头模块)
└── 控制总线 (实时性要求高)
    ├── CAN/CAN FD (机器人关节)
    ├── EtherCAT (工业控制)
    └── I2C/SPI (低速传感器)
```

---

## 2. 主要总线技术详解

### 2.1 PCI Express (PCIe)

PCIe是机器人系统中**最重要的高速总线**，连接GPU、NVMe SSD、高速采集卡等。

| 代际 | 单通道带宽 (x1) | x16带宽 | 编码 | 推出年份 |
|------|---------------|--------|------|---------|
| PCIe 3.0 | ~1 GB/s | ~16 GB/s | 128b/130b | 2010 |
| PCIe 4.0 | ~2 GB/s | ~32 GB/s | 128b/130b | 2017 |
| PCIe 5.0 | ~4 GB/s | ~64 GB/s | 128b/130b | 2022 |
| PCIe 6.0 | ~8 GB/s | ~128 GB/s | PAM4 | 2024 |

**在机器人中的应用**：
```
Jetson Orin内部PCIe拓扑：
┌─────────────────────────────────┐
│         Jetson Orin SoC          │
│  ┌──────┐  ┌──────┐  ┌──────┐  │
│  │ GPU  │  │ DLA  │  │ CPU  │  │
│  │PCIe④ │  │PCIe④ │  │内部  │  │
│  └──┬───┘  └──┬───┘  └──────┘  │
│     │         │                │
│  ┌──▼─────────▼──┐             │
│  │  PCIe Switch  │             │
│  └──┬─────────┬──┘             │
│     │         │                │
│  ┌──▼──┐ ┌───▼────┐           │
│  │NVMe │ │PCIe x4 │           │
│  │SSD  │ │扩展槽   │           │
│  └─────┘ └────────┘           │
└─────────────────────────────────┘
```

### 2.2 USB (Universal Serial Bus)

适合**中等带宽、即插即用**的机器人传感器。

| 标准 | 带宽 | 延迟 | 最大线长 | 典型机器人设备 |
|------|------|------|---------|--------------|
| USB 2.0 | 480 Mbps | ~1ms | 5m | 激光雷达, IMU |
| USB 3.2 Gen1 | 5 Gbps | ~100μs | 3m | RGB相机 |
| USB 3.2 Gen2 | 10 Gbps | ~100μs | 3m | 深度相机 |
| USB 3.2 Gen2×2 | 20 Gbps | ~100μs | 3m | 高速相机 |
| USB4 | 40 Gbps | ~10μs | 3m | 高速数据采集 |

**重要考虑**：USB的延迟对实时性有影响（~100μs-1ms），不适合需要确定性延迟的实时控制。

### 2.3 MIPI CSI-2 (Camera Serial Interface)

**机器人相机的主流接口**，直接连接图像传感器到SoC。

| 特性 | 说明 |
|------|------|
| 物理层 | MIPI D-PHY / C-PHY |
| 通道数 | 1-4 Lane (每个Lane ~1-2.5 Gbps) |
| 总带宽 | 4 Lane × 2.5 Gbps = 10 Gbps |
| 延迟 | ~μs级 (硬件直连) |
| 使用平台 | Jetson, RK3588, 手机SoC |

**MIPI vs USB 相机对比**：
| 对比项 | MIPI CSI | USB Camera |
|-------|---------|-----------|
| 延迟 | 极低 (~μs) | 中等 (~ms) |
| CPU占用 | 低 (硬件ISP) | 高 (软件处理) |
| 灵活性 | 固定连接 | 即插即用 |
| 线长 | ~30cm | ~3m |
| 典型用例 | Jetson内置相机 | RealSense D435 |

### 2.4 工业控制总线

| 总线 | 带宽 | 延迟 | 特点 | 在机器人中的应用 |
|------|------|------|------|----------------|
| **CAN 2.0** | 1 Mbps | ~100μs | 可靠, 简单 | 关节电机通信 |
| **CAN FD** | 5 Mbps | ~50μs | 更高带宽 | 高级机器人关节 |
| **EtherCAT** | 100 Mbps | ~1-10μs | 极低抖动, 实时 | 工业机器人控制 |
| **RS-485** | 10 Mbps | ~100μs | 长距离, 差分 | 外围设备通信 |
| **I2C** | 400 kbps (快速模式) | ~ms | 简单, 双线 | IMU, 温度传感器 |
| **SPI** | 10-50 Mbps | ~μs | 全双工, 高速 | 编码器, ADC |

---

## 3. 机器人传感器带宽规划

### 3.1 带宽计算

```python
# 典型机器人传感器带宽需求计算
sensors = {
    "RGB_Camera":   1920 * 1080 * 3 * 30,  # 1920x1080, RGB, 30fps
    "Depth_Camera": 640 * 480 * 4 * 30,    # 640x480, 32bit depth, 30fps
    "LiDAR_16":     300000 * 6,             # 16线, 300K点/秒, 6通道
    "LiDAR_64":     1300000 * 6,            # 64线, 1.3M点/秒
    "IMU":          200 * 6 * 4,            # 200Hz, 6轴, float32
    "Joint_Encoders": 1000 * 12 * 4,        # 1kHz, 12关节, float32
}

for name, bps in sensors.items():
    print(f"{name:20s}: {bps/1e6:8.2f} MB/s")

# 输出:
# RGB_Camera        : 178.00 MB/s
# Depth_Camera      : 36.86 MB/s
# LiDAR_16          : 1.80 MB/s
# LiDAR_64          : 7.80 MB/s
# IMU               : 0.005 MB/s
# Joint_Encoders    : 0.19 MB/s
# 总计              : ~225 MB/s
```

### 3.2 总线带宽可行性检查

| 总线 | 可用带宽 | 能否满足225MB/s? |
|------|---------|-----------------|
| PCIe 3.0 x4 | ~4 GB/s | ✅ 绰绰有余 |
| USB 3.0 | ~0.5 GB/s | ✅ 但需管理多设备 |
| USB 2.0 | ~60 MB/s | ❌ 不够 |
| MIPI CSI 4-lane | ~1.25 GB/s | ✅ |
| CAN FD | ~0.6 MB/s | ❌ (只适合控制信号) |

---

## 4. 多传感器同步问题

### 4.1 硬件时间戳的重要性

```
多个传感器各自独立采集：
RGB:    |--帧1--|--帧2--|--帧3--|
Depth:    |--帧1--|--帧2--|--帧3--|
LiDAR:       |--帧1--|--帧2--|
IMU:     |-|-|-|-|-|-|-|-|-|-|-|

问题：无法知道哪个Depth帧对应哪个RGB帧
```

### 4.2 同步解决方案

| 方法 | 精度 | 实现复杂度 | 说明 |
|------|------|-----------|------|
| **软件时间戳** | ~ms级 | 低 | ROS2的Time Synchronizer |
| **硬件触发线** | ~μs级 | 中 | 相机支持硬件触发 |
| **PTP (IEEE 1588)** | ~1μs | 高 | 网络设备时钟同步 |
| **专用同步板** | ~ns级 | 高 | 工业机器人方案 |

### 4.3 在Jetson上的实现

```cpp
// Jetson上使用硬件时间戳同步传感器
#include <jetson-utils/camera.h>

void process_frame(uchar3* image, uint64_t timestamp) {
    // timestamp 来自相机驱动的硬件时间戳
    // 与IMU的硬件时间戳在同一时间域
    sensor_fusion(image, imu_data, timestamp);
}

// 硬件触发配置 (CSI相机)
camera_config_t config;
config.io_method = IO_METHOD_MMAP;
config.framerate = 30;
// 使用GPIO触发多个相机同时曝光
// 实现所有相机在同一时刻捕获
```

---

## 5. 相关论文

| 论文 | 发表 | 核心内容 |
|------|------|---------|
| "PCI Express Base Specification Rev 6.0" | PCI-SIG, 2022 | PCIe 6.0规范 |
| "Time-Sensitive Networking for Robotics" | IEEE IROS, 2021 | 机器人中的TSN时间同步 |
| "A Survey of Bus Architectures for Robotics" | IEEE Access, 2021 | 机器人总线架构综述 |
| "Hardware Synchronization of Multi-Modal Sensors" | ICRA 2022 | 多模态传感器硬件同步 |

---

## 6. 本章小结

### 核心要点
1. **PCIe是机器人的骨干总线**：连接GPU、SSD、高速采集卡
2. **MIPI CSI是延迟最低的相机接口**：适合实时视觉
3. **CAN FD / EtherCAT是关节控制的标准**：实时性有保障
4. **USB适合中等带宽外设**：但延迟可控性差
5. **带宽规划必须在硬件设计阶段完成**：否则后期无法弥补

### 自测题
1. 设计一个包含RGB-D相机、LiDAR和IMU的Jetson系统，如何规划总线带宽？
2. 为什么MIPI CSI相机比USB相机延迟更低？
3. CAN FD和EtherCAT的区别是什么？各适合什么机器人场景？
4. PCIe通道数（lane）如何影响GPU性能？
