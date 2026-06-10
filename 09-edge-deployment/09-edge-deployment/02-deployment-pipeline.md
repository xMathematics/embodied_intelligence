# 9.2 端到端部署流程

## 1. 标准部署管线

```
训练(PyTorch) → 导出(ONNX) → 优化(TensorRT) → 部署(ROS2 Node)
                                                      │
                                                      ▼
                                              验证(延迟+精度)
```

## 2. 详细步骤

### Step 1: 训练与导出
```python
import torch
model = YOLOv8m()
model.train(data='coco.yaml', epochs=100)

# 导出ONNX (动态batch支持)
torch.onnx.export(model, dummy_input, "model.onnx",
                  opset_version=17,
                  dynamic_axes={'input': {0: 'batch', 2: 'h', 3: 'w'}})
```

### Step 2: TensorRT优化
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine \
        --fp16 --workspace=1024
```

### Step 3: ROS2推理节点
```cpp
class PerceptionNode : public rclcpp::Node {
    // 加载TensorRT engine
    // 订阅/camera/image_raw
    // 预处理→推理→后处理
    // 发布/detections
};
```

### Step 4: 系统启动
```bash
ros2 launch robot_bringup robot.launch.py
ros2 topic delay /detections --window 100  # 测量延迟
```

## 3. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "End-to-End Deployment on Edge Robots" | ICRA 2022 | 端到端部署流程 |
| "ROS2 Integration with TensorRT" | IROS 2021 | ROS2+TensorRT |
