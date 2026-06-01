# 7.4 机器人操控

## 目录

- [1. 引言](#1-引言)
- [2. 机器人操控概述](#2-机器人操控概述)
- [3. 操控方法](#3-操控方法)
- [4. 代表性模型](#4-代表性模型)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 机器人操控的重要性

**机器人操控**是指机器人与环境交互以改变环境状态的能力。这是具身智能的核心能力，包括抓取、放置、操作等任务。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **抓取** | 抓取物体 | 拿起杯子 |
| **放置** | 放置物体 | 把杯子放在桌子上 |
| **操作** | 操作物体 | 打开门、拧螺丝 |
| **装配** | 组装物体 | 组装家具 |

---

## 2. 机器人操控概述

### 2.1 定义

**机器人操控**：机器人通过执行动作来改变环境状态的过程。

**形式化表达**：
```
Manipulation: (State, Action) → NewState
```

### 2.2 操控任务类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **抓取** | 抓取并保持物体 | 拿起杯子 |
| **放置** | 将物体放置到目标位置 | 放置杯子 |
| **推拉** | 推或拉物体 | 推箱子 |
| **旋转** | 旋转物体 | 拧开瓶盖 |
| **插入** | 将物体插入孔中 | 插入钥匙 |

---

## 3. 操控方法

### 3.1 基于学习的操控

**端到端学习**：从图像直接到动作。

```python
import torch
import torch.nn as nn

class EndToEndManipulation(nn.Module):
    def __init__(self, image_dim, action_dim, hidden_dim=512):
        super().__init__()
        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 策略网络
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, image):
        """
        前向传播
        
        参数:
            image: 图像特征 [batch, image_dim]
        
        返回:
            动作 [batch, action_dim]
        """
        features = self.image_encoder(image)
        action = self.policy(features)
        return action
    
    def select_action(self, image):
        """选择动作"""
        with torch.no_grad():
            action = self.forward(image)
        return action
```

### 3.2 基于模型的操控

**使用世界模型进行规划**。

```python
class ModelBasedManipulation:
    def __init__(self, dynamics_model, planner):
        self.dynamics = dynamics_model
        self.planner = planner
    
    def plan_action(self, current_state, goal_state, horizon=10):
        """
        规划动作序列
        
        参数:
            current_state: 当前状态
            goal_state: 目标状态
            horizon: 规划步数
        
        返回:
            动作序列
        """
        # 使用规划器在动态模型中搜索
        action_sequence = self.planner.plan(
            self.dynamics,
            current_state,
            goal_state,
            horizon
        )
        
        return action_sequence
    
    def execute(self, environment, action_sequence):
        """
        执行动作序列
        
        参数:
            environment: 环境
            action_sequence: 动作序列
        
        返回:
            执行结果
        """
        state = environment.get_state()
        
        for action in action_sequence:
            state, reward, done, _ = environment.step(action)
            
            if done:
                break
        
        return state, reward
```

### 3.3 分层操控

**将复杂任务分解为子任务**。

```python
class HierarchicalManipulation:
    def __init__(self):
        self.high_level_planner = None
        self.low_level_controllers = {}
    
    def set_high_level_planner(self, planner):
        """设置高层规划器"""
        self.high_level_planner = planner
    
    def add_low_level_controller(self, task, controller):
        """
        添加低层控制器
        
        参数:
            task: 任务类型
            controller: 控制器
        """
        self.low_level_controllers[task] = controller
    
    def execute_task(self, task, environment):
        """
        执行任务
        
        参数:
            task: 任务描述
            environment: 环境
        
        返回:
            执行结果
        """
        # 高层规划
        subtasks = self.high_level_planner.plan(task, environment)
        
        # 执行子任务
        for subtask in subtasks:
            if subtask in self.low_level_controllers:
                controller = self.low_level_controllers[subtask]
                controller.execute(environment)
            else:
                print(f"未知子任务: {subtask}")
        
        return True
```

---

## 4. 代表性模型

### 4.1 RT-1 (Robotic Transformer 1)

**论文**：RT-1: Robotics Transformer for Real-World Control (Brohan et al., 2022)

**核心特点**：
- 基于Transformer的机器人控制
- 处理多模态输入
- 实时推理

### 4.2 BC-Z

**论文**：BC-Z: Zero-Shot Generalization to Robotic Skills (Lynch et al., 2022)

**核心特点**：
- 行为克隆方法
- 零样本泛化
- 多任务学习

### 4.3 Diffusion Policy

**论文**：Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (Chi et al., 2023)

**核心特点**：
- 使用扩散模型生成动作
- 处理多模态动作分布
- 高质量动作生成

---

## 5. 实践练习

### 练习1：实现抓取策略

```python
import torch
import torch.nn as nn

class GraspPolicy(nn.Module):
    def __init__(self, image_dim, hidden_dim=256):
        super().__init__()
        # 特征提取
        self.feature_net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 抓取点预测
        self.grasp_point_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # x, y, z
        )
        
        # 抓取角度预测
        self.grasp_angle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # quaternion
        )
        
        # 抓取宽度预测
        self.grasp_width_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image):
        """
        前向传播
        
        参数:
            image: 图像特征 [batch, image_dim]
        
        返回:
            抓取参数
        """
        features = self.feature_net(image)
        
        grasp_point = self.grasp_point_head(features)
        grasp_angle = self.grasp_angle_head(features)
        grasp_angle = F.normalize(grasp_angle, dim=-1)
        grasp_width = self.grasp_width_head(features) * 0.1  # 最大宽度0.1米
        
        return {
            'grasp_point': grasp_point,
            'grasp_angle': grasp_angle,
            'grasp_width': grasp_width
        }
    
    def plan_grasp(self, image):
        """规划抓取"""
        with torch.no_grad():
            grasp_params = self.forward(image)
        
        return grasp_params

# 测试
policy = GraspPolicy(image_dim=512)
image = torch.randn(1, 512)

grasp_params = policy.plan_grasp(image)
print(f"抓取点: {grasp_params['grasp_point'].shape}")  # [1, 3]
print(f"抓取角度: {grasp_params['grasp_angle'].shape}")  # [1, 4]
print(f"抓取宽度: {grasp_params['grasp_width'].shape}")  # [1, 1]
```

### 练习2：实现放置策略

```python
class PlacePolicy(nn.Module):
    def __init__(self, image_dim, hidden_dim=256):
        super().__init__()
        # 特征提取
        self.feature_net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 放置点预测
        self.place_point_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        
        # 放置角度预测
        self.place_angle_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
        # 放置速度预测
        self.place_velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, grasp_params):
        """
        前向传播
        
        参数:
            image: 图像特征
            grasp_params: 抓取参数
        
        返回:
            放置参数
        """
        features = self.feature_net(image)
        
        place_point = self.place_point_head(features)
        place_angle = self.place_angle_head(features)
        place_angle = F.normalize(place_angle, dim=-1)
        place_velocity = self.place_velocity_head(features) * 0.5
        
        return {
            'place_point': place_point,
            'place_angle': place_angle,
            'place_velocity': place_velocity
        }

# 测试
place_policy = PlacePolicy(image_dim=512)
image = torch.randn(1, 512)
grasp_params = {
    'grasp_point': torch.randn(1, 3),
    'grasp_angle': torch.randn(1, 4),
    'grasp_width': torch.randn(1, 1)
}

place_params = place_policy(image, grasp_params)
print(f"放置点: {place_params['place_point'].shape}")  # [1, 3]
```

### 练习3：实现完整的Pick-Place策略

```python
class PickPlacePolicy:
    def __init__(self, grasp_policy, place_policy):
        self.grasp_policy = grasp_policy
        self.place_policy = place_policy
    
    def plan_pick_place(self, image_before, image_after):
        """
        规划Pick-Place动作
        
        参数:
            image_before: 抓取前的图像
            image_after: 放置后的图像
        
        返回:
            完整的动作序列
        """
        action_sequence = []
        
        # 1. 规划抓取
        grasp_params = self.grasp_policy.plan_grasp(image_before)
        
        # 2. 移动到抓取点
        action_sequence.append({
            'type': 'move',
            'target': grasp_params['grasp_point'].squeeze().tolist(),
            'angle': grasp_params['grasp_angle'].squeeze().tolist()
        })
        
        # 3. 执行抓取
        action_sequence.append({
            'type': 'grasp',
            'width': grasp_params['grasp_width'].item()
        })
        
        # 4. 规划放置
        place_params = self.place_policy(image_after, grasp_params)
        
        # 5. 移动到放置点
        action_sequence.append({
            'type': 'move',
            'target': place_params['place_point'].squeeze().tolist(),
            'angle': place_params['place_angle'].squeeze().tolist()
        })
        
        # 6. 执行放置
        action_sequence.append({
            'type': 'release',
            'velocity': place_params['place_velocity'].item()
        })
        
        return action_sequence
    
    def execute(self, environment, action_sequence):
        """
        执行动作序列
        
        参数:
            environment: 环境
            action_sequence: 动作序列
        
        返回:
            执行结果
        """
        for action in action_sequence:
            if action['type'] == 'move':
                environment.move_to(
                    action['target'],
                    action['angle']
                )
            elif action['type'] == 'grasp':
                environment.grasp(action['width'])
            elif action['type'] == 'release':
                environment.release(action['velocity'])
        
        return True

# 测试
grasp_policy = GraspPolicy(image_dim=512)
place_policy = PlacePolicy(image_dim=512)
pick_place_policy = PickPlacePolicy(grasp_policy, place_policy)

image_before = torch.randn(1, 512)
image_after = torch.randn(1, 512)

action_sequence = pick_place_policy.plan_pick_place(image_before, image_after)
print(f"动作序列:")
for i, action in enumerate(action_sequence):
    print(f"{i+1}. {action['type']}")
```

---

**下一节**：[具身规划](05-embodied-planning.md)

---

## 参考文献

1. Brohan, A., et al. (2022). RT-1: Robotics Transformer for Real-World Control.
2. Lynch, C., et al. (2022). BC-Z: Zero-Shot Generalization to Robotic Skills.
3. Chi, J., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.