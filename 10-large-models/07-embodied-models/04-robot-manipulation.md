# 7.4 机器人操控 (Robot Manipulation)

## 目录

- [1. 引言](#1-引言)
- [2. 机器人操控概述](#2-机器人操控概述)
- [3. 操控方法](#3-操控方法)
- [4. 代表性模型](#4-代表性模型)
- [5. 抓取规划](#5-抓取规划)
- [6. 操作技能学习](#6-操作技能学习)
- [7. 灵巧操作](#7-灵巧操作)
- [8. 人机协作操控](#8-人机协作操控)
- [9. 实践练习](#9-实践练习)

---

## 1. 引言

### 1.1 机器人操控的重要性

**机器人操控**是指机器人与环境交互以改变环境状态的能力。这是具身智能的核心能力，包括抓取、放置、操作等任务。机器人操控技术是连接人工智能与物理世界的桥梁，是实现自动化、智能化生产和服务的关键。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **工业自动化** | 生产线上的装配和搬运 | 汽车零部件装配 |
| **物流仓储** | 仓库中的货物搬运 | 货架上的物品拣选 |
| **医疗护理** | 辅助医疗操作 | 手术器械传递 |
| **家庭服务** | 家庭中的日常任务 | 整理物品、烹饪辅助 |
| **危险环境** | 危险场景中的操作 | 核设施维护、排爆 |

### 1.3 发展历程

```python
class ManipulationHistory:
    """
    机器人操控发展历程
    """
    
    def __init__(self):
        self.milestones = [
            {
                'year': 1961,
                'event': 'Unimate机器人',
                'description': '世界上第一台工业机器人，用于汽车装配线',
                'impact': '开启了工业自动化时代',
            },
            {
                'year': 1980,
                'event': 'PUMA 560',
                'description': '第一台商业化6自由度机械臂',
                'impact': '推动了机器人操控技术的普及',
            },
            {
                'year': 2000,
                'event': 'Honda ASIMO',
                'description': '人形机器人，具备复杂操控能力',
                'impact': '展示了机器人技术的巨大潜力',
            },
            {
                'year': 2020,
                'event': 'OpenAI Dactyl',
                'description': '基于强化学习的灵巧手操控',
                'impact': '证明了机器学习在机器人操控中的有效性',
            },
            {
                'year': 2023,
                'event': 'Google RT-X',
                'description': '通用机器人基础模型',
                'impact': '推动了机器人操控的通用化',
            },
        ]
    
    def get_milestone(self, year):
        """获取特定年份的里程碑"""
        for milestone in self.milestones:
            if milestone['year'] == year:
                return milestone
        return None
```

---

## 2. 机器人操控概述

### 2.1 定义

**机器人操控**：机器人通过执行动作来改变环境状态的过程。

**形式化表达**：
```
Manipulation: (State, Action) → NewState
```

### 2.2 操控任务类型

| 类型 | 描述 | 示例 | 技术挑战 |
|------|------|------|----------|
| **抓取** | 抓取并保持物体 | 拿起杯子 | 姿态估计、力控制 |
| **放置** | 将物体放置到目标位置 | 放置杯子 | 精确定位、稳定性 |
| **推拉** | 推或拉物体 | 推箱子 | 摩擦力建模、力控制 |
| **旋转** | 旋转物体 | 拧开瓶盖 | 力矩控制、精度 |
| **插入** | 将物体插入孔中 | 插入钥匙 | 对准、力反馈 |
| **装配** | 组装多个物体 | 组装家具 | 多步骤协调 |

### 2.3 操控系统架构

```python
class ManipulationSystem:
    """
    机器人操控系统架构
    """
    
    def __init__(self):
        self.components = [
            {
                'name': '感知模块',
                'description': '获取环境信息',
                'components': ['视觉传感器', '深度传感器', '力传感器'],
                'output': '物体位姿、场景结构',
            },
            {
                'name': '规划模块',
                'description': '生成动作序列',
                'components': ['抓取规划器', '运动规划器', '任务规划器'],
                'output': '动作序列',
            },
            {
                'name': '控制模块',
                'description': '执行动作',
                'components': ['关节控制器', '力控制器', '视觉伺服'],
                'output': '控制指令',
            },
            {
                'name': '学习模块',
                'description': '优化操控策略',
                'components': ['模仿学习', '强化学习', '自适应控制'],
                'output': '策略更新',
            },
        ]
    
    def execute_task(self, task_description):
        """
        执行操控任务
        
        参数:
            task_description: 任务描述
        
        返回:
            result: 执行结果
        """
        # 感知
        perception = self._perceive()
        
        # 规划
        plan = self._plan(task_description, perception)
        
        # 执行
        result = self._execute(plan)
        
        # 学习
        self._learn(result)
        
        return result
    
    def _perceive(self):
        """感知环境"""
        return {'objects': [], 'scene': {}}
    
    def _plan(self, task, perception):
        """规划动作"""
        return []
    
    def _execute(self, plan):
        """执行动作"""
        return {'success': True}
    
    def _learn(self, result):
        """从结果中学习"""
        pass
```

---

## 3. 操控方法

### 3.1 基于学习的操控

**端到端学习**：从图像直接到动作。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EndToEndManipulation(nn.Module):
    """
    端到端操控模型
    """
    
    def __init__(self, image_dim=2048, action_dim=7, hidden_dim=512):
        super().__init__()
        
        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 语言编码器（用于指令）
        self.language_encoder = nn.Sequential(
            nn.Embedding(10000, 256),
            nn.LSTM(256, 256, batch_first=True),
        )
        
        # 策略网络
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim + 256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
    
    def forward(self, image_features, language_input):
        """
        前向传播
        
        参数:
            image_features: 图像特征 [batch, image_dim]
            language_input: 语言输入 [batch, seq_len]
        
        返回:
            action: 动作 [batch, action_dim]
        """
        # 编码图像
        visual_feat = self.image_encoder(image_features)
        
        # 编码语言
        lang_out, _ = self.language_encoder(language_input)
        lang_feat = lang_out[:, -1, :]  # 取最后一个时间步
        
        # 融合特征
        combined = torch.cat([visual_feat, lang_feat], dim=-1)
        
        # 预测动作
        action = self.policy(combined)
        
        return action
    
    def select_action(self, image_features, language_input, deterministic=False):
        """选择动作"""
        with torch.no_grad():
            action = self.forward(image_features, language_input)
            
            if not deterministic:
                # 添加噪声探索
                action += torch.randn_like(action) * 0.1
            
        return action
    
    def train_step(self, image_features, language_input, target_action):
        """
        训练步骤
        
        参数:
            image_features: 图像特征
            language_input: 语言输入
            target_action: 目标动作
        
        返回:
            loss: 损失值
        """
        pred_action = self.forward(image_features, language_input)
        loss = F.mse_loss(pred_action, target_action)
        
        return loss
```

### 3.2 基于模型的操控

**使用世界模型进行规划**。

```python
class ModelBasedManipulation:
    """
    基于模型的操控系统
    """
    
    def __init__(self, dynamics_model, planner, controller):
        self.dynamics_model = dynamics_model
        self.planner = planner
        self.controller = controller
        self.history = []
    
    def plan_action(self, current_state, goal_state, horizon=10):
        """
        规划动作序列
        
        参数:
            current_state: 当前状态
            goal_state: 目标状态
            horizon: 规划步数
        
        返回:
            action_sequence: 动作序列
        """
        # 使用规划器在动态模型中搜索
        action_sequence = self.planner.plan(
            self.dynamics_model,
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
            result: 执行结果
        """
        state = environment.get_state()
        trajectory = []
        
        for action in action_sequence:
            # 使用控制器执行动作
            control_signal = self.controller.compute(state, action)
            
            # 执行动作
            state, reward, done, info = environment.step(control_signal)
            
            # 记录轨迹
            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
            })
            
            if done:
                break
        
        # 更新动态模型
        self._update_dynamics(trajectory)
        
        return {
            'trajectory': trajectory,
            'success': info.get('success', False),
        }
    
    def _update_dynamics(self, trajectory):
        """更新动态模型"""
        self.dynamics_model.update(trajectory)

class DynamicsModel:
    """
    动态模型
    """
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 神经网络动态模型
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim),
        )
    
    def predict(self, state, action):
        """
        预测下一状态
        
        参数:
            state: 当前状态
            action: 动作
        
        返回:
            next_state: 下一状态预测
        """
        input_feat = torch.cat([state, action], dim=-1)
        next_state = self.model(input_feat)
        return next_state
    
    def update(self, trajectory):
        """
        更新模型
        
        参数:
            trajectory: 轨迹数据
        """
        # 简化实现
        pass
```

### 3.3 分层操控

**将复杂任务分解为子任务**。

```python
class HierarchicalManipulation:
    """
    分层操控系统
    """
    
    def __init__(self):
        self.high_level_planner = None
        self.mid_level_controller = None
        self.low_level_controller = None
        
        self.subtask_controllers = {
            'grasp': None,
            'move': None,
            'release': None,
            'rotate': None,
            'insert': None,
        }
    
    def set_planner(self, planner):
        """设置高层规划器"""
        self.high_level_planner = planner
    
    def set_controller(self, level, controller):
        """
        设置控制器
        
        参数:
            level: 控制层级 ('mid', 'low')
            controller: 控制器
        """
        if level == 'mid':
            self.mid_level_controller = controller
        elif level == 'low':
            self.low_level_controller = controller
    
    def add_subtask_controller(self, task_type, controller):
        """
        添加子任务控制器
        
        参数:
            task_type: 任务类型
            controller: 控制器
        """
        self.subtask_controllers[task_type] = controller
    
    def execute_task(self, task_description, environment):
        """
        执行任务
        
        参数:
            task_description: 任务描述
            environment: 环境
        
        返回:
            result: 执行结果
        """
        # 高层规划：分解任务
        subtasks = self.high_level_planner.plan(task_description, environment)
        
        if subtasks is None:
            return {'success': False, 'error': '规划失败'}
        
        # 执行子任务
        results = []
        for subtask in subtasks:
            task_type = subtask['type']
            params = subtask.get('parameters', {})
            
            # 检查是否有对应的控制器
            if task_type not in self.subtask_controllers:
                return {'success': False, 'error': f'未知任务类型: {task_type}'}
            
            # 执行子任务
            controller = self.subtask_controllers[task_type]
            result = controller.execute(environment, params)
            results.append(result)
            
            if not result.get('success', False):
                return {'success': False, 'error': f'子任务失败: {task_type}'}
        
        return {'success': True, 'results': results}
```

---

## 4. 代表性模型

### 4.1 RT-1 (Robotic Transformer 1)

**论文**：RT-1: Robotics Transformer for Real-World Control (Brohan et al., 2022)

**核心特点**：
- 基于Transformer架构
- 处理多模态输入（图像、语言）
- 实时推理能力
- 在真实机器人上训练

```python
class RT1Model:
    """
    RT-1 机器人Transformer模型
    """
    
    def __init__(self, config):
        self.config = config
        
        # 图像编码器
        self.image_encoder = self._build_image_encoder()
        
        # 语言编码器
        self.language_encoder = self._build_language_encoder()
        
        # Transformer编码器
        self.transformer = nn.Transformer(
            d_model=config['hidden_dim'],
            nhead=config['num_heads'],
            num_encoder_layers=config['num_layers'],
        )
        
        # 动作解码器
        self.action_decoder = nn.Sequential(
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['action_dim']),
        )
    
    def _build_image_encoder(self):
        """构建图像编码器"""
        return nn.Sequential(
            # 简化的视觉Transformer
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, self.config['hidden_dim']),
        )
    
    def _build_language_encoder(self):
        """构建语言编码器"""
        return nn.Sequential(
            nn.Embedding(self.config['vocab_size'], self.config['hidden_dim']),
            nn.LSTM(
                self.config['hidden_dim'],
                self.config['hidden_dim'],
                batch_first=True
            ),
        )
    
    def forward(self, image, instruction):
        """
        前向传播
        
        参数:
            image: 图像输入 [batch, 3, H, W]
            instruction: 语言指令 [batch, seq_len]
        
        返回:
            action: 动作 [batch, action_dim]
        """
        # 编码图像
        visual_feat = self.image_encoder(image)
        
        # 编码语言
        lang_out, _ = self.language_encoder(instruction)
        lang_feat = lang_out[:, -1, :]
        
        # 融合特征
        # 将特征转换为Transformer格式
        visual_seq = visual_feat.unsqueeze(1)  # [batch, 1, hidden_dim]
        lang_seq = lang_feat.unsqueeze(1)      # [batch, 1, hidden_dim]
        
        # Transformer处理
        combined = torch.cat([visual_seq, lang_seq], dim=1)
        transformer_out = self.transformer(combined, combined)
        
        # 解码动作
        action = self.action_decoder(transformer_out[:, 0, :])
        
        return action
```

### 4.2 BC-Z

**论文**：BC-Z: Zero-Shot Generalization to Robotic Skills (Lynch et al., 2022)

**核心特点**：
- 行为克隆方法
- 零样本泛化能力
- 多任务学习
- 数据高效

```python
class BCZModel:
    """
    BC-Z 行为克隆模型
    """
    
    def __init__(self, config):
        self.config = config
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
        )
        
        # 任务嵌入
        self.task_embedding = nn.Embedding(config['num_tasks'], 256)
        
        # 动作预测器
        self.action_predictor = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, config['action_dim']),
        )
    
    def forward(self, image, task_id):
        """
        前向传播
        
        参数:
            image: 图像输入
            task_id: 任务ID
        
        返回:
            action: 动作
        """
        # 提取视觉特征
        visual_feat = self.feature_extractor(image)
        
        # 获取任务嵌入
        task_feat = self.task_embedding(task_id)
        
        # 融合特征
        combined = torch.cat([visual_feat, task_feat], dim=-1)
        
        # 预测动作
        action = self.action_predictor(combined)
        
        return action
    
    def adapt_to_new_task(self, demonstrations, task_id, epochs=10):
        """
        适配到新任务
        
        参数:
            demonstrations: 演示数据
            task_id: 新任务ID
            epochs: 训练轮数
        """
        optimizer = torch.optim.Adam(
            list(self.task_embedding.parameters()) + 
            list(self.action_predictor.parameters()),
            lr=1e-4
        )
        
        for epoch in range(epochs):
            total_loss = 0
            
            for image, action in demonstrations:
                pred_action = self(image, task_id)
                loss = F.mse_loss(pred_action, action)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(demonstrations)
            print(f"Task {task_id} - Epoch {epoch}: Loss = {avg_loss:.4f}")
```

### 4.3 Diffusion Policy

**论文**：Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (Chi et al., 2023)

**核心特点**：
- 使用扩散模型生成动作
- 处理多模态动作分布
- 高质量动作生成
- 不确定性建模

```python
class DiffusionPolicy(nn.Module):
    """
    扩散策略模型
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
        )
        
        # 扩散模型
        self.diffusion_model = self._build_diffusion_model()
        
        # 时间步嵌入
        self.time_embedding = nn.Embedding(config['num_timesteps'], 256)
    
    def _build_diffusion_model(self):
        """构建扩散模型"""
        return nn.Sequential(
            nn.Linear(512 + 256 + self.config['action_dim'], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.config['action_dim']),
        )
    
    def forward(self, image, action, timestep):
        """
        前向传播（扩散过程）
        
        参数:
            image: 图像输入
            action: 带噪声的动作
            timestep: 时间步
        
        返回:
            noise_pred: 噪声预测
        """
        # 提取特征
        visual_feat = self.feature_extractor(image)
        
        # 获取时间步嵌入
        time_feat = self.time_embedding(timestep)
        
        # 融合特征
        combined = torch.cat([visual_feat, time_feat, action], dim=-1)
        
        # 预测噪声
        noise_pred = self.diffusion_model(combined)
        
        return noise_pred
    
    def sample_action(self, image, num_samples=1):
        """
        采样动作
        
        参数:
            image: 图像输入
            num_samples: 采样数量
        
        返回:
            actions: 采样的动作
        """
        # 初始化随机动作
        actions = torch.randn(num_samples, self.config['action_dim'])
        
        # 反向扩散过程
        for t in reversed(range(self.config['num_timesteps'])):
            timestep = torch.tensor([t] * num_samples)
            
            # 预测噪声
            noise_pred = self.forward(image.repeat(num_samples, 1, 1, 1), actions, timestep)
            
            # 更新动作（简化的DDPM采样）
            actions = actions - noise_pred * self.config['beta'][t]
        
        return actions
```

---

## 5. 抓取规划

### 5.1 抓取点检测

```python
class GraspDetector(nn.Module):
    """
    抓取点检测模型
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 骨干网络
        self.backbone = self._build_backbone()
        
        # 抓取点预测头
        self.grasp_point_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=1),  # x, y, confidence
        )
        
        # 抓取角度预测头
        self.grasp_angle_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),  # angle
        )
        
        # 抓取宽度预测头
        self.grasp_width_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),  # width
            nn.Sigmoid(),
        )
    
    def _build_backbone(self):
        """构建骨干网络"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 残差块1
            self._build_residual_block(64, 64),
            self._build_residual_block(64, 64),
            
            # 残差块2
            self._build_residual_block(64, 128, stride=2),
            self._build_residual_block(128, 128),
            
            # 残差块3
            self._build_residual_block(128, 256, stride=2),
            self._build_residual_block(256, 256),
        )
    
    def _build_residual_block(self, in_channels, out_channels, stride=1):
        """构建残差块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, image):
        """
        前向传播
        
        参数:
            image: 图像输入 [batch, 3, H, W]
        
        返回:
            grasp_params: 抓取参数
        """
        # 提取特征
        features = self.backbone(image)
        
        # 预测抓取点
        grasp_points = self.grasp_point_head(features)
        
        # 预测抓取角度
        grasp_angles = self.grasp_angle_head(features)
        
        # 预测抓取宽度
        grasp_widths = self.grasp_width_head(features) * self.config['max_width']
        
        return {
            'points': grasp_points,
            'angles': grasp_angles,
            'widths': grasp_widths,
        }
    
    def detect_grasps(self, image, top_k=5):
        """
        检测最佳抓取点
        
        参数:
            image: 图像输入
            top_k: 返回前k个最佳抓取点
        
        返回:
            grasps: 抓取点列表
        """
        with torch.no_grad():
            output = self.forward(image)
        
        # 获取置信度最高的抓取点
        confidence = output['points'][:, 2, :, :]
        flatten_conf = confidence.flatten()
        top_indices = torch.topk(flatten_conf, top_k).indices
        
        grasps = []
        for idx in top_indices:
            h = idx // confidence.size(-1)
            w = idx % confidence.size(-1)
            
            grasps.append({
                'x': w.item(),
                'y': h.item(),
                'angle': output['angles'][:, 0, h, w].item(),
                'width': output['widths'][:, 0, h, w].item(),
                'confidence': confidence[:, h, w].item(),
            })
        
        return grasps
```

### 5.2 抓取规划器

```python
class GraspPlanner:
    """
    抓取规划器
    """
    
    def __init__(self, grasp_detector, robot_arm):
        self.grasp_detector = grasp_detector
        self.robot_arm = robot_arm
        
        # 抓取质量评估器
        self.quality_evaluator = GraspQualityEvaluator()
    
    def plan_grasp(self, image, camera_intrinsics):
        """
        规划抓取
        
        参数:
            image: 图像
            camera_intrinsics: 相机内参
        
        返回:
            grasp_plan: 抓取计划
        """
        # 检测抓取点
        candidates = self.grasp_detector.detect_grasps(image, top_k=10)
        
        # 转换为3D坐标
        grasp_candidates_3d = []
        for grasp in candidates:
            # 将图像坐标转换为3D坐标
            point_3d = self._image_to_3d(
                grasp['x'], 
                grasp['y'], 
                camera_intrinsics
            )
            
            grasp_candidates_3d.append({
                **grasp,
                'point_3d': point_3d,
            })
        
        # 评估抓取质量
        evaluated = []
        for grasp in grasp_candidates_3d:
            quality = self.quality_evaluator.evaluate(
                grasp, 
                self.robot_arm.get_joint_limits()
            )
            evaluated.append({
                **grasp,
                'quality': quality,
            })
        
        # 选择最佳抓取
        best_grasp = max(evaluated, key=lambda x: x['quality'])
        
        # 生成抓取轨迹
        trajectory = self._generate_grasp_trajectory(best_grasp)
        
        return {
            'grasp_params': best_grasp,
            'trajectory': trajectory,
        }
    
    def _image_to_3d(self, x, y, camera_intrinsics):
        """将图像坐标转换为3D坐标"""
        # 简化实现
        return [x / camera_intrinsics['fx'], y / camera_intrinsics['fy'], 0.5]
    
    def _generate_grasp_trajectory(self, grasp):
        """生成抓取轨迹"""
        # 生成接近、抓取、抬起等动作
        trajectory = [
            {'type': 'move', 'target': grasp['point_3d'] + [0, 0, 0.1]},
            {'type': 'approach', 'target': grasp['point_3d']},
            {'type': 'grasp', 'width': grasp['width']},
            {'type': 'lift', 'height': 0.2},
        ]
        
        return trajectory

class GraspQualityEvaluator:
    """
    抓取质量评估器
    """
    
    def evaluate(self, grasp, joint_limits):
        """
        评估抓取质量
        
        参数:
            grasp: 抓取参数
            joint_limits: 关节限制
        
        返回:
            quality: 质量分数 (0-1)
        """
        quality = 0.0
        
        # 1. 可达性评估
        reachability = self._evaluate_reachability(grasp, joint_limits)
        quality += 0.3 * reachability
        
        # 2. 稳定性评估
        stability = self._evaluate_stability(grasp)
        quality += 0.3 * stability
        
        # 3. 碰撞风险评估
        collision_risk = self._evaluate_collision_risk(grasp)
        quality += 0.2 * (1 - collision_risk)
        
        # 4. 置信度
        quality += 0.2 * grasp['confidence']
        
        return min(quality, 1.0)
    
    def _evaluate_reachability(self, grasp, joint_limits):
        """评估可达性"""
        # 简化实现
        return 1.0
    
    def _evaluate_stability(self, grasp):
        """评估稳定性"""
        # 简化实现：宽度适中的抓取更稳定
        if 0.05 < grasp['width'] < 0.15:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_collision_risk(self, grasp):
        """评估碰撞风险"""
        # 简化实现
        return 0.1
```

---

## 6. 操作技能学习

### 6.1 模仿学习

```python
class ImitationLearningTrainer:
    """
    模仿学习训练器
    """
    
    def __init__(self, model, optimizer, data_loader):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.loss_fn = F.mse_loss
    
    def train(self, epochs=50):
        """
        训练模型
        
        参数:
            epochs: 训练轮数
        """
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in self.data_loader:
                images, instructions, actions = batch
                
                # 前向传播
                pred_actions = self.model(images, instructions)
                
                # 计算损失
                loss = self.loss_fn(pred_actions, actions)
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.data_loader)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    def evaluate(self, eval_data):
        """
        评估模型
        
        参数:
            eval_data: 评估数据
        
        返回:
            metrics: 评估指标
        """
        self.model.eval()
        
        total_loss = 0
        correct = 0
        
        with torch.no_grad():
            for images, instructions, actions in eval_data:
                pred_actions = self.model(images, instructions)
                
                loss = self.loss_fn(pred_actions, actions)
                total_loss += loss.item()
                
                # 简单的成功判断
                if torch.norm(pred_actions - actions) < 0.1:
                    correct += 1
        
        return {
            'loss': total_loss / len(eval_data),
            'accuracy': correct / len(eval_data),
        }
```

### 6.2 强化学习

```python
class RLManipulationTrainer:
    """
    强化学习操控训练器
    """
    
    def __init__(self, model, env, config):
        self.model = model
        self.env = env
        self.config = config
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        self.gamma = config['gamma']
        self.entropy_weight = config['entropy_weight']
    
    def train(self, episodes=1000):
        """
        训练模型
        
        参数:
            episodes: 训练回合数
        """
        self.model.train()
        
        for episode in range(episodes):
            state = self.env.reset()
            trajectory = []
            episode_reward = 0
            
            while True:
                # 获取观察
                image = state['image']
                instruction = state['instruction']
                
                # 选择动作
                action = self.model.select_action(image, instruction)
                
                # 执行动作
                next_state, reward, done, info = self.env.step(action)
                
                # 记录轨迹
                trajectory.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done,
                })
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # 更新策略
            loss = self._update_policy(trajectory)
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}, Loss = {loss:.4f}")
    
    def _update_policy(self, trajectory):
        """
        更新策略
        
        参数:
            trajectory: 轨迹数据
        
        返回:
            loss: 损失值
        """
        # 计算回报
        returns = self._compute_returns(trajectory)
        
        # 计算损失
        total_loss = 0
        
        for i, transition in enumerate(trajectory):
            image = transition['state']['image']
            instruction = transition['state']['instruction']
            action = transition['action']
            target_return = returns[i]
            
            # 前向传播
            pred_action = self.model(image, instruction)
            
            # 策略损失
            policy_loss = F.mse_loss(pred_action, action) * target_return
            
            total_loss += policy_loss
        
        # 更新
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def _compute_returns(self, trajectory):
        """计算折扣回报"""
        returns = []
        running_return = 0
        
        for transition in reversed(trajectory):
            running_return = transition['reward'] + self.gamma * running_return
            returns.insert(0, running_return)
        
        # 标准化
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
```

### 6.3 自我监督学习

```python
class SelfSupervisedManipulation:
    """
    自我监督操控学习
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    def train(self, data_generator, iterations=10000):
        """
        训练模型
        
        参数:
            data_generator: 数据生成器
            iterations: 训练迭代次数
        """
        self.model.train()
        
        for iteration in range(iterations):
            # 生成自我监督数据
            obs, action, next_obs = data_generator.generate()
            
            # 预测下一观察
            pred_next_obs = self.model.predict_next(obs, action)
            
            # 计算损失
            loss = F.mse_loss(pred_next_obs, next_obs)
            
            # 更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss.item():.4f}")
    
    def generate_pseudo_labels(self, unlabeled_data):
        """
        生成伪标签
        
        参数:
            unlabeled_data: 未标记数据
        
        返回:
            pseudo_labels: 伪标签
        """
        self.model.eval()
        
        pseudo_labels = []
        
        with torch.no_grad():
            for obs in unlabeled_data:
                # 预测动作
                action = self.model.predict_action(obs)
                pseudo_labels.append(action)
        
        return pseudo_labels
```

---

## 7. 灵巧操作

### 7.1 灵巧手模型

```python
class DexterousHandModel:
    """
    灵巧手模型
    """
    
    def __init__(self, num_fingers=5, num_joints_per_finger=3):
        self.num_fingers = num_fingers
        self.num_joints_per_finger = num_joints_per_finger
        self.total_joints = num_fingers * num_joints_per_finger
        
        # 手指动力学模型
        self.finger_dynamics = self._build_finger_dynamics()
        
        # 触觉传感器模型
        self.tactile_sensors = TactileSensorModel()
    
    def _build_finger_dynamics(self):
        """构建手指动力学模型"""
        dynamics = {}
        for finger in range(self.num_fingers):
            dynamics[finger] = FingerDynamics()
        return dynamics
    
    def forward_kinematics(self, joint_angles):
        """
        正向运动学
        
        参数:
            joint_angles: 关节角度
        
        返回:
            fingertip_positions: 指尖位置
        """
        positions = []
        
        for i in range(self.num_fingers):
            start_idx = i * self.num_joints_per_finger
            end_idx = start_idx + self.num_joints_per_finger
            angles = joint_angles[start_idx:end_idx]
            
            # 计算指尖位置
            pos = self.finger_dynamics[i].forward(angles)
            positions.append(pos)
        
        return positions
    
    def inverse_kinematics(self, target_positions):
        """
        反向运动学
        
        参数:
            target_positions: 目标指尖位置
        
        返回:
            joint_angles: 关节角度
        """
        angles = []
        
        for i in range(self.num_fingers):
            target_pos = target_positions[i]
            joint_angle = self.finger_dynamics[i].inverse(target_pos)
            angles.extend(joint_angle)
        
        return angles
    
    def apply_force(self, forces):
        """
        应用力
        
        参数:
            forces: 力向量
        
        返回:
            tactile_data: 触觉数据
        """
        # 计算触觉反馈
        tactile_data = self.tactile_sensors.measure(forces)
        
        return tactile_data

class FingerDynamics:
    """
    单指动力学模型
    """
    
    def __init__(self):
        self.link_lengths = [0.05, 0.05, 0.03]  # 指节长度
    
    def forward(self, angles):
        """
        正向运动学
        
        参数:
            angles: 关节角度 [3]
        
        返回:
            position: 指尖位置
        """
        # 使用DH参数计算
        x = 0.0
        y = 0.0
        z = 0.0
        
        for i in range(3):
            x += self.link_lengths[i] * torch.cos(angles[i])
            y += self.link_lengths[i] * torch.sin(angles[i])
        
        return torch.tensor([x, y, z])
    
    def inverse(self, target_pos):
        """
        反向运动学
        
        参数:
            target_pos: 目标位置
        
        返回:
            angles: 关节角度
        """
        # 简化实现
        return torch.zeros(3)
```

### 7.2 触觉反馈控制

```python
class TactileFeedbackController:
    """
    触觉反馈控制器
    """
    
    def __init__(self, hand_model):
        self.hand_model = hand_model
        
        # PID控制器参数
        self.kp = 1.0
        self.ki = 0.1
        self.kd = 0.01
        
        # 积分误差
        self.integral_error = 0.0
        self.previous_error = 0.0
    
    def control(self, target_force, current_tactile_data):
        """
        触觉反馈控制
        
        参数:
            target_force: 目标力
            current_tactile_data: 当前触觉数据
        
        返回:
            control_signal: 控制信号
        """
        # 提取当前力
        current_force = self._extract_force(current_tactile_data)
        
        # 计算误差
        error = target_force - current_force
        
        # 积分
        self.integral_error += error
        
        # 微分
        derivative = error - self.previous_error
        self.previous_error = error
        
        # PID控制
        control_signal = (
            self.kp * error +
            self.ki * self.integral_error +
            self.kd * derivative
        )
        
        return control_signal
    
    def _extract_force(self, tactile_data):
        """从触觉数据中提取力"""
        # 简化实现
        return tactile_data.get('force', 0.0)
    
    def grasp_with_feedback(self, object_stiffness=1.0):
        """
        使用触觉反馈进行抓取
        
        参数:
            object_stiffness: 物体刚度
        
        返回:
            success: 是否成功
        """
        # 目标力（根据物体刚度调整）
        target_force = 5.0 * object_stiffness
        
        # 初始化
        self.integral_error = 0.0
        self.previous_error = 0.0
        
        for step in range(100):
            # 获取触觉数据
            tactile_data = self.hand_model.get_tactile_data()
            
            # 计算控制信号
            control_signal = self.control(target_force, tactile_data)
            
            # 应用控制
            self.hand_model.apply_control(control_signal)
            
            # 检查是否稳定
            if abs(self.previous_error) < 0.1:
                return True
        
        return False
```

---

## 8. 人机协作操控

### 8.1 意图识别

```python
class IntentRecognizer(nn.Module):
    """
    人类意图识别器
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 视觉编码器
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
        )
        
        # 骨架编码器（用于人体姿态）
        self.skeleton_encoder = nn.Sequential(
            nn.Linear(3 * 25, 256),  # 25个关键点，每个3D坐标
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        
        # 意图分类器
        self.intent_classifier = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, config['num_intents']),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, image, skeleton):
        """
        前向传播
        
        参数:
            image: 图像输入
            skeleton: 人体骨架
        
        返回:
            intent_probs: 意图概率分布
        """
        # 编码图像
        visual_feat = self.visual_encoder(image)
        
        # 编码骨架
        skeleton_feat = self.skeleton_encoder(skeleton)
        
        # 融合特征
        combined = torch.cat([visual_feat, skeleton_feat], dim=-1)
        
        # 分类意图
        intent_probs = self.intent_classifier(combined)
        
        return intent_probs
    
    def recognize(self, image, skeleton):
        """
        识别意图
        
        参数:
            image: 图像
            skeleton: 骨架
        
        返回:
            intent: 识别的意图
        """
        with torch.no_grad():
            probs = self.forward(image, skeleton)
        
        # 获取最可能的意图
        intent_idx = torch.argmax(probs, dim=-1)
        
        return intent_idx.item()
```

### 8.2 协作策略

```python
class CollaborativeStrategy:
    """
    人机协作策略
    """
    
    def __init__(self, robot, intent_recognizer):
        self.robot = robot
        self.intent_recognizer = intent_recognizer
        
        # 协作模式
        self.modes = {
            'assist': self._assist_mode,
            'cooperate': self._cooperate_mode,
            'observe': self._observe_mode,
        }
    
    def collaborate(self, human_action, task_context):
        """
        执行协作
        
        参数:
            human_action: 人类动作
            task_context: 任务上下文
        
        返回:
            robot_action: 机器人动作
        """
        # 识别人类意图
        intent = self.intent_recognizer.recognize(
            human_action['image'],
            human_action['skeleton']
        )
        
        # 选择协作模式
        mode = self._select_mode(intent, task_context)
        
        # 执行协作动作
        robot_action = self.modes[mode](human_action, task_context)
        
        return robot_action
    
    def _select_mode(self, intent, task_context):
        """选择协作模式"""
        # 简化实现
        if intent == 'request_help':
            return 'assist'
        elif intent == 'working_together':
            return 'cooperate'
        else:
            return 'observe'
    
    def _assist_mode(self, human_action, task_context):
        """辅助模式"""
        # 分析人类需要帮助的地方
        target_object = self._find_target_object(human_action, task_context)
        
        return {
            'action': 'assist',
            'target': target_object,
            'parameters': {'speed': 0.5},
        }
    
    def _cooperate_mode(self, human_action, task_context):
        """协作模式"""
        # 与人类协同工作
        return {
            'action': 'cooperate',
            'target': task_context['current_object'],
            'parameters': {'role': 'partner'},
        }
    
    def _observe_mode(self, human_action, task_context):
        """观察模式"""
        # 等待并观察
        return {
            'action': 'observe',
            'parameters': {'duration': 1.0},
        }
    
    def _find_target_object(self, human_action, task_context):
        """找到目标物体"""
        # 简化实现
        return task_context.get('target_object', 'unknown')
```

---

## 9. 实践练习

### 练习1：实现抓取策略

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraspPolicy(nn.Module):
    """
    抓取策略模型
    """
    
    def __init__(self, image_dim=512, hidden_dim=256):
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
        
        # 抓取角度预测（四元数）
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
            grasp_params: 抓取参数
        """
        features = self.feature_net(image)
        
        grasp_point = self.grasp_point_head(features)
        grasp_angle = self.grasp_angle_head(features)
        grasp_angle = F.normalize(grasp_angle, dim=-1)  # 归一化四元数
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
print(f"抓取点形状: {grasp_params['grasp_point'].shape}")  # [1, 3]
print(f"抓取角度形状: {grasp_params['grasp_angle'].shape}")  # [1, 4]
print(f"抓取宽度形状: {grasp_params['grasp_width'].shape}")  # [1, 1]
```

### 练习2：实现放置策略

```python
class PlacePolicy(nn.Module):
    """
    放置策略模型
    """
    
    def __init__(self, image_dim=512, hidden_dim=256):
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
            place_params: 放置参数
        """
        features = self.feature_net(image)
        
        place_point = self.place_point_head(features)
        place_angle = self.place_angle_head(features)
        place_angle = F.normalize(place_angle, dim=-1)
        place_velocity = self.place_velocity_head(features) * 0.5  # 最大速度0.5m/s
        
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
print(f"放置点形状: {place_params['place_point'].shape}")  # [1, 3]
print(f"放置角度形状: {place_params['place_angle'].shape}")  # [1, 4]
```

### 练习3：实现完整的Pick-Place策略

```python
class PickPlacePolicy:
    """
    完整的拾取-放置策略
    """
    
    def __init__(self, grasp_policy, place_policy):
        self.grasp_policy = grasp_policy
        self.place_policy = place_policy
    
    def plan_pick_place(self, image_before, image_after):
        """
        规划Pick-Place动作序列
        
        参数:
            image_before: 抓取前的图像
            image_after: 放置后的图像（或目标场景图像）
        
        返回:
            action_sequence: 动作序列
        """
        action_sequence = []
        
        # 1. 规划抓取
        grasp_params = self.grasp_policy.plan_grasp(image_before)
        
        # 2. 移动到抓取点上方
        pre_grasp_pose = grasp_params['grasp_point'].squeeze().tolist()
        pre_grasp_pose[2] += 0.1  # 抬高10cm
        
        action_sequence.append({
            'type': 'move',
            'target': pre_grasp_pose,
            'speed': 0.2,
            'interpolate': True,
        })
        
        # 3. 移动到抓取点
        action_sequence.append({
            'type': 'move',
            'target': grasp_params['grasp_point'].squeeze().tolist(),
            'speed': 0.1,
            'interpolate': True,
        })
        
        # 4. 调整抓取角度
        action_sequence.append({
            'type': 'rotate',
            'quaternion': grasp_params['grasp_angle'].squeeze().tolist(),
        })
        
        # 5. 张开夹爪
        action_sequence.append({
            'type': 'open_gripper',
            'width': float(grasp_params['grasp_width'].squeeze().item()),
        })
        
        # 6. 执行抓取
        action_sequence.append({
            'type': 'grasp',
            'force': 10.0,  # 牛顿
        })
        
        # 7. 抬起物体
        lift_pose = grasp_params['grasp_point'].squeeze().tolist()
        lift_pose[2] += 0.2  # 抬起20cm
        
        action_sequence.append({
            'type': 'move',
            'target': lift_pose,
            'speed': 0.15,
            'interpolate': True,
        })
        
        # 8. 规划放置
        place_params = self.place_policy(image_after, grasp_params)
        
        # 9. 移动到放置点上方
        pre_place_pose = place_params['place_point'].squeeze().tolist()
        pre_place_pose[2] += 0.15
        
        action_sequence.append({
            'type': 'move',
            'target': pre_place_pose,
            'speed': 0.2,
            'interpolate': True,
        })
        
        # 10. 调整放置角度
        action_sequence.append({
            'type': 'rotate',
            'quaternion': place_params['place_angle'].squeeze().tolist(),
        })
        
        # 11. 移动到放置点
        action_sequence.append({
            'type': 'move',
            'target': place_params['place_point'].squeeze().tolist(),
            'speed': 0.1,
            'interpolate': True,
            'velocity': float(place_params['place_velocity'].squeeze().item()),
        })
        
        # 12. 释放物体
        action_sequence.append({
            'type': 'release',
            'speed': 0.05,
        })
        
        # 13. 退避
        retreat_pose = place_params['place_point'].squeeze().tolist()
        retreat_pose[2] += 0.2
        
        action_sequence.append({
            'type': 'move',
            'target': retreat_pose,
            'speed': 0.2,
            'interpolate': True,
        })
        
        return action_sequence
    
    def execute(self, environment, action_sequence):
        """
        执行动作序列
        
        参数:
            environment: 环境接口
            action_sequence: 动作序列
        
        返回:
            success: 是否成功
        """
        for i, action in enumerate(action_sequence):
            print(f"执行动作 {i+1}/{len(action_sequence)}: {action['type']}")
            
            try:
                if action['type'] == 'move':
                    environment.move_to(
                        action['target'],
                        speed=action.get('speed', 0.1),
                        interpolate=action.get('interpolate', False)
                    )
                
                elif action['type'] == 'rotate':
                    environment.rotate(action['quaternion'])
                
                elif action['type'] == 'open_gripper':
                    environment.open_gripper(action['width'])
                
                elif action['type'] == 'grasp':
                    environment.grasp(action['force'])
                
                elif action['type'] == 'release':
                    environment.release(action.get('speed', 0.05))
                
                else:
                    print(f"未知动作类型: {action['type']}")
                    return False
            
            except Exception as e:
                print(f"执行动作失败: {e}")
                return False
        
        return True

# 测试
grasp_policy = GraspPolicy(image_dim=512)
place_policy = PlacePolicy(image_dim=512)
pick_place_policy = PickPlacePolicy(grasp_policy, place_policy)

# 生成测试图像
image_before = torch.randn(1, 512)
image_after = torch.randn(1, 512)

# 规划动作序列
action_sequence = pick_place_policy.plan_pick_place(image_before, image_after)
print(f"\n生成的动作序列包含 {len(action_sequence)} 个动作:")
for i, action in enumerate(action_sequence):
    print(f"{i+1}. {action['type']}")

# 模拟执行
class MockEnvironment:
    def move_to(self, target, speed=0.1, interpolate=False):
        print(f"  移动到: {target[:2]} (z={target[2]:.2f}m)")
    
    def rotate(self, quaternion):
        print(f"  旋转到: {quaternion}")
    
    def open_gripper(self, width):
        print(f"  张开夹爪: {width:.2f}m")
    
    def grasp(self, force):
        print(f"  抓取力: {force}N")
    
    def release(self, speed):
        print(f"  释放物体")

environment = MockEnvironment()
print("\n执行动作序列:")
success = pick_place_policy.execute(environment, action_sequence)
print(f"\n执行结果: {'成功' if success else '失败'}")
```

---

## 参考文献

1. Brohan, A., et al. (2022). RT-1: Robotics Transformer for Real-World Control.
2. Lynch, C., et al. (2022). BC-Z: Zero-Shot Generalization to Robotic Skills.
3. Chi, J., et al. (2023). Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.
4. Levine, S., et al. (2016). End-to-End Training of Deep Visuomotor Policies.
5. OpenAI. (2020). Learning Dexterous In-Hand Manipulation.
6. Toussaint, M. (2009). Robot Trajectory Optimization using Approximate Inference.
7. Khatib, O. (1986). Real-Time Obstacle Avoidance for Manipulators and Mobile Robots.