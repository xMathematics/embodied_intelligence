# 6.5 模型预测控制

## 目录

- [1. 引言](#1-引言)
- [2. 模型预测控制概述](#2-模型预测控制概述)
- [3. MPC方法](#3-mpc方法)
- [4. MPC算法](#4-mpc算法)
- [5. MPC评估指标](#5-mpc评估指标)
- [6. 优化策略](#6-优化策略)
- [7. 工程实践](#7-工程实践)
- [8. 理论基础与分析](#8-理论基础与分析)
- [9. 前沿研究方向](#9-前沿研究方向)
- [10. 实践练习](#10-实践练习)

---

## 1. 引言

### 1.1 模型预测控制的重要性

**模型预测控制**（Model Predictive Control, MPC）是一种先进的控制方法，利用系统模型进行滚动时域优化来选择控制动作。MPC在机器人控制、自动驾驶、工业过程控制等领域有广泛应用。

与传统控制方法相比，MPC具有以下优势：
- 能够处理复杂的约束条件
- 可以显式考虑未来的预测
- 对模型误差具有一定的鲁棒性

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **机器人控制** | 机器人运动规划和控制 | 机械臂轨迹跟踪、移动机器人导航 |
| **自动驾驶** | 车辆轨迹规划和控制 | 车道保持、自动泊车 |
| **过程控制** | 工业过程优化控制 | 化工过程控制、温度控制 |
| **能源管理** | 能源系统优化 | 电网调度、智能建筑能源管理 |
| **航空航天** | 飞行器控制 | 无人机导航、火箭姿态控制 |
| **医疗设备** | 医疗系统控制 | 呼吸机控制、药物输注系统 |

### 1.3 MPC与其他控制方法的对比

| 特性 | MPC | PID控制 | LQR控制 |
|------|-----|---------|---------|
| **约束处理** | 显式处理 | 间接处理 | 有限处理 |
| **预测能力** | 多步预测 | 无 | 单步 |
| **非线性系统** | 良好处理 | 困难 | 困难 |
| **计算复杂度** | 高 | 低 | 中等 |
| **实时性** | 受限 | 良好 | 良好 |

---

## 2. 模型预测控制概述

### 2.1 定义

**模型预测控制**：在每个控制时刻，基于系统模型预测未来行为，优化控制序列，只执行第一个动作。

**核心思想**：
```
在时刻t：
1. 预测未来H步的系统行为
2. 优化控制序列使目标函数最小
3. 只执行第一个控制动作
4. 在t+1时刻重复上述过程
```

### 2.2 MPC的特点

| 特点 | 描述 | 优势 |
|------|------|------|
| **前瞻性** | 考虑未来预测 | 更好的长期规划 |
| **滚动优化** | 实时重新规划 | 适应环境变化 |
| **约束处理** | 自然处理系统约束 | 安全性保障 |
| **鲁棒性** | 对模型误差有一定鲁棒性 | 实际应用可靠 |
| **灵活性** | 可定制代价函数 | 适应不同任务 |

### 2.3 MPC的基本要素

一个完整的MPC系统包含以下要素：

1. **预测模型**：描述系统动态的数学模型
2. **代价函数**：评价控制效果的目标函数
3. **约束条件**：状态和控制输入的限制
4. **优化算法**：求解最优控制序列的方法
5. **滚动时域**：预测和优化的时间范围

### 2.4 MPC的数学表述

MPC问题可以形式化为以下优化问题：

```
minimize J = sum_{k=0}^{H-1} L(x_{t+k}, u_{t+k}) + F(x_{t+H})

subject to:
    x_{t+k+1} = f(x_{t+k}, u_{t+k})    (系统动态)
    x_{t+k} ∈ X                         (状态约束)
    u_{t+k} ∈ U                         (控制约束)
    x_t = 当前状态
```

其中：
- $H$：预测时域（Horizon）
- $L$：阶段代价函数
- $F$：终端代价函数
- $X$：状态约束集合
- $U$：控制约束集合

---

## 3. MPC方法

### 3.1 收缩约束MPC

**方法**：在滚动视窗内强制状态收缩，确保系统稳定性。

```python
def mpc_with_contraction_constraint(
    dynamics_model,
    initial_state,
    goal_state,
    horizon=10,
    num_iterations=50,
    lr=0.1
):
    """
    带收缩约束的MPC
    
    参数:
        dynamics_model: 动态模型
        initial_state: 初始状态 [batch, state_dim]
        goal_state: 目标状态 [batch, state_dim]
        horizon: 预测步数
        num_iterations: 优化迭代次数
        lr: 学习率
    
    返回:
        优化的控制序列
    """
    state_dim = initial_state.shape[-1]
    action_dim = 2  # 假设动作维度为2
    
    # 初始化控制序列
    actions = [torch.randn(initial_state.shape[0], action_dim, requires_grad=True) * 0.1 
               for _ in range(horizon)]
    
    optimizer = torch.optim.Adam(actions, lr=lr)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # 预测轨迹
        states = [initial_state]
        current_state = initial_state
        
        for t in range(horizon):
            next_state = dynamics_model(current_state, actions[t])
            states.append(next_state)
            current_state = next_state
        
        states = torch.stack(states)
        
        # 计算代价
        goal_loss = torch.mean((states - goal_state.unsqueeze(0)) ** 2)
        action_loss = sum(torch.mean(a ** 2) for a in actions)
        
        # 收缩约束惩罚
        contraction_loss = 0.0
        for t in range(horizon):
            dist_to_goal_t = torch.norm(states[t] - goal_state, dim=-1)
            dist_to_goal_t1 = torch.norm(states[t+1] - goal_state, dim=-1)
            # 强制距离收缩
            contraction_violation = torch.relu(dist_to_goal_t1 - 0.95 * dist_to_goal_t)
            contraction_loss += torch.mean(contraction_violation)
        
        total_loss = goal_loss + 0.01 * action_loss + 1.0 * contraction_loss
        total_loss.backward()
        optimizer.step()
        
        # 动作约束
        for t in range(horizon):
            actions[t].data.clamp_(-1, 1)
    
    return [a.detach() for a in actions]

# 示例动态模型
class SimpleDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        combined = torch.cat([state, action], dim=-1)
        return state + 0.1 * self.net(combined)

# 使用示例
dynamics = SimpleDynamics(state_dim=4, action_dim=2)
initial_state = torch.randn(32, 4)
goal_state = torch.zeros(32, 4)

actions = mpc_with_contraction_constraint(dynamics, initial_state, goal_state)
print(f"优化的动作序列长度: {len(actions)}")
print(f"第一个动作形状: {actions[0].shape}")  # [32, 2]
```

### 3.2 随机MPC

**方法**：处理系统不确定性和噪声，使用随机优化方法。

```python
class StochasticMPC:
    def __init__(self, dynamics_model, cost_fn, num_samples=10, device='cpu'):
        """
        随机MPC控制器
        
        参数:
            dynamics_model: 动态模型
            cost_fn: 代价函数
            num_samples: Monte Carlo采样数量
            device: 计算设备
        """
        self.dynamics = dynamics_model.to(device)
        self.cost_fn = cost_fn
        self.num_samples = num_samples
        self.device = device
    
    def optimize(self, initial_state, goal_state, horizon=10, action_dim=2, num_iterations=100):
        """
        随机优化
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
            horizon: 预测步数
            action_dim: 动作维度
            num_iterations: 优化迭代次数
        
        返回:
            优化的动作序列
        """
        batch_size = initial_state.shape[0]
        
        # 初始化动作序列
        actions = [torch.randn(batch_size, action_dim, requires_grad=True, device=self.device) * 0.1 
                   for _ in range(horizon)]
        
        optimizer = torch.optim.Adam(actions, lr=0.1)
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            total_cost = 0.0
            
            for _ in range(self.num_samples):
                current_state = initial_state.clone()
                trajectory_cost = 0.0
                
                for t in range(horizon):
                    # 添加过程噪声
                    noise = torch.randn_like(current_state) * 0.01
                    noisy_state = current_state + noise
                    
                    action = actions[t]
                    next_state = self.dynamics(noisy_state, action)
                    
                    # 添加观测噪声
                    obs_noise = torch.randn_like(next_state) * 0.01
                    observed_state = next_state + obs_noise
                    
                    step_cost = self.cost_fn(observed_state, goal_state)
                    trajectory_cost += step_cost
                    current_state = next_state
                
                total_cost += trajectory_cost
            
            # 平均代价
            expected_cost = total_cost / self.num_samples
            expected_cost.backward()
            optimizer.step()
            
            # 动作约束
            for t in range(horizon):
                actions[t].data.clamp_(-1.0, 1.0)
        
        return [a.detach() for a in actions]
    
    def get_action(self, initial_state, goal_state, horizon=10, action_dim=2):
        """获取第一个优化动作"""
        actions = self.optimize(initial_state, goal_state, horizon, action_dim)
        return actions[0]

# 示例代价函数
def quadratic_cost(state, goal):
    """二次代价函数"""
    return torch.mean((state - goal) ** 2)

# 使用示例
stochastic_mpc = StochasticMPC(dynamics, quadratic_cost, num_samples=5)
initial_state = torch.randn(32, 4)
goal_state = torch.zeros(32, 4)

action = stochastic_mpc.get_action(initial_state, goal_state)
print(f"随机MPC动作形状: {action.shape}")  # [32, 2]
```

### 3.3 Tube MPC

**方法**：使用鲁棒控制理论处理不确定性，保证状态在鲁棒管内。

```python
class TubeMPC:
    def __init__(self, nominal_dynamics, controller, disturbance_bound=0.1):
        """
        Tube MPC控制器
        
        参数:
            nominal_dynamics: 标称动态模型
            controller: 反馈控制器
            disturbance_bound: 扰动边界
        """
        self.nominal_dynamics = nominal_dynamics
        self.controller = controller
        self.disturbance_bound = disturbance_bound
    
    def compute_tube(self, state, goal, horizon=10):
        """
        计算鲁棒控制管
        
        参数:
            state: 当前状态 [batch, state_dim]
            goal: 目标状态 [batch, state_dim]
            horizon: 预测步数
        
        返回:
            标称轨迹和控制序列
        """
        batch_size = state.shape[0]
        state_dim = state.shape[-1]
        
        nominal_trajectory = [state]
        control_sequence = []
        tube_radii = []
        
        current_state = state
        
        for t in range(horizon):
            # 计算标称控制
            action = self.controller(current_state, goal)
            control_sequence.append(action)
            
            # 预测标称轨迹
            next_nominal = self.nominal_dynamics(current_state, action)
            nominal_trajectory.append(next_nominal)
            
            # 计算鲁棒管半径（简化版本）
            tube_radius = self.disturbance_bound * (1.1 ** t)
            tube_radii.append(tube_radius)
            
            current_state = next_nominal
        
        return {
            'nominal_trajectory': torch.stack(nominal_trajectory),
            'control_sequence': torch.stack(control_sequence),
            'tube_radii': torch.tensor(tube_radii)
        }
    
    def robust_control(self, state, goal, horizon=10):
        """
        计算鲁棒控制动作
        
        参数:
            state: 当前状态
            goal: 目标状态
            horizon: 预测步数
        
        返回:
            鲁棒控制动作
        """
        tube_info = self.compute_tube(state, goal, horizon)
        
        # 获取第一个控制动作
        first_action = tube_info['control_sequence'][0]
        
        # 计算扰动补偿
        nominal_state = tube_info['nominal_trajectory'][0]
        error = state - nominal_state
        compensation = -0.5 * error  # 简化的扰动补偿
        
        # 组合控制
        robust_action = first_action + compensation
        
        # 约束动作
        robust_action = torch.clamp(robust_action, -1.0, 1.0)
        
        return robust_action

# 示例反馈控制器
class LinearController(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.K = nn.Parameter(torch.randn(action_dim, state_dim) * 0.1)
    
    def forward(self, state, goal):
        error = goal - state
        return self.K @ error.T

# 使用示例
controller = LinearController(state_dim=4, action_dim=2)
tube_mpc = TubeMPC(dynamics, controller, disturbance_bound=0.05)

state = torch.randn(32, 4)
goal = torch.zeros(32, 4)
action = tube_mpc.robust_control(state, goal)
print(f"Tube MPC动作形状: {action.shape}")  # [32, 2]
```

### 3.4 混合MPC

**方法**：结合多个控制器的优势，自适应选择控制策略。

```python
class HybridMPC:
    def __init__(self, controllers, weights=None):
        """
        混合MPC控制器
        
        参数:
            controllers: 控制器列表
            weights: 控制器权重（可选）
        """
        self.controllers = controllers
        
        if weights is None:
            self.weights = torch.ones(len(controllers)) / len(controllers)
        else:
            self.weights = torch.tensor(weights)
    
    def compute_action(self, state, goal):
        """
        计算混合控制动作
        
        参数:
            state: 当前状态 [batch, state_dim]
            goal: 目标状态 [batch, state_dim]
        
        返回:
            混合控制动作
        """
        actions = []
        
        for controller in self.controllers:
            action = controller.get_action(state, goal)
            actions.append(action)
        
        # 加权组合
        actions = torch.stack(actions)
        weights = self.weights.view(-1, 1, 1)
        combined_action = (weights * actions).sum(dim=0)
        
        return combined_action
    
    def adaptive_compute_action(self, state, goal, performance_metrics):
        """
        自适应计算控制动作
        
        参数:
            state: 当前状态
            goal: 目标状态
            performance_metrics: 各控制器的性能指标
        
        返回:
            自适应控制动作
        """
        # 根据性能调整权重（软max）
        weights = torch.softmax(torch.tensor(performance_metrics), dim=0)
        self.weights = weights
        
        return self.compute_action(state, goal)

# 使用示例
mpc1 = StochasticMPC(dynamics, quadratic_cost, num_samples=3)
mpc2 = TubeMPC(dynamics, controller)

# 包装器使接口一致
class ControllerWrapper:
    def __init__(self, controller):
        self.controller = controller
    
    def get_action(self, state, goal):
        if isinstance(self.controller, StochasticMPC):
            return self.controller.get_action(state, goal)
        elif isinstance(self.controller, TubeMPC):
            return self.controller.robust_control(state, goal)

hybrid_mpc = HybridMPC([ControllerWrapper(mpc1), ControllerWrapper(mpc2)])
action = hybrid_mpc.compute_action(state, goal)
print(f"混合MPC动作形状: {action.shape}")  # [32, 2]
```

---

## 4. MPC算法

### 4.1 基于梯度的MPC

**使用梯度下降进行优化**。

```python
class GradientMPC:
    def __init__(self, dynamics_model, horizon=10, lr=0.1):
        """
        基于梯度的MPC控制器
        
        参数:
            dynamics_model: 动态模型
            horizon: 预测步数
            lr: 学习率
        """
        self.dynamics = dynamics_model
        self.horizon = horizon
        self.lr = lr
    
    def optimize(self, initial_state, goal_state, action_dim):
        """
        优化控制序列
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
            action_dim: 动作维度
        
        返回:
            优化的动作序列
        """
        batch_size = initial_state.shape[0]
        
        # 初始化动作
        actions = [torch.randn(batch_size, action_dim, requires_grad=True) * 0.1 
                   for _ in range(self.horizon)]
        
        optimizer = torch.optim.Adam(actions, lr=self.lr)
        
        for iteration in range(100):
            optimizer.zero_grad()
            
            # 前向传播
            states = [initial_state]
            current_state = initial_state
            
            for t in range(self.horizon):
                next_state = self.dynamics(current_state, actions[t])
                states.append(next_state)
                current_state = next_state
            
            states = torch.stack(states)
            
            # 计算损失
            goal_loss = torch.mean((states - goal_state.unsqueeze(0)) ** 2)
            action_loss = sum(torch.mean(a ** 2) for a in actions)
            total_loss = goal_loss + 0.01 * action_loss
            
            # 梯度下降
            total_loss.backward()
            optimizer.step()
            
            # 动作约束
            for a in actions:
                a.data.clamp_(-1.0, 1.0)
        
        return [a.detach() for a in actions]
    
    def get_action(self, initial_state, goal_state, action_dim):
        """获取第一个控制动作"""
        actions = self.optimize(initial_state, goal_state, action_dim)
        return actions[0]

# 使用示例
gradient_mpc = GradientMPC(dynamics, horizon=10, lr=0.05)
action = gradient_mpc.get_action(torch.randn(32, 4), torch.zeros(32, 4), action_dim=2)
print(f"梯度MPC动作形状: {action.shape}")  # [32, 2]
```

### 4.2 Cross-Entropy MPC

**使用交叉熵方法进行优化**。

```python
class CEMPC:
    def __init__(self, dynamics_model, horizon=10, num_samples=100, elite_ratio=0.1):
        """
        交叉熵MPC控制器
        
        参数:
            dynamics_model: 动态模型
            horizon: 预测步数
            num_samples: 采样数量
            elite_ratio: 精英比例
        """
        self.dynamics = dynamics_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.elite_ratio = elite_ratio
    
    def optimize(self, initial_state, goal_state, action_dim):
        """
        使用交叉熵方法优化
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
            action_dim: 动作维度
        
        返回:
            优化的动作序列
        """
        batch_size = initial_state.shape[0]
        
        # 初始化分布
        mean = torch.zeros(self.horizon, action_dim)
        std = torch.ones(self.horizon, action_dim) * 0.5
        
        for iteration in range(50):
            # 采样动作序列
            action_samples = []
            costs = []
            
            for _ in range(self.num_samples):
                # 采样
                actions = torch.randn(self.horizon, action_dim) * std + mean
                action_samples.append(actions)
                
                # 计算代价
                current_state = initial_state
                total_cost = 0.0
                
                for t in range(self.horizon):
                    next_state = self.dynamics(current_state, actions[t].unsqueeze(0).repeat(batch_size, 1))
                    total_cost += torch.mean((next_state - goal_state) ** 2)
                    current_state = next_state
                
                costs.append(total_cost.item())
            
            # 选择精英样本
            costs = torch.tensor(costs)
            num_elite = int(self.num_samples * self.elite_ratio)
            elite_indices = costs.topk(num_elite, largest=False)[1]
            elite_actions = torch.stack([action_samples[i] for i in elite_indices])
            
            # 更新分布
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6  # 防止方差为0
            
            # 限制标准差
            std = torch.clamp(std, 0.01, 1.0)
        
        # 返回均值作为最优动作序列
        return mean
    
    def get_action(self, initial_state, goal_state, action_dim):
        """获取第一个控制动作"""
        actions = self.optimize(initial_state, goal_state, action_dim)
        return actions[0].repeat(initial_state.shape[0], 1)

# 使用示例
cem_mpc = CEMPC(dynamics, horizon=10, num_samples=50, elite_ratio=0.1)
action = cem_mpc.get_action(torch.randn(32, 4), torch.zeros(32, 4), action_dim=2)
print(f"CEM MPC动作形状: {action.shape}")  # [32, 2]
```

### 4.3 iLQR MPC

**迭代线性二次调节器**，使用二阶方法进行优化。

```python
class ILQRMPC:
    def __init__(self, dynamics_model, horizon=10):
        """
        iLQR MPC控制器
        
        参数:
            dynamics_model: 动态模型
            horizon: 预测步数
        """
        self.dynamics = dynamics_model
        self.horizon = horizon
    
    def compute_ilqr(self, initial_state, goal_state, action_dim):
        """
        计算iLQR控制序列
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
            action_dim: 动作维度
        
        返回:
            优化的动作序列
        """
        batch_size = initial_state.shape[0]
        state_dim = initial_state.shape[-1]
        
        # 初始化动作序列
        actions = [torch.randn(batch_size, action_dim) * 0.1 for _ in range(self.horizon)]
        
        for iteration in range(50):
            # 前向传播，计算轨迹和代价
            states = [initial_state]
            current_state = initial_state
            
            for t in range(self.horizon):
                next_state = self.dynamics(current_state, actions[t])
                states.append(next_state)
                current_state = next_state
            
            states = torch.stack(states)
            
            # 计算代价
            goal_cost = torch.mean((states - goal_state.unsqueeze(0)) ** 2)
            action_cost = sum(torch.mean(a ** 2) for a in actions)
            total_cost = goal_cost + 0.01 * action_cost
            
            # 反向传播计算梯度（简化版本）
            # 实际iLQR会计算Hessian并使用二次近似
            actions_tensor = torch.stack([a.requires_grad_(True) for a in actions])
            
            # 重新计算前向传播以获取梯度
            states = [initial_state]
            current_state = initial_state
            
            for t in range(self.horizon):
                next_state = self.dynamics(current_state, actions_tensor[t])
                states.append(next_state)
                current_state = next_state
            
            states = torch.stack(states)
            goal_cost = torch.mean((states - goal_state.unsqueeze(0)) ** 2)
            action_cost = sum(torch.mean(a ** 2) for a in actions_tensor)
            total_cost = goal_cost + 0.01 * action_cost
            
            total_cost.backward()
            
            # 更新动作（梯度下降）
            for t in range(self.horizon):
                grad = actions_tensor[t].grad
                if grad is not None:
                    actions[t] = actions[t] - 0.1 * grad
                    actions[t] = torch.clamp(actions[t], -1.0, 1.0)
        
        return actions
    
    def get_action(self, initial_state, goal_state, action_dim):
        """获取第一个控制动作"""
        actions = self.compute_ilqr(initial_state, goal_state, action_dim)
        return actions[0]

# 使用示例
ilqr_mpc = ILQRMPC(dynamics, horizon=10)
action = ilqr_mpc.get_action(torch.randn(32, 4), torch.zeros(32, 4), action_dim=2)
print(f"iLQR MPC动作形状: {action.shape}")  # [32, 2]
```

### 4.4 采样-Based MPC（MPPI）

**模型预测路径积分控制**，一种基于采样的高效MPC方法。

```python
class MPPI:
    def __init__(self, dynamics_model, horizon=10, num_samples=100, noise_std=0.5):
        """
        MPPI控制器（模型预测路径积分控制）
        
        参数:
            dynamics_model: 动态模型
            horizon: 预测步数
            num_samples: 采样数量
            noise_std: 噪声标准差
        """
        self.dynamics = dynamics_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.noise_std = noise_std
    
    def optimize(self, initial_state, goal_state, action_dim, prev_actions=None):
        """
        MPPI优化
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
            action_dim: 动作维度
            prev_actions: 上一时刻的动作序列（用于时间一致性）
        
        返回:
            优化的动作序列
        """
        batch_size = initial_state.shape[0]
        
        # 初始化基线动作
        if prev_actions is None:
            base_actions = torch.zeros(self.horizon, action_dim)
        else:
            # 向前滚动
            base_actions = torch.cat([prev_actions[1:], torch.zeros(1, action_dim)])
        
        # 采样噪声
        noise = torch.randn(self.num_samples, self.horizon, action_dim) * self.noise_std
        
        # 添加探索噪声
        exploration_noise = torch.randn(self.horizon, action_dim) * 0.1
        
        costs = []
        trajectories = []
        
        for i in range(self.num_samples):
            # 采样动作 = 基线动作 + 噪声
            actions = base_actions + noise[i] + exploration_noise
            
            # 前向传播
            states = [initial_state]
            current_state = initial_state
            total_cost = 0.0
            
            for t in range(self.horizon):
                action = actions[t].unsqueeze(0).repeat(batch_size, 1)
                next_state = self.dynamics(current_state, action)
                states.append(next_state)
                
                # 计算代价
                step_cost = torch.mean((next_state - goal_state) ** 2)
                action_cost = torch.mean(action ** 2)
                total_cost += step_cost + 0.01 * action_cost
                
                current_state = next_state
            
            trajectories.append(states)
            costs.append(total_cost.item())
        
        # 计算权重（指数加权）
        costs = torch.tensor(costs)
        min_cost = costs.min()
        weights = torch.exp(-(costs - min_cost) / 0.1)
        weights = weights / weights.sum()
        
        # 加权平均得到新的基线动作
        new_base_actions = base_actions.clone()
        for i in range(self.num_samples):
            new_base_actions += weights[i] * noise[i]
        
        # 添加阻尼
        new_base_actions = 0.9 * new_base_actions
        
        return new_base_actions
    
    def get_action(self, initial_state, goal_state, action_dim, prev_actions=None):
        """获取第一个控制动作"""
        actions = self.optimize(initial_state, goal_state, action_dim, prev_actions)
        return actions[0].repeat(initial_state.shape[0], 1), actions

# 使用示例
mppi = MPPI(dynamics, horizon=10, num_samples=50, noise_std=0.3)
action, all_actions = mppi.get_action(torch.randn(32, 4), torch.zeros(32, 4), action_dim=2)
print(f"MPPI动作形状: {action.shape}")  # [32, 2]
print(f"所有动作形状: {all_actions.shape}")  # [10, 2]
```

---

## 5. MPC评估指标

### 5.1 控制性能指标

```python
class MPCMetrics:
    def __init__(self):
        self.metrics = {}
    
    def compute_tracking_error(self, states, goal_state):
        """
        计算跟踪误差
        
        参数:
            states: 状态序列 [horizon+1, batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
        
        返回:
            跟踪误差
        """
        errors = states - goal_state.unsqueeze(0)
        rmse = torch.sqrt(torch.mean(errors ** 2, dim=-1))
        
        return {
            'rmse_per_step': rmse.mean(dim=-1).tolist(),
            'final_rmse': rmse[-1].mean().item(),
            'average_rmse': rmse.mean().item()
        }
    
    def compute_control_effort(self, actions):
        """
        计算控制努力（能量消耗）
        
        参数:
            actions: 动作序列 [horizon, batch, action_dim]
        
        返回:
            控制努力指标
        """
        actions_tensor = torch.stack(actions)
        effort = torch.mean(actions_tensor ** 2, dim=-1)
        
        return {
            'average_effort': effort.mean().item(),
            'max_effort': effort.max().item(),
            'total_effort': effort.sum().item()
        }
    
    def compute_time_to_target(self, states, goal_state, threshold=0.1):
        """
        计算到达目标的时间
        
        参数:
            states: 状态序列 [horizon+1, batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
            threshold: 收敛阈值
        
        返回:
            到达目标的平均时间
        """
        errors = torch.norm(states - goal_state.unsqueeze(0), dim=-1)
        below_threshold = errors < threshold
        
        # 找到第一个低于阈值的时间步
        times_to_target = []
        for batch_idx in range(states.shape[1]):
            for t in range(states.shape[0]):
                if below_threshold[t, batch_idx]:
                    times_to_target.append(t)
                    break
        
        if not times_to_target:
            return float('inf')
        
        return sum(times_to_target) / len(times_to_target)
    
    def compute_stability_margin(self, states):
        """
        计算稳定性裕度
        
        参数:
            states: 状态序列 [horizon+1, batch, state_dim]
        
        返回:
            稳定性裕度
        """
        # 计算状态变化的平滑度
        diffs = states[1:] - states[:-1]
        diff_norms = torch.norm(diffs, dim=-1)
        
        # 稳定性裕度 = 1 / (1 + 变化率)
        stability_margin = 1 / (1 + diff_norms.mean().item())
        
        return stability_margin
    
    def evaluate(self, states, actions, goal_state):
        """
        综合评估MPC性能
        
        参数:
            states: 状态序列 [horizon+1, batch, state_dim]
            actions: 动作序列 [horizon, batch, action_dim]
            goal_state: 目标状态 [batch, state_dim]
        
        返回:
            综合评估结果
        """
        tracking_metrics = self.compute_tracking_error(states, goal_state)
        effort_metrics = self.compute_control_effort(actions)
        time_to_target = self.compute_time_to_target(states, goal_state)
        stability_margin = self.compute_stability_margin(states)
        
        # 综合评分
        score = (
            0.3 * (1 / (1 + tracking_metrics['final_rmse'])) +
            0.2 * (1 / (1 + effort_metrics['average_effort'])) +
            0.3 * (1 if time_to_target < float('inf') else 0) +
            0.2 * stability_margin
        )
        
        return {
            'tracking': tracking_metrics,
            'effort': effort_metrics,
            'time_to_target': time_to_target,
            'stability_margin': stability_margin,
            'overall_score': score
        }

# 使用示例
metrics = MPCMetrics()

# 模拟数据
states = torch.randn(11, 32, 4)  # [horizon+1, batch, state_dim]
actions = [torch.randn(32, 2) for _ in range(10)]
goal_state = torch.zeros(32, 4)

results = metrics.evaluate(states, actions, goal_state)
print(f"最终RMSE: {results['tracking']['final_rmse']:.4f}")
print(f"平均控制努力: {results['effort']['average_effort']:.4f}")
print(f"到达目标时间: {results['time_to_target']:.2f}")
print(f"稳定性裕度: {results['stability_margin']:.4f}")
print(f"综合评分: {results['overall_score']:.4f}")
```

### 5.2 鲁棒性指标

```python
class RobustnessMetrics:
    def __init__(self):
        pass
    
    def compute_robustness_margin(self, nominal_states, perturbed_states, threshold=0.1):
        """
        计算鲁棒性裕度
        
        参数:
            nominal_states: 标称状态序列
            perturbed_states: 扰动后的状态序列
            threshold: 性能阈值
        
        返回:
            鲁棒性裕度
        """
        diff = torch.norm(perturbed_states - nominal_states, dim=-1)
        max_deviation = diff.max().item()
        
        # 鲁棒性裕度 = 阈值 / 最大偏差
        robustness_margin = threshold / max_deviation if max_deviation > 0 else float('inf')
        
        return robustness_margin
    
    def compute_disturbance_rejection(self, states, disturbances):
        """
        计算扰动抑制能力
        
        参数:
            states: 状态序列
            disturbances: 扰动序列
        
        返回:
            扰动抑制比
        """
        disturbance_norm = torch.norm(disturbances, dim=-1).mean()
        state_deviation = torch.norm(states - states[0], dim=-1).mean()
        
        # 扰动抑制比 = 扰动大小 / 状态偏差
        rejection_ratio = disturbance_norm / (state_deviation + 1e-6)
        
        return rejection_ratio.item()
    
    def compute_sensitivity(self, model, state, action, epsilon=1e-6):
        """
        计算模型灵敏度
        
        参数:
            model: 动态模型
            state: 当前状态
            action: 当前动作
            epsilon: 扰动大小
        
        返回:
            灵敏度度量
        """
        original_pred = model(state, action)
        
        # 计算状态灵敏度
        state_perturbed = state + epsilon * torch.randn_like(state)
        perturbed_pred = model(state_perturbed, action)
        state_sensitivity = torch.norm(perturbed_pred - original_pred) / epsilon
        
        # 计算动作灵敏度
        action_perturbed = action + epsilon * torch.randn_like(action)
        perturbed_pred = model(state, action_perturbed)
        action_sensitivity = torch.norm(perturbed_pred - original_pred) / epsilon
        
        return {
            'state_sensitivity': state_sensitivity.item(),
            'action_sensitivity': action_sensitivity.item()
        }
    
    def evaluate_robustness(self, nominal_states, perturbed_states, disturbances, model, state, action):
        """
        综合评估鲁棒性
        
        参数:
            nominal_states: 标称状态序列
            perturbed_states: 扰动后的状态序列
            disturbances: 扰动序列
            model: 动态模型
            state: 当前状态
            action: 当前动作
        
        返回:
            鲁棒性评估结果
        """
        robustness_margin = self.compute_robustness_margin(nominal_states, perturbed_states)
        rejection_ratio = self.compute_disturbance_rejection(nominal_states, disturbances)
        sensitivity = self.compute_sensitivity(model, state, action)
        
        # 综合鲁棒性评分
        robustness_score = (
            0.4 * min(robustness_margin, 1.0) +
            0.3 * min(rejection_ratio, 1.0) +
            0.3 * (1 - min(sensitivity['state_sensitivity'], 1.0))
        )
        
        return {
            'robustness_margin': robustness_margin,
            'disturbance_rejection': rejection_ratio,
            'sensitivity': sensitivity,
            'robustness_score': robustness_score
        }

# 使用示例
robustness_metrics = RobustnessMetrics()

nominal_states = torch.randn(11, 32, 4)
perturbed_states = nominal_states + torch.randn_like(nominal_states) * 0.05
disturbances = torch.randn(10, 32, 4) * 0.1

robustness_results = robustness_metrics.evaluate_robustness(
    nominal_states, perturbed_states, disturbances, dynamics, 
    torch.randn(32, 4), torch.randn(32, 2)
)
print(f"鲁棒性裕度: {robustness_results['robustness_margin']:.4f}")
print(f"扰动抑制比: {robustness_results['disturbance_rejection']:.4f}")
print(f"状态灵敏度: {robustness_results['sensitivity']['state_sensitivity']:.4f}")
print(f"鲁棒性评分: {robustness_results['robustness_score']:.4f}")
```

---

## 6. 优化策略

### 6.1 自适应控制策略

```python
class AdaptiveMPC:
    def __init__(self, dynamics_model, base_horizon=10, min_horizon=5, max_horizon=20):
        """
        自适应MPC控制器
        
        参数:
            dynamics_model: 动态模型
            base_horizon: 基础预测步数
            min_horizon: 最小预测步数
            max_horizon: 最大预测步数
        """
        self.dynamics = dynamics_model
        self.base_horizon = base_horizon
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.current_horizon = base_horizon
    
    def update_horizon(self, error, error_rate):
        """
        根据误差自适应调整预测步数
        
        参数:
            error: 当前跟踪误差
            error_rate: 误差变化率
        """
        # 如果误差大或误差在增长，增加预测步数
        if error > 0.5 or error_rate > 0.1:
            self.current_horizon = min(self.max_horizon, self.current_horizon + 2)
        # 如果误差小且稳定，减少预测步数
        elif error < 0.1 and abs(error_rate) < 0.01:
            self.current_horizon = max(self.min_horizon, self.current_horizon - 1)
        
        return self.current_horizon
    
    def optimize(self, initial_state, goal_state, action_dim):
        """优化控制序列"""
        horizon = self.current_horizon
        
        # 使用当前预测步数进行优化
        actions = [torch.randn(initial_state.shape[0], action_dim, requires_grad=True) * 0.1 
                   for _ in range(horizon)]
        
        optimizer = torch.optim.Adam(actions, lr=0.1)
        
        for iteration in range(100):
            optimizer.zero_grad()
            
            states = [initial_state]
            current_state = initial_state
            
            for t in range(horizon):
                next_state = self.dynamics(current_state, actions[t])
                states.append(next_state)
                current_state = next_state
            
            states = torch.stack(states)
            goal_loss = torch.mean((states - goal_state.unsqueeze(0)) ** 2)
            action_loss = sum(torch.mean(a ** 2) for a in actions)
            total_loss = goal_loss + 0.01 * action_loss
            
            total_loss.backward()
            optimizer.step()
            
            for a in actions:
                a.data.clamp_(-1.0, 1.0)
        
        return [a.detach() for a in actions]
    
    def get_action(self, initial_state, goal_state, action_dim, error, error_rate):
        """获取自适应控制动作"""
        self.update_horizon(error, error_rate)
        actions = self.optimize(initial_state, goal_state, action_dim)
        return actions[0], self.current_horizon

# 使用示例
adaptive_mpc = AdaptiveMPC(dynamics, base_horizon=10)

current_error = 0.3
error_rate = 0.05
action, horizon = adaptive_mpc.get_action(
    torch.randn(32, 4), 
    torch.zeros(32, 4), 
    action_dim=2,
    error=current_error,
    error_rate=error_rate
)
print(f"自适应MPC动作形状: {action.shape}")  # [32, 2]
print(f"当前预测步数: {horizon}")
```

### 6.2 约束处理策略

```python
class ConstrainedMPC:
    def __init__(self, dynamics_model, state_bounds, action_bounds):
        """
        带约束的MPC控制器
        
        参数:
            dynamics_model: 动态模型
            state_bounds: 状态边界 [(min, max), ...]
            action_bounds: 动作边界 [(min, max), ...]
        """
        self.dynamics = dynamics_model
        self.state_bounds = torch.tensor(state_bounds)
        self.action_bounds = torch.tensor(action_bounds)
    
    def project_to_bounds(self, state):
        """
        将状态投影到可行域
        
        参数:
            state: 当前状态
        
        返回:
            投影后的状态
        """
        lower = self.state_bounds[:, 0].unsqueeze(0)
        upper = self.state_bounds[:, 1].unsqueeze(0)
        return torch.clamp(state, lower, upper)
    
    def compute_constraint_violation(self, states, actions):
        """
        计算约束违反程度
        
        参数:
            states: 状态序列
            actions: 动作序列
        
        返回:
            约束违反代价
        """
        # 状态约束违反
        state_lower = self.state_bounds[:, 0].unsqueeze(0).unsqueeze(0)
        state_upper = self.state_bounds[:, 1].unsqueeze(0).unsqueeze(0)
        state_violation_lower = torch.relu(state_lower - states)
        state_violation_upper = torch.relu(states - state_upper)
        state_violation = torch.mean(state_violation_lower + state_violation_upper)
        
        # 动作约束违反
        action_lower = self.action_bounds[:, 0].unsqueeze(0).unsqueeze(0)
        action_upper = self.action_bounds[:, 1].unsqueeze(0).unsqueeze(0)
        actions_tensor = torch.stack(actions)
        action_violation_lower = torch.relu(action_lower - actions_tensor)
        action_violation_upper = torch.relu(actions_tensor - action_upper)
        action_violation = torch.mean(action_violation_lower + action_violation_upper)
        
        return state_violation + action_violation
    
    def optimize(self, initial_state, goal_state, horizon=10):
        """
        带约束优化
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
            horizon: 预测步数
        
        返回:
            优化的动作序列
        """
        action_dim = self.action_bounds.shape[0]
        batch_size = initial_state.shape[0]
        
        actions = [torch.randn(batch_size, action_dim, requires_grad=True) * 0.1 
                   for _ in range(horizon)]
        
        optimizer = torch.optim.Adam(actions, lr=0.1)
        
        for iteration in range(100):
            optimizer.zero_grad()
            
            # 前向传播（带约束投影）
            states = [self.project_to_bounds(initial_state)]
            current_state = states[0]
            
            for t in range(horizon):
                next_state = self.dynamics(current_state, actions[t])
                next_state = self.project_to_bounds(next_state)  # 投影到可行域
                states.append(next_state)
                current_state = next_state
            
            states = torch.stack(states)
            
            # 计算代价
            goal_loss = torch.mean((states - goal_state.unsqueeze(0)) ** 2)
            action_loss = sum(torch.mean(a ** 2) for a in actions)
            constraint_loss = self.compute_constraint_violation(states, actions)
            
            # 带约束的总代价
            total_loss = goal_loss + 0.01 * action_loss + 10.0 * constraint_loss
            
            total_loss.backward()
            optimizer.step()
            
            # 硬约束动作
            for a in actions:
                lower = self.action_bounds[:, 0].unsqueeze(0)
                upper = self.action_bounds[:, 1].unsqueeze(0)
                a.data.clamp_(lower, upper)
        
        return [a.detach() for a in actions]

# 使用示例
state_bounds = [[-1.0, 1.0], [-1.0, 1.0], [-0.5, 0.5], [-0.5, 0.5]]  # 4维状态边界
action_bounds = [[-1.0, 1.0], [-0.5, 0.5]]  # 2维动作边界

constrained_mpc = ConstrainedMPC(dynamics, state_bounds, action_bounds)
actions = constrained_mpc.optimize(torch.randn(32, 4), torch.zeros(32, 4))
print(f"约束MPC动作数量: {len(actions)}")
print(f"第一个动作: {actions[0].shape}")  # [32, 2]
```

---

## 7. 工程实践

### 7.1 实时MPC实现

```python
class RealTimeMPC:
    def __init__(self, dynamics_model, horizon=10, max_compute_time=0.1):
        """
        实时MPC控制器
        
        参数:
            dynamics_model: 动态模型
            horizon: 预测步数
            max_compute_time: 最大计算时间（秒）
        """
        self.dynamics = dynamics_model
        self.horizon = horizon
        self.max_compute_time = max_compute_time
        self.warm_start = None
    
    def optimize(self, initial_state, goal_state, action_dim):
        """
        实时优化
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
            action_dim: 动作维度
        
        返回:
            优化的动作序列
        """
        batch_size = initial_state.shape[0]
        
        # 使用热启动
        if self.warm_start is not None:
            actions = [self.warm_start[t].detach().requires_grad_(True) 
                       for t in range(self.horizon)]
        else:
            actions = [torch.randn(batch_size, action_dim, requires_grad=True) * 0.1 
                       for _ in range(self.horizon)]
        
        optimizer = torch.optim.Adam(actions, lr=0.1)
        
        import time
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < self.max_compute_time:
            optimizer.zero_grad()
            
            states = [initial_state]
            current_state = initial_state
            
            for t in range(self.horizon):
                next_state = self.dynamics(current_state, actions[t])
                states.append(next_state)
                current_state = next_state
            
            states = torch.stack(states)
            goal_loss = torch.mean((states - goal_state.unsqueeze(0)) ** 2)
            action_loss = sum(torch.mean(a ** 2) for a in actions)
            total_loss = goal_loss + 0.01 * action_loss
            
            total_loss.backward()
            optimizer.step()
            
            for a in actions:
                a.data.clamp_(-1.0, 1.0)
            
            iteration += 1
        
        # 更新热启动（向前滚动）
        self.warm_start = [a.detach() for a in actions]
        if len(self.warm_start) > 1:
            self.warm_start = self.warm_start[1:] + [torch.zeros_like(self.warm_start[0])]
        
        print(f"实时MPC完成 {iteration} 次迭代")
        return [a.detach() for a in actions]
    
    def get_action(self, initial_state, goal_state, action_dim):
        """获取控制动作"""
        actions = self.optimize(initial_state, goal_state, action_dim)
        return actions[0]

# 使用示例
realtime_mpc = RealTimeMPC(dynamics, horizon=10, max_compute_time=0.05)
action = realtime_mpc.get_action(torch.randn(32, 4), torch.zeros(32, 4), action_dim=2)
print(f"实时MPC动作形状: {action.shape}")  # [32, 2]
```

### 7.2 分布式MPC

```python
class DistributedMPC:
    def __init__(self, dynamics_models, num_workers=4):
        """
        分布式MPC控制器
        
        参数:
            dynamics_models: 动态模型列表（每个worker一个）
            num_workers: worker数量
        """
        self.models = dynamics_models
        self.num_workers = num_workers
    
    def parallel_optimize(self, initial_state, goal_state, horizon=10, action_dim=2):
        """
        并行优化
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
            horizon: 预测步数
            action_dim: 动作维度
        
        返回:
            优化的动作序列
        """
        batch_size = initial_state.shape[0]
        
        # 将批次分配到各个worker
        batch_per_worker = batch_size // self.num_workers
        
        results = []
        
        for i in range(self.num_workers):
            start_idx = i * batch_per_worker
            end_idx = (i + 1) * batch_per_worker if i < self.num_workers - 1 else batch_size
            
            local_state = initial_state[start_idx:end_idx]
            local_goal = goal_state[start_idx:end_idx]
            
            # 每个worker独立优化
            model = self.models[i]
            actions = [torch.randn(local_state.shape[0], action_dim, requires_grad=True) * 0.1 
                       for _ in range(horizon)]
            
            optimizer = torch.optim.Adam(actions, lr=0.1)
            
            for iteration in range(50):
                optimizer.zero_grad()
                
                states = [local_state]
                current_state = local_state
                
                for t in range(horizon):
                    next_state = model(current_state, actions[t])
                    states.append(next_state)
                    current_state = next_state
                
                states = torch.stack(states)
                loss = torch.mean((states - local_goal.unsqueeze(0)) ** 2)
                loss.backward()
                optimizer.step()
            
            results.append([a.detach() for a in actions])
        
        # 合并结果
        final_actions = []
        for t in range(horizon):
            action_parts = [results[i][t] for i in range(self.num_workers)]
            final_actions.append(torch.cat(action_parts, dim=0))
        
        return final_actions

# 使用示例
models = [SimpleDynamics(4, 2) for _ in range(4)]
distributed_mpc = DistributedMPC(models, num_workers=4)

initial_state = torch.randn(32, 4)
goal_state = torch.zeros(32, 4)
actions = distributed_mpc.parallel_optimize(initial_state, goal_state)
print(f"分布式MPC动作数量: {len(actions)}")
print(f"第一个动作形状: {actions[0].shape}")  # [32, 2]
```

---

## 8. 理论基础与分析

### 8.1 稳定性分析

```python
class StabilityAnalyzer:
    def __init__(self, dynamics_model):
        self.dynamics = dynamics_model
    
    def compute_lyapunov_function(self, state, goal_state, Q=None, R=None):
        """
        计算李雅普诺夫函数值
        
        参数:
            state: 当前状态
            goal_state: 目标状态
            Q: 状态权重矩阵
            R: 动作权重矩阵
        
        返回:
            李雅普诺夫函数值
        """
        if Q is None:
            Q = torch.eye(state.shape[-1])
        if R is None:
            R = torch.eye(2)  # 假设动作维度为2
        
        error = state - goal_state
        value = error @ Q @ error.T
        
        return value.mean().item()
    
    def check_stability(self, initial_state, goal_state, horizon=10):
        """
        检查闭环稳定性
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
            horizon: 预测步数
        
        返回:
            稳定性分析结果
        """
        # 模拟闭环系统
        states = [initial_state]
        current_state = initial_state
        
        for t in range(horizon):
            # 简化的控制律：比例控制
            action = -0.5 * (current_state - goal_state)
            
            next_state = self.dynamics(current_state, action)
            states.append(next_state)
            current_state = next_state
        
        states = torch.stack(states)
        
        # 计算李雅普诺夫函数值序列
        lyapunov_values = []
        for t in range(horizon + 1):
            value = self.compute_lyapunov_function(states[t], goal_state)
            lyapunov_values.append(value)
        
        # 检查是否单调递减
        is_stable = all(lyapunov_values[i+1] <= lyapunov_values[i] for i in range(horizon))
        
        return {
            'lyapunov_values': lyapunov_values,
            'is_stable': is_stable,
            'final_value': lyapunov_values[-1],
            'decay_rate': lyapunov_values[0] / (lyapunov_values[-1] + 1e-6)
        }
    
    def compute_region_of_attraction(self, goal_state, num_samples=100, threshold=0.1):
        """
        估计吸引域
        
        参数:
            goal_state: 目标状态
            num_samples: 采样数量
            threshold: 收敛阈值
        
        返回:
            吸引域估计
        """
        state_dim = goal_state.shape[-1]
        successes = 0
        
        for _ in range(num_samples):
            # 从随机初始状态开始
            initial_state = goal_state + (torch.rand_like(goal_state) - 0.5) * 4
            
            # 模拟闭环系统
            current_state = initial_state
            converged = False
            
            for t in range(50):
                action = -0.5 * (current_state - goal_state)
                current_state = self.dynamics(current_state, action)
                
                if torch.norm(current_state - goal_state) < threshold:
                    converged = True
                    break
            
            if converged:
                successes += 1
        
        return successes / num_samples

# 使用示例
stability_analyzer = StabilityAnalyzer(dynamics)
result = stability_analyzer.check_stability(torch.randn(32, 4), torch.zeros(32, 4))
print(f"是否稳定: {result['is_stable']}")
print(f"李雅普诺夫值序列: {result['lyapunov_values']}")
print(f"衰减率: {result['decay_rate']:.4f}")

attraction_region = stability_analyzer.compute_region_of_attraction(torch.zeros(4))
print(f"吸引域估计: {attraction_region:.4f}")
```

### 8.2 性能边界分析

```python
class PerformanceAnalyzer:
    def __init__(self, dynamics_model):
        self.dynamics = dynamics_model
    
    def compute_theoretical_bound(self, state_dim, action_dim, horizon):
        """
        计算理论性能边界
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            horizon: 预测步数
        
        返回:
            理论边界
        """
        # 简化的性能边界估计
        # 基于线性系统理论
        bound = (state_dim * horizon) / (action_dim + 1)
        
        return bound
    
    def compute_practical_bound(self, initial_state, goal_state, horizon):
        """
        计算实际性能边界
        
        参数:
            initial_state: 初始状态
            goal_state: 目标状态
            horizon: 预测步数
        
        返回:
            实际边界
        """
        # 最优控制的理论下界（简化）
        initial_error = torch.norm(initial_state - goal_state).mean()
        minimal_cost = initial_error ** 2 / horizon
        
        return minimal_cost.item()
    
    def analyze_gap(self, actual_cost, initial_state, goal_state, horizon):
        """
        分析实际性能与理论边界的差距
        
        参数:
            actual_cost: 实际代价
            initial_state: 初始状态
            goal_state: 目标状态
            horizon: 预测步数
        
        返回:
            差距分析结果
        """
        theoretical_bound = self.compute_theoretical_bound(
            initial_state.shape[-1], 2, horizon
        )
        practical_bound = self.compute_practical_bound(initial_state, goal_state, horizon)
        
        # 计算差距
        gap_to_theoretical = actual_cost - theoretical_bound
        gap_to_practical = actual_cost - practical_bound
        
        # 相对差距
        relative_gap = gap_to_practical / (practical_bound + 1e-6)
        
        return {
            'actual_cost': actual_cost,
            'theoretical_bound': theoretical_bound,
            'practical_bound': practical_bound,
            'gap_to_theoretical': gap_to_theoretical,
            'gap_to_practical': gap_to_practical,
            'relative_gap': relative_gap
        }

# 使用示例
performance_analyzer = PerformanceAnalyzer(dynamics)

# 计算理论边界
theoretical_bound = performance_analyzer.compute_theoretical_bound(4, 2, 10)
print(f"理论边界: {theoretical_bound:.4f}")

# 计算实际边界
practical_bound = performance_analyzer.compute_practical_bound(
    torch.randn(32, 4), torch.zeros(32, 4), 10
)
print(f"实际边界: {practical_bound:.4f}")

# 分析差距
gap_analysis = performance_analyzer.analyze_gap(
    actual_cost=0.5,
    initial_state=torch.randn(32, 4),
    goal_state=torch.zeros(32, 4),
    horizon=10
)
print(f"差距分析: {gap_analysis}")
```

---

## 9. 前沿研究方向

### 9.1 学习增强MPC

```python
class LearnedMPC:
    def __init__(self, dynamics_model, cost_model, horizon=10):
        """
        学习增强MPC控制器
        
        参数:
            dynamics_model: 动态模型
            cost_model: 学习到的代价模型
            horizon: 预测步数
        """
        self.dynamics = dynamics_model
        self.cost_model = cost_model
        self.horizon = horizon
    
    def optimize(self, initial_state, goal_state, action_dim):
        """
        使用学习到的代价模型优化
        
        参数:
            initial_state: 初始状态 [batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
            action_dim: 动作维度
        
        返回:
            优化的动作序列
        """
        batch_size = initial_state.shape[0]
        
        actions = [torch.randn(batch_size, action_dim, requires_grad=True) * 0.1 
                   for _ in range(self.horizon)]
        
        optimizer = torch.optim.Adam(actions, lr=0.1)
        
        for iteration in range(100):
            optimizer.zero_grad()
            
            states = [initial_state]
            current_state = initial_state
            total_cost = 0.0
            
            for t in range(self.horizon):
                next_state = self.dynamics(current_state, actions[t])
                states.append(next_state)
                
                # 使用学习到的代价模型
                step_cost = self.cost_model(next_state, goal_state)
                total_cost += step_cost
                
                current_state = next_state
            
            # 添加动作正则化
            action_cost = sum(torch.mean(a ** 2) for a in actions)
            total_cost = total_cost + 0.01 * action_cost
            
            total_cost.backward()
            optimizer.step()
            
            for a in actions:
                a.data.clamp_(-1.0, 1.0)
        
        return [a.detach() for a in actions]

# 学习到的代价模型示例
class LearnedCostModel(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, goal):
        combined = torch.cat([state, goal], dim=-1)
        return torch.mean(self.net(combined))

# 使用示例
cost_model = LearnedCostModel(state_dim=4)
learned_mpc = LearnedMPC(dynamics, cost_model)
action = learned_mpc.optimize(
    torch.randn(32, 4), torch.zeros(32, 4), action_dim=2
)[0]
print(f"学习增强MPC动作形状: {action.shape}")  # [32, 2]
```

### 9.2 元MPC

```python
class MetaMPC:
    def __init__(self, dynamics_model, horizon=10):
        """
        元学习MPC控制器
        
        参数:
            dynamics_model: 动态模型
            horizon: 预测步数
        """
        self.dynamics = dynamics_model
        self.horizon = horizon
        self.meta_params = None
    
    def meta_train(self, tasks, num_meta_iterations=100):
        """
        元训练
        
        参数:
            tasks: 任务列表，每个任务包含 (initial_states, goal_states)
            num_meta_iterations: 元迭代次数
        """
        # 初始化元参数（学习率）
        meta_lr = nn.Parameter(torch.tensor(0.1))
        
        optimizer = torch.optim.Adam([meta_lr], lr=0.01)
        
        for meta_iter in range(num_meta_iterations):
            optimizer.zero_grad()
            total_loss = 0.0
            
            for initial_states, goal_states in tasks:
                # 快速适应（在当前任务上优化）
                action_dim = 2
                actions = [torch.randn(initial_states.shape[0], action_dim, requires_grad=True) * 0.1 
                           for _ in range(self.horizon)]
                
                for inner_iter in range(10):
                    # 使用元学习的学习率
                    opt = torch.optim.Adam(actions, lr=meta_lr.item())
                    opt.zero_grad()
                    
                    states = [initial_states]
                    current_state = initial_states
                    
                    for t in range(self.horizon):
                        next_state = self.dynamics(current_state, actions[t])
                        states.append(next_state)
                        current_state = next_state
                    
                    loss = torch.mean((torch.stack(states) - goal_states.unsqueeze(0)) ** 2)
                    loss.backward()
                    opt.step()
                
                # 计算元损失
                final_states = torch.stack(states)
                meta_loss = torch.mean((final_states[-1] - goal_states) ** 2)
                total_loss += meta_loss
            
            total_loss.backward()
            optimizer.step()
            
            # 确保学习率为正
            meta_lr.data.clamp_(0.01, 1.0)
        
        self.meta_params = {'lr': meta_lr.item()}
        print(f"元学习完成，学习率: {self.meta_params['lr']:.4f}")
    
    def optimize(self, initial_state, goal_state, action_dim):
        """使用元学习参数优化"""
        lr = self.meta_params['lr'] if self.meta_params else 0.1
        
        actions = [torch.randn(initial_state.shape[0], action_dim, requires_grad=True) * 0.1 
                   for _ in range(self.horizon)]
        
        optimizer = torch.optim.Adam(actions, lr=lr)
        
        for iteration in range(100):
            optimizer.zero_grad()
            
            states = [initial_state]
            current_state = initial_state
            
            for t in range(self.horizon):
                next_state = self.dynamics(current_state, actions[t])
                states.append(next_state)
                current_state = next_state
            
            states = torch.stack(states)
            loss = torch.mean((states - goal_state.unsqueeze(0)) ** 2)
            loss.backward()
            optimizer.step()
            
            for a in actions:
                a.data.clamp_(-1.0, 1.0)
        
        return [a.detach() for a in actions]

# 使用示例
meta_mpc = MetaMPC(dynamics, horizon=10)

# 模拟元训练任务
tasks = []
for _ in range(10):
    initial_states = torch.randn(32, 4)
    goal_states = torch.zeros(32, 4)
    tasks.append((initial_states, goal_states))

meta_mpc.meta_train(tasks, num_meta_iterations=50)
action = meta_mpc.optimize(torch.randn(32, 4), torch.zeros(32, 4), action_dim=2)[0]
print(f"元MPC动作形状: {action.shape}")  # [32, 2]
```

---

## 10. 实践练习

### 练习1：实现基础MPC控制器

```python
class BasicMPC:
    def __init__(self, dynamics_model, horizon=10, lr=0.1):
        self.dynamics = dynamics_model
        self.horizon = horizon
        self.lr = lr
    
    def control(self, current_state, goal_state, action_dim):
        """
        计算控制动作
        
        参数:
            current_state: 当前状态 [batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
            action_dim: 动作维度
        
        返回:
            第一个控制动作
        """
        actions = self._optimize(current_state, goal_state, action_dim)
        return actions[0]
    
    def _optimize(self, current_state, goal_state, action_dim):
        """优化动作序列"""
        batch_size = current_state.shape[0]
        
        actions = [torch.randn(batch_size, action_dim, requires_grad=True) * 0.1 
                   for _ in range(self.horizon)]
        optimizer = torch.optim.Adam(actions, lr=self.lr)
        
        for _ in range(100):
            optimizer.zero_grad()
            
            # 前向传播
            state = current_state
            total_cost = 0
            
            for t in range(self.horizon):
                next_state = self.dynamics(state, actions[t])
                total_cost += torch.mean((next_state - goal_state) ** 2)
                state = next_state
            
            total_cost.backward()
            optimizer.step()
            
            for a in actions:
                a.data.clamp_(-1.0, 1.0)
        
        return [a.detach() for a in actions]

# 测试
dynamics = SimpleDynamics(state_dim=4, action_dim=2)
mpc = BasicMPC(dynamics, horizon=10)

current_state = torch.randn(32, 4)
goal_state = torch.zeros(32, 4)

action = mpc.control(current_state, goal_state, action_dim=2)
print(f"选择的动作: {action.shape}")  # [32, 2]
```

### 练习2：实现CEM优化器

```python
class CEMOptimizer:
    def __init__(self, num_samples=100, elite_ratio=0.1, lr=0.1):
        self.num_samples = num_samples
        self.elite_ratio = elite_ratio
        self.lr = lr
    
    def optimize(self, cost_fn, action_dim, horizon, initial_mean=None, initial_std=None):
        """
        交叉熵方法优化
        
        参数:
            cost_fn: 代价函数，输入动作序列，返回代价
            action_dim: 动作维度
            horizon: 时间步数
            initial_mean: 初始均值
            initial_std: 初始标准差
        
        返回:
            优化的动作序列
        """
        if initial_mean is None:
            mean = torch.zeros(horizon, action_dim)
        else:
            mean = initial_mean
        
        if initial_std is None:
            std = torch.ones(horizon, action_dim)
        else:
            std = initial_std
        
        for iteration in range(50):
            # 采样
            samples = []
            costs = []
            
            for _ in range(self.num_samples):
                actions = torch.randn(horizon, action_dim) * std + mean
                samples.append(actions)
                cost = cost_fn(actions)
                costs.append(cost.item())
            
            samples = torch.stack(samples)
            costs = torch.tensor(costs)
            
            # 选择精英
            elite_idx = costs.topk(int(self.num_samples * self.elite_ratio), largest=False)[1]
            elite_samples = samples[elite_idx]
            
            # 更新分布
            mean = elite_samples.mean(dim=0)
            std = elite_samples.std(dim=0) + 1e-6  # 防止方差为0
        
        return mean

# 测试
def simple_cost_fn(actions):
    """简单的代价函数"""
    return torch.sum(actions ** 2)

optimizer = CEMOptimizer(num_samples=100)
optimal_actions = optimizer.optimize(simple_cost_fn, action_dim=2, horizon=10)
print(f"优化后的动作序列形状: {optimal_actions.shape}")  # [10, 2]
```

### 练习3：实现完整的MPC控制系统

```python
class MPC控制系统:
    def __init__(self, dynamics_model, mpc_horizon=10, num_samples=50):
        self.dynamics = dynamics_model
        self.mpc_horizon = mpc_horizon
        self.num_samples = num_samples
    
    def compute_action(self, current_state, goal_state):
        """
        计算最优动作
        
        参数:
            current_state: 当前状态 [batch, state_dim]
            goal_state: 目标状态 [batch, state_dim]
        
        返回:
            最优动作
        """
        action_dim = goal_state.shape[-1]
        
        # 定义代价函数
        def cost_fn(actions):
            state = current_state
            total_cost = 0
            for t in range(self.mpc_horizon):
                next_state = self.dynamics(state, actions[t].unsqueeze(0).repeat(state.shape[0], 1))
                total_cost += torch.mean((next_state - goal_state) ** 2)
                state = next_state
            return total_cost
        
        # 使用CEM优化
        optimizer = CEMOptimizer(num_samples=self.num_samples)
        actions = optimizer.optimize(cost_fn, action_dim, self.mpc_horizon)
        
        return actions[0].repeat(current_state.shape[0], 1)
    
    def run_episode(self, env, goal_state, max_steps=100):
        """
        运行一个episode
        
        参数:
            env: 环境
            goal_state: 目标状态
            max_steps: 最大步数
        
        返回:
            总奖励
        """
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # 计算动作
            action = self.compute_action(
                torch.FloatTensor(state).unsqueeze(0),
                torch.FloatTensor(goal_state).unsqueeze(0)
            ).numpy()[0]
            
            # 执行动作
            state, reward, done, _ = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        return total_reward

# 测试（需要环境）
# env = SimpleEnvironment()
# mpc_system = MPC控制系统(dynamics, mpc_horizon=10)
# goal = torch.randn(4)
# total_reward = mpc_system.run_episode(env, goal)
# print(f"总奖励: {total_reward}")
```

---

**返回**：[世界模型概念](01-world-model-concepts.md)

---

## 参考文献

1. Rawlings, J. B., & Mayne, D. Q. (2009). Model Predictive Control: Theory and Design.
2. Richalet, J., et al. (1978). Model Predictive Heuristic Control.
3. Garcia, C. E., et al. (1989). Model Predictive Control: Theory and Practice.
4. KPN (2021). Model Predictive Control for Robotics.
5. Li, W., et al. (2022). Learning-based Model Predictive Control for Autonomous Systems.
6. Wang, Z., et al. (2023). Robust Model Predictive Control with Learned Dynamics.
7. Zhang, L., et al. (2023). Meta-Learning for Adaptive Model Predictive Control.