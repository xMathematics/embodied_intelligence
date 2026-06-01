# 6.5 模型预测控制

## 目录

- [1. 引言](#1-引言)
- [2. 模型预测控制概述](#2-模型预测控制概述)
- [3. MPC方法](#3-mpc方法)
- [4. MPC算法](#4-mpc算法)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 模型预测控制的重要性

**模型预测控制**（Model Predictive Control, MPC）是一种先进的控制方法，利用系统模型进行滚动时域优化来选择控制动作。MPC在机器人控制、自动驾驶、工业过程控制等领域有广泛应用。

### 1.2 应用场景

| 场景 | 描述 | 示例 |
|------|------|------|
| **机器人控制** | 机器人运动规划和控制 | 机械臂轨迹跟踪 |
| **自动驾驶** | 车辆轨迹规划和控制 | 车道保持 |
| **过程控制** | 工业过程优化控制 | 化工过程控制 |
| **能源管理** | 能源系统优化 | 电网调度 |

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

| 特点 | 描述 |
|------|------|
| **前瞻性** | 考虑未来预测 |
| **滚动优化** | 实时重新规划 |
| **约束处理** | 自然处理系统约束 |
| **鲁棒性** | 对模型误差有一定鲁棒性 |

---

## 3. MPC方法

### 3.1 收缩约束MPC

**方法**：在滚动视窗内强制状态收缩。

```python
def mpc_with_contraction_constraint(
    dynamics_model,
    initial_state,
    goal_state,
    horizon=10,
    num_iterations=5
):
    """
    带收缩约束的MPC
    
    参数:
        dynamics_model: 动态模型
        initial_state: 初始状态
        goal_state: 目标状态
        horizon: 预测步数
        num_iterations: 优化迭代次数
    
    返回:
        优化的控制序列
    """
    state_dim = initial_state.shape[-1]
    action_dim = 2  # 假设动作维度为2
    
    # 初始化控制序列
    actions = [torch.randn(1, action_dim) * 0.1 for _ in range(horizon)]
    
    for iteration in range(num_iterations):
        # 预测轨迹
        states = [initial_state]
        current_state = initial_state
        
        for t in range(horizon):
            next_state = dynamics_model(current_state, actions[t])
            states.append(next_state)
            current_state = next_state
        
        states = torch.stack(states)
        
        # 计算损失
        goal_loss = torch.sum((states - goal_state) ** 2)
        action_loss = sum(torch.sum(a ** 2) for a in actions)
        total_loss = goal_loss + 0.01 * action_loss
        
        # 简化的梯度更新
        # 实际应用中应使用更复杂的优化器
        total_loss.backward()
        
        # 更新动作
        for t in range(horizon):
            if actions[t].grad is not None:
                actions[t] = actions[t] - 0.1 * actions[t].grad
                actions[t] = torch.clamp(actions[t], -1, 1)
                actions[t].grad = None
    
    return actions
```

### 3.2 随机MPC

**方法**：处理系统不确定性和噪声。

```python
class StochasticMPC:
    def __init__(self, dynamics_model, cost_fn, num_samples=10):
        self.dynamics = dynamics_model
        self.cost_fn = cost_fn
        self.num_samples = num_samples
    
    def optimize(self, initial_state, goal_state, horizon=10):
        """
        随机优化
        
        参数:
            initial_state: 初始状态
            goal_state: 目标状态
            horizon: 预测步数
        
        返回:
            优化的动作序列
        """
        action_dim = 2
        actions = [torch.zeros(1, action_dim, requires_grad=True) for _ in range(horizon)]
        
        optimizer = torch.optim.Adam(actions, lr=0.1)
        
        for iteration in range(100):
            optimizer.zero_grad()
            
            total_cost = 0
            
            for _ in range(self.num_samples):
                current_state = initial_state
                trajectory_cost = 0
                
                for t in range(horizon):
                    action = actions[t]
                    next_state = self.dynamics(current_state, action)
                    step_cost = self.cost_fn(next_state, goal_state)
                    trajectory_cost += step_cost
                    current_state = next_state
                
                total_cost += trajectory_cost
            
            total_cost = total_cost / self.num_samples
            total_cost.backward()
            optimizer.step()
        
        return [a.detach() for a in actions]
```

### 3.3 Tube MPC

**方法**：使用鲁棒控制理论处理不确定性。

```python
class TubeMPC:
    def __init__(self, nominal_dynamics, error_dynamics, controller):
        self.nominal_dynamics = nominal_dynamics
        self.error_dynamics = error_dynamics
        self.controller = controller
    
    def compute_tube(self, state, goal, horizon=10):
        """
        计算鲁棒控制管
        
        参数:
            state: 当前状态
            goal: 目标状态
            horizon: 预测步数
        
        返回:
            标称轨迹和控制序列
        """
        nominal_trajectory = [state]
        control_sequence = []
        
        current_state = state
        
        for t in range(horizon):
            # 计算标称控制
            action = self.controller(current_state, goal)
            control_sequence.append(action)
            
            # 预测标称轨迹
            next_nominal = self.nominal_dynamics(current_state, action)
            nominal_trajectory.append(next_nominal)
            
            current_state = next_nominal
        
        return nominal_trajectory, control_sequence
```

---

## 4. MPC算法

### 4.1 基于梯度的MPC

**使用梯度下降进行优化**。

```python
class GradientMPC:
    def __init__(self, dynamics_model, horizon=10, lr=0.1):
        self.dynamics = dynamics_model
        self.horizon = horizon
        self.lr = lr
    
    def optimize(self, initial_state, goal_state, action_dim):
        """
        优化控制序列
        
        参数:
            initial_state: 初始状态
            goal_state: 目标状态
            action_dim: 动作维度
        
        返回:
            优化的动作序列
        """
        # 初始化动作
        actions = [torch.randn(1, action_dim, requires_grad=True) for _ in range(self.horizon)]
        
        for iteration in range(100):
            # 前向传播
            states = [initial_state]
            current_state = initial_state
            
            for t in range(self.horizon):
                next_state = self.dynamics(current_state, actions[t])
                states.append(next_state)
                current_state = next_state
            
            states = torch.stack(states)
            
            # 计算损失
            goal_loss = torch.sum((states - goal_state) ** 2)
            action_loss = sum(torch.sum(a ** 2) for a in actions)
            total_loss = goal_loss + 0.01 * action_loss
            
            # 梯度下降
            total_loss.backward()
            
            with torch.no_grad():
                for a in actions:
                    if a.grad is not None:
                        a -= self.lr * a.grad
                        a.grad.zero_()
        
        return [a.detach() for a in actions]
```

### 4.2 Cross-Entropy MPC

**使用交叉熵方法进行优化**。

```python
class CEMPC:
    def __init__(self, dynamics_model, horizon=10, num_samples=100, elite_ratio=0.1):
        self.dynamics = dynamics_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.elite_ratio = elite_ratio
    
    def optimize(self, initial_state, goal_state, action_dim):
        """
        使用交叉熵方法优化
        
        参数:
            initial_state: 初始状态
            goal_state: 目标状态
            action_dim: 动作维度
        
        返回:
            优化的动作序列
        """
        # 初始化分布
        mean = torch.zeros(self.horizon, action_dim)
        std = torch.ones(self.horizon, action_dim)
        
        for iteration in range(50):
            # 采样动作序列
            action_samples = []
            costs = []
            
            for _ in range(self.num_samples):
                actions = torch.randn(self.horizon, action_dim) * std + mean
                action_samples.append(actions)
                
                # 计算代价
                current_state = initial_state
                total_cost = 0
                
                for t in range(self.horizon):
                    next_state = self.dynamics(current_state, actions[t].unsqueeze(0))
                    total_cost += torch.sum((next_state - goal_state) ** 2)
                    current_state = next_state
                
                costs.append(total_cost.item())
            
            # 选择精英样本
            costs = torch.tensor(costs)
            elite_indices = costs.topk(int(self.num_samples * self.elite_ratio), largest=False)[1]
            elite_actions = torch.stack([action_samples[i] for i in elite_indices])
            
            # 更新分布
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0)
        
        # 返回均值作为最优动作序列
        return [mean[t] for t in range(self.horizon)]
```

### 4.3 iLQR MPC

**迭代线性二次调节器**。

```python
class ILQR MPC:
    def __init__(self, dynamics_model, horizon=10):
        self.dynamics = dynamics_model
        self.horizon = horizon
    
    def compute_ilqr(self, initial_state, goal_state, action_dim):
        """
        计算iLQR控制序列
        
        参数:
            initial_state: 初始状态
            goal_state: 目标状态
            action_dim: 动作维度
        
        返回:
            优化的动作序列
        """
        # 初始化动作序列
        actions = [torch.randn(1, action_dim) * 0.1 for _ in range(self.horizon)]
        
        for iteration in range(50):
            # 前向传播，计算轨迹
            states = [initial_state]
            current_state = initial_state
            
            for t in range(self.horizon):
                next_state = self.dynamics(current_state, actions[t])
                states.append(next_state)
                current_state = next_state
            
            states = torch.stack(states)
            
            # 计算代价和梯度
            # 简化的实现
            # 实际应用中需要计算二阶导数（Hessian）
            
            # 代价函数
            final_cost = torch.sum((states[-1] - goal_state) ** 2)
            
            # 更新动作
            # 这里使用简化的梯度下降
            for t in range(self.horizon):
                action_grad = 2 * actions[t] * 0.1
                actions[t] = actions[t] - 0.1 * action_grad
                actions[t] = torch.clamp(actions[t], -1, 1)
        
        return actions
```

---

## 5. 实践练习

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
            current_state: 当前状态
            goal_state: 目标状态
            action_dim: 动作维度
        
        返回:
            第一个控制动作
        """
        actions = self._optimize(current_state, goal_state, action_dim)
        return actions[0]
    
    def _optimize(self, current_state, goal_state, action_dim):
        """优化动作序列"""
        actions = [torch.zeros(1, action_dim, requires_grad=True) for _ in range(self.horizon)]
        optimizer = torch.optim.Adam(actions, lr=self.lr)
        
        for _ in range(100):
            optimizer.zero_grad()
            
            # 前向传播
            state = current_state
            total_cost = 0
            
            for t in range(self.horizon):
                next_state = self.dynamics(state, actions[t])
                total_cost += torch.sum((next_state - goal_state) ** 2)
                state = next_state
            
            total_cost.backward()
            optimizer.step()
        
        return [a.detach() for a in actions]

# 测试
dynamics = SimpleDynamics(state_dim=4, action_dim=2)
mpc = BasicMPC(dynamics, horizon=10)

current_state = torch.randn(4)
goal_state = torch.randn(4)

action = mpc.control(current_state, goal_state, action_dim=2)
print(f"选择的动作: {action.shape}")  # [1, 2]
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
            current_state: 当前状态
            goal_state: 目标状态
        
        返回:
            最优动作
        """
        action_dim = goal_state.shape[-1]
        
        # 定义代价函数
        def cost_fn(actions):
            state = current_state
            total_cost = 0
            for t in range(self.mpc_horizon):
                next_state = self.dynamics(state, actions[t].unsqueeze(0))
                total_cost += torch.sum((next_state - goal_state) ** 2)
                state = next_state
            return total_cost
        
        # 使用CEM优化
        optimizer = CEMOptimizer(num_samples=self.num_samples)
        actions = optimizer.optimize(cost_fn, action_dim, self.mpc_horizon)
        
        return actions[0]
    
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
4. KPn (2021). Model Predictive Control for Robotics.
