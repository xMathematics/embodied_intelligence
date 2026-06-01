# 9.2 感知-规划-闭环

## 目录

- [1. 引言](#1-引言)
- [2. 感知-规划-控制闭环](#2-感知-规划-控制闭环)
- [3. 状态估计](#3-状态估计)
- [4. 重规划机制](#4-重规划机制)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 闭环系统

完整的机器人系统是一个闭环：
```
感知 → 状态估计 → 规划 → 控制 → 执行 → 环境 → 感知 → ...
```

```python
import numpy as np
import time
import threading
from typing import List, Dict, Any
```

---

## 2. 感知-规划-控制闭环

### 2.1 闭环系统类

```python
class PerceptionPlanningControlLoop:
    """感知-规划-控制闭环"""
    
    def __init__(self):
        self.state = np.zeros(3)  # x, y, theta
        self.goal = np.array([5.0, 5.0, 0.0])
        self.running = False
        
        self.perception_thread = None
        self.planning_thread = None
        self.control_thread = None
        
        self.latest_sensor_data = None
        self.latest_plan = None
        self.latest_command = None
    
    def start(self):
        """启动闭环"""
        self.running = True
        print("Starting closed loop...")
        
        # 启动各个线程
        self.perception_thread = threading.Thread(target=self._perception_loop)
        self.planning_thread = threading.Thread(target=self._planning_loop)
        self.control_thread = threading.Thread(target=self._control_loop)
        
        self.perception_thread.start()
        self.planning_thread.start()
        self.control_thread.start()
    
    def stop(self):
        """停止闭环"""
        self.running = False
        print("Stopping closed loop...")
        
        if self.perception_thread:
            self.perception_thread.join()
        if self.planning_thread:
            self.planning_thread.join()
        if self.control_thread:
            self.control_thread.join()
    
    def _perception_loop(self):
        """感知循环"""
        while self.running:
            # 模拟传感器数据
            sensor_data = self._simulate_sensor_data()
            self.latest_sensor_data = sensor_data
            print(f"[Perception] Got data: {sensor_data}")
            
            time.sleep(0.2)
    
    def _simulate_sensor_data(self):
        """模拟传感器数据（含噪声）"""
        noise = np.random.normal(0, 0.1, 3)
        return self.state + noise
    
    def _planning_loop(self):
        """规划循环"""
        while self.running:
            if self.latest_sensor_data is not None:
                # 规划
                plan = self._create_plan(self.latest_sensor_data)
                self.latest_plan = plan
                print(f"[Planning] Created plan: {plan}")
            
            time.sleep(0.5)
    
    def _create_plan(self, current):
        """创建简单计划"""
        diff = self.goal - current
        distance = np.linalg.norm(diff[:2])
        
        if distance < 0.5:
            return 'stop'
        else:
            return 'move to goal'
    
    def _control_loop(self):
        """控制循环"""
        while self.running:
            if self.latest_plan is not None:
                # 执行控制
                command = self._generate_command(self.latest_plan)
                self.latest_command = command
                print(f"[Control] Executing command: {command}")
                
                # 执行动作（模拟）
                self._execute_action(command)
            
            time.sleep(0.1)
    
    def _generate_command(self, plan):
        """生成命令"""
        if plan == 'stop':
            return {'v': 0, 'w': 0}
        
        # 简单控制：朝目标走
        direction = self.goal - self.state
        theta = np.arctan2(direction[1], direction[0])
        return {'v': 0.5, 'w': theta - self.state[2]}
    
    def _execute_action(self, command):
        """执行动作（模拟）"""
        v = command['v']
        w = command['w']
        
        # 更新状态
        dt = 0.1
        self.state[0] += v * np.cos(self.state[2]) * dt
        self.state[1] += v * np.sin(self.state[2]) * dt
        self.state[2] += w * dt
```

---

## 3. 状态估计

### 3.1 简单滤波器

```python
class SimpleStateEstimator:
    """简单状态估计器"""
    
    def __init__(self):
        self.state = np.zeros(3)
        self.covariance = np.eye(3) * 0.1
    
    def predict(self, v, w, dt):
        """预测"""
        # 简单运动模型
        self.state[0] += v * np.cos(self.state[2]) * dt
        self.state[1] += v * np.sin(self.state[2]) * dt
        self.state[2] += w * dt
        
        # 增加协方差（简化）
        self.covariance += np.eye(3) * 0.01
    
    def update(self, z):
        """测量更新（简化）"""
        # 简单卡尔曼滤波形式
        K = 0.5
        self.state = (1 - K) * self.state + K * z
        self.covariance = (1 - K) * self.covariance
```

---

## 4. 重规划机制

### 4.1 重规划

```python
class Replanner:
    """重规划器"""
    
    def __init__(self):
        self.current_plan = None
        self.replan_threshold = 1.0  # 当误差超过这个值时重规划
    
    def check_replan(self, state, expected_state):
        """检查是否需要重规划"""
        error = np.linalg.norm(state - expected_state)
        return error > self.replan_threshold
    
    def replan(self, new_state):
        """重规划"""
        print(f"[Replanner] Re-planning...")
        # 这里是新的计划
        return f"replan from {new_state}"
```

---

## 5. 实践练习

### 练习1：运行闭环系统

```python
def exercise_closed_loop():
    """闭环系统练习"""
    print("=== 感知-规划-控制闭环 ===")
    
    robot_loop = PerceptionPlanningControlLoop()
    robot_loop.start()
    
    # 运行一会儿
    time.sleep(3)
    
    robot_loop.stop()

# exercise_closed_loop()
```

### 练习2：状态估计与重规划

```python
def exercise_state_replan():
    """状态估计和重规划练习"""
    print("=== 状态估计和重规划 ===")
    
    estimator = SimpleStateEstimator()
    replanner = Replanner()
    
    true_state = np.array([0.0, 0.0, 0.0])
    goal = np.array([5.0, 5.0])
    expected_path = np.linspace(true_state[:2], goal, 20)
    
    for i in range(20):
        if i < len(expected_path):
            expected = np.array([*expected_path[i], 0.0])
        
        # 模拟运动（有误差）
        true_state += np.array([0.25, 0.2, 0.0])
        
        # 模拟传感器
        sensor_data = true_state + np.random.normal(0, 0.1, 3)
        
        # 估计
        estimator.update(sensor_data)
        
        # 检查是否需要重规划
        if replanner.check_replan(estimator.state[:2], expected[:2]):
            new_plan = replanner.replan(estimator.state)
            print(f"Step {i}: {new_plan}")
        
        print(f"Step {i}: Estimated {estimator.state[:2]}, True {true_state[:2]}")
    
    print("Done.")

# exercise_state_replan()
```

---

**下一节**：[ROS集成](03-ros-integration.md)

---

## 参考文献

1. Thrun, S., Burgard, W., & Fox, D. (2005). Probabilistic Robotics.
2. Siciliano, B., et al. (2016). Springer Handbook of Robotics.
3. LaValle, S. M. (2006). Planning Algorithms.
4. Siegwart, R., Nourbakhsh, I. R., & Scaramuzza, D. (2011). Introduction to Autonomous Mobile Robots.
