# 9.4 仿真测试

## 目录

- [1. 引言](#1-引言)
- [2. 简单仿真环境](#2-简单仿真环境)
- [3. Gazebo仿真](#3-gazebo仿真)
- [4. 硬件在环](#4-硬件在环)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 仿真的好处

- 安全
- 低成本
- 快速
- 可复现

```python
import numpy as np
import time
import random
```

---

## 2. 简单仿真环境

### 2.1 2D网格世界

```python
class GridEnvironment:
    """2D网格环境"""
    
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.robot_pos = (1, 1)
        self.goal_pos = (width-2, height-2)
        self.obstacles = set()
        
        # 随机添加一些障碍
        for _ in range(width*height // 10):
            x = random.randint(1, width-2)
            y = random.randint(1, height-2)
            if (x,y) != self.robot_pos and (x,y) != self.goal_pos:
                self.obstacles.add((x,y))
    
    def reset(self):
        """重置环境"""
        self.robot_pos = (1,1)
        return self.robot_pos
    
    def step(self, action):
        """执行动作"""
        actions = {
            0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)
        }
        
        dx, dy = actions[action]
        nx, ny = self.robot_pos[0] + dx, self.robot_pos[1] + dy
        
        # 检查边界和障碍
        if 0 <= nx < self.width and 0 <= ny < self.height and (nx, ny) not in self.obstacles:
            self.robot_pos = (nx, ny)
        
        # 计算奖励
        done = (self.robot_pos == self.goal_pos)
        if done:
            reward = 10
        elif self.robot_pos in self.obstacles:
            reward = -10
        else:
            reward = -1
        
        return self.robot_pos, reward, done
    
    def render(self):
        """可视化"""
        s = ""
        for y in range(self.height):
            for x in range(self.width):
                pos = (x,y)
                if pos == self.robot_pos:
                    s += "R"
                elif pos == self.goal_pos:
                    s += "G"
                elif pos in self.obstacles:
                    s += "#"
                else:
                    s += "."
            s += "\n"
        print(s)
```

---

## 3. Gazebo仿真

### 3.1 模拟Gazebo接口

```python
class MockGazebo:
    """模拟Gazebo"""
    
    def __init__(self):
        self.world = None
        self.running = False
        self.robots = {}
    
    def spawn_robot(self, name, x, y, z):
        """加载机器人"""
        self.robots[name] = {'pos': (x,y,z)}
        print(f"[Gazebo] Spawned robot {name} at {x,y,z}")
    
    def start(self):
        """启动仿真"""
        self.running = True
        print("[Gazebo] Starting simulation")
    
    def step(self):
        """仿真步长"""
        if not self.running:
            return
        
        # 简单模拟：机器人随机动一点
        for name in self.robots:
            pos = self.robots[name]['pos']
            new_pos = (
                pos[0] + random.uniform(-0.1, 0.1),
                pos[1] + random.uniform(-0.1, 0.1),
                pos[2]
            )
            self.robots[name]['pos'] = new_pos
        
        print(f"[Gazebo] Step complete")
    
    def get_robot_pose(self, name):
        """获取机器人位姿"""
        if name in self.robots:
            return self.robots[name]['pos']
        return None
    
    def stop(self):
        """停止"""
        self.running = False
        print("[Gazebo] Stopped simulation")
```

---

## 4. 硬件在环

### 4.1 HIL模拟

```python
class HILSimulator:
    """硬件在环模拟"""
    
    def __init__(self):
        self.simulator = MockGazebo()
        self.hardware_connected = False
    
    def connect_hardware(self):
        """连接硬件（模拟）"""
        print("[HIL] Connecting to hardware...")
        self.hardware_connected = True
        return True
    
    def step(self):
        """步长"""
        if self.hardware_connected:
            print("[HIL] Stepping with hardware")
            self.simulator.step()
        else:
            print("[HIL] No hardware connected")
```

---

## 5. 实践练习

### 练习1：网格环境仿真

```python
def exercise_grid_sim():
    """网格环境练习"""
    print("=== 简单网格仿真 ===")
    
    env = GridEnvironment(10,10)
    
    state = env.reset()
    env.render()
    done = False
    
    # 简单策略：随机走
    for _ in range(100):
        action = random.randint(0,3)
        state, reward, done = env.step(action)
        
        env.render()
        print(f"Reward: {reward}, Done: {done}")
        
        if done:
            print("Reached goal!")
            break

# exercise_grid_sim()
```

### 练习2：Gazebo模拟

```python
def exercise_gazebo_sim():
    """Gazebo模拟"""
    print("=== Gazebo仿真 ===")
    
    gazebo = MockGazebo()
    gazebo.start()
    gazebo.spawn_robot('robot1', 0,0,0)
    
    for i in range(10):
        gazebo.step()
        pose = gazebo.get_robot_pose('robot1')
        print(f"Step {i}: Pose {pose}")
    
    gazebo.stop()

# exercise_gazebo_sim()
```

---

**下一节**：[实际部署](05-deployment.md)

---

## 参考文献

1. Koenig, N., & Howard, A. (2004). Design and Use Paradigms for Gazebo, An Open-Source Multi-Robot Simulator.
2. Quigley, M., et al. (2009). ROS: An Open-Source Robot Operating System.
3. Steinbaeck, J., et al. (2013). The ADE Environment for Development of Distributed Control Systems.
4. Krotkov, E., & Simmons, R. (1995). Lessons Learned in Designing and Using a System for Visual Task Execution.
