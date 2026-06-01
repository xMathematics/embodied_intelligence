# 9.3 ROS集成

## 目录

- [1. 引言](#1-引言)
- [2. ROS基础](#2-ros基础)
- [3. 导航与MoveIt!](#3-导航与moveit)
- [4. ROS编程示例](#4-ros编程示例)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 ROS简介

ROS（Robot Operating System）是机器人软件开发的标准框架：
- 节点间通信
- 常用库
- 丰富工具
- 工具链

```python
from typing import Dict, List, Any
import time
```

---

## 2. ROS基础

### 2.1 模拟ROS节点

```python
class ROSNode:
    """模拟ROS节点"""
    
    def __init__(self, name):
        self.name = name
        self.publishers = {}
        self.subscribers = {}
    
    def create_publisher(self, topic, msg_type):
        """创建发布者"""
        self.publishers[topic] = {'type': msg_type}
    
    def create_subscriber(self, topic, msg_type, callback):
        """创建订阅者"""
        self.subscribers[topic] = {'type': msg_type, 'callback': callback}


class ROSBridge:
    """ROS通信模拟"""
    
    def __init__(self):
        self.topics = {}
        self._lock = threading.Lock()
    
    def publish(self, topic, message):
        """发布消息"""
        with self._lock.acquire()
        if topic not in self.topics:
            self.topics[topic] = []
        
        self.topics[topic].append(message)
        self._lock.release()
    
    def call_callbacks(self, topic, msg):
        """调用回调"""
        pass


class MockROSTopic:
    """模拟Topic"""
    
    def __init__(self, name):
        self.name = name
        self.subscribers = []
    
    def publish(self, msg):
        for sub in self.subscribers:
            sub(msg)
    
    def subscribe(self, callback):
        self.subscribers.append(callback)


# 全局模拟
_ros = {}


def create_topic(name):
    if name not in _ros:
        return _ros[name]
    topic = MockROSTopic(name)
    _ros[name] = topic
    return topic
```

---

## 3. 导航与MoveIt!

### 3.1 导航模拟

```python
class MockNavigation:
    """模拟导航系统"""
    
    def __init__(self):
        self.goal_topic = create_topic('/move_base_simple/goal')
        self.status_topic = create_topic('/move_base/status')
        
        self.current_pose = (0.0, 0.0, 0.0)
        self.goal = None
    
    def send_goal(self, x, y, theta):
        """发送导航目标"""
        self.goal = (x, y, theta)
        print(f"[Navigation] Goal received: ({x},{y},{theta}")
    
    def simulate(self):
        """模拟导航"""
        if self.goal:
            x, y, theta = self.goal
            dx = x - self.current_pose[0]
            dy = y - self.current_pose[1]
            dist = (dx**2 + dy**2)**0.5
            if dist < 0.5:
                print("[Navigation] Goal reached!")
                self.goal = None
            else:
                self.current_pose = (
                    self.current_pose[0] + 0.1*(dx/dist),
                    self.current_pose[1] + 0.1*(dy/dist),
                    theta
                )


class MockMoveIt:
    """模拟MoveIt!"""
    
    def __init__(self):
        self.plan_topic = create_topic('/move_group/goal')
    
    def plan_and_execute(self, target_joints):
        """规划并执行"""
        print(f"[MoveIt!] Planning to: {target_joints}")
        return {'status": "success"}
```

---

## 4. ROS编程示例

### 4.1 示例节点

```python
class MockLaserScan:
    """模拟激光扫描"""
    
    def __init__(self):
        self.scan_topic = create_topic('/scan')
    
    def publish_scan(self, ranges):
        """发布扫描数据"""
        msg = {'ranges': ranges}
        self.scan_topic.publish(msg)


class MockOdometry:
    """模拟里程计"""
    
    def __init__(self):
        self.odom_topic = create_topic('/odom')
    
    def publish_odom(self, x, y, theta):
        """发布里程计"""
        msg = {
            'pose': {'position': {'x': x, 'y': y},
            'orientation': {'theta': theta}
        }
        self.odom_topic.publish(msg)


class SimpleNavigatorNode:
    """简单导航节点"""
    
    def __init__(self):
        self.scan_sub = create_topic('/scan').subscribe(self.scan_callback)
        self.cmd_pub = create_topic('/cmd_vel')
    
    def scan_callback(self, msg):
        """激光回调"""
        ranges = msg['ranges']
        
        # 简单避障
        front = ranges[len(ranges)//2]
        if front < 1.0:
            self.avoid()
    
    def avoid(self):
        """避障"""
        cmd = {'linear': {'x': 0.0}, 'angular': {'z': 0.5}}
        self.cmd_pub.publish(cmd)
```

---

## 5. 实践练习

### 练习1：简单ROS系统

```python
def exercise_ros():
    """ROS示例"""
    print("=== ROS集成示例")
    
    nav = MockNavigation()
    
    print("Setting goal...")
    nav.send_goal(5.0, 5.0, 0.0)
    
    for i in range(20):
        nav.simulate()
        print(f"Step {i}: Pose {nav.current_pose}")
        time.sleep(0.1)
    
    print("Done.")

# exercise_ros()
```

### 练习2：MoveIt!示例

```python
def exercise_moveit():
    """MoveIt!示例"""
    print("=== MoveIt!示例")
    
    moveit = MockMoveIt()
    
    result = moveit.plan_and_execute([0.5, -0.3, 0.2])
    print(f"MoveIt! result: {result}")
    
    print("Done.")

# exercise_moveit()
```

---

**下一节**：[仿真测试](04-simulation-testing.md)

---

## 参考文献

1. Quigley, M., et al. (2009). ROS: An Open-Source Robot Operating System.
2. Gerkey, B., et al. (2010). The Player/Stage Project: Tools for Multi-Robot and Distributed Sensor Systems.
3. Coleman, D., et al. (2014). Reducing the Barrier to Entry of Complex Robotic Software: A Case Study of MoveIt!
4. Marder-Eppstein, E., et al. (2010). The Office Marathon: Robust Navigation in an Indoor Office Environment.
