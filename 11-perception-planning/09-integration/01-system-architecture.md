# 9.1 系统架构

## 目录

- [1. 引言](#1-引言)
- [2. 分层架构](#2-分层架构)
- [3. 模块化设计](#3-模块化设计)
- [4. 系统设计模式](#4-系统设计模式)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 机器人系统架构

一个完整的机器人系统需要：
- 分层的组织结构
- 清晰的接口定义
- 模块化的组件设计
- 容错和恢复机制

```python
import time
import threading
import queue
from abc import ABC, abstractmethod
import json
```

---

## 2. 分层架构

### 2.1 分层系统

```python
class Layer(ABC):
    """层基类"""
    
    def __init__(self, name):
        self.name = name
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.thread = None
    
    @abstractmethod
    def process(self, data):
        """处理数据"""
        pass
    
    def start(self):
        """启动层"""
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()
    
    def stop(self):
        """停止层"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _run(self):
        """运行主循环"""
        while self.running:
            try:
                data = self.input_queue.get(timeout=0.1)
                result = self.process(data)
                if result is not None:
                    self.output_queue.put(result)
            except queue.Empty:
                continue


class PerceptionLayer(Layer):
    """感知层"""
    
    def __init__(self):
        super().__init__('Perception')
    
    def process(self, data):
        """感知处理"""
        print(f"[Perception] Processing: {data}")
        return {'type': 'perception', 'data': data}


class PlanningLayer(Layer):
    """规划层"""
    
    def __init__(self):
        super().__init__('Planning')
    
    def process(self, data):
        """规划处理"""
        print(f"[Planning] Processing: {data}")
        return {'type': 'plan', 'plan': 'sample plan'}


class ControlLayer(Layer):
    """控制层"""
    
    def __init__(self):
        super().__init__('Control')
    
    def process(self, data):
        """控制处理"""
        print(f"[Control] Processing: {data}")
        return {'type': 'command', 'value': 'move forward'}


class RobotSystem:
    """机器人系统"""
    
    def __init__(self):
        self.perception = PerceptionLayer()
        self.planning = PlanningLayer()
        self.control = ControlLayer()
        
        # 连接层
        self._connect_layers()
    
    def _connect_layers(self):
        """连接层"""
        # Perception -> Planning
        self._connect(self.perception, self.planning)
        # Planning -> Control
        self._connect(self.planning, self.control)
    
    def _connect(self, source, target):
        """连接两个层"""
        # 在后台线程中转发
        def forward():
            while True:
                data = source.output_queue.get()
                target.input_queue.put(data)
        
        t = threading.Thread(target=forward, daemon=True)
        t.start()
    
    def start(self):
        """启动系统"""
        print("Starting Robot System...")
        self.control.start()
        self.planning.start()
        self.perception.start()
    
    def stop(self):
        """停止系统"""
        print("Stopping Robot System...")
        self.perception.stop()
        self.planning.stop()
        self.control.stop()
    
    def add_sensor_data(self, data):
        """添加传感器数据"""
        self.perception.input_queue.put(data)
```

---

## 3. 模块化设计

### 3.1 模块接口

```python
class Module(ABC):
    """模块基类"""
    
    def __init__(self, name):
        self.name = name
        self.subscribers = []
    
    def publish(self, message):
        """发布消息"""
        for sub in self.subscribers:
            sub.receive(self.name, message)
    
    def subscribe(self, publisher):
        """订阅模块"""
        publisher.subscribers.append(self)
    
    @abstractmethod
    def receive(self, source, message):
        """接收消息"""
        pass


class SensorModule(Module):
    """传感器模块"""
    
    def __init__(self):
        super().__init__('Sensor')
        self.running = False
    
    def run(self):
        """模拟传感器数据"""
        self.running = True
        t = 0
        while self.running:
            data = {'time': t, 'sensor': 'camera', 'value': [t, t*2]}
            self.publish(data)
            t += 1
            time.sleep(0.5)
    
    def receive(self, source, message):
        pass


class PerceptionModule(Module):
    """感知模块"""
    
    def __init__(self):
        super().__init__('Perception')
    
    def receive(self, source, message):
        print(f"[Perception] Received from {source}: {message}")
        processed = {'detected': 'object', 'source': source}
        self.publish(processed)


class PlanningModule(Module):
    """规划模块"""
    
    def __init__(self):
        super().__init__('Planning')
    
    def receive(self, source, message):
        print(f"[Planning] Received from {source}: {message}")
        plan = {'steps': ['step1', 'step2'], 'source': source}
        self.publish(plan)


class ControlModule(Module):
    """控制模块"""
    
    def __init__(self):
        super().__init__('Control')
    
    def receive(self, source, message):
        print(f"[Control] Executing plan from {source}: {message}")
```

---

## 4. 系统设计模式

### 4.1 管道与过滤器模式

```python
class Component(ABC):
    """组件基类"""
    
    @abstractmethod
    def process(self, data):
        pass


class Pipeline:
    """管道"""
    
    def __init__(self):
        self.components = []
    
    def add(self, component):
        """添加组件"""
        self.components.append(component)
    
    def execute(self, initial_data):
        """执行管道"""
        data = initial_data
        for comp in self.components:
            data = comp.process(data)
            if data is None:
                break
        return data
```

---

## 5. 实践练习

### 练习1：分层系统

```python
def exercise_layered_system():
    """分层系统练习"""
    print("=== 分层系统架构 ===")
    
    robot = RobotSystem()
    robot.start()
    
    # 模拟一些数据
    for i in range(5):
        sensor_data = {'id': i, 'value': i*10}
        robot.add_sensor_data(sensor_data)
        time.sleep(0.5)
    
    time.sleep(2)
    robot.stop()

# exercise_layered_system()
```

### 练习2：模块化设计

```python
def exercise_modular():
    """模块化设计练习"""
    print("=== 模块化系统 ===")
    
    sensor = SensorModule()
    perception = PerceptionModule()
    planning = PlanningModule()
    control = ControlModule()
    
    # 连接模块
    perception.subscribe(sensor)
    planning.subscribe(perception)
    control.subscribe(planning)
    
    # 启动传感器
    sensor_thread = threading.Thread(target=sensor.run, daemon=True)
    sensor_thread.start()
    
    # 运行一会儿
    time.sleep(3)
    sensor.running = False

# exercise_modular()
```

---

**下一节**：[感知-规划-闭环](02-perception-planning-loop.md)

---

## 参考文献

1. Shaw, M., & Garlan, D. (1996). Software Architecture: Perspectives on an Emerging Discipline.
2. Brooks, R. A. (1986). A Robust Layered Control System for a Mobile Robot.
3. Siciliano, B., & Khatib, O. (2016). Springer Handbook of Robotics.
4. Quigley, M., et al. (2009). ROS: An Open-Source Robot Operating System.
