# 9.5 实际部署

## 目录

- [1. 引言](#1-引言)
- [2. 部署清单](#2-部署清单)
- [3. 调试与测试](#3-调试与测试)
- [4. 系统监控](#4-系统监控)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 实际部署考虑

从仿真到真实机器人的关键考虑：
- 硬件验证
- 安全措施
- 容错设计
- 持续监控
- 可解释性

```python
import time
from typing import Dict, Any
```

---

## 2. 部署清单

### 2.1 清单类

```python
class DeploymentChecklist:
    """部署清单"""
    
    def __init__(self):
        self.items = {}
        self.items['hardware'] = [
            ('Power supply', False),
            ('Motors/Actuators', False),
            ('Sensors', False),
            ('EStop', False),
            ('Network', False)
        ]
        self.items['software'] = [
            ('ROS Master', False),
            ('Perception nodes', False),
            ('SLAM node', False),
            ('Navigation node', False),
            ('Safety monitors', False)
        ]
    
    def check(self, category, item):
        """检查项"""
        if category in self.items:
            for i, (name, val) in enumerate(self.items[category]):
                if name == item:
                    self.items[category][i] = (name, True)
                    return True
        return False
    
    def check_all(self):
        """检查所有项"""
        all_ok = True
        print("Deployment Checklist:")
        
        for cat in self.items:
            print(f"\n  {cat.upper()}:")
            for name, ok in self.items[cat]:
                status = "✓" if ok else "✗"
                print(f"    {status} {name}")
                if not ok:
                    all_ok = False
        
        return all_ok
```

---

## 3. 调试与测试

### 3.1 调试工具

```python
class DebugLogger:
    """调试日志"""
    
    def __init__(self):
        self.logs = []
        self.start_time = time.time()
    
    def log(self, level, msg):
        """记录日志"""
        t = time.time() - self.start_time
        entry = (t, level, msg)
        self.logs.append(entry)
        print(f"[{t:.2f}] {level}: {msg}")
    
    def info(self, msg):
        self.log("INFO", msg)
    
    def warn(self, msg):
        self.log("WARN", msg)
    
    def error(self, msg):
        self.log("ERROR", msg)


class SimpleTestSuite:
    """简单测试套件"""
    
    def __init__(self):
        self.tests = []
        self.results = []
    
    def add_test(self, name, test_function):
        """添加测试"""
        self.tests.append((name, test_function))
    
    def run_all(self):
        """运行所有测试"""
        print("Running Test Suite...")
        
        for name, test in self.tests:
            try:
                result = test()
                self.results.append((name, result))
                if result:
                    print(f"  [PASS] {name}")
                else:
                    print(f"  [FAIL] {name}")
            except Exception as e:
                self.results.append((name, False))
                print(f"  [EXCEPTION] {name}: {e}")
        
        return sum(1 for name, ok in self.results if ok), len(self.tests)
```

---

## 4. 系统监控

### 4.1 监控系统

```python
class SystemMonitor:
    """系统监控"""
    
    def __init__(self):
        self.cpu = 0
        self.memory = 0
        self.battery = 100
        self.connections = {}
    
    def update(self):
        """更新（模拟）"""
        import random
        self.cpu = random.uniform(30, 70)
        self.memory = random.uniform(40, 80)
        self.battery = max(0, self.battery - random.uniform(0.5, 2.0))
    
    def check_health(self):
        """检查健康状态"""
        status = []
        
        if self.battery < 20:
            status.append(('WARN', 'Low battery'))
        
        if self.cpu > 90:
            status.append(('WARN', 'High CPU usage'))
        
        if not status:
            status.append(('OK', 'System nominal'))
        
        return status
    
    def print_status(self):
        """打印状态"""
        print(f"\n=== SYSTEM STATUS ===")
        print(f"CPU: {self.cpu:.1f}%, Mem: {self.memory:.1f}%")
        print(f"Battery: {self.battery:.1f}%")
        health = self.check_health()
        for level, msg in health:
            print(f"  {level}: {msg}")
```

---

## 5. 实践练习

### 练习1：部署清单

```python
def exercise_deployment():
    """部署练习"""
    print("=== 实际部署 ===")
    
    checklist = DeploymentChecklist()
    
    checklist.check('hardware', 'Power supply')
    checklist.check('hardware', 'EStop')
    checklist.check('hardware', 'Network')
    checklist.check('software', 'Safety monitors')
    
    checklist.check_all()
    print()
    
    logger = DebugLogger()
    logger.info("System starting...")
    logger.warn("Sensor data rate lower than expected")
    logger.info("System initialized")
    
    print()
    
    monitor = SystemMonitor()
    for i in range(3):
        monitor.update()
        monitor.print_status()
        time.sleep(0.5)

# exercise_deployment()
```

### 练习2：测试套件

```python
def exercise_tests():
    """测试练习"""
    print("=== 测试与调试 ===")
    
    suite = SimpleTestSuite()
    
    suite.add_test("Sensor test", lambda: True)
    suite.add_test("Motor test", lambda: True)
    suite.add_test("Safety test", lambda: True)
    
    pass_count, total = suite.run_all()
    print(f"\nPassed {pass_count}/{total} tests")

# exercise_tests()
```

---

恭喜！你已经完成了第九部分：系统集成（1周）！也完成了整个**11-perception-planning**模块的所有内容！

---

## 参考文献

1. Siegemund, J., et al. (2016). A Survey of Reconfigurable Manufacturing Systems in Robotics.
2. Quigley, M., et al. (2009). ROS: An Open-Source Robot Operating System.
3. Marques, L., et al. (2010). The 3D Simulators Stage, Gazebo, and MORSE.
4. Nourbakhsh, I. R. (2013). Introduction to Autonomous Mobile Robots.
5. Siegwart, R., Nourbakhsh, I. R., & Scaramuzza, D. (2011). Introduction to Autonomous Mobile Robots.
