
# 机器人控制模块

## 模块概述

机器人控制是具身智能的执行层，本模块涵盖基础控制、运动规划、力控制等内容。

## 学习目标

完成本模块学习后，您将能够：
- 理解基础控制理论
- 掌握运动规划算法
- 了解力控制原理

## 学习路径

### 第一部分：基础控制（1周）

#### 1.1 PID控制

**核心概念：**
- 比例-积分-微分控制
- 参数整定
- PID变种

**推荐资源：**
- 《Control Systems Engineering》- Nise
- 《PID Controllers: Theory, Design, and Tuning》- Astrom

**实践练习：**
- 实现PID控制器
- 调节参数优化控制效果

#### 1.2 状态反馈控制

**核心概念：**
- 状态空间表示
- 极点配置
- LQR控制

**实践练习：**
- 实现状态反馈控制器
- 设计LQR控制器

---

### 第二部分：运动规划（2周）

#### 2.1 路径规划

**核心概念：**
- 图搜索算法（A*、Dijkstra）
- 采样规划（RRT、RRT*）
- 概率路径规划

**推荐论文：**
- "Rapidly-exploring Random Trees: A New Tool for Path Planning"
- "Sampling-Based Algorithms for Optimal Motion Planning"

**实践练习：**
- 实现A*算法
- 实现RRT算法

#### 2.2 轨迹规划

**核心概念：**
- 多项式轨迹
- 轨迹优化
- 时间最优控制

**实践练习：**
- 生成多项式轨迹
- 优化轨迹平滑性

---

### 第三部分：力控制（1周）

#### 3.1 阻抗控制

**核心概念：**
- 阻抗模型
- 力/位置混合控制
- 自适应阻抗

**推荐论文：**
- "Impedance Control: An Approach to Manipulation"

**实践练习：**
- 理解阻抗控制原理

#### 3.2 柔顺控制

**核心概念：**
- 柔顺运动
- 力传感器反馈
- 接触力控制

**实践练习：**
- 模拟力控制场景

---

## 推荐学习资源

### 书籍
1. 《Robot Modeling and Control》- Spong et al.
2. 《Planning Algorithms》- LaValle
3. 《Control Systems Engineering》- Nise

### 在线课程
1. Coursera - Control of Mobile Robots
2. edX - Control Systems

### 工具
- ROS Control - 机器人控制框架
- MoveIt! - 运动规划框架
- Drake - 控制与规划库

## 学习评估

### 自测题
1. 解释PID控制器的三个组成部分
2. 比较A*与RRT的优缺点
3. 解释阻抗控制的原理

### 实践项目
1. 使用PID控制机械臂关节
2. 实现A*路径规划
3. 使用MoveIt!进行运动规划

---

**下一个模块：** [机器人硬件](../08-robot-hardware/README.md)
