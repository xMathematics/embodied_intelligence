
# 机器人学基础模块

## 模块概述

机器人学是具身智能的载体，本模块涵盖机器人运动学、动力学、感知等核心内容。

## 学习目标

完成本模块学习后，您将能够：
- 理解机器人运动学原理
- 掌握机器人动力学建模
- 了解机器人感知系统

## 学习路径

### 第一部分：机器人运动学（2周）

#### 1.1 正运动学

**核心概念：**
- 连杆参数（DH参数）
- 齐次变换
- 正向运动学求解

**推荐资源：**
- 《Introduction to Robotics: Mechanics and Control》- Craig
- 《Modern Robotics》- Lynch & Park

**实践练习：**
- 推导PUMA机械臂的正运动学
- 实现正运动学求解器

#### 1.2 逆运动学

**核心概念：**
- 解析解与数值解
- Jacobian矩阵
- 封闭解条件

**实践练习：**
- 实现2自由度机械臂的逆运动学
- 理解Jacobian矩阵的作用

---

### 第二部分：机器人动力学（1周）

#### 2.1 动力学建模

**核心概念：**
- Lagrange-Euler方程
- Newton-Euler方程
- 动力学方程的简化

**推荐资源：**
- 《Robot Dynamics and Control》- Spong

**实践练习：**
- 推导简单机械臂的动力学方程

#### 2.2 操作空间控制

**核心概念：**
- 操作空间动力学
- 雅可比转置控制
- 阻抗控制

**实践练习：**
- 理解操作空间控制原理

---

### 第三部分：机器人感知（1周）

#### 3.1 传感器技术

**核心概念：**
- 激光雷达（LiDAR）
- RGB-D相机
- 惯性测量单元（IMU）

**实践练习：**
- 理解不同传感器的工作原理

#### 3.2 状态估计

**核心概念：**
- 卡尔曼滤波
- 传感器融合
- 位姿估计

**实践练习：**
- 实现简单的传感器融合

---

## 推荐学习资源

### 书籍
1. 《Introduction to Robotics: Mechanics and Control》- Craig
2. 《Modern Robotics: Mechanics, Planning, and Control》- Lynch & Park
3. 《Probabilistic Robotics》- Thrun et al.

### 在线课程
1. Coursera - Robotics Specialization
2. edX - Robotics Fundamentals

### 工具
- ROS - 机器人操作系统
- PyKDL - 运动学库
- OpenRAVE - 机器人仿真

## 学习评估

### 自测题
1. 解释DH参数的含义
2. 比较解析逆运动学与数值逆运动学
3. 解释传感器融合的必要性

### 实践项目
1. 使用ROS实现机械臂运动学
2. 实现简单的状态估计器
3. 模拟机器人传感器数据

---

**下一个模块：** [机器人控制](../07-robot-control/README.md)
