
# 2.4 优化理论

## 目录

- [1. 优化问题概述](#1-优化问题概述)
- [2. 无约束优化](#2-无约束优化)
- [3. 约束优化](#3-约束优化)
- [4. 凸优化](#4-凸优化)
- [5. 数值优化方法](#5-数值优化方法)
- [6. 在具身智能中的应用](#6-在具身智能中的应用)
- [7. 相关论文](#7-相关论文)
- [8. 实践练习](#8-实践练习)

---

## 1. 优化问题概述

### 1.1 定义

**优化问题**是寻找使目标函数取得极值的变量值：

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{或} \quad \max_{\mathbf{x}} f(\mathbf{x})$$

其中 $\mathbf{x} \in \mathbb{R}^n$ 是决策变量，$f: \mathbb{R}^n \to \mathbb{R}$ 是目标函数。

### 1.2 问题分类

**按约束分**：
- 无约束优化：$\min_{\mathbf{x}} f(\mathbf{x})$
- 约束优化：$\min_{\mathbf{x}} f(\mathbf{x}) \text{ s.t. } g_i(\mathbf{x}) \leq 0, h_j(\mathbf{x}) = 0$

**按函数性质分**：
- 线性规划：目标和约束都是线性的
- 非线性规划：目标或约束是非线性的
- 凸优化：目标函数凸，约束集凸

**按变量类型分**：
- 连续优化：变量连续
- 离散优化：变量离散
- 整数规划：变量取整数值

### 1.3 最优性条件

**局部最优**：存在邻域 $N$，使得 $\forall \mathbf{x} \in N, f(\mathbf{x}^*) \leq f(\mathbf{x})$

**全局最优**：$\forall \mathbf{x} \in \mathbb{R}^n, f(\mathbf{x}^*) \leq f(\mathbf{x})$

---

## 2. 无约束优化

### 2.1 最优性条件

**一阶必要条件**：若 $f$ 在 $\mathbf{x}^*$ 可微且取极值，则：

$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

**二阶充分条件**：若 $\nabla f(\mathbf{x}^*) = \mathbf{0}$ 且 Hessian 矩阵 $\nabla^2 f(\mathbf{x}^*)$ 正定，则 $\mathbf{x}^*$ 是严格局部极小点。

### 2.2 梯度下降法

**基本思想**：沿负梯度方向更新

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \eta_k \nabla f(\mathbf{x}_k)$$

**步长选择**：
- 固定步长：$\eta_k = \eta$
- 线搜索：找到合适的 $\eta_k$
- Armijo 条件：$f(\mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)) \leq f(\mathbf{x}_k) - c\eta \|\nabla f(\mathbf{x}_k)\|^2$

**收敛性**：
- 凸函数：收敛到全局最优
- 非凸函数：可能收敛到局部最优

### 2.3 牛顿法

**基本思想**：使用二阶信息

$$\mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$$

**优点**：
- 二次收敛速度
- 不需要步长选择

**缺点**：
- 需要计算和求逆 Hessian 矩阵
- Hessian 可能不正定

### 2.4 拟牛顿法

**BFGS算法**：

用近似矩阵 $\mathbf{B}_k \approx \nabla^2 f(\mathbf{x}_k)$ 代替 Hessian：

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{B}_k^{-1} \nabla f(\mathbf{x}_k)$$

**更新公式**：

$$\mathbf{B}_{k+1} = \mathbf{B}_k + \frac{\mathbf{y}_k\mathbf{y}_k^T}{\mathbf{y}_k^T\mathbf{s}_k} - \frac{\mathbf{B}_k\mathbf{s}_k\mathbf{s}_k^T\mathbf{B}_k}{\mathbf{s}_k^T\mathbf{B}_k\mathbf{s}_k}$$

其中 $\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k$，$\mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$。

### 2.5 共轭梯度法

**基本思想**：搜索方向共轭

$$\mathbf{d}_{k+1} = -\nabla f(\mathbf{x}_{k+1}) + \beta_k \mathbf{d}_k$$

**FR公式**：

$$\beta_k^{FR} = \frac{\|\nabla f(\mathbf{x}_{k+1})\|^2}{\|\nabla f(\mathbf{x}_k)\|^2}$$

**PRP公式**：

$$\beta_k^{PRP} = \frac{\nabla f(\mathbf{x}_{k+1})^T(\nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k))}{\|\nabla f(\mathbf{x}_k)\|^2}$$

---

## 3. 约束优化

### 3.1 等式约束

**问题**：

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad h(\mathbf{x}) = 0$$

**拉格朗日乘数法**：

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda h(\mathbf{x})$$

**KKT条件**：

$$\nabla_{\mathbf{x}} \mathcal{L} = \mathbf{0}, \quad h(\mathbf{x}) = 0$$

### 3.2 不等式约束

**问题**：

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0$$

**KKT条件**：

1. 平稳性：$\nabla f(\mathbf{x}^*) + \sum \mu_i \nabla g_i(\mathbf{x}^*) = \mathbf{0}$
2. 原始可行性：$g_i(\mathbf{x}^*) \leq 0$
3. 对偶可行性：$\mu_i \geq 0$
4. 互补松弛：$\mu_i g_i(\mathbf{x}^*) = 0$

### 3.3 一般约束问题

**问题**：

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, h_j(\mathbf{x}) = 0$$

**增广拉格朗日法**：

$$\mathcal{L}_\rho(\mathbf{x}, \lambda, \mu) = f(\mathbf{x}) + \sum \lambda_j h_j(\mathbf{x}) + \sum \mu_i g_i(\mathbf{x}) + \frac{\rho}{2}\left(\sum h_j(\mathbf{x})^2 + \sum g_i(\mathbf{x})^2\right)$$

### 3.4 内点法

**障碍函数法**：

$$\min_{\mathbf{x}} f(\mathbf{x}) - \mu \sum \ln(-g_i(\mathbf{x}))$$

当 $\mu \to 0$ 时，解趋近于原问题的解。

---

## 4. 凸优化

### 4.1 凸集

**定义**：集合 $C$ 是凸集，如果 $\forall \mathbf{x}, \mathbf{y} \in C, \theta \in [0, 1]$：

$$\theta \mathbf{x} + (1-\theta)\mathbf{y} \in C$$

**性质**：
- 凸集的交集是凸集
- 超平面是凸集
- 半空间是凸集

### 4.2 凸函数

**定义**：$f$ 是凸函数，如果定义域是凸集且：

$$f(\theta \mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y})$$

**判定**：
- 一阶条件：$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x})$
- 二阶条件：$\nabla^2 f(\mathbf{x}) \succeq 0$（半正定）

### 4.3 凸优化问题

**标准形式**：

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \mathbf{A}\mathbf{x} = \mathbf{b}$$

其中 $f$ 和 $g_i$ 是凸函数。

**性质**：
- 局部最优即全局最优
- KKT条件是充要条件

### 4.4 对偶理论

**拉格朗日对偶**：

$$g(\lambda, \mu) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \lambda, \mu)$$

**对偶问题**：

$$\max_{\lambda, \mu} g(\lambda, \mu) \quad \text{s.t.} \quad \mu \geq 0$$

**强对偶**：原问题和对偶问题的最优值相等。

---

## 5. 数值优化方法

### 5.1 线搜索方法

**精确线搜索**：

$$\eta_k = \arg\min_{\eta > 0} f(\mathbf{x}_k - \eta \nabla f(\mathbf{x}_k))$$

**非精确线搜索**（Wolfe条件）：
1. 充分下降：$f(\mathbf{x}_k + \eta \mathbf{d}_k) \leq f(\mathbf{x}_k) + c_1 \eta \nabla f(\mathbf{x}_k)^T \mathbf{d}_k$
2. 曲率条件：$\nabla f(\mathbf{x}_k + \eta \mathbf{d}_k)^T \mathbf{d}_k \geq c_2 \nabla f(\mathbf{x}_k)^T \mathbf{d}_k$

### 5.2 信赖域方法

**思想**：在当前点附近用二次模型近似

$$\min_{\mathbf{p}} m_k(\mathbf{p}) = f(\mathbf{x}_k) + \nabla f(\mathbf{x}_k)^T \mathbf{p} + \frac{1}{2}\mathbf{p}^T \mathbf{B}_k \mathbf{p}$$

$$\text{s.t.} \quad \|\mathbf{p}\| \leq \Delta_k$$

### 5.3 随机优化

**随机梯度下降（SGD）**：

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \eta_k \nabla f_{i_k}(\mathbf{x}_k)$$

其中 $i_k$ 是随机选择的样本索引。

**Adam优化器**：

$$m_{k+1} = \beta_1 m_k + (1-\beta_1)\nabla f(\mathbf{x}_k)$$

$$v_{k+1} = \beta_2 v_k + (1-\beta_2)\nabla f(\mathbf{x}_k)^2$$

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \frac{m_{k+1}}{\sqrt{v_{k+1}} + \epsilon}$$

---

## 6. 在具身智能中的应用

### 6.1 神经网络训练

**损失函数最小化**：

$$\min_{\theta} \frac{1}{N}\sum_{i=1}^{N} L(f(\mathbf{x}_i; \theta), y_i)$$

**优化器**：SGD、Adam、RMSprop

### 6.2 机器人轨迹优化

**问题**：

$$\min_{\mathbf{q}(t)} \int_0^T \|\ddot{\mathbf{q}}(t)\|^2 dt$$

$$\text{s.t.} \quad \mathbf{q}(0) = \mathbf{q}_0, \mathbf{q}(T) = \mathbf{q}_T$$

### 6.3 SLAM优化

**位姿图优化**：

$$\min_{\mathbf{T}_1, \ldots, \mathbf{T}_n} \sum_{(i,j) \in \mathcal{E}} \|\mathbf{T}_{ij} - \mathbf{T}_i^{-1}\mathbf{T}_j\|^2$$

### 6.4 模型预测控制

**问题**：

$$\min_{\mathbf{u}_0, \ldots, \mathbf{u}_{N-1}} \sum_{k=0}^{N-1} \|\mathbf{x}_k - \mathbf{x}_{ref}\|^2 + \|\mathbf{u}_k\|^2$$

$$\text{s.t.} \quad \mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k)$$

---

## 7. 相关论文

### 优化理论

1. **"Convex Optimization"** - Boyd & Vandenberghe (2004)
   - 凸优化经典教材

2. **"Numerical Optimization"** - Nocedal & Wright (2006)
   - 数值优化权威参考书

3. **"Nonlinear Programming"** - Bertsekas (1999)
   - 非线性规划经典

### 机器学习优化

4. **"Adam: A Method for Stochastic Optimization"** - Kingma & Ba (2015)
   - Adam优化器

5. **"On the Convergence of Adam and Beyond"** - Reddi et al. (2018)
   - Adam收敛性分析

### 机器人优化

6. **"Trajectory Optimization for Robotics"** - Betts (1998)
   - 机器人轨迹优化

7. **"Factor Graphs for Robot Perception"** - Dellaert & Kaess (2017)
   - 因子图在SLAM中的应用

---

## 8. 实践练习

### 练习1：梯度下降实现

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x0, learning_rate=0.1, max_iter=1000, tol=1e-6):
    """
    梯度下降算法
    """
    x = x0.copy()
    trajectory = [x.copy()]
    f_values = [f(x)]
    
    for i in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tol:
            print(f"收敛于第 {i} 次迭代")
            break
        
        x = x - learning_rate * grad
        trajectory.append(x.copy())
        f_values.append(f(x))
    
    return x, np.array(trajectory), np.array(f_values)

# 测试函数：Rosenbrock函数
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

# 运行优化
x0 = np.array([-1.5, 1.5])
x_opt, trajectory, f_values = gradient_descent(rosenbrock, rosenbrock_grad, x0, learning_rate=0.0002)

print(f"初始点: {x0}")
print(f"最优解: {x_opt}")
print(f"最优值: {f_values[-1]:.6f}")

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', markersize=3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Optimization Trajectory')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(f_values)
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.title('Objective Function Value')
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 练习2：牛顿法实现

```python
import numpy as np

def newton_method(f, grad_f, hess_f, x0, max_iter=100, tol=1e-6):
    """
    牛顿法
    """
    x = x0.copy()
    trajectory = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            print(f"收敛于第 {i} 次迭代")
            break
        
        # 求解牛顿方向
        try:
            delta_x = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            print("Hessian矩阵奇异，添加正则化")
            delta_x = np.linalg.solve(hess + 0.01*np.eye(len(x)), -grad)
        
        # 线搜索
        alpha = 1.0
        while f(x + alpha*delta_x) > f(x) + 0.0001*alpha*grad.dot(delta_x):
            alpha *= 0.5
        
        x = x + alpha * delta_x
        trajectory.append(x.copy())
    
    return x, np.array(trajectory)

# 测试函数
def f(x):
    return x[0]**2 + 2*x[1]**2

def grad_f(x):
    return np.array([2*x[0], 4*x[1]])

def hess_f(x):
    return np.array([[2, 0], [0, 4]])

# 运行优化
x0 = np.array([3.0, 3.0])
x_opt, trajectory = newton_method(f, grad_f, hess_f, x0)

print(f"初始点: {x0}")
print(f"最优解: {x_opt}")
print(f"迭代次数: {len(trajectory)-1}")
```

### 练习3：约束优化（拉格朗日乘数法）

```python
import sympy as sp

# 定义符号
x, y, lam = sp.symbols('x y lambda')

# 目标函数
f = x**2 + y**2

# 约束条件
g = x + y - 1

# 拉格朗日函数
L = f + lam * g

# 求解KKT条件
eq1 = sp.diff(L, x)
eq2 = sp.diff(L, y)
eq3 = sp.diff(L, lam)

solution = sp.solve([eq1, eq2, eq3], [x, y, lam])

print(f"优化问题: min {f} s.t. {g} = 0")
print(f"\n解: {solution}")

# 验证二阶条件
x_opt = solution[0][0]
y_opt = solution[0][1]

print(f"\n最优解: x = {x_opt}, y = {y_opt}")
print(f"最优值: {f.subs({x: x_opt, y: y_opt})}")
```

### 练习4：凸优化问题

```python
import numpy as np
from scipy.optimize import minimize

# 定义凸优化问题
# min x^2 + y^2
# s.t. x + y >= 1

def objective(x):
    return x[0]**2 + x[1]**2

def grad_objective(x):
    return np.array([2*x[0], 2*x[1]])

# 约束：x + y - 1 >= 0
constraints = {
    'type': 'ineq',
    'fun': lambda x: x[0] + x[1] - 1
}

# 初始点
x0 = np.array([0.0, 0.0])

# 求解
result = minimize(objective, x0, method='SLSQP', jac=grad_objective, constraints=constraints)

print(f"优化结果:")
print(f"  最优解: x = {result.x[0]:.6f}, y = {result.x[1]:.6f}")
print(f"  最优值: {result.fun:.6f}")
print(f"  收敛: {result.success}")
print(f"  迭代次数: {result.nit}")

# 验证约束
print(f"\n约束验证: x + y = {result.x[0] + result.x[1]:.6f} >= 1")
```

### 练习5：Adam优化器实现

```python
import numpy as np
import matplotlib.pyplot as plt

def adam_optimizer(f, grad_f, x0, learning_rate=0.01, beta1=0.9, beta2=0.999, 
                   epsilon=1e-8, max_iter=1000):
    """
    Adam优化器
    """
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    trajectory = [x.copy()]
    
    for t in range(1, max_iter + 1):
        grad = grad_f(x)
        
        # 更新一阶矩估计
        m = beta1 * m + (1 - beta1) * grad
        
        # 更新二阶矩估计
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        
        # 偏差修正
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # 更新参数
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        trajectory.append(x.copy())
    
    return x, np.array(trajectory)

# 测试函数
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    return np.array([
        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

# 运行优化
x0 = np.array([0.0, 0.0])
x_opt, trajectory = adam_optimizer(rosenbrock, rosenbrock_grad, x0)

print(f"Adam优化结果:")
print(f"  最优解: x = {x_opt[0]:.6f}, y = {x_opt[1]:.6f}")
print(f"  最优值: {rosenbrock(x_opt):.6f}")
print(f"  迭代次数: {len(trajectory)-1}")

# 可视化轨迹
plt.figure(figsize=(8, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.5)
plt.plot(trajectory[0, 0], trajectory[0, 1], 'ro', label='Start')
plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'go', label='End')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Adam Optimization Trajectory')
plt.legend()
plt.grid(True)
plt.show()
```

---

**微积分部分完成！** 

接下来可以学习 [概率论与统计](../probability/01-probability-basics.md)
