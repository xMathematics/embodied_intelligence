# 4.2 数值优化

## 目录

- [1. 优化问题概述](#1-优化问题概述)
- [2. 无约束优化](#2-无约束优化)
- [3. 约束优化](#3-约束优化)
- [4. 凸优化](#4-凸优化)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 优化问题概述

### 1.1 问题定义

**一般形式**：

$$\min_{x} f(x)$$
$$\text{s.t. } g_i(x) \leq 0, \quad i = 1, \ldots, m$$
$$h_j(x) = 0, \quad j = 1, \ldots, p$$

**分类**：
- **无约束优化**：$m = p = 0$
- **约束优化**：$m > 0$ 或 $p > 0$
- **凸优化**：目标函数和约束集都是凸的

### 1.2 局部最优与全局最优

**局部最优**：

$$f(x^*) \leq f(x) \quad \forall x \in B(x^*, \epsilon)$$

**全局最优**：

$$f(x^*) \leq f(x) \quad \forall x \in \text{dom}(f)$$

### 1.3 一阶必要条件

**梯度**：$\nabla f(x) = 0$

### 1.4 二阶必要条件

**梯度**：$\nabla f(x) = 0$

**Hessian矩阵**：$\nabla^2 f(x)$ 半正定

### 1.5 二阶充分条件

**梯度**：$\nabla f(x) = 0$

**Hessian矩阵**：$\nabla^2 f(x)$ 正定

---

## 2. 无约束优化

### 2.1 梯度下降法

**迭代公式**：

$$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$$

**步长选择**：
- 固定步长
- 线搜索（精确线搜索、Armijo-Goldstein条件）

**收敛性**：线性收敛。

### 2.2 牛顿法

**迭代公式**：

$$x_{k+1} = x_k - (\nabla^2 f(x_k))^{-1} \nabla f(x_k)$$

**收敛性**：二次收敛。

**缺点**：需要计算Hessian矩阵，计算量大。

### 2.3 拟牛顿法

**BFGS算法**：
- 近似Hessian矩阵的逆
- 迭代公式：
  $$x_{k+1} = x_k - \alpha_k B_k \nabla f(x_k)$$
  $$B_{k+1} = B_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}$$

其中 $s_k = x_{k+1} - x_k$，$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$。

**L-BFGS**：有限内存BFGS，适用于大规模问题。

### 2.4 共轭梯度法

**算法步骤**：
1. 初始化：$x_0$，$g_0 = \nabla f(x_0)$，$d_0 = -g_0$
2. 迭代：
   $$\alpha_k = \frac{g_k^T g_k}{d_k^T \nabla^2 f(x_k) d_k}$$
   $$x_{k+1} = x_k + \alpha_k d_k$$
   $$g_{k+1} = \nabla f(x_{k+1})$$
   $$\beta_k = \frac{g_{k+1}^T g_{k+1}}{g_k^T g_k}$$
   $$d_{k+1} = -g_{k+1} + \beta_k d_k$$

**收敛性**：最多 $n$ 次迭代收敛（二次函数）。

### 2.5 随机梯度下降（SGD）

**目标函数**：

$$f(x) = \frac{1}{n} \sum_{i=1}^{n} f_i(x)$$

**迭代公式**：

$$x_{k+1} = x_k - \alpha_k \nabla f_{i_k}(x_k)$$

其中 $i_k$ 是随机选取的样本索引。

**优点**：计算量小，适用于大规模数据。

**缺点**：收敛不稳定，需要学习率衰减。

---

## 3. 约束优化

### 3.1 拉格朗日乘数法

**等式约束**：

$$\mathcal{L}(x, \lambda) = f(x) - \lambda^T h(x)$$

**最优性条件**：

$$\nabla_x \mathcal{L} = 0$$
$$h(x) = 0$$

### 3.2 KKT条件

**不等式约束**：

$$\mathcal{L}(x, \lambda, \mu) = f(x) - \lambda^T g(x) - \mu^T h(x)$$

**KKT条件**：
1. $\nabla_x \mathcal{L} = 0$
2. $g(x) \leq 0$
3. $h(x) = 0$
4. $\lambda \geq 0$
5. $\lambda_i g_i(x) = 0$（互补松弛）

### 3.3 内点法

**障碍函数**：

$$\phi(x) = -\sum_{i=1}^{m} \log(-g_i(x))$$

**目标函数**：

$$f(x) + \mu \phi(x)$$

**迭代**：逐步减小 $\mu$，求解无约束问题。

### 3.4 序列二次规划（SQP）

**思想**：在每次迭代中求解二次规划子问题。

**子问题**：

$$\min_d \frac{1}{2} d^T \nabla^2 \mathcal{L} d + \nabla_x \mathcal{L}^T d$$
$$\text{s.t. } \nabla h_j(x)^T d + h_j(x) = 0$$
$$\nabla g_i(x)^T d + g_i(x) \leq 0$$

---

## 4. 凸优化

### 4.1 凸集

**定义**：集合 $C$ 是凸集，如果：

$$\theta x + (1-\theta) y \in C \quad \forall x, y \in C, \forall \theta \in [0, 1]$$

**常见凸集**：
- 仿射子空间
- 凸锥
- 半正定锥

### 4.2 凸函数

**定义**：函数 $f$ 是凸函数，如果：

$$f(\theta x + (1-\theta) y) \leq \theta f(x) + (1-\theta) f(y) \quad \forall x, y, \forall \theta \in [0, 1]$$

**一阶条件**：

$$f(y) \geq f(x) + \nabla f(x)^T (y - x)$$

**二阶条件**：

$$\nabla^2 f(x) \text{ 半正定}$$

### 4.3 凸优化问题

**定义**：

$$\min_{x \in C} f(x)$$

其中 $C$ 是凸集，$f$ 是凸函数。

**性质**：局部最优就是全局最优。

### 4.4 对偶理论

**拉格朗日对偶函数**：

$$g(\lambda, \mu) = \inf_x \mathcal{L}(x, \lambda, \mu)$$

**对偶问题**：

$$\max_{\lambda \geq 0, \mu} g(\lambda, \mu)$$

**强对偶性**：当原问题和对偶问题都可行且满足Slater条件时，对偶间隙为零。

---

## 5. 在具身智能中的应用

### 5.1 SLAM优化

**目标函数**：

$$\min_x \sum_{i,j} \rho(\|e_{ij}(x)\|^2)$$

其中 $e_{ij}(x)$ 是误差项。

**方法**：使用LM算法或高斯-牛顿法。

### 5.2 运动规划

**轨迹优化**：

$$\min_{x(\cdot)} \int_0^T (u^T R u + x^T Q x) dt$$

**方法**：使用间接法或直接法。

### 5.3 模型预测控制

**优化问题**：

$$\min_{u_0, \ldots, u_{N-1}} \sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + x_N^T P x_N$$
$$\text{s.t. } x_{k+1} = f(x_k, u_k)$$
$$u_k \in U$$

**方法**：使用SQP或内点法。

### 5.4 神经网络训练

**损失函数**：

$$L(\theta) = \frac{1}{n} \sum_{i=1}^{n} l(f(x_i; \theta), y_i)$$

**方法**：使用SGD、Adam、L-BFGS等。

---

## 6. 相关论文

### 优化理论

1. **"Convex Optimization"** - Boyd & Vandenberghe (2004)
   - 凸优化经典教材

2. **"Numerical Optimization"** - Nocedal & Wright (2006)
   - 数值优化权威教材

### 应用相关

3. **"Bundle Methods for Convex Optimization"** - Lemaréchal (2001)
   - 束方法

4. **"Sequential Quadratic Programming"** - Gill et al. (1981)
   - SQP算法

---

## 7. 实践练习

### 练习1：梯度下降法

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x0, alpha=0.01, max_iter=1000, tol=1e-6):
    """
    梯度下降法
    """
    x = x0.copy()
    history = [x.copy()]
    
    for _ in range(max_iter):
        grad = grad_f(x)
        x = x - alpha * grad
        history.append(x.copy())
        
        if np.linalg.norm(grad) < tol:
            break
    
    return x, np.array(history)

# 目标函数: f(x) = x1^2 + x2^2
f = lambda x: x[0]**2 + x[1]**2
grad_f = lambda x: np.array([2*x[0], 2*x[1]])

# 初始点
x0 = np.array([2, 1])

# 梯度下降
x_opt, history = gradient_descent(f, grad_f, x0, alpha=0.1)

print(f"最优解: x = {x_opt}")
print(f"目标函数值: f(x) = {f(x_opt):.6f}")

# 可视化
plt.figure(figsize=(8, 8))

# 等高线
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = X1**2 + X2**2
plt.contour(X1, X2, Z, levels=20)

# 迭代轨迹
plt.plot(history[:, 0], history[:, 1], 'ro-', markersize=5, label='Gradient Descent')
plt.scatter(x0[0], x0[1], color='green', s=100, label='Start')
plt.scatter(x_opt[0], x_opt[1], color='blue', s=100, label='Optimum')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Gradient Descent')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 练习2：牛顿法

```python
import numpy as np

def newton_method(f, grad_f, hess_f, x0, max_iter=100, tol=1e-6):
    """
    牛顿法
    """
    x = x0.copy()
    history = [x.copy()]
    
    for _ in range(max_iter):
        grad = grad_f(x)
        hess = hess_f(x)
        
        if np.linalg.norm(grad) < tol:
            break
        
        # 求解牛顿方程
        d = np.linalg.solve(hess, -grad)
        x = x + d
        history.append(x.copy())
    
    return x, np.array(history)

# 目标函数: f(x) = x1^2 + x2^2
f = lambda x: x[0]**2 + x[1]**2
grad_f = lambda x: np.array([2*x[0], 2*x[1]])
hess_f = lambda x: np.array([[2, 0], [0, 2]])

# 初始点
x0 = np.array([2, 1])

# 牛顿法
x_opt, history = newton_method(f, grad_f, hess_f, x0)

print(f"最优解: x = {x_opt}")
print(f"目标函数值: f(x) = {f(x_opt):.6f}")
print(f"迭代次数: {len(history)-1}")

# 比较梯度下降
x_opt_gd, history_gd = gradient_descent(f, grad_f, x0, alpha=0.1)
print(f"\n梯度下降迭代次数: {len(history_gd)-1}")
```

### 练习3：约束优化

```python
import numpy as np
from scipy.optimize import minimize

# 目标函数: f(x) = x1^2 + x2^2
def objective(x):
    return x[0]**2 + x[1]**2

# 约束: x1 + x2 >= 1
def constraint(x):
    return x[0] + x[1] - 1

# 初始点
x0 = np.array([0, 0])

# 约束定义
cons = ({'type': 'ineq', 'fun': constraint})

# 求解
result = minimize(objective, x0, constraints=cons)

print("约束优化结果:")
print(f"最优解: x = {result.x}")
print(f"目标函数值: f(x) = {result.fun:.6f}")
print(f"约束满足: x1 + x2 = {result.x[0] + result.x[1]:.6f}")

# 可视化
import matplotlib.pyplot as plt

x1 = np.linspace(-1, 3, 100)
x2 = np.linspace(-1, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = X1**2 + X2**2

plt.figure(figsize=(8, 8))
plt.contour(X1, X2, Z, levels=20)
plt.fill_between(x1, 1 - x1, 3, alpha=0.2, label='Feasible Region')
plt.scatter(result.x[0], result.x[1], color='red', s=100, label='Optimum')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Constrained Optimization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 练习4：凸优化

```python
import numpy as np
from scipy.optimize import minimize

# 凸函数: f(x) = exp(x1) + exp(x2)
def convex_objective(x):
    return np.exp(x[0]) + np.exp(x[1])

# 线性约束: x1 + x2 = 1
def linear_constraint(x):
    return x[0] + x[1] - 1

# 初始点
x0 = np.array([0, 0])

# 约束定义
cons = ({'type': 'eq', 'fun': linear_constraint})

# 求解
result = minimize(convex_objective, x0, constraints=cons)

print("凸优化结果:")
print(f"最优解: x = {result.x}")
print(f"目标函数值: f(x) = {result.fun:.6f}")
print(f"约束满足: x1 + x2 = {result.x[0] + result.x[1]:.6f}")

# 验证凸性：Hessian矩阵
def hessian(x):
    return np.diag([np.exp(x[0]), np.exp(x[1])])

hess = hessian(result.x)
eigenvalues = np.linalg.eigvals(hess)
print(f"\nHessian矩阵特征值: {eigenvalues}")
print(f"Hessian矩阵是否正定: {all(eigenvalues > 0)}")
```

### 练习5：L-BFGS算法

```python
import numpy as np
from scipy.optimize import minimize

# Rosenbrock函数
def rosenbrock(x):
    return sum(100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

# 梯度
def rosenbrock_grad(x):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n-1):
        grad[i] = -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
        grad[i+1] += 200*(x[i+1] - x[i]**2)
    return grad

# 初始点
x0 = np.array([-1.2, 1.0, 0.5, -0.8])

# 使用L-BFGS
result_lbfgs = minimize(rosenbrock, x0, jac=rosenbrock_grad, method='L-BFGS-B')

print("L-BFGS结果:")
print(f"最优解: x = {result_lbfgs.x}")
print(f"目标函数值: f(x) = {result_lbfgs.fun:.6f}")
print(f"迭代次数: {result_lbfgs.nit}")

# 使用BFGS对比
result_bfgs = minimize(rosenbrock, x0, jac=rosenbrock_grad, method='BFGS')

print("\nBFGS结果:")
print(f"最优解: x = {result_bfgs.x}")
print(f"目标函数值: f(x) = {result_bfgs.fun:.6f}")
print(f"迭代次数: {result_bfgs.nit}")

# 使用SGD对比
def sgd_rosenbrock(x0, lr=0.0001, max_iter=10000):
    x = x0.copy()
    for _ in range(max_iter):
        grad = rosenbrock_grad(x)
        x = x - lr * grad
    return x, rosenbrock(x)

x_sgd, f_sgd = sgd_rosenbrock(x0)
print(f"\nSGD结果:")
print(f"最优解: x = {x_sgd}")
print(f"目标函数值: f(x) = {f_sgd:.6f}")
```

---

**下一步**：[微分方程求解](03-differential-equations.md)