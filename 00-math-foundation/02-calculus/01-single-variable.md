
# 2.1 一元微积分

## 目录

- [1. 极限与连续性](#1-极限与连续性)
- [2. 导数](#2-导数)
- [3. 积分](#3-积分)
- [4. 级数](#4-级数)
- [5. 在具身智能中的应用](#5-在具身智能中的应用)
- [6. 相关论文](#6-相关论文)
- [7. 实践练习](#7-实践练习)

---

## 1. 极限与连续性

### 1.1 极限的定义

**直观理解**：当 $x$ 接近 $a$ 时，$f(x)$ 接近 $L$。

**严格定义**（$\epsilon$-$\delta$ 定义）：

$$\lim_{x \to a} f(x) = L \iff \forall \epsilon > 0, \exists \delta > 0: 0 < |x - a| < \delta \Rightarrow |f(x) - L| < \epsilon$$

### 1.2 极限的性质

- **唯一性**：极限若存在则唯一
- **线性性**：$\lim_{x \to a} [\alpha f(x) + \beta g(x)] = \alpha \lim_{x \to a} f(x) + \beta \lim_{x \to a} g(x)$
- **乘积**：$\lim_{x \to a} [f(x)g(x)] = \lim_{x \to a} f(x) \cdot \lim_{x \to a} g(x)$
- **商**：$\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{\lim_{x \to a} f(x)}{\lim_{x \to a} g(x)}$（分母不为零）

### 1.3 重要极限

$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

$$\lim_{x \to \infty} \left(1 + \frac{1}{x}\right)^x = e$$

$$\lim_{x \to 0} \frac{e^x - 1}{x} = 1$$

### 1.4 连续性

**定义**：$f$ 在 $a$ 点连续，如果：

$$\lim_{x \to a} f(x) = f(a)$$

**等价条件**：
- $\lim_{x \to a^+} f(x) = \lim_{x \to a^-} f(x) = f(a)$

**性质**：
- 连续函数的和、差、积、商（分母不为零）仍连续
- 连续函数的复合仍连续

---

## 2. 导数

### 2.1 导数的定义

**几何意义**：切线的斜率

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**等价形式**：

$$f'(x) = \lim_{x \to a} \frac{f(x) - f(a)}{x - a}$$

### 2.2 求导法则

#### 基本导数

| 函数 $f(x)$ | 导数 $f'(x)$ |
|------------|-------------|
| $c$（常数） | $0$ |
| $x^n$ | $nx^{n-1}$ |
| $e^x$ | $e^x$ |
| $\ln x$ | $\frac{1}{x}$ |
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x$ |

#### 运算法则

**和差法则**：

$$(f \pm g)' = f' \pm g'$$

**积法则**：

$$(fg)' = f'g + fg'$$

**商法则**：

$$\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$$

**链式法则**：

$$(f \circ g)' = (f' \circ g) \cdot g'$$

或记为：

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

### 2.3 高阶导数

**二阶导数**：

$$f''(x) = \frac{d^2f}{dx^2} = \frac{d}{dx}\left(\frac{df}{dx}\right)$$

**n阶导数**：

$$f^{(n)}(x) = \frac{d^n f}{dx^n}$$

### 2.4 导数的应用

#### 函数的单调性

- $f'(x) > 0$：$f$ 单调递增
- $f'(x) < 0$：$f$ 单调递减

#### 极值

**必要条件**：若 $f$ 在 $x_0$ 可导且取极值，则 $f'(x_0) = 0$

**充分条件**：
- 一阶导数变号：$f'$ 从正变负为极大值，从负变正为极小值
- 二阶导数：$f''(x_0) > 0$ 为极小值，$f''(x_0) < 0$ 为极大值

#### 凹凸性

- $f''(x) > 0$：凹函数（convex）
- $f''(x) < 0$：凸函数（concave）

**拐点**：$f''(x) = 0$ 且变号

#### 泰勒展开

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots + \frac{f^{(n)}(a)}{n!}(x-a)^n + R_n(x)$$

**麦克劳林展开**（$a=0$）：

$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$$

$$\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$$

$$\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots$$

$$\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots$$

---

## 3. 积分

### 3.1 不定积分

**定义**：若 $F'(x) = f(x)$，则 $F$ 是 $f$ 的原函数。

$$\int f(x) dx = F(x) + C$$

**基本积分公式**：

| 函数 $f(x)$ | 积分 $\int f(x) dx$ |
|------------|-------------------|
| $x^n$（$n \neq -1$） | $\frac{x^{n+1}}{n+1} + C$ |
| $\frac{1}{x}$ | $\ln|x| + C$ |
| $e^x$ | $e^x + C$ |
| $\sin x$ | $-\cos x + C$ |
| $\cos x$ | $\sin x + C$ |

### 3.2 积分技巧

#### 换元法

$$\int f(g(x))g'(x)dx = \int f(u)du$$

#### 分部积分

$$\int u \, dv = uv - \int v \, du$$

#### 部分分式分解

用于有理函数的积分。

### 3.3 定积分

**定义**：

$$\int_a^b f(x) dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i^*)\Delta x$$

**几何意义**：曲线与x轴之间的有向面积

**微积分基本定理**：

$$\int_a^b f(x) dx = F(b) - F(a)$$

其中 $F$ 是 $f$ 的原函数。

### 3.4 定积分的性质

- **线性性**：$\int_a^b [\alpha f(x) + \beta g(x)] dx = \alpha \int_a^b f(x) dx + \beta \int_a^b g(x) dx$
- **区间可加性**：$\int_a^b f(x) dx + \int_b^c f(x) dx = \int_a^c f(x) dx$
- **对称性**：若 $f$ 为奇函数，$\int_{-a}^a f(x) dx = 0$；若 $f$ 为偶函数，$\int_{-a}^a f(x) dx = 2\int_0^a f(x) dx$

### 3.5 数值积分

#### 矩形法

$$\int_a^b f(x) dx \approx \sum_{i=1}^n f(x_i) \Delta x$$

#### 梯形法

$$\int_a^b f(x) dx \approx \frac{\Delta x}{2} \left[f(x_0) + 2\sum_{i=1}^{n-1}f(x_i) + f(x_n)\right]$$

#### 辛普森法

$$\int_a^b f(x) dx \approx \frac{\Delta x}{3} \left[f(x_0) + 4\sum_{i=1,3,5...}f(x_i) + 2\sum_{i=2,4,6...}f(x_i) + f(x_n)\right]$$

---

## 4. 级数

### 4.1 数项级数

**定义**：$\sum_{n=1}^{\infty} a_n = a_1 + a_2 + a_3 + \cdots$

**收敛**：部分和序列 $S_N = \sum_{n=1}^{N} a_n$ 收敛

**重要级数**：

**几何级数**：

$$\sum_{n=0}^{\infty} r^n = \frac{1}{1-r}, \quad |r| < 1$$

**p-级数**：

$$\sum_{n=1}^{\infty} \frac{1}{n^p}$$

- $p > 1$：收敛
- $p \leq 1$：发散

### 4.2 函数项级数

**幂级数**：

$$\sum_{n=0}^{\infty} c_n (x-a)^n$$

**收敛半径**：

$$R = \frac{1}{\limsup_{n \to \infty} |c_n|^{1/n}}$$

或

$$R = \lim_{n \to \infty} \left|\frac{c_n}{c_{n+1}}\right|$$

### 4.3 傅里叶级数

**定义**：周期函数 $f(x)$（周期 $2L$）的傅里叶级数：

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left(a_n \cos\frac{n\pi x}{L} + b_n \sin\frac{n\pi x}{L}\right)$$

其中：

$$a_n = \frac{1}{L} \int_{-L}^{L} f(x) \cos\frac{n\pi x}{L} dx$$

$$b_n = \frac{1}{L} \int_{-L}^{L} f(x) \sin\frac{n\pi x}{L} dx$$

---

## 5. 在具身智能中的应用

### 5.1 运动学

**位置、速度、加速度**：

$$v(t) = \frac{dx}{dt}, \quad a(t) = \frac{dv}{dt} = \frac{d^2x}{dt^2}$$

### 5.2 优化

**梯度下降**：

$$x_{n+1} = x_n - \eta f'(x_n)$$

**牛顿法**：

$$x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}$$

### 5.3 概率密度

**归一化条件**：

$$\int_{-\infty}^{\infty} p(x) dx = 1$$

**期望值**：

$$E[X] = \int_{-\infty}^{\infty} x p(x) dx$$

### 5.4 信号处理

**傅里叶变换**：

$$\hat{f}(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt$$

---

## 6. 相关论文

### 数值分析

1. **"Numerical Recipes: The Art of Scientific Computing"** - Press et al.
   - 科学计算经典参考书

2. **"Introduction to Numerical Analysis"** - Stoer & Bulirsch
   - 数值分析教材

### 优化

3. **"Convex Optimization"** - Boyd & Vandenberghe
   - 凸优化经典教材

4. **"Numerical Optimization"** - Nocedal & Wright
   - 数值优化教材

---

## 7. 实践练习

### 练习1：导数计算

```python
import numpy as np
import sympy as sp

# 定义符号
x = sp.Symbol('x')

# 定义函数
f = x**3 - 2*x**2 + x - 1

# 计算导数
f_prime = sp.diff(f, x)
f_double_prime = sp.diff(f, x, 2)

print(f"f(x) = {f}")
print(f"f'(x) = {f_prime}")
print(f"f''(x) = {f_double_prime}")

# 求极值点
critical_points = sp.solve(f_prime, x)
print(f"\n临界点: {critical_points}")

# 判断极值类型
for cp in critical_points:
    second_deriv = f_double_prime.subs(x, cp)
    if second_deriv > 0:
        print(f"x = {cp} 是极小值点")
    elif second_deriv < 0:
        print(f"x = {cp} 是极大值点")
    else:
        print(f"x = {cp} 需要进一步判断")
```

### 练习2：数值微分

```python
import numpy as np

def numerical_derivative(f, x, h=1e-5):
    """
    数值微分（中心差分）
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return np.sin(x)

def f_exact(x):
    return np.cos(x)

# 测试
x_test = np.pi / 4
numerical = numerical_derivative(f, x_test)
exact = f_exact(x_test)

print(f"数值导数: {numerical:.10f}")
print(f"精确导数: {exact:.10f}")
print(f"误差: {abs(numerical - exact):.2e}")

# 不同步长的影响
h_values = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
errors = []

for h in h_values:
    approx = (f(x_test + h) - f(x_test - h)) / (2 * h)
    error = abs(approx - exact)
    errors.append(error)
    print(f"h = {h:.0e}, 误差 = {error:.2e}")
```

### 练习3：数值积分

```python
import numpy as np

def f(x):
    return np.sin(x)

def rectangle_rule(f, a, b, n):
    """矩形法"""
    x = np.linspace(a, b, n, endpoint=False)
    dx = (b - a) / n
    return np.sum(f(x)) * dx

def trapezoid_rule(f, a, b, n):
    """梯形法"""
    x = np.linspace(a, b, n+1)
    dx = (b - a) / n
    return (dx/2) * (f(x[0]) + 2*np.sum(f(x[1:-1])) + f(x[-1]))

def simpson_rule(f, a, b, n):
    """辛普森法（n必须为偶数）"""
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n+1)
    dx = (b - a) / n
    return (dx/3) * (f(x[0]) + 4*np.sum(f(x[1:-1:2])) + 2*np.sum(f(x[2:-1:2])) + f(x[-1]))

# 测试：积分 sin(x) 从 0 到 pi
a, b = 0, np.pi
exact = 2  # 精确值

n_values = [10, 100, 1000]

for n in n_values:
    rect = rectangle_rule(f, a, b, n)
    trap = trapezoid_rule(f, a, b, n)
    simp = simpson_rule(f, a, b, n)
    
    print(f"\nn = {n}:")
    print(f"矩形法: {rect:.10f}, 误差: {abs(rect - exact):.2e}")
    print(f"梯形法: {trap:.10f}, 误差: {abs(trap - exact):.2e}")
    print(f"辛普森法: {simp:.10f}, 误差: {abs(simp - exact):.2e}")
```

### 练习4：梯度下降实现

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f_prime, x0, learning_rate=0.1, num_iterations=100):
    """
    梯度下降算法
    """
    x = x0
    trajectory = [x]
    
    for i in range(num_iterations):
        gradient = f_prime(x)
        x = x - learning_rate * gradient
        trajectory.append(x)
    
    return x, trajectory

# 测试函数: f(x) = x^2 - 4x + 4 = (x-2)^2
def f(x):
    return x**2 - 4*x + 4

def f_prime(x):
    return 2*x - 4

# 运行梯度下降
x0 = 0
learning_rate = 0.3
num_iterations = 20

x_opt, trajectory = gradient_descent(f_prime, x0, learning_rate, num_iterations)

print(f"初始值: {x0}")
print(f"最优解: {x_opt:.6f}")
print(f"真实最优解: 2.0")

# 可视化
x_range = np.linspace(-1, 5, 100)
y_range = f(x_range)

plt.figure(figsize=(10, 6))
plt.plot(x_range, y_range, 'b-', label='f(x) = (x-2)^2')
plt.plot(trajectory, [f(x) for x in trajectory], 'ro-', markersize=4, label='Gradient Descent')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gradient Descent Optimization')
plt.legend()
plt.grid(True)
plt.show()
```

---

**下一步**：[多元微积分](02-multivariable.md)
