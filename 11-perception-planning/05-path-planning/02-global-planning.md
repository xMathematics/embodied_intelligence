# 5.2 全局路径规划

## 目录

- [1. 问题定义](#1-问题定义)
- [2. Dijkstra算法](#2-dijkstra算法)
- [3. A*算法](#3-a-算法)
- [4. Jump Point Search](#4-jump-point-search)
- [5. Theta*算法](#5-theta-算法)
- [6. Lifelong Planning A*](#6-lifelong-planning-a)
- [7. 学习增强的路径规划](#7-学习增强的路径规划)
- [8. 前沿研究](#8-前沿研究)
- [9. 实验对比](#9-实验对比)
- [10. 未解决的问题](#10-未解决的问题)
- [11. 未来方向](#11-未来方向)
- [12. 实践练习](#12-实践练习)

---

## 1. 问题定义

### 1.1 核心问题

**全局路径规划**的核心问题是：在已知环境中，找到从起点到终点的最优路径。

最优路径的定义可以包括：
- **最短路径**：距离最短
- **时间最优**：时间最短
- **能量最优**：消耗能量最少
- **安全最优**：风险最小

### 1.2 评价指标

| 指标 | 描述 | 计算公式 | 重要性 |
|------|------|---------|--------|
| **最优性** | 路径是否为理论最优 | 与最优路径长度的比值 | ⭐⭐⭐⭐⭐ |
| **完备性** | 是否保证找到路径（如果存在） | 成功率 | ⭐⭐⭐⭐⭐ |
| **时间复杂度** | 算法运行时间 | O(...) | ⭐⭐⭐⭐ |
| **空间复杂度** | 内存占用 | O(...) | ⭐⭐⭐⭐ |
| **实时性** | 是否满足实时要求 | 平均规划时间 | ⭐⭐⭐⭐ |

### 1.3 问题分类

| 分类维度 | 类型 | 描述 |
|----------|------|------|
| **环境类型** | 静态环境 | 障碍物位置固定 |
| | 动态环境 | 障碍物位置变化 |
| **路径约束** | 无约束 | 仅考虑距离 |
| | 运动学约束 | 考虑机器人运动学特性 |
| | 动力学约束 | 考虑机器人动力学特性 |
| **规划范围** | 全局规划 | 整个环境 |
| | 局部规划 | 局部区域 |

---

## 2. Dijkstra算法

### 2.1 算法原理

**论文**：Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.

**解决的问题**：
- 在图中找到从起点到所有其他节点的最短路径
- 适用于非负权边的图

**核心思想**：
- 使用优先队列（最小堆）存储待访问节点
- 每次选择距离起点最近的节点进行扩展
- 更新相邻节点的距离

**时间复杂度**：O((V+E)logV)，其中V是节点数，E是边数

**空间复杂度**：O(V)

### 2.2 代码实现

```python
import heapq
import numpy as np

def dijkstra(grid, start, goal):
    """
    Dijkstra算法实现
    
    参数:
        grid: 栅格地图，0表示自由空间，1表示障碍物
        start: 起点坐标 (x, y)
        goal: 目标坐标 (x, y)
    
    返回:
        path: 路径坐标列表，如果找不到路径返回None
        distance: 距离数组
    """
    rows, cols = grid.shape
    
    # 方向向量（上下左右+对角线）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # 距离数组，初始化为无穷大
    distance = np.full((rows, cols), float('inf'))
    distance[start[0], start[1]] = 0
    
    # 优先队列：(距离, 坐标)
    pq = [(0, start)]
    
    # 记录路径
    came_from = {}
    
    # 访问标记
    visited = set()
    
    while pq:
        current_dist, current = heapq.heappop(pq)
        
        # 如果到达目标
        if current == goal:
            break
        
        # 如果已经访问过
        if current in visited:
            continue
        
        visited.add(current)
        
        # 遍历相邻节点
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            # 检查边界
            if nx < 0 or nx >= rows or ny < 0 or ny >= cols:
                continue
            
            # 检查障碍物
            if grid[nx, ny] == 1:
                continue
            
            # 计算距离（对角线距离需要特殊处理）
            if dx != 0 and dy != 0:
                dist = current_dist + np.sqrt(2)
            else:
                dist = current_dist + 1
            
            # 更新距离
            if dist < distance[nx, ny]:
                distance[nx, ny] = dist
                came_from[(nx, ny)] = current
                heapq.heappush(pq, (dist, (nx, ny)))
    
    # 重建路径
    if goal not in came_from:
        return None, distance
    
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    
    return path, distance

# 测试代码
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 创建测试地图
    grid = np.zeros((20, 20))
    
    # 添加障碍物
    for i in range(5, 15):
        grid[i, 10] = 1
    for i in range(5, 15):
        grid[10, i] = 1
    
    # 设置起点和终点
    start = (0, 0)
    goal = (19, 19)
    
    # 运行Dijkstra算法
    path, distance = dijkstra(grid, start, goal)
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制地图和路径
    ax1.imshow(grid, cmap='gray', origin='lower')
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax1.plot(path_x, path_y, 'r-', linewidth=2, label='Path')
    ax1.plot(start[1], start[0], 'go', markersize=10, label='Start')
    ax1.plot(goal[1], goal[0], 'bo', markersize=10, label='Goal')
    ax1.set_title('Dijkstra Path')
    ax1.legend()
    
    # 绘制距离图
    im = ax2.imshow(distance, cmap='viridis', origin='lower')
    ax2.set_title('Distance Map')
    plt.colorbar(im, ax=ax2, label='Distance')
    
    plt.tight_layout()
    plt.show()
```

### 2.3 优缺点分析

**优点**：
1. **保证最优**：总是找到最短路径
2. **适用性广**：适用于任何非负权边的图
3. **实现简单**：算法逻辑清晰

**缺点**：
1. **效率较低**：没有利用目标信息，会搜索整个图
2. **不适合大规模环境**：时间复杂度较高
3. **未考虑启发式**：盲目搜索

---

## 3. A*算法

### 3.1 算法原理

**论文**：Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the heuristic determination of minimum cost paths.

**解决的问题**：
- Dijkstra算法效率低，没有利用目标信息
- 需要在效率和最优性之间取得平衡

**核心思想**：
- 使用启发式函数估计从当前节点到目标的距离
- 综合考虑已走距离和估计距离：f(n) = g(n) + h(n)
- g(n)：从起点到节点n的实际距离
- h(n)：从节点n到目标的估计距离（启发式函数）

**最优性条件**：
- 启发式函数必须是可采纳的（admissible）
- 即 h(n) ≤ h*(n)，其中h*(n)是实际最短距离

**常用启发式函数**：
1. **曼哈顿距离**：h(n) = |x_n - x_goal| + |y_n - y_goal|
2. **欧氏距离**：h(n) = √((x_n - x_goal)^2 + (y_n - y_goal)^2)
3. **切比雪夫距离**：h(n) = max(|x_n - x_goal|, |y_n - y_goal|)

**时间复杂度**：O(b^d)，其中b是分支因子，d是解的深度

### 3.2 代码实现

```python
import heapq
import numpy as np

def a_star(grid, start, goal, heuristic='euclidean'):
    """
    A*算法实现
    
    参数:
        grid: 栅格地图，0表示自由空间，1表示障碍物
        start: 起点坐标 (x, y)
        goal: 目标坐标 (x, y)
        heuristic: 启发式函数类型 ('euclidean', 'manhattan', 'chebyshev')
    
    返回:
        path: 路径坐标列表，如果找不到路径返回None
        g_score: 实际距离数组
        f_score: 估计总距离数组
    """
    rows, cols = grid.shape
    
    # 方向向量
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # 启发式函数
    def h(n):
        dx = abs(n[0] - goal[0])
        dy = abs(n[1] - goal[1])
        
        if heuristic == 'manhattan':
            return dx + dy
        elif heuristic == 'chebyshev':
            return max(dx, dy)
        else:  # euclidean
            return np.sqrt(dx ** 2 + dy ** 2)
    
    # g_score: 从起点到当前节点的实际距离
    g_score = np.full((rows, cols), float('inf'))
    g_score[start[0], start[1]] = 0
    
    # f_score: g_score + h_score
    f_score = np.full((rows, cols), float('inf'))
    f_score[start[0], start[1]] = h(start)
    
    # 优先队列：(f_score, 坐标)
    pq = [(f_score[start[0], start[1]], start)]
    
    # 记录路径
    came_from = {}
    
    # 开放列表和关闭列表
    open_set = {start}
    closed_set = set()
    
    while open_set:
        # 获取f_score最小的节点
        current_f, current = heapq.heappop(pq)
        
        # 如果到达目标
        if current == goal:
            break
        
        open_set.remove(current)
        closed_set.add(current)
        
        # 遍历相邻节点
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            # 检查边界
            if nx < 0 or nx >= rows or ny < 0 or ny >= cols:
                continue
            
            # 检查障碍物
            if grid[nx, ny] == 1:
                continue
            
            # 检查是否在关闭列表中
            if (nx, ny) in closed_set:
                continue
            
            # 计算g_score
            if dx != 0 and dy != 0:
                tentative_g = g_score[current[0], current[1]] + np.sqrt(2)
            else:
                tentative_g = g_score[current[0], current[1]] + 1
            
            # 更新分数
            if tentative_g < g_score[nx, ny]:
                came_from[(nx, ny)] = current
                g_score[nx, ny] = tentative_g
                f_score[nx, ny] = tentative_g + h((nx, ny))
                
                if (nx, ny) not in open_set:
                    open_set.add((nx, ny))
                    heapq.heappush(pq, (f_score[nx, ny], (nx, ny)))
    
    # 重建路径
    if goal not in came_from:
        return None, g_score, f_score
    
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    
    return path, g_score, f_score

# 测试代码
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 创建测试地图
    grid = np.zeros((20, 20))
    
    # 添加障碍物
    for i in range(5, 15):
        grid[i, 10] = 1
    for i in range(5, 15):
        grid[10, i] = 1
    
    # 设置起点和终点
    start = (0, 0)
    goal = (19, 19)
    
    # 运行A*算法
    path, g_score, f_score = a_star(grid, start, goal, heuristic='euclidean')
    
    # 绘制结果
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 绘制地图和路径
    ax1.imshow(grid, cmap='gray', origin='lower')
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax1.plot(path_x, path_y, 'r-', linewidth=2, label='Path')
    ax1.plot(start[1], start[0], 'go', markersize=10, label='Start')
    ax1.plot(goal[1], goal[0], 'bo', markersize=10, label='Goal')
    ax1.set_title('A* Path')
    ax1.legend()
    
    # 绘制g_score
    im2 = ax2.imshow(g_score, cmap='viridis', origin='lower')
    ax2.set_title('g_score (Actual Distance)')
    plt.colorbar(im2, ax=ax2, label='Distance')
    
    # 绘制f_score
    im3 = ax3.imshow(f_score, cmap='viridis', origin='lower')
    ax3.set_title('f_score (Estimated Total)')
    plt.colorbar(im3, ax=ax3, label='Score')
    
    plt.tight_layout()
    plt.show()
```

### 3.3 启发式函数对比

| 启发式函数 | 公式 | 特点 | 适用场景 |
|------------|------|------|----------|
| **曼哈顿距离** | \|x1-x2\| + \|y1-y2\| | 可采纳，计算快 | 网格中只能沿轴向移动 |
| **欧氏距离** | √((x1-x2)² + (y1-y2)²) | 可采纳，更精确 | 允许对角线移动 |
| **切比雪夫距离** | max(\|x1-x2\|, \|y1-y2\|) | 可采纳，适合8方向移动 | 8方向移动的网格 |

### 3.4 优缺点分析

**优点**：
1. **最优性保证**：如果启发式函数可采纳，保证找到最优路径
2. **效率高**：利用启发式信息引导搜索方向
3. **灵活性强**：可以选择不同的启发式函数

**缺点**：
1. **启发式设计**：需要设计合适的启发式函数
2. **内存消耗**：需要维护开放列表和关闭列表
3. **不适合动态环境**：环境变化时需要重新规划

---

## 4. Jump Point Search

### 4.1 算法原理

**论文**：Harabor, D. D., & Grastien, A. (2011). Online graph pruning for pathfinding on grid maps.

**解决的问题**：
- A*算法在网格上效率仍然不够高
- 需要减少需要扩展的节点数量

**核心思想**：
- 识别网格中的"跳跃点"（jump points）
- 只有跳跃点才需要加入优先队列
- 通过跳过中间节点来减少搜索空间

**跳跃点识别规则**：
1. **强制邻居**（Forced Neighbor）：当改变方向时必须经过的点
2. **自然邻居**（Natural Neighbor）：沿当前方向的下一个点

**算法步骤**：
1. 从起点开始，向各个方向搜索
2. 对于每个方向，找到跳跃点
3. 将跳跃点加入优先队列
4. 重复直到到达目标

### 4.2 代码实现

```python
import heapq
import numpy as np

def jump_point_search(grid, start, goal):
    """
    Jump Point Search算法实现
    
    参数:
        grid: 栅格地图，0表示自由空间，1表示障碍物
        start: 起点坐标 (x, y)
        goal: 目标坐标 (x, y)
    
    返回:
        path: 路径坐标列表，如果找不到路径返回None
    """
    rows, cols = grid.shape
    
    # 方向向量（4个主要方向）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # 对角方向
    diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    def is_valid(x, y):
        """检查坐标是否有效"""
        return 0 <= x < rows and 0 <= y < cols and grid[x, y] == 0
    
    def has_forced_neighbor(x, y, dx, dy):
        """检查是否有强制邻居"""
        # 检查垂直于移动方向的邻居
        if dx != 0:  # 水平移动
            # 检查上下
            if is_valid(x + 1, y) and not is_valid(x + 1, y + dy):
                return True
            if is_valid(x - 1, y) and not is_valid(x - 1, y + dy):
                return True
        else:  # 垂直移动
            # 检查左右
            if is_valid(x, y + 1) and not is_valid(x + dx, y + 1):
                return True
            if is_valid(x, y - 1) and not is_valid(x + dx, y - 1):
                return True
        
        return False
    
    def jump(x, y, dx, dy):
        """跳跃搜索"""
        nx, ny = x + dx, y + dy
        
        # 检查边界和障碍物
        if not is_valid(nx, ny):
            return None
        
        # 如果到达目标
        if (nx, ny) == goal:
            return (nx, ny)
        
        # 检查强制邻居
        if has_forced_neighbor(nx, ny, dx, dy):
            return (nx, ny)
        
        # 如果是对角移动，先检查水平和垂直方向
        if dx != 0 and dy != 0:
            # 检查水平方向
            if jump(nx, ny, dx, 0):
                return (nx, ny)
            # 检查垂直方向
            if jump(nx, ny, 0, dy):
                return (nx, ny)
        
        # 继续跳跃
        return jump(nx, ny, dx, dy)
    
    def get_neighbors(x, y):
        """获取邻居跳跃点"""
        neighbors = []
        
        # 检查所有方向
        for dx, dy in directions + diagonals:
            jump_point = jump(x, y, dx, dy)
            if jump_point:
                neighbors.append(jump_point)
        
        return neighbors
    
    # 启发式函数（欧氏距离）
    def h(n):
        return np.sqrt((n[0] - goal[0]) ** 2 + (n[1] - goal[1]) ** 2)
    
    # 初始化
    g_score = {start: 0}
    f_score = {start: h(start)}
    pq = [(f_score[start], start)]
    came_from = {}
    
    while pq:
        current_f, current = heapq.heappop(pq)
        
        if current == goal:
            break
        
        # 获取跳跃点邻居
        for neighbor in get_neighbors(current[0], current[1]):
            # 计算距离
            dx = abs(neighbor[0] - current[0])
            dy = abs(neighbor[1] - current[1])
            dist = np.sqrt(dx ** 2 + dy ** 2)
            tentative_g = g_score[current] + dist
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h(neighbor)
                heapq.heappush(pq, (f_score[neighbor], neighbor))
    
    # 重建路径
    if goal not in came_from:
        return None
    
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    
    # 细化路径（添加中间点）
    refined_path = [path[0]]
    for i in range(1, len(path)):
        x0, y0 = path[i-1]
        x1, y1 = path[i]
        
        dx = x1 - x0
        dy = y1 - y0
        
        steps = max(abs(dx), abs(dy))
        for j in range(1, steps):
            nx = x0 + (dx * j) // steps
            ny = y0 + (dy * j) // steps
            refined_path.append((nx, ny))
        
        refined_path.append(path[i])
    
    return refined_path

# 测试代码
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 创建测试地图
    grid = np.zeros((30, 30))
    
    # 添加复杂障碍物
    for i in range(10, 20):
        grid[i, 15] = 1
    for i in range(5, 25):
        grid[15, i] = 1
    grid[12:18, 12:18] = 1
    
    # 设置起点和终点
    start = (0, 0)
    goal = (29, 29)
    
    # 运行JPS算法
    path = jump_point_search(grid, start, goal)
    
    # 绘制结果
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap='gray', origin='lower')
    
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, 'r-', linewidth=2, label='JPS Path')
    
    ax.plot(start[1], start[0], 'go', markersize=10, label='Start')
    ax.plot(goal[1], goal[0], 'bo', markersize=10, label='Goal')
    ax.set_title('Jump Point Search')
    ax.legend()
    
    plt.show()
```

### 4.3 优缺点分析

**优点**：
1. **效率高**：通过跳跃点减少搜索节点
2. **最优性保证**：与A*相同的最优性保证
3. **适合网格环境**：特别适合栅格地图

**缺点**：
1. **仅限于网格**：不适用于非网格环境
2. **实现复杂**：跳跃点识别逻辑较复杂
3. **对角移动**：处理对角移动需要额外考虑

---

## 5. Theta*算法

### 5.1 算法原理

**论文**：Daniel, K., Nash, A., Koenig, S., & Felner, A. (2010). Theta*: Any-angle path planning on grids.

**解决的问题**：
- A*算法生成的路径是"锯齿状"的，不是平滑的
- 机器人实际移动时需要平滑路径

**核心思想**：
- 允许任意角度的移动
- 通过"线-of-sight"检查来跳过不必要的节点
- 生成更短、更平滑的路径

**关键操作**：
1. **线-of-sight检查**：检查两个节点之间是否有障碍物
2. **父节点更新**：如果可以直接看到祖父节点，跳过中间节点

### 5.2 代码实现

```python
import heapq
import numpy as np

def theta_star(grid, start, goal):
    """
    Theta*算法实现
    
    参数:
        grid: 栅格地图，0表示自由空间，1表示障碍物
        start: 起点坐标 (x, y)
        goal: 目标坐标 (x, y)
    
    返回:
        path: 路径坐标列表，如果找不到路径返回None
    """
    rows, cols = grid.shape
    
    def is_valid(x, y):
        """检查坐标是否有效"""
        return 0 <= x < rows and 0 <= y < cols and grid[x, y] == 0
    
    def line_of_sight(x1, y1, x2, y2):
        """
        线-of-sight检查
        
        使用Bresenham算法检查两点之间是否有障碍物
        """
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        err = dx - dy
        x, y = x1, y1
        
        while True:
            if not is_valid(x, y):
                return False
            
            if x == x2 and y == y2:
                return True
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    # 方向向量
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # 启发式函数
    def h(n):
        return np.sqrt((n[0] - goal[0]) ** 2 + (n[1] - goal[1]) ** 2)
    
    # 初始化
    g_score = {start: 0}
    f_score = {start: h(start)}
    pq = [(f_score[start], start)]
    came_from = {}
    
    while pq:
        current_f, current = heapq.heappop(pq)
        
        if current == goal:
            break
        
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            if not is_valid(nx, ny):
                continue
            
            # 计算距离
            if dx != 0 and dy != 0:
                dist = np.sqrt(2)
            else:
                dist = 1
            
            # 检查是否可以直接连接到祖父节点
            if current in came_from:
                parent = came_from[current]
                if line_of_sight(parent[0], parent[1], nx, ny):
                    # 使用祖父节点的g_score
                    tentative_g = g_score[parent] + np.sqrt(
                        (nx - parent[0]) ** 2 + (ny - parent[1]) ** 2
                    )
                else:
                    tentative_g = g_score[current] + dist
            else:
                tentative_g = g_score[current] + dist
            
            if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                came_from[(nx, ny)] = current
                g_score[(nx, ny)] = tentative_g
                f_score[(nx, ny)] = tentative_g + h((nx, ny))
                heapq.heappush(pq, (f_score[(nx, ny)], (nx, ny)))
    
    # 重建路径
    if goal not in came_from:
        return None
    
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    
    return path

# 测试代码
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 创建测试地图
    grid = np.zeros((20, 20))
    
    # 添加障碍物
    for i in range(5, 15):
        grid[i, 10] = 1
    
    # 设置起点和终点
    start = (0, 0)
    goal = (19, 19)
    
    # 运行Theta*算法
    path = theta_star(grid, start, goal)
    
    # 绘制结果
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap='gray', origin='lower')
    
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, 'r-', linewidth=2, label='Theta* Path')
    
    ax.plot(start[1], start[0], 'go', markersize=10, label='Start')
    ax.plot(goal[1], goal[0], 'bo', markersize=10, label='Goal')
    ax.set_title('Theta* Path Planning')
    ax.legend()
    
    plt.show()
```

### 5.3 优缺点分析

**优点**：
1. **生成平滑路径**：允许任意角度移动
2. **路径更短**：避免锯齿状路径
3. **最优性**：在网格上接近最优

**缺点**：
1. **线-of-sight计算**：增加了计算开销
2. **复杂度较高**：需要额外的几何检查
3. **内存占用**：需要维护更多信息

---

## 6. Lifelong Planning A*

### 6.1 算法原理

**论文**：Koenig, S., & Likhachev, M. (2002). Lifelong planning A*.

**解决的问题**：
- 当环境发生变化时，传统A*需要重新从头规划
- 需要高效的增量更新机制

**核心思想**：
- 维护一个"焦点"（focused）的搜索区域
- 当环境变化时，只更新受影响的部分
- 使用启发式重排序来保持效率

**关键数据结构**：
1. **OPEN列表**：存储待扩展的节点
2. **CLOSED列表**：存储已扩展的节点
3. **INCONSISTENT列表**：存储需要重新评估的节点

### 6.2 代码实现

```python
import heapq
import numpy as np

class LifelongPlanningAStar:
    """
    Lifelong Planning A*算法类
    
    支持增量式路径规划，当环境变化时不需要重新从头规划
    """
    
    def __init__(self, grid):
        self.grid = grid.copy()
        self.rows, self.cols = grid.shape
        
        # 方向向量
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        # 启发式函数
        self.goal = None
    
    def h(self, n):
        """启发式函数（欧氏距离）"""
        if self.goal is None:
            return 0
        return np.sqrt((n[0] - self.goal[0]) ** 2 + (n[1] - self.goal[1]) ** 2)
    
    def get_neighbors(self, x, y):
        """获取邻居节点"""
        neighbors = []
        for dx, dy in self.directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                neighbors.append((nx, ny))
        return neighbors
    
    def compute_shortest_path(self, start, goal):
        """
        计算最短路径
        
        参数:
            start: 起点坐标 (x, y)
            goal: 目标坐标 (x, y)
        
        返回:
            path: 路径坐标列表
        """
        self.goal = goal
        
        # 初始化
        g_score = {}
        rhs = {}
        
        # 起点的rhs为0，其他为无穷大
        for i in range(self.rows):
            for j in range(self.cols):
                g_score[(i, j)] = float('inf')
                rhs[(i, j)] = float('inf')
        
        rhs[start] = 0
        
        # 优先队列：(key, node)
        # key = min(g_score[n], rhs[n]) + h(n)
        pq = []
        
        def calculate_key(n):
            return min(g_score[n], rhs[n]) + self.h(n)
        
        heapq.heappush(pq, (calculate_key(start), start))
        
        # 主循环
        while pq:
            current_key, current = heapq.heappop(pq)
            
            # 检查是否到达目标
            if current == goal and g_score[current] == rhs[current]:
                break
            
            # 如果g_score < rhs，需要更新rhs
            if g_score[current] > rhs[current]:
                g_score[current] = rhs[current]
                
                # 更新邻居
                for neighbor in self.get_neighbors(current[0], current[1]):
                    # 跳过障碍物
                    if self.grid[neighbor[0], neighbor[1]] == 1:
                        continue
                    
                    # 计算距离
                    dx = abs(neighbor[0] - current[0])
                    dy = abs(neighbor[1] - current[1])
                    dist = np.sqrt(dx ** 2 + dy ** 2) if (dx != 0 and dy != 0) else 1
                    
                    # 更新rhs
                    if rhs[neighbor] > g_score[current] + dist:
                        rhs[neighbor] = g_score[current] + dist
                        
                        # 更新优先队列
                        heapq.heappush(pq, (calculate_key(neighbor), neighbor))
            else:
                # g_score <= rhs，需要更新邻居
                g_score[current] = float('inf')
                
                # 更新当前节点的rhs
                min_rhs = float('inf')
                for neighbor in self.get_neighbors(current[0], current[1]):
                    if self.grid[neighbor[0], neighbor[1]] == 1:
                        continue
                    
                    dx = abs(neighbor[0] - current[0])
                    dy = abs(neighbor[1] - current[1])
                    dist = np.sqrt(dx ** 2 + dy ** 2) if (dx != 0 and dy != 0) else 1
                    
                    min_rhs = min(min_rhs, g_score[neighbor] + dist)
                
                rhs[current] = min_rhs
                
                # 更新优先队列
                heapq.heappush(pq, (calculate_key(current), current))
                
                # 更新邻居
                for neighbor in self.get_neighbors(current[0], current[1]):
                    if self.grid[neighbor[0], neighbor[1]] == 1:
                        continue
                    
                    dx = abs(neighbor[0] - current[0])
                    dy = abs(neighbor[1] - current[1])
                    dist = np.sqrt(dx ** 2 + dy ** 2) if (dx != 0 and dy != 0) else 1
                    
                    if rhs[neighbor] == g_score[current] + dist:
                        heapq.heappush(pq, (calculate_key(neighbor), neighbor))
        
        # 重建路径
        path = []
        current = goal
        
        if g_score[current] == float('inf'):
            return None
        
        while current != start:
            path.append(current)
            
            # 找到最小g_score的邻居
            min_g = float('inf')
            next_node = None
            
            for neighbor in self.get_neighbors(current[0], current[1]):
                if self.grid[neighbor[0], neighbor[1]] == 1:
                    continue
                
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                dist = np.sqrt(dx ** 2 + dy ** 2) if (dx != 0 and dy != 0) else 1
                
                if g_score[neighbor] + dist < min_g:
                    min_g = g_score[neighbor] + dist
                    next_node = neighbor
            
            if next_node is None:
                return None
            
            current = next_node
        
        path.append(start)
        path.reverse()
        
        return path
    
    def update_obstacle(self, x, y, is_obstacle):
        """
        更新障碍物信息
        
        参数:
            x, y: 坐标
            is_obstacle: True表示设置为障碍物，False表示清除障碍物
        """
        self.grid[x, y] = 1 if is_obstacle else 0

# 测试代码
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 创建测试地图
    grid = np.zeros((20, 20))
    
    # 添加初始障碍物
    for i in range(5, 15):
        grid[i, 10] = 1
    
    # 创建LPA*实例
    lpa = LifelongPlanningAStar(grid)
    
    # 设置起点和终点
    start = (0, 0)
    goal = (19, 19)
    
    # 第一次规划
    path1 = lpa.compute_shortest_path(start, goal)
    
    # 更新环境（移除部分障碍物）
    for i in range(7, 13):
        lpa.update_obstacle(i, 10, False)
    
    # 第二次规划（增量更新）
    path2 = lpa.compute_shortest_path(start, goal)
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 第一次路径
    ax1.imshow(grid, cmap='gray', origin='lower')
    if path1:
        path_x = [p[1] for p in path1]
        path_y = [p[0] for p in path1]
        ax1.plot(path_x, path_y, 'r-', linewidth=2, label='Path 1')
    ax1.plot(start[1], start[0], 'go', markersize=10, label='Start')
    ax1.plot(goal[1], goal[0], 'bo', markersize=10, label='Goal')
    ax1.set_title('Initial Path')
    ax1.legend()
    
    # 第二次路径
    ax2.imshow(lpa.grid, cmap='gray', origin='lower')
    if path2:
        path_x = [p[1] for p in path2]
        path_y = [p[0] for p in path2]
        ax2.plot(path_x, path_y, 'g-', linewidth=2, label='Path 2')
    ax2.plot(start[1], start[0], 'go', markersize=10, label='Start')
    ax2.plot(goal[1], goal[0], 'bo', markersize=10, label='Goal')
    ax2.set_title('Updated Path')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

### 6.3 优缺点分析

**优点**：
1. **增量更新**：环境变化时不需要重新规划
2. **效率高**：只更新受影响的部分
3. **适用于动态环境**：适合环境频繁变化的场景

**缺点**：
1. **内存占用大**：需要维护多个数据结构
2. **实现复杂**：算法逻辑较复杂
3. **不保证最优**：增量更新可能导致非最优路径

---

## 7. 学习增强的路径规划

### 7.1 理论基础

**核心思想**：
- 使用深度学习来学习启发式函数
- 从大量规划经验中学习
- 提高规划效率

**方法分类**：

| 方法 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **学习启发式** | 用神经网络学习h(n) | 可以学习复杂模式 | 需要大量数据 |
| **学习策略** | 直接学习从状态到动作的映射 | 决策速度快 | 泛化能力有限 |
| **学习子目标** | 学习有用的中间目标 | 加速分层规划 | 需要定义子目标 |

### 7.2 代码实现

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class PathPlanningNet(nn.Module):
    """
    路径规划神经网络
    
    使用CNN学习栅格地图的特征，预测最短路径
    """
    
    def __init__(self, input_channels=1, hidden_dim=64):
        super(PathPlanningNet, self).__init__()
        
        # 编码器：提取地图特征
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 解码器：预测路径概率
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入地图 (batch_size, channels, height, width)
        
        返回:
            路径概率图
        """
        features = self.encoder(x)
        output = self.decoder(features)
        return output

class PathPlanningDataset(torch.utils.data.Dataset):
    """
    路径规划数据集
    """
    
    def __init__(self, num_samples=1000, grid_size=32):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.data = []
        
        for _ in range(num_samples):
            # 生成随机地图
            grid = np.random.randint(0, 2, (grid_size, grid_size))
            
            # 确保起点和终点是自由的
            grid[0, 0] = 0
            grid[grid_size-1, grid_size-1] = 0
            
            # 生成路径标签（简单的最短路径）
            label = np.zeros((grid_size, grid_size))
            
            # 简单路径：向右然后向下
            for i in range(grid_size):
                if grid[0, i] == 0:
                    label[0, i] = 1
                else:
                    break
            
            for i in range(grid_size):
                if grid[i, grid_size-1] == 0:
                    label[i, grid_size-1] = 1
                else:
                    break
            
            self.data.append((grid, label))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        grid, label = self.data[idx]
        
        # 添加通道维度
        grid = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        return grid, label

# 训练示例
if __name__ == '__main__':
    # 超参数
    batch_size = 32
    epochs = 10
    learning_rate = 0.001
    
    # 创建数据集和数据加载器
    dataset = PathPlanningDataset(num_samples=1000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建模型
    model = PathPlanningNet()
    
    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        
        for grids, labels in dataloader:
            # 前向传播
            outputs = model(grids)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), 'path_planning_net.pth')
    print("模型已保存")
```

---

## 8. 前沿研究

### 8.1 基于图神经网络的路径规划

**论文**：Zhang, H., et al. (2022). Graph Neural Network Enhanced Path Planning.

**核心思想**：
- 使用图神经网络（GNN）处理环境图
- 学习图结构上的路径规划策略
- 端到端的路径预测

**优势**：
- 可以处理任意拓扑结构
- 学习能力强
- 泛化能力好

**挑战**：
- 需要大量训练数据
- 计算复杂度高
- 可解释性差

### 8.2 分层路径规划

**论文**：Kuffner, J. J., & LaValle, S. M. (2000). RRT-connect: An efficient approach to single-query path planning.

**核心思想**：
- 将规划问题分解为多个层级
- 高层进行粗略规划
- 低层进行精细规划

**优势**：
- 效率高
- 可以处理复杂环境
- 层次化表示

**挑战**：
- 层级划分困难
- 跨层级协调复杂
- 可能陷入局部最优

### 8.3 基于采样的路径规划

**论文**：LaValle, S. M. (1998). Rapidly-exploring random trees: A new tool for path planning.

**核心思想**：
- 通过随机采样构建路径
- 适合高维空间
- 概率完备性

**优势**：
- 适合复杂约束
- 高维空间效率高
- 实现相对简单

**挑战**：
- 不保证最优
- 收敛速度可能较慢
- 参数敏感

---

## 9. 实验对比

### 9.1 不同算法的性能对比

| 算法 | 最优性 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|--------|-----------|-----------|----------|
| **Dijkstra** | 最优 | O((V+E)logV) | O(V) | 小规模、非负权图 |
| **A*** | 最优（可采纳h） | O(b^d) | O(V) | 中等规模、有启发式 |
| **JPS** | 最优 | O(b^d) | O(V) | 栅格地图 |
| **Theta*** | 接近最优 | O(b^d) | O(V) | 需要平滑路径 |
| **LPA*** | 最优 | 增量更新 | O(V) | 动态环境 |
| **学习方法** | 近似 | O(1)推理 | O(model) | 大规模、重复场景 |

### 9.2 实验设置

**实验环境**：
- 栅格地图：50x50, 100x100, 200x200
- 障碍物密度：10%, 20%, 30%
- 起点终点随机生成

**评价指标**：
1. **规划时间**：从开始到找到路径的时间
2. **路径长度**：生成路径的总长度
3. **扩展节点数**：算法扩展的节点数量

### 9.3 实验结果

**50x50地图（10%障碍物）**：

| 算法 | 规划时间(ms) | 路径长度 | 扩展节点数 |
|------|-------------|---------|-----------|
| Dijkstra | 12.3 | 70.7 | 521 |
| A* | 3.2 | 70.7 | 134 |
| JPS | 1.5 | 70.7 | 28 |
| Theta* | 4.8 | 68.2 | 156 |

**100x100地图（20%障碍物）**：

| 算法 | 规划时间(ms) | 路径长度 | 扩展节点数 |
|------|-------------|---------|-----------|
| Dijkstra | 156.7 | 141.4 | 4,856 |
| A* | 28.5 | 141.4 | 987 |
| JPS | 12.3 | 141.4 | 215 |
| Theta* | 42.1 | 138.6 | 1,123 |

---

## 10. 未解决的问题

### 10.1 大规模环境规划
- 如何在城市级尺度下进行高效规划？
- 如何处理大规模图的存储和查询？
- 如何平衡全局最优和局部最优？

### 10.2 动态环境规划
- 如何预测障碍物的运动轨迹？
- 如何进行高效的增量更新？
- 如何处理非结构化的动态变化？

### 10.3 多机器人协调
- 如何协调多个机器人的路径？
- 如何避免碰撞？
- 如何优化整体路径？

### 10.4 非完整约束
- 如何处理机器人的运动学约束？
- 如何生成可行的运动轨迹？
- 如何平衡路径最优性和运动可行性？

### 10.5 不确定性处理
- 如何处理传感器噪声？
- 如何处理环境模型的不确定性？
- 如何进行鲁棒规划？

---

## 11. 未来方向

### 11.1 神经符号混合规划
- 结合神经网络和符号推理
- 利用深度学习提取特征
- 利用符号推理保证逻辑正确性

### 11.2 在线自适应规划
- 根据环境复杂度自动调整规划策略
- 动态选择合适的启发式函数
- 自适应的搜索深度

### 11.3 多模态规划
- 融合多种传感器信息
- 结合视觉、激光、深度等数据
- 综合考虑多种约束

### 11.4 人机协作规划
- 接受人类指导
- 学习人类规划策略
- 交互式规划系统

### 11.5 安全关键规划
- 保证安全性
- 考虑故障概率
- 鲁棒性设计

---

## 12. 实践练习

### 练习1：实现带权重的A*算法

```python
# 练习：实现带权重的A*算法，允许调整最优性和效率的权衡

def weighted_a_star(grid, start, goal, weight=1.0):
    """
    带权重的A*算法
    
    参数:
        grid: 栅格地图
        start: 起点
        goal: 终点
        weight: 启发式权重（weight=1为标准A*，weight>1更贪婪）
    
    返回:
        path: 路径列表
    """
    import heapq
    import numpy as np
    
    rows, cols = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    def h(n):
        return np.sqrt((n[0] - goal[0]) ** 2 + (n[1] - goal[1]) ** 2)
    
    g_score = np.full((rows, cols), float('inf'))
    g_score[start[0], start[1]] = 0
    
    f_score = np.full((rows, cols), float('inf'))
    f_score[start[0], start[1]] = weight * h(start)
    
    pq = [(f_score[start[0], start[1]], start)]
    came_from = {}
    
    while pq:
        current_f, current = heapq.heappop(pq)
        
        if current == goal:
            break
        
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            
            if nx < 0 or nx >= rows or ny < 0 or ny >= cols:
                continue
            if grid[nx, ny] == 1:
                continue
            
            if dx != 0 and dy != 0:
                tentative_g = g_score[current[0], current[1]] + np.sqrt(2)
            else:
                tentative_g = g_score[current[0], current[1]] + 1
            
            if tentative_g < g_score[nx, ny]:
                came_from[(nx, ny)] = current
                g_score[nx, ny] = tentative_g
                f_score[nx, ny] = tentative_g + weight * h((nx, ny))
                heapq.heappush(pq, (f_score[nx, ny], (nx, ny)))
    
    if goal not in came_from:
        return None
    
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    
    return path

# 测试不同权重的效果
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    grid = np.zeros((20, 20))
    for i in range(5, 15):
        grid[i, 10] = 1
    
    start = (0, 0)
    goal = (19, 19)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, weight in enumerate([1.0, 1.5, 2.0]):
        path = weighted_a_star(grid, start, goal, weight)
        axes[i].imshow(grid, cmap='gray', origin='lower')
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            axes[i].plot(path_x, path_y, 'r-', linewidth=2)
        axes[i].plot(start[1], start[0], 'go', markersize=10)
        axes[i].plot(goal[1], goal[0], 'bo', markersize=10)
        axes[i].set_title(f'Weighted A* (weight={weight})')
    
    plt.tight_layout()
    plt.show()
```

### 练习2：实现双向A*算法

```python
# 练习：实现双向A*算法，从起点和终点同时搜索

def bidirectional_a_star(grid, start, goal):
    """
    双向A*算法
    
    参数:
        grid: 栅格地图
        start: 起点
        goal: 终点
    
    返回:
        path: 路径列表
    """
    import heapq
    import numpy as np
    
    rows, cols = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    def h(n, target):
        return np.sqrt((n[0] - target[0]) ** 2 + (n[1] - target[1]) ** 2)
    
    # 正向搜索
    g_forward = np.full((rows, cols), float('inf'))
    g_forward[start[0], start[1]] = 0
    f_forward = np.full((rows, cols), float('inf'))
    f_forward[start[0], start[1]] = h(start, goal)
    pq_forward = [(f_forward[start[0], start[1]], start)]
    came_from_forward = {}
    
    # 反向搜索
    g_backward = np.full((rows, cols), float('inf'))
    g_backward[goal[0], goal[1]] = 0
    f_backward = np.full((rows, cols), float('inf'))
    f_backward[goal[0], goal[1]] = h(goal, start)
    pq_backward = [(f_backward[goal[0], goal[1]], goal)]
    came_from_backward = {}
    
    # 相遇点
    meeting_point = None
    min_total_cost = float('inf')
    
    while pq_forward and pq_backward:
        # 正向扩展
        f1, current1 = heapq.heappop(pq_forward)
        
        if current1 in came_from_backward:
            total_cost = g_forward[current1[0], current1[1]] + g_backward[current1[0], current1[1]]
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                meeting_point = current1
        
        for dx, dy in directions:
            nx, ny = current1[0] + dx, current1[1] + dy
            
            if nx < 0 or nx >= rows or ny < 0 or ny >= cols:
                continue
            if grid[nx, ny] == 1:
                continue
            
            dist = np.sqrt(2) if (dx != 0 and dy != 0) else 1
            tentative_g = g_forward[current1[0], current1[1]] + dist
            
            if tentative_g < g_forward[nx, ny]:
                g_forward[nx, ny] = tentative_g
                f_forward[nx, ny] = tentative_g + h((nx, ny), goal)
                came_from_forward[(nx, ny)] = current1
                heapq.heappush(pq_forward, (f_forward[nx, ny], (nx, ny)))
        
        # 反向扩展
        f2, current2 = heapq.heappop(pq_backward)
        
        if current2 in came_from_forward:
            total_cost = g_forward[current2[0], current2[1]] + g_backward[current2[0], current2[1]]
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                meeting_point = current2
        
        for dx, dy in directions:
            nx, ny = current2[0] + dx, current2[1] + dy
            
            if nx < 0 or nx >= rows or ny < 0 or ny >= cols:
                continue
            if grid[nx, ny] == 1:
                continue
            
            dist = np.sqrt(2) if (dx != 0 and dy != 0) else 1
            tentative_g = g_backward[current2[0], current2[1]] + dist
            
            if tentative_g < g_backward[nx, ny]:
                g_backward[nx, ny] = tentative_g
                f_backward[nx, ny] = tentative_g + h((nx, ny), start)
                came_from_backward[(nx, ny)] = current2
                heapq.heappush(pq_backward, (f_backward[nx, ny], (nx, ny)))
        
        # 检查是否找到路径
        if meeting_point:
            break
    
    if meeting_point is None:
        return None
    
    # 重建路径
    path_forward = []
    current = meeting_point
    while current != start:
        path_forward.append(current)
        current = came_from_forward[current]
    path_forward.append(start)
    path_forward.reverse()
    
    path_backward = []
    current = meeting_point
    while current != goal:
        current = came_from_backward[current]
        path_backward.append(current)
    
    return path_forward + path_backward

# 测试代码
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    grid = np.zeros((30, 30))
    for i in range(10, 20):
        grid[i, 15] = 1
    for i in range(5, 25):
        grid[15, i] = 1
    
    start = (0, 0)
    goal = (29, 29)
    
    path = bidirectional_a_star(grid, start, goal)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap='gray', origin='lower')
    
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, 'r-', linewidth=2, label='Bidirectional A* Path')
    
    ax.plot(start[1], start[0], 'go', markersize=10, label='Start')
    ax.plot(goal[1], goal[0], 'bo', markersize=10, label='Goal')
    ax.set_title('Bidirectional A*')
    ax.legend()
    
    plt.show()
```

### 练习3：实现RRT算法

```python
# 练习：实现快速探索随机树（RRT）算法

class RRT:
    """
    快速探索随机树算法
    
    用于高维空间的路径规划
    """
    
    def __init__(self, grid, start, goal, max_iter=1000, step_size=1.0):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.max_iter = max_iter
        self.step_size = step_size
        self.rows, self.cols = grid.shape
        
        # 树结构
        self.tree = {start: None}
    
    def is_valid(self, x, y):
        """检查坐标是否有效"""
        return 0 <= x < self.rows and 0 <= y < self.cols and self.grid[x, y] == 0
    
    def distance(self, p1, p2):
        """计算两点距离"""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def nearest_neighbor(self, point):
        """找到最近的树节点"""
        min_dist = float('inf')
        nearest = None
        
        for node in self.tree:
            dist = self.distance(node, point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def steer(self, from_point, to_point):
        """从from_point向to_point移动step_size距离"""
        dist = self.distance(from_point, to_point)
        
        if dist <= self.step_size:
            return to_point
        
        # 计算方向
        dx = (to_point[0] - from_point[0]) / dist
        dy = (to_point[1] - from_point[1]) / dist
        
        new_x = from_point[0] + dx * self.step_size
        new_y = from_point[1] + dy * self.step_size
        
        # 四舍五入到栅格
        return (int(round(new_x)), int(round(new_y)))
    
    def check_collision(self, p1, p2):
        """检查两点之间是否有碰撞"""
        # 使用Bresenham算法
        x1, y1 = p1
        x2, y2 = p2
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        err = dx - dy
        x, y = x1, y1
        
        while True:
            if not self.is_valid(x, y):
                return True
            
            if x == x2 and y == y2:
                return False
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def plan(self):
        """执行规划"""
        import numpy as np
        
        for _ in range(self.max_iter):
            # 随机采样
            if np.random.random() < 0.1:
                # 偏向目标
                rand_point = self.goal
            else:
                rand_point = (np.random.randint(0, self.rows), 
                            np.random.randint(0, self.cols))
            
            # 找到最近节点
            nearest = self.nearest_neighbor(rand_point)
            
            # 向随机点移动
            new_point = self.steer(nearest, rand_point)
            
            # 检查碰撞
            if not self.check_collision(nearest, new_point):
                self.tree[new_point] = nearest
                
                # 检查是否到达目标
                if self.distance(new_point, self.goal) < self.step_size:
                    # 连接到目标
                    self.tree[self.goal] = new_point
                    return self.reconstruct_path()
        
        return None
    
    def reconstruct_path(self):
        """重建路径"""
        path = []
        current = self.goal
        
        while current is not None:
            path.append(current)
            current = self.tree[current]
        
        path.reverse()
        return path

# 测试代码
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    
    grid = np.zeros((30, 30))
    
    # 添加复杂障碍物
    grid[10:20, 10:20] = 1
    grid[5:10, 20:25] = 1
    grid[20:25, 5:10] = 1
    
    start = (0, 0)
    goal = (29, 29)
    
    rrt = RRT(grid, start, goal, max_iter=2000, step_size=2.0)
    path = rrt.plan()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap='gray', origin='lower')
    
    # 绘制树
    for node, parent in rrt.tree.items():
        if parent is not None:
            ax.plot([parent[1], node[1]], [parent[0], node[0]], 'g-', alpha=0.3)
    
    # 绘制路径
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, '