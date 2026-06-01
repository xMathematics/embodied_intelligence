# 5.2 全局规划

## 目录

- [1. 引言](#1-引言)
- [2. Dijkstra算法](#2-dijkstra算法)
- [3. A*算法](#3-a算法)
- [4. Jump Point Search](#4-jump-point-search)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 全局规划的定义

全局规划在已知的环境地图上找到从起点到目标点的最优或近似最优路径。

### 1.2 常用算法

```python
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import deque

class GlobalPlanner:
    """全局规划器基类"""
    
    def __init__(self, gridmap):
        self.map = gridmap
    
    def plan(self, start, goal):
        """规划路径"""
        raise NotImplementedError
```

---

## 2. Dijkstra算法

### 2.1 Dijkstra实现

```python
class Dijkstra(GlobalPlanner):
    """Dijkstra算法"""
    
    def __init__(self, gridmap):
        super().__init__(gridmap)
        self.costmap = None
    
    def heuristic(self, a, b):
        """启发式函数（Dijkstra不使用）"""
        return 0
    
    def get_neighbors(self, node):
        """获取邻居节点（8邻域）"""
        neighbors = []
        gx, gy = node
        
        # 8个方向
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                ngx, ngy = gx + dx, gy + dy
                
                if 0 <= ngx < self.map.width and 0 <= ngy < self.map.height:
                    if not self.map.is_occupied(ngx, ngy):
                        # 移动成本
                        cost = np.sqrt(dx*dx + dy*dy)
                        neighbors.append((ngx, ngy, cost))
        
        return neighbors
    
    def plan(self, start, goal):
        """规划路径"""
        start_gx, start_gy = self.map.world_to_grid(start[0], start[1])
        goal_gx, goal_gy = self.map.world_to_grid(goal[0], goal[1])
        
        # 开放列表和关闭列表
        open_list = []
        heapq.heappush(open_list, (0, (start_gx, start_gy)))
        
        came_from = {}
        g_score = {}
        g_score[(start_gx, start_gy)] = 0
        
        while open_list:
            current_cost, current = heapq.heappop(open_list)
            
            if current == (goal_gx, goal_gy):
                # 回溯路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append((start_gx, start_gy))
                path.reverse()
                
                # 转换为世界坐标
                world_path = []
                for gx, gy in path:
                    wx, wy = self.map.grid_to_world(gx, gy)
                    world_path.append((wx, wy))
                
                return world_path
            
            for ngx, ngy, move_cost in self.get_neighbors(current):
                tentative_g = g_score[current] + move_cost
                
                if (ngx, ngy) not in g_score or tentative_g < g_score[(ngx, ngy)]:
                    g_score[(ngx, ngy)] = tentative_g
                    f = tentative_g + self.heuristic((ngx, ngy), (goal_gx, goal_gy))
                    heapq.heappush(open_list, (f, (ngx, ngy)))
                    came_from[(ngx, ngy)] = current
        
        return None  # 无法到达
```

---

## 3. A*算法

### 3.1 A*实现

```python
class AStar(Dijkstra):
    """A*算法"""
    
    def __init__(self, gridmap, heuristic_type='euclidean'):
        super().__init__(gridmap)
        self.heuristic_type = heuristic_type
    
    def heuristic(self, a, b):
        """启发式函数"""
        ax, ay = a
        bx, by = b
        
        if self.heuristic_type == 'euclidean':
            return np.sqrt((ax - bx)**2 + (ay - by)**2)
        elif self.heuristic_type == 'manhattan':
            return abs(ax - bx) + abs(ay - by)
        elif self.heuristic_type == 'octile':
            dx = abs(ax - bx)
            dy = abs(ay - by)
            return dx + dy + (np.sqrt(2) - 2) * min(dx, dy)
        else:
            return 0
```

### 3.2 A*变体

```python
class ThetaStar(AStar):
    """Theta*算法 (任意角度)"""
    
    def line_of_sight(self, a, b):
        """视线检查"""
        ax, ay = a
        bx, by = b
        
        dx = abs(bx - ax)
        dy = abs(by - ay)
        x = ax
        y = ay
        sx = 1 if ax < bx else -1
        sy = 1 if ay < by else -1
        
        if dx > dy:
            err = dx / 2
            while x != bx:
                if self.map.is_occupied(x, y):
                    return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2
            while y != by:
                if self.map.is_occupied(x, y):
                    return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        
        return True
    
    def plan(self, start, goal):
        """Theta*规划"""
        start_gx, start_gy = self.map.world_to_grid(start[0], start[1])
        goal_gx, goal_gy = self.map.world_to_grid(goal[0], goal[1])
        
        open_list = []
        heapq.heappush(open_list, (0, (start_gx, start_gy)))
        
        came_from = {}
        g_score = {}
        g_score[(start_gx, start_gy)] = 0
        
        while open_list:
            current_f, current = heapq.heappop(open_list)
            
            if current == (goal_gx, goal_gy):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append((start_gx, start_gy))
                path.reverse()
                
                world_path = []
                for gx, gy in path:
                    wx, wy = self.map.grid_to_world(gx, gy)
                    world_path.append((wx, wy))
                
                return world_path
            
            for ngx, ngy, move_cost in self.get_neighbors(current):
                # Theta*逻辑: 检查父节点到邻居是否有视线
                if current in came_from:
                    parent = came_from[current]
                    if self.line_of_sight(parent, (ngx, ngy)):
                        # 直接连接
                        cost = np.sqrt((parent[0] - ngx)**2 + (parent[1] - ngy)**2)
                        tentative_g = g_score[parent] + cost
                        
                        if (ngx, ngy) not in g_score or tentative_g < g_score[(ngx, ngy)]:
                            g_score[(ngx, ngy)] = tentative_g
                            h = self.heuristic((ngx, ngy), (goal_gx, goal_gy))
                            f = tentative_g + h
                            heapq.heappush(open_list, (f, (ngx, ngy)))
                            came_from[(ngx, ngy)] = parent
                            continue
                
                # 标准A*逻辑
                tentative_g = g_score[current] + move_cost
                
                if (ngx, ngy) not in g_score or tentative_g < g_score[(ngx, ngy)]:
                    g_score[(ngx, ngy)] = tentative_g
                    h = self.heuristic((ngx, ngy), (goal_gx, goal_gy))
                    f = tentative_g + h
                    heapq.heappush(open_list, (f, (ngx, ngy)))
                    came_from[(ngx, ngy)] = current
        
        return None
```

---

## 4. Jump Point Search

### 4.1 JPS实现

```python
class JumpPointSearch(AStar):
    """Jump Point Search算法"""
    
    def __init__(self, gridmap, heuristic_type='octile'):
        super().__init__(gridmap, heuristic_type)
    
    def has_forced_neighbors(self, x, y, dx, dy):
        """检查是否有强迫邻居"""
        # 简化实现
        # 实际JPS需要检查特定的强迫邻居
        return False
    
    def jump(self, x, y, dx, dy, goal_x, goal_y):
        """跳跃"""
        nx, ny = x + dx, y + dy
        
        if not (0 <= nx < self.map.width and 0 <= ny < self.map.height):
            return None
        
        if self.map.is_occupied(nx, ny):
            return None
        
        if (nx, ny) == (goal_x, goal_y):
            return (nx, ny)
        
        # 检查强迫邻居
        if self.has_forced_neighbors(nx, ny, dx, dy):
            return (nx, ny)
        
        # 对角移动
        if dx != 0 and dy != 0:
            if self.jump(nx, ny, dx, 0, goal_x, goal_y) or \
               self.jump(nx, ny, 0, dy, goal_x, goal_y):
                return (nx, ny)
        
        # 继续跳跃
        return self.jump(nx, ny, dx, dy, goal_x, goal_y)
    
    def plan(self, start, goal):
        """JPS规划"""
        start_gx, start_gy = self.map.world_to_grid(start[0], start[1])
        goal_gx, goal_gy = self.map.world_to_grid(goal[0], goal[1])
        
        open_list = []
        heapq.heappush(open_list, (0, (start_gx, start_gy)))
        
        came_from = {}
        g_score = {}
        g_score[(start_gx, start_gy)] = 0
        
        while open_list:
            current_f, current = heapq.heappop(open_list)
            
            if current == (goal_gx, goal_gy):
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append((start_gx, start_gy))
                path.reverse()
                
                world_path = []
                for gx, gy in path:
                    wx, wy = self.map.grid_to_world(gx, gy)
                    world_path.append((wx, wy))
                
                return world_path
            
            # 寻找跳跃点
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    jump_point = self.jump(current[0], current[1], dx, dy, goal_gx, goal_gy)
                    if jump_point:
                        jx, jy = jump_point
                        move_cost = np.sqrt((jx - current[0])**2 + (jy - current[1])**2)
                        tentative_g = g_score[current] + move_cost
                        
                        if (jx, jy) not in g_score or tentative_g < g_score[(jx, jy)]:
                            g_score[(jx, jy)] = tentative_g
                            h = self.heuristic((jx, jy), (goal_gx, goal_gy))
                            f = tentative_g + h
                            heapq.heappush(open_list, (f, (jx, jy)))
                            came_from[(jx, jy)] = current
        
        return None
```

---

## 5. 实践练习

### 练习1：Dijkstra vs A*

```python
def exercise_dijkstra_astar():
    """Dijkstra与A*比较"""
    print("=== Dijkstra vs A* ===")
    
    # 创建简单地图
    from .01-environment-representation import OccupancyGrid
    
    grid = OccupancyGrid(80, 80, 0.1)
    
    # 添加障碍物墙
    for i in range(80):
        grid.set_occupancy(-2, i * 0.1 - 4, True)
        grid.set_occupancy(2, i * 0.1 - 4, True)
        grid.set_occupancy(i * 0.1 - 4, -2, True)
        grid.set_occupancy(i * 0.1 - 4, 2, True)
    
    # 添加中间障碍物
    for x in np.linspace(-0.5, 0.5, 20):
        for y in np.linspace(-0.5, 0.5, 20):
            grid.set_occupancy(x, y, True)
    
    # 创建规划器
    import time
    start = (-3, -3)
    goal = (3, 3)
    
    # Dijkstra
    dijkstra = Dijkstra(grid)
    t1 = time.time()
    path_d = dijkstra.plan(start, goal)
    t2 = time.time()
    print(f"Dijkstra 耗时: {(t2 - t1)*1000:.2f}ms, 路径长度: {len(path_d) if path_d else 'N/A'}")
    
    # A* 欧几里得
    astar_e = AStar(grid, heuristic_type='euclidean')
    t1 = time.time()
    path_ae = astar_e.plan(start, goal)
    t2 = time.time()
    print(f"A* (Euclidean) 耗时: {(t2 - t1)*1000:.2f}ms, 路径长度: {len(path_ae) if path_ae else 'N/A'}")
    
    # A* 对角
    astar_o = AStar(grid, heuristic_type='octile')
    t1 = time.time()
    path_ao = astar_o.plan(start, goal)
    t2 = time.time()
    print(f"A* (Octile) 耗时: {(t2 - t1)*1000:.2f}ms, 路径长度: {len(path_ao) if path_ao else 'N/A'}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(131)
    plt.imshow(grid.grid, origin='lower', cmap='gray')
    if path_d:
        path_g = [grid.world_to_grid(x, y) for x, y in path_d]
        path_g = np.array(path_g)
        plt.plot(path_g[:, 0], path_g[:, 1], 'b-', label='Dijkstra')
    plt.title('Dijkstra')
    plt.legend()
    
    plt.subplot(132)
    plt.imshow(grid.grid, origin='lower', cmap='gray')
    if path_ae:
        path_g = [grid.world_to_grid(x, y) for x, y in path_ae]
        path_g = np.array(path_g)
        plt.plot(path_g[:, 0], path_g[:, 1], 'g-', label='A* (Euclidean)')
    plt.title('A* (Euclidean)')
    plt.legend()
    
    plt.subplot(133)
    plt.imshow(grid.grid, origin='lower', cmap='gray')
    if path_ao:
        path_g = [grid.world_to_grid(x, y) for x, y in path_ao]
        path_g = np.array(path_g)
        plt.plot(path_g[:, 0], path_g[:, 1], 'r-', label='A* (Octile)')
    plt.title('A* (Octile)')
    plt.legend()
    
    plt.savefig('dijkstra_astar.png')
    print("对比图已保存")

# exercise_dijkstra_astar()
```

### 练习2：A*可视化

```python
def exercise_astar_visualization():
    """A*可视化"""
    print("=== A*可视化 ===")
    
    # 创建环境
    from .01-environment-representation import OccupancyGrid
    grid = OccupancyGrid(60, 60, 0.1)
    
    # 添加随机障碍物
    np.random.seed(42)
    for _ in range(100):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        grid.set_occupancy(x, y, True)
    
    # 规划
    start = (-2.5, -2.5)
    goal = (2.5, 2.5)
    
    planner = AStar(grid)
    path = planner.plan(start, goal)
    
    # 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.grid, origin='lower', cmap='gray_r')
    
    if path:
        path_g = [grid.world_to_grid(x, y) for x, y in path]
        path_g = np.array(path_g)
        plt.plot(path_g[:, 0], path_g[:, 1], 'r-', linewidth=2, label='Path')
        
        start_g = grid.world_to_grid(start[0], start[1])
        goal_g = grid.world_to_grid(goal[0], goal[1])
        plt.plot(start_g[0], start_g[1], 'go', markersize=10, label='Start')
        plt.plot(goal_g[0], goal_g[1], 'bo', markersize=10, label='Goal')
    
    plt.title('A* Path Planning')
    plt.legend()
    plt.savefig('astar_planning.png')
    print("路径图已保存")

# exercise_astar_visualization()
```

---

**下一节**：[局部规划](03-local-planning.md)

---

## 参考文献

1. Dijkstra, E. W. (1959). A Note on Two Problems in Connexion with Graphs.
2. Hart, P. E., et al. (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths.
3. Daniel, K., et al. (2010). Theta*: Any-Angle Path Planning on Grids.
4. Harabor, D., & Grastien, A. (2011). Online Graph Pruning for Pathfinding on Grid Maps.
5. Lozano-Pérez, T. (1983). Spatial Planning: A Configuration Space Approach.
