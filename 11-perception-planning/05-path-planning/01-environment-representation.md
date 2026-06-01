# 5.1 环境表示

## 目录

- [1. 引言](#1-引言)
- [2. 栅格地图](#2-栅格地图)
- [3. 拓扑地图](#3-拓扑地图)
- [4. 混合表示](#4-混合表示)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 环境表示的重要性

环境表示是路径规划的基础，它决定了机器人如何理解和抽象周围环境。好的环境表示方法能让规划算法既高效又准确。

### 1.2 常见表示方法

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2

class EnvRepresentation:
    """环境表示"""
    
    @staticmethod
    def grid_map():
        """栅格地图"""
        return {
            "name": "栅格地图",
            "description": "将环境分为均匀的栅格",
            "pros": ["简单", "便于规划"],
            "cons": ["内存大", "分辨率和精度矛盾"]
        }
    
    @staticmethod
    def topological_map():
        """拓扑地图"""
        return {
            "name": "拓扑地图",
            "description": "用节点和边表示环境",
            "pros": ["内存小", "层次化"],
            "cons": ["细节少", "构建困难"]
        }
    
    @staticmethod
    def hybrid_map():
        """混合地图"""
        return {
            "name": "混合地图",
            "description": "结合多种表示方法",
            "pros": ["优点互补"],
            "cons": ["复杂"]
        }
```

---

## 2. 栅格地图

### 2.1 占据栅格

```python
class OccupancyGrid:
    """占据栅格地图"""
    
    def __init__(self, width, height, resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution
        
        # 0: 空闲, 1: 占据, -1: 未知
        self.grid = np.zeros((height, width), dtype=np.int8)
        
        # 原点在左下角
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2
    
    def world_to_grid(self, x, y):
        """世界坐标转栅格坐标"""
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy
    
    def grid_to_world(self, gx, gy):
        """栅格坐标转世界坐标"""
        x = gx * self.resolution + self.origin_x + self.resolution / 2
        y = gy * self.resolution + self.origin_y + self.resolution / 2
        return x, y
    
    def set_occupancy(self, x, y, occupied):
        """设置占据状态"""
        gx, gy = self.world_to_grid(x, y)
        if 0 <= gx < self.width and 0 <= gy < self.height:
            self.grid[gy, gx] = 1 if occupied else 0
    
    def is_occupied(self, gx, gy):
        """检查是否占据"""
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.grid[gy, gx] == 1
        return True
    
    def inflate_obstacles(self, inflation_radius):
        """膨胀障碍物"""
        from scipy.ndimage import binary_dilation
        
        kernel_size = int(inflation_radius / self.resolution)
        if kernel_size > 0:
            kernel = np.ones((2 * kernel_size + 1, 2 * kernel_size + 1), dtype=np.uint8)
            self.grid = binary_dilation(self.grid > 0, kernel).astype(np.int8)
```

### 2.2 代价地图

```python
class Costmap(OccupancyGrid):
    """代价地图"""
    
    def __init__(self, width, height, resolution=0.1):
        super().__init__(width, height, resolution)
        self.cost = np.zeros((height, width), dtype=np.float32)
        self.max_cost = 255
    
    def compute_cost(self, obstacle_radius=0.5):
        """计算代价值"""
        from scipy.ndimage import distance_transform_edt
        
        # 距离变换
        binary_occ = (self.grid == 1).astype(np.uint8)
        dist = distance_transform_edt(1 - binary_occ) * self.resolution
        
        # 计算代价
        self.cost = self.max_cost * np.exp(-dist / obstacle_radius)
        self.cost[binary_occ > 0] = self.max_cost
    
    def get_cost(self, x, y):
        """获取代价"""
        gx, gy = self.world_to_grid(x, y)
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.cost[gy, gx]
        return self.max_cost
```

---

## 3. 拓扑地图

### 3.1 图表示

```python
class TopologicalNode:
    """拓扑节点"""
    
    def __init__(self, node_id, position, neighbors=None):
        self.id = node_id
        self.position = np.array(position)
        self.neighbors = neighbors if neighbors is not None else []
        self.data = {}

class TopologicalMap:
    """拓扑地图"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = []
    
    def add_node(self, node_id, position):
        """添加节点"""
        if node_id not in self.nodes:
            self.nodes[node_id] = TopologicalNode(node_id, position)
    
    def add_edge(self, id1, id2, bidirectional=True):
        """添加边"""
        if id1 in self.nodes and id2 in self.nodes:
            self.nodes[id1].neighbors.append(id2)
            if bidirectional:
                self.nodes[id2].neighbors.append(id1)
            
            self.edges.append((id1, id2))
    
    def find_nearest_node(self, position):
        """找到最近的节点"""
        best_id = None
        best_dist = float('inf')
        
        for node_id, node in self.nodes.items():
            dist = np.linalg.norm(node.position - position)
            if dist < best_dist:
                best_dist = dist
                best_id = node_id
        
        return best_id
    
    def shortest_path(self, start_id, end_id):
        """拓扑路径（BFS）"""
        from collections import deque
        
        visited = {start_id: None}
        queue = deque([start_id])
        
        while queue:
            current = queue.popleft()
            
            if current == end_id:
                # 回溯路径
                path = []
                while current is not None:
                    path.append(current)
                    current = visited[current]
                return list(reversed(path))
            
            for neighbor in self.nodes[current].neighbors:
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
        
        return None
```

### 3.2 Voronoi图

```python
class VoronoiRoadmap:
    """Voronoi路线图"""
    
    def __init__(self, obstacles):
        self.obstacles = np.array(obstacles)
    
    def compute_voronoi(self, points):
        """计算Voronoi图"""
        from scipy.spatial import Voronoi
        
        vor = Voronoi(np.vstack([points, self.obstacles]))
        return vor
    
    def extract_roads(self, vor, free_space_mask=None):
        """提取道路"""
        roads = []
        
        for i, ridge in enumerate(vor.ridge_vertices):
            if -1 not in ridge:  # 不包含无穷远
                p1 = vor.vertices[ridge[0]]
                p2 = vor.vertices[ridge[1]]
                roads.append((p1, p2))
        
        return roads
```

---

## 4. 混合表示

### 4.1 分层地图

```python
class HierarchicalMap:
    """分层地图"""
    
    def __init__(self):
        # 高层：拓扑地图
        self.topological = TopologicalMap()
        
        # 低层：栅格地图
        self.gridmaps = {}  # 每个区域的局部栅格地图
    
    def add_region(self, region_id, gridmap):
        """添加区域"""
        self.gridmaps[region_id] = gridmap
    
    def plan_global(self, start, end):
        """全局规划（拓扑层）"""
        start_id = self.topological.find_nearest_node(start)
        end_id = self.topological.find_nearest_node(end)
        
        if start_id and end_id:
            return self.topological.shortest_path(start_id, end_id)
        return None
    
    def plan_local(self, region_id, start, end):
        """局部规划（栅格层）"""
        if region_id in self.gridmaps:
            # 在此区域内使用栅格规划
            pass
        return None
```

---

## 5. 实践练习

### 练习1：创建简单栅格地图

```python
def exercise_grid_map():
    """栅格地图练习"""
    print("=== 栅格地图练习 ===")
    
    # 创建地图
    grid = OccupancyGrid(100, 100, 0.1)
    
    # 添加一些障碍物
    # 中心矩形
    for x in np.linspace(-2, 2, 20):
        for y in np.linspace(-1, 1, 20):
            grid.set_occupancy(x, y, True)
    
    # 可视化
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.grid, origin='lower', cmap='gray')
    plt.title('Occupancy Grid')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.savefig('grid_map.png')
    print("地图已保存到 grid_map.png")

# exercise_grid_map()
```

### 练习2：代价地图膨胀

```python
def exercise_costmap():
    """代价地图练习"""
    print("=== 代价地图练习 ===")
    
    # 创建代价地图
    costmap = Costmap(100, 100, 0.1)
    
    # 添加障碍物
    for x in np.linspace(-1, 1, 10):
        for y in np.linspace(-1, 1, 10):
            costmap.set_occupancy(x, y, True)
    
    # 计算代价
    costmap.compute_cost(inflation_radius=0.8)
    
    # 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(costmap.grid, origin='lower', cmap='gray')
    plt.title('Occupancy')
    plt.subplot(122)
    plt.imshow(costmap.cost, origin='lower', cmap='hot')
    plt.title('Costmap')
    plt.savefig('costmap.png')
    print("代价地图已保存")

# exercise_costmap()
```

### 练习3：拓扑路径

```python
def exercise_topological_map():
    """拓扑地图练习"""
    print("=== 拓扑地图练习 ===")
    
    # 创建拓扑地图
    topo = TopologicalMap()
    
    # 添加节点
    nodes = [
        (0, (0, 0)),
        (1, (1, 0)),
        (2, (2, 0)),
        (3, (1, 1)),
        (4, (2, 1)),
        (5, (0, 2)),
        (6, (1, 2))
    ]
    
    for node_id, pos in nodes:
        topo.add_node(node_id, pos)
    
    # 添加边
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4),
             (0, 5), (3, 6), (5, 6)]
    for id1, id2 in edges:
        topo.add_edge(id1, id2)
    
    # 找路径
    path = topo.shortest_path(0, 4)
    print(f"路径: {path}")
    
    # 可视化
    plt.figure(figsize=(8, 8))
    
    # 画节点
    for node_id, node in topo.nodes.items():
        plt.plot(node.position[0], node.position[1], 'bo', markersize=10)
        plt.text(node.position[0] + 0.1, node.position[1] + 0.1, 
                 str(node_id), fontsize=12)
    
    # 画边
    for id1, id2 in topo.edges:
        p1 = topo.nodes[id1].position
        p2 = topo.nodes[id2].position
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')
    
    # 画路径
    if path:
        path_points = [topo.nodes[i].position for i in path]
        path_points = np.array(path_points)
        plt.plot(path_points[:, 0], path_points[:, 1], 'r-', linewidth=3, label='Path')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Topological Map')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('topological_map.png')
    print("拓扑地图已保存")

# exercise_topological_map()
```

---

**下一节**：[全局规划](02-global-planning.md)

---

## 参考文献

1. Elfes, A. (1989). Occupancy Grids: A Probabilistic Framework for Robot Perception and Navigation.
2. Thrun, S. (1998). Learning Metric-Topological Maps for Indoor Mobile Robot Navigation.
3. Lozano-Pérez, T., & Wesley, M. A. (1979). An Algorithm for Planning Collision-Free Paths Among Polyhedral Obstacles.
4. Aurenhammer, F. (1991). Voronoi Diagrams — A Survey of a Fundamental Geometric Data Structure.
