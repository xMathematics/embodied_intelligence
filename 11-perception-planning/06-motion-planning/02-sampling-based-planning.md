# 6.2 采样规划

## 目录

- [1. 引言](#1-引言)
- [2. 概率路标图 (PRM)](#2-概率路标图-prm)
- [3. 快速随机探索树 (RRT)](#3-快速随机探索树-rrt)
- [4. RRT* 和其他改进](#4-rrt-和其他改进)
- [5. 实践练习](#5-实践练习)

---

## 1. 引言

### 1.1 为什么采样规划？

高维运动规划（如6-DoF机械臂）无法用栅格法高效求解，需要采样规划。

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
```

---

## 2. 概率路标图 (PRM)

### 2.1 PRM实现

```python
class PRM:
    """概率路标图"""
    
    def __init__(self, problem):
        self.problem = problem
        self.graph = defaultdict(list)
        self.nodes = []
    
    def sample_node(self):
        """随机采样节点"""
        q = self.problem.cspace.sample()
        if self.problem.is_valid(q):
            return q
        return None
    
    def build_roadmap(self, num_samples, k=5):
        """构建路标图"""
        self.nodes = []
        self.graph = defaultdict(list)
        
        # 采样节点
        while len(self.nodes) < num_samples:
            q = self.sample_node()
            if q is not None:
                self.nodes.append(q.copy())
        
        # 连接近邻
        for i, q_i in enumerate(self.nodes):
            # 找最近的k个
            dists = []
            for j, q_j in enumerate(self.nodes):
                if i != j:
                    d = self.problem.distance(q_i, q_j)
                    dists.append((d, j))
            
            dists.sort()
            
            for d, j in dists[:k]:
                q_j = self.nodes[j]
                if self.problem.is_edge_valid(q_i, q_j):
                    self.graph[i].append(j)
                    self.graph[j].append(i)
    
    def query(self, q_start, q_goal):
        """查询路径"""
        # 将起点和终点加入图
        temp_nodes = self.nodes.copy()
        temp_graph = defaultdict(list, dict(self.graph))
        
        start_idx = len(temp_nodes)
        goal_idx = len(temp_nodes) + 1
        temp_nodes.append(q_start)
        temp_nodes.append(q_goal)
        
        # 连接起点
        dists = []
        for i, q in enumerate(self.nodes):
            d = self.problem.distance(q_start, q)
            dists.append((d, i))
        dists.sort()
        for d, i in dists[:10]:
            if self.problem.is_edge_valid(q_start, self.nodes[i]):
                temp_graph[start_idx].append(i)
                temp_graph[i].append(start_idx)
        
        # 连接终点
        dists = []
        for i, q in enumerate(self.nodes):
            d = self.problem.distance(q_goal, q)
            dists.append((d, i))
        dists.sort()
        for d, i in dists[:10]:
            if self.problem.is_edge_valid(q_goal, self.nodes[i]):
                temp_graph[goal_idx].append(i)
                temp_graph[i].append(goal_idx)
        
        # A*搜索
        path = self._astar_search(temp_nodes, temp_graph, start_idx, goal_idx)
        if path:
            return [temp_nodes[idx] for idx in path]
        return None
    
    def _astar_search(self, nodes, graph, start_idx, goal_idx):
        """A*搜索"""
        import heapq
        
        open_set = []
        heapq.heappush(open_set, (0, start_idx))
        
        came_from = {}
        g_score = {start_idx: 0}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal_idx:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_idx)
                return list(reversed(path))
            
            for neighbor in graph[current]:
                tentative_g = g_score[current] + \
                    self.problem.distance(nodes[current], nodes[neighbor])
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    h = self.problem.distance(nodes[neighbor], nodes[goal_idx])
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current
        
        return None
```

---

## 3. 快速随机探索树 (RRT)

### 3.1 RRT实现

```python
class RRT:
    """快速随机探索树"""
    
    def __init__(self, problem, step_size=0.2):
        self.problem = problem
        self.step_size = step_size
        self.tree = []
        self.parent = {}
    
    def plan(self, q_start, q_goal, max_iterations=10000, goal_sample_rate=0.05):
        """规划"""
        self.tree = [np.array(q_start)]
        self.parent = {0: None}
        
        for i in range(max_iterations):
            # 采样
            if np.random.random() < goal_sample_rate:
                q_rand = np.array(q_goal)
            else:
                q_rand = self.problem.cspace.sample()
            
            # 找最近节点
            nearest_idx = self._nearest_neighbor(q_rand)
            q_nearest = self.tree[nearest_idx]
            
            # 扩展
            q_new = self._steer(q_nearest, q_rand)
            
            if self.problem.is_edge_valid(q_nearest, q_new):
                self.tree.append(q_new)
                self.parent[len(self.tree) - 1] = nearest_idx
                
                # 检查是否接近目标
                if self.problem.distance(q_new, q_goal) < self.step_size:
                    if self.problem.is_edge_valid(q_new, q_goal):
                        self.tree.append(np.array(q_goal))
                        self.parent[len(self.tree) - 1] = len(self.tree) - 2
                        return self._extract_path(len(self.tree) - 1)
        
        return None
    
    def _nearest_neighbor(self, q_rand):
        """找最近邻居"""
        distances = [self.problem.distance(q, q_rand) for q in self.tree]
        return np.argmin(distances)
    
    def _steer(self, q_from, q_to):
        """朝目标方向扩展"""
        d = self.problem.distance(q_from, q_to)
        if d < self.step_size:
            return np.array(q_to)
        return q_from + (q_to - q_from) / d * self.step_size
    
    def _extract_path(self, goal_idx):
        """提取路径"""
        path = []
        current = goal_idx
        while current is not None:
            path.append(self.tree[current])
            current = self.parent[current]
        return list(reversed(path))
```

---

## 4. RRT* 和其他改进

### 4.1 RRT*实现

```python
class RRTStar(RRT):
    """RRT* (渐近最优)"""
    
    def __init__(self, problem, step_size=0.2, rewire_radius=0.5):
        super().__init__(problem, step_size)
        self.rewire_radius = rewire_radius
    
    def plan(self, q_start, q_goal, max_iterations=10000, goal_sample_rate=0.05):
        """规划"""
        self.tree = [np.array(q_start)]
        self.parent = {0: None}
        self.cost = {0: 0.0}
        
        for i in range(max_iterations):
            if np.random.random() < goal_sample_rate:
                q_rand = np.array(q_goal)
            else:
                q_rand = self.problem.cspace.sample()
            
            # 找最近节点
            nearest_idx = self._nearest_neighbor(q_rand)
            q_nearest = self.tree[nearest_idx]
            q_new = self._steer(q_nearest, q_rand)
            
            if self.problem.is_edge_valid(q_nearest, q_new):
                # 找近邻
                near_indices = self._near_vertices(q_new)
                
                # 选择最佳父节点
                min_cost = float('inf')
                best_parent = nearest_idx
                
                for idx in near_indices:
                    q = self.tree[idx]
                    if self.problem.is_edge_valid(q, q_new):
                        c = self.cost[idx] + self.problem.distance(q, q_new)
                        if c < min_cost:
                            min_cost = c
                            best_parent = idx
                
                # 添加新节点
                self.tree.append(q_new)
                new_idx = len(self.tree) - 1
                self.parent[new_idx] = best_parent
                self.cost[new_idx] = min_cost
                
                # 重连
                self._rewire(q_new, new_idx, near_indices)
                
                # 检查目标
                if self.problem.distance(q_new, q_goal) < self.step_size:
                    if self.problem.is_edge_valid(q_new, q_goal):
                        self.tree.append(np.array(q_goal))
                        self.parent[len(self.tree) - 1] = new_idx
                        self.cost[len(self.tree) - 1] = self.cost[new_idx] + \
                            self.problem.distance(q_new, q_goal)
                        return self._extract_path(len(self.tree) - 1)
        
        return None
    
    def _near_vertices(self, q_new):
        """找近邻节点"""
        near = []
        for i, q in enumerate(self.tree):
            if self.problem.distance(q, q_new) < self.rewire_radius:
                near.append(i)
        return near
    
    def _rewire(self, q_new, new_idx, near_indices):
        """重连"""
        for idx in near_indices:
            q = self.tree[idx]
            if self.problem.is_edge_valid(q_new, q):
                c_new = self.cost[new_idx] + self.problem.distance(q_new, q)
                if c_new < self.cost[idx]:
                    self.parent[idx] = new_idx
                    self.cost[idx] = c_new
```

---

## 5. 实践练习

### 练习1：PRM演示

```python
def exercise_prm():
    """PRM练习"""
    print("=== PRM 演示 ===")
    
    from .01-motion-planning-basics import CSpace, BoxObstacle, SimpleRobot, MotionPlanningProblem
    
    # 环境
    cspace = CSpace(bounds=[(-2, 6), (-2, 6)])
    robot = SimpleRobot(size=0.2)
    obstacles = [
        BoxObstacle(center=(1.5, 1.5), size=(2, 2)),
        BoxObstacle(center=(3.5, 3.5), size=(2, 2)),
    ]
    problem = MotionPlanningProblem(cspace, robot, obstacles)
    
    # PRM
    prm = PRM(problem)
    print("构建路标图...")
    prm.build_roadmap(num_samples=100, k=5)
    
    # 查询
    q_start = np.array([0, 0])
    q_goal = np.array([5, 5])
    print("查询路径...")
    path = prm.query(q_start, q_goal)
    
    # 可视化
    plt.figure(figsize=(10, 10))
    
    # 画障碍物
    for obs in obstacles:
        rect = plt.Rectangle(
            (obs.center[0] - obs.size[0]/2, obs.center[1] - obs.size[1]/2),
            obs.size[0], obs.size[1], facecolor='red', alpha=0.5
        )
        plt.gca().add_patch(rect)
    
    # 画PRM图
    nodes = np.array(prm.nodes)
    plt.scatter(nodes[:, 0], nodes[:, 1], c='blue', s=20, alpha=0.5)
    
    for i, neighbors in prm.graph.items():
        for j in neighbors:
            if i < j:
                plt.plot([nodes[i, 0], nodes[j, 0]], 
                        [nodes[i, 1], nodes[j, 1]], 'b-', alpha=0.3)
    
    # 画路径
    if path:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'g-', linewidth=3, label='Path')
    
    plt.plot(q_start[0], q_start[1], 'go', markersize=10, label='Start')
    plt.plot(q_goal[0], q_goal[1], 'r*', markersize=15, label='Goal')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('PRM Planning')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('prm_demo.png')
    print("PRM演示图已保存")

# exercise_prm()
```

### 练习2：RRT演示

```python
def exercise_rrt():
    """RRT练习"""
    print("=== RRT 演示 ===")
    
    from .01-motion-planning-basics import CSpace, BoxObstacle, SimpleRobot, MotionPlanningProblem
    
    cspace = CSpace(bounds=[(-2, 6), (-2, 6)])
    robot = SimpleRobot(size=0.2)
    obstacles = [
        BoxObstacle(center=(1.5, 1.5), size=(2, 2)),
        BoxObstacle(center=(3.5, 3.5), size=(2, 2)),
    ]
    problem = MotionPlanningProblem(cspace, robot, obstacles)
    
    rrt = RRT(problem, step_size=0.3)
    
    q_start = np.array([0, 0])
    q_goal = np.array([5, 5])
    print("RRT规划...")
    path = rrt.plan(q_start, q_goal, max_iterations=5000)
    
    plt.figure(figsize=(10, 10))
    
    for obs in obstacles:
        rect = plt.Rectangle(
            (obs.center[0] - obs.size[0]/2, obs.center[1] - obs.size[1]/2),
            obs.size[0], obs.size[1], facecolor='red', alpha=0.5
        )
        plt.gca().add_patch(rect)
    
    # 画树
    tree = np.array(rrt.tree)
    plt.scatter(tree[:, 0], tree[:, 1], c='blue', s=10, alpha=0.3)
    
    for i, parent_idx in rrt.parent.items():
        if parent_idx is not None:
            plt.plot([tree[i, 0], tree[parent_idx, 0]], 
                    [tree[i, 1], tree[parent_idx, 1]], 'b-', alpha=0.2)
    
    if path:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'g-', linewidth=3, label='Path')
    
    plt.plot(q_start[0], q_start[1], 'go', markersize=10, label='Start')
    plt.plot(q_goal[0], q_goal[1], 'r*', markersize=15, label='Goal')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('RRT Planning')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('rrt_demo.png')
    print("RRT演示图已保存")

# exercise_rrt()
```

---

**下一节**：[轨迹优化](03-trajectory-optimization.md)

---

## 参考文献

1. Kavraki, L. E., et al. (1996). Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces.
2. LaValle, S. M. (1998). Rapidly-Exploring Random Trees: A New Tool for Path Planning.
3. Karaman, S., & Frazzoli, E. (2011). Sampling-Based Algorithms for Optimal Motion Planning.
4. Karaman, S., & Frazzoli, E. (2010). Incremental Sampling-Based Algorithms for a Class of Pursuit-Evasion Games.
