# 3.8 优化库详解：g2o、Ceres、GTSAM

## 1. 概述

现代SLAM系统通常使用现成的优化库进行后端优化。本章详细介绍三个最主流的优化库g2o、Ceres Solver和GTSAM的特点、用法和选型建议。

## 2. g2o

### 2.1 简介

g2o（General Graph Optimization, 2011）是SLAM领域最常用的图优化库，由Kümmerle等人开发。

**特点**：
- 专门为SLAM设计
- 支持顶点（Vertex）和边（Edge）的抽象
- 内置多种求解器和线性代数接口

### 2.2 核心架构

```
g2o框架
├── SparseOptimizer       # 优化器主入口
├── Vertex                # 顶点基类
│   ├── SE3Quat           # SE(3)位姿顶点
│   ├── PointXYZ          # 3D点顶点
│   └── SBAPointXYZ       # 稀疏BA点顶点
└── Edge                  # 边基类
    ├── EdgeSE3           # SE(3)相对位姿边
    ├── EdgeProjectXYZ2UV # 重投影误差边
    └── EdgeSE3ProjectXYZ # SE(3)重投影边
```

### 2.3 使用示例

```cpp
// 创建优化器
g2o::SparseOptimizer optimizer;
auto* solver = g2o::make_unique<BlockSolver<BlockSolverTraits<-1, -1>>>();
optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(solver));

// 添加顶点
g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
pose->setId(0);
pose->setEstimate(Eigen::Isometry3d::Identity());
optimizer.addVertex(pose);

// 添加边
g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
edge->setVertex(0, point);
edge->setVertex(1, pose);
edge->setMeasurement(Eigen::Vector2d(u, v));
edge->setInformation(Eigen::Matrix2d::Identity());
optimizer.addEdge(edge);

// 执行优化
optimizer.initializeOptimization();
optimizer.optimize(10);
```

## 3. Ceres Solver

### 3.1 简介

Ceres Solver是Google开发的开源非线性最小二乘求解器，广泛应用于SLAM和SfM。

**特点**：
- 自动微分（AutoDiff）
- 数值微分和解析微分
- 鲁棒核函数
- 优秀的稀疏求解器
- 支持流形优化（LocalParameterization）

### 3.2 核心概念

```cpp
// 创建问题
ceres::Problem problem;

// 定义代价函数 - 使用自动微分
struct ReprojectionError {
    ReprojectionError(double u, double v) : u_(u), v_(v) {}
    
    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        T p[3];
        // 投影计算...
        residuals[0] = T(u_) - p[0] / p[2];
        residuals[1] = T(v_) - p[1] / p[2];
        return true;
    }
private:
    double u_, v_;
};

// 添加残差块
ceres::CostFunction* cost_function =
    new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
        new ReprojectionError(u, v));
problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), camera, point);

// 配置求解器
ceres::Solver::Options options;
options.linear_solver_type = ceres::SPARSE_SCHUR;
options.minimizer_progress_to_stdout = true;

// 求解
ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);
```

### 3.3 鲁棒核函数

Ceres内置多种鲁棒核函数：

```cpp
// 可选核函数
new ceres::HuberLoss(1.0);      // Huber损失
new ceres::CauchyLoss(0.5);     // Cauchy损失
new ceres::TukeyLoss(1.0);      // Tukey损失
new ceres::SoftLOneLoss(1.0);   // Soft L1损失
```

## 4. GTSAM

### 4.1 简介

GTSAM（Georgia Tech Smoothing and Mapping）是基于因子图的优化库，由Dellaert等人开发。

**特点**：
- 原生支持因子图
- iSAM2增量求解器
- 丰富的SLAM因子类型
- 支持IMU预积分
- 开源且维护活跃

### 4.2 核心概念

```cpp
// 创建因子图
NonlinearFactorGraph graph;

// 添加先验因子
auto prior = PriorFactor<Pose3>(
    symbol('x', 0), Pose3(), noiseModel::Isotropic::Sigma(6, 0.1));
graph.push_back(prior);

// 添加里程计因子
auto odometry = BetweenFactor<Pose3>(
    symbol('x', 0), symbol('x', 1), 
    Pose3(Rot3::identity(), Point3(1, 0, 0)),
    noiseModel::Diagonal::Sigmas(Vector6(0.1, 0.1, 0.1, 0.01, 0.01, 0.01)));
graph.push_back(odometry);

// 添加观测因子
auto measurement = GenericProjectionFactor<Pose3, Point3>(
    Point2(u, v), noiseModel::Isotropic::Sigma(2, 1.0),
    symbol('x', 0), symbol('l', 0), K);
graph.push_back(measurement);

// 创建初始估计
Values initial;
initial.insert(symbol('x', 0), Pose3());
initial.insert(symbol('x', 1), Pose3(Rot3::identity(), Point3(1, 0, 0)));

// 使用iSAM2增量求解
ISAM2 isam;
isam.update(graph, initial);
```

### 4.3 IMU预积分

GTSAM对IMU预积分有原生支持：

```cpp
// 创建IMU预积分器
auto preintegrated = std::make_shared<PreintegratedImuMeasurements>(
    imu_params, bias);

// 添加IMU测量
preintegrated->integrateMeasurement(accel, gyro, dt);

// 创建IMU因子
auto imu_factor = ImuFactor(
    symbol('x', i), symbol('v', i), symbol('b', i),
    symbol('x', j), symbol('v', j), symbol('b', j),
    preintegrated);
graph.push_back(imu_factor);
```

## 5. 库对比

| 特性 | g2o | Ceres | GTSAM |
|------|-----|-------|-------|
| 自动微分 | 否 | 是 | 是 |
| 因子图 | 有限 | 否 | 原生支持 |
| iSAM2 | 否 | 否 | 是 |
| IMU预积分 | 手动 | 手动 | 原生支持 |
| 学习曲线 | 中 | 低 | 高 |
| SLAM生态 | 好 | 中 | 好 |
| 灵活性 | 中 | 高 | 中 |
| 文档 | 中 | 好 | 好 |

## 6. 选型建议

- **实时视觉SLAM**：g2o（成熟稳定）
- **多传感器融合**：GTSAM（因子图、IMU预积分）
- **通用优化**：Ceres（灵活、文档好）
- **增量SLAM**：GTSAM + iSAM2
- **学习型SLAM**：Ceres（可结合PyTorch）

## 7. 参考文献

1. Kümmerle, R., Grisetti, G., Strasdat, H., Konolige, K., & Burgard, W. (2011). g2o: A general framework for graph optimization. *ICRA*.
2. Agarwal, S., Mierle, K., & Others. Ceres Solver. http://ceres-solver.org
3. Dellaert, F. (2012). Factor graphs and GTSAM: A hands-on introduction. *Georgia Tech Technical Report*.
