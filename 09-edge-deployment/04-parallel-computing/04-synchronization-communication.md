# 4.4 同步与通信

## 1. 同步原语

| 机制 | 粒度 | 开销 | 用途 | 在机器人中 |
|------|------|------|------|-----------|
| **原子操作** | 单变量 | 低 | 计数器 | 帧计数 |
| **互斥锁** | 临界区 | 中 | 共享资源 | 地图访问 |
| **读写锁** | 区分读写 | 中 | 读多写少 | 全局地图 |
| **屏障** | 线程组 | 中 | 同步点 | 传感器对齐 |
| **信号量** | 资源计数 | 中 | 生产者-消费者 | 数据缓冲区 |
| **无锁队列** | 无阻塞 | 低 | 高吞吐 | ROS2消息 |

## 2. 无锁队列在ROS2中

```cpp
// ROS2使用无锁队列 (原子CAS)
// 核心: compare-and-swap
bool cas(int* ptr, int expected, int desired) {
    return atomic_compare_exchange(ptr, expected, desired);
    // 如果 *ptr == expected, 则 *ptr = desired, 返回true
    // 否则返回false
}

// 优点: 无阻塞, 适合实时控制
// 缺点: 实现复杂, ABA问题
```

## 3. 避免死锁

```cpp
// ❌ 死锁风险
void robot_update() {
    lock(map_mutex);     // 线程1持map锁
    // ... 需要sensor锁
    lock(sensor_mutex);  // 线程2持sensor锁等map锁
}

// ✅ 固定顺序
void robot_update() {
    // 所有线程按相同顺序加锁
    lock(sensor_mutex);  // 先sensor
    lock(map_mutex);     // 后map
}
```

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "Lock-Free Data Structures for Robotics" | ICRA 2020 | 无锁数据结构 |
| "Wait-Free Synchronization in ROS" | IEEE RAM 2021 | ROS无锁同步 |
