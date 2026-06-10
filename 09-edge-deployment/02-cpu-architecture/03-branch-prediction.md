# 2.3 分支预测

## 1. 为什么需要分支预测

分支指令在程序中占比约15-20%，而每次分支误预测可能导致10-25个周期的流水线冲刷。

```asm
loop:                    ; 循环分支 (90%概率跳转)
    ADD R1, R1, #1
    CMP R1, #100
    BNE loop             ; 预测跳转 → 正确
                          ; 预测不跳转 → 错误, 冲刷20+条指令
    
if_sensor:               ; 传感器阈值分支
    LDR R2, [sensor_val]
    CMP R2, #THRESHOLD
    BGT trigger_action   ; 预测正确率依赖传感器模式
```

---

## 2. 分支预测技术演进

| 技术 | 准确率 | 硬件开销 | 原理 |
|------|-------|---------|------|
| 静态预测 (BTFN) | ~60% | 极小 | 总是预测不跳转 |
| 2-bit饱和计数器 | ~85-90% | ~2bit/条目 | 基于局部历史 |
| 两级自适应 | ~93-97% | ~10-100KB | 全局+局部历史 |
| TAGE | ~95-98% | ~100KB-1MB | 多种长度历史组合 |
| 神经预测器 | ~97-99% | ~MB级 | 小型神经网络 |

---

## 3. 在机器人中的影响

### 3.1 行为树中的分支

```cpp
// 机器人行为树节点
BT::NodeStatus ConditionNode::tick() {
    // 传感器值比较 - 分支结果取决于环境
    if (battery_level_ < 0.2)      // 20%概率进入
        return NodeStatus::FAILURE;
    if (obstacle_detected_)         // 30%概率
        return NodeStatus::SUCCESS;
    return NodeStatus::RUNNING;     // 50%概率
    // 分支模式随机 → 预测准确率低
}
```

### 3.2 无分支编程优化

```cpp
// ❌ 有分支版本 (碰撞检测)
float clamp_branch(float val, float min, float max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

// ✅ 无分支版本 (使用条件移动)
float clamp_nobranch(float val, float min, float max) {
    val = fmaxf(val, min);  // 无分支
    val = fminf(val, max);  // 无分支
    return val;
}
// 现代ARM有 FMAX/FMIN 指令, 无条件执行
```

---

## 4. 相关论文

| 论文 | 发表 | 内容 |
|------|------|------|
| "TAGE-SC-L Branch Predictors" | Seznec, JILP 2014 | TAGE预测器 |
| "Branch Prediction for Real-Time Systems" | ECRTS 2019 | 实时系统中的分支预测 |
| "Neural Branch Prediction" | ACM CS 2021 | 神经分支预测综述 |
