# 5.2 词袋模型（Bag of Words）

## 1. 概述

词袋模型（Bag of Words, BoW）是回环检测中最经典的方法。它将图像表示为视觉词汇的直方图，通过比较直方图相似度检测回环。

## 2. 词袋模型原理

### 2.1 三个核心步骤

1. **词汇构建**：对所有训练图像的局部特征进行聚类，构建视觉词典
2. **图像编码**：将每幅图像表示为视觉词汇的频率直方图
3. **相似度计算**：比较两幅图像直方图的相似度

### 2.2 视觉词典构建

使用k-means聚类构建层次化词汇树：

```
根节点 ─→ 聚类K个词 ─→ 每个词再聚类K个词 ─→ ... ─→ 叶子节点
```

叶子节点数量（词汇量）通常为 $10^4$ 到 $10^6$。

### 2.3 TF-IDF加权

**词频（TF）**：词汇在图像中出现的频率

$$ \text{TF}_i = \frac{n_i}{n} $$

**逆文档频率（IDF）**：词汇在数据库中出现的频率的倒数

$$ \text{IDF}_i = \log \frac{N}{N_i} $$

**TF-IDF权重**：

$$ w_i = \text{TF}_i \times \text{IDF}_i $$

## 3. DBoW

### 3.1 DBoW（2010）

DBoW是SLAM中使用最广泛的词袋模型实现。

**特点**：
- 层次化词汇树
- TF-IDF加权
- 倒排索引加速查询
- 增量式词典更新

### 3.2 DBoW2（2012）

**改进**：
- 使用二进制特征（ORB、BRIEF）
- 改进的相似度评分
- 更高效的数据库管理

**相似度评分**：

$$ s(v_1, v_2) = 1 - \frac{1}{2} \left\| \frac{v_1}{\|v_1\|} - \frac{v_2}{\|v_2\|} \right\| $$

### 3.3 DBoW相似度归一化

$$ \eta = \frac{s(v_t, v_c)}{s(v_t, v_{t-1})} $$

其中 $s(v_t, v_c)$ 是当前帧与候选帧的相似度，$s(v_t, v_{t-1})$ 是与上一帧的相似度。

## 4. 优缺点

| 优点 | 缺点 |
|------|------|
| 速度快（倒排索引） | 对视角变化敏感 |
| 内存占用可控 | 忽略特征空间关系 |
| 经过广泛验证 | 受感知混叠影响 |
| 易于实现 | 光照变化鲁棒性一般 |

## 5. 参考文献

1. Sivic, J., & Zisserman, A. (2003). Video Google: A text retrieval approach to object matching in videos. *ICCV*.
2. Gálvez-López, D., & Tardós, J. D. (2012). Bags of binary words for fast place recognition in image sequences. *IEEE Transactions on Robotics*, 28(5), 1188-1197.
3. Mur-Artal, R., & Tardós, J. D. (2017). Fast relocalisation and loop closing in keyframe-based SLAM. *IEEE Transactions on Robotics*.
