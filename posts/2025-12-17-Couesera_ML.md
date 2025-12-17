---
title: Coursera#1 多元线性回归
date: 2025-12-17
category: 学习
tags: [ML, Vectorization]
---

# ML-多元线性回归

***

## 1. 矢量化

在多元线性回归中，首先要被提及的就是矢量化的概念，将特征值与权重值两组数据看作两个向量，计算时进行点积，最后加上偏移量b。这种方式比手敲公式以及for循环来的快的多，因为numpy中的dot方法可以并行地调用内存，从而提升效率。
$$
\vec{w} = [w_1, w_2, w_3, ...]\\
\vec{x} = [x_1, x_2, x_3, ...]\\
f_\vec{w},_b = w_1x_1 + w_2,x_2 + ... + b
$$

### Without Vectorization

```python
f = w[0] * x[0] +
	w[1] * x[1] +
    w[2] * x[2] + 
    w[3] * x[3] +
    ...			+ b
```

```python
f = 0
for j in range(n):
    f = f + w[j] * x[j]
f = f + b
```

### Vectorization

```python
f = np.dot(w,x) + b
```

//to be continued...