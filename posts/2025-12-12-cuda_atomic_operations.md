---
title: CUDA#3 cuda atomic operations
date: 2025-12-12
category: 学习
tags: [CUDA, c++, 原子操作]
---

# CUDA Atomic Operations

***

## 0. 引入

在编写传统的单线程应用时通常不需要使用原子操作。在进入原子操作之前，先回顾一下传统的C/C++中最先接触的操作之一：

```c++
	x++;
```

这是一个标准C语言中的单行表达式，x的值在执行递增操作之后应当比之前大1，这很简单的逻辑中具体需要执行哪些操作呢？

+ 读取值`x`。首先需要知道`x`当前的值。
+ 将第一步中读取的值加1。
+ 将结果写回`x`。

有时候这个操作被称为读-修改-写操作。因为步骤2可以包含任何改变从`x`中读取的值的操作。现在想象一种情况：线程A和B都要执行对`x`值的递增操作，为了达到目的，它们需要执行前面描述的三个操作。假设`x`的初始值为7，理想情况下A和B依次执行(`A先读改写，然后B读改写`)的预期值为9。之前的顺序操作中，我们确实会得到9这一个结果，然而如果线程A和B交错执行，会让递增操作实际上只达到执行一次的效果(`A读-B读-A改-B改-A写-B写`)，最后结果为8。

因此如果我们的线程调度不当，最终会导致计算结果错误，这一系列操作还可以有好多不同的顺序，有些顺序能带来正确的结果，有些则不能。

> 当从单线程版本迁移到多线程版本时，如果多个线程需要读取或写入共享值，就可能出现不可预测的结果。

所以在这个例子中，我们需要使用一种方法来顺序执行读改写操作，而不会被其他线程中断。也就是说在一个线程完成操作之前，任何其他线程都不能读取或写入`x`的值。由于其他线程无法将这些操作的执行拆分成更小的部分，我们将满足此约束的操作称为原子操作。CUDA C支持多种原子操作，即使成千上万个线程可能同时争用内存，也能安全地进行内存操作。



## 1. 计算直方图

直方图表示每个元素出现的频率，下图是如果我们绘制短语“Programming with CUDA C”中字母的直方图。

| 2    | 2    | 1    | 2    | 1    | 2    | 2    | 1    | 1    | 1    | 2    | 1    | 1    | 1    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| A    | C    | D    | G    | H    | I    | M    | N    | O    | P    | R    | T    | U    | W    |

这种直方图简单易懂，但是在计算机科学领域频繁地出现。它被广泛地应用于图像处理、数据压缩、计算机视觉、机器学习、音频编码等诸多算法中。

### 1.1 CPU直方图计算

```c++
#include "../common/book.h"

#define SIZE (100*1024*1024)//规定数据的规模，100MB

int main(void){
    //初始化一个随机字符流
    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);
    //初始化存放历史的直方图数组
    unsigned int histo[256];
    //循环将直方图构建
    for (int i = 0; i < 256; i++) histo[i] = 0;
    for (int i = 0; i < SIZE; i++) histo[buffer[i]] ++;
    //计算历史数组中的总和
    long histoCount = 0;
    for (int i = 0; i < 256; i++) histoCount += histo[i];
    printf( "Histogram Sum: %ld\n", histoCount );
    //释放内存
    free (buffer);
    return 0;
}
```

### 1.2 GPU计算直方图

在GPU上计算直方图，让不同的线程分别处理缓冲区的不同部分可以节省大量时间，但问题在于，多个不同的线程可能想在同一时间递增输出直方图的同一个bin。这就需要原子递增来避免冲突。总之我们先来关注main函数部分。

```c++
int main(void){
    unsigned char *buffer = (unsigned char*)big_random_block(SIZE);
    ...
```

```c++
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
```

//to be filled

### 完整代码

```c++
#include "../../common/book.h"

#define SIZE (100 * 1024 * 1024)

__global__ void histo_kernel(unsigned char *buffer, long size, unsigned int *histo){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < size){
        //将histo数组中的对应位置加1，同时原子操作保证了硬件在执行这些操作期间没有任何其他线程可以对当前地址的值进行读写
        atomicAdd(&(histo[buffer[i]]), 1);
        i += stride;
    }

    // __shared__ unsigned int temp[256];
    // temp[threadIdx.x] = 0;
    // __syncthreads();
    // int i = threadIdx.x + blockIdx.x * blockDim.x;
    // int stride = blockDim.x * gridDim.x;
    // while (i < size){
    //     atomicAdd(&temp[buffer[i]], 1);
    //     i += stride;
    // }

    // __syncthreads();
    // atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);

}

int main(void){
    unsigned char *buffer = (unsigned char*) big_random_block(SIZE);

    //衡量代码性能
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    //设置好需要输入的数据和事件后，转头看向GPU内存操作
    
    //分配文件数据在gpu上的内存空间
    //一个存储字节流，一个存储计数流
    unsigned char *dev_buffer;
    unsigned int *dev_histo;

    cudaMalloc((void**)&dev_buffer, SIZE);
    cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_histo, 256 * sizeof(int));
    cudaMemset(dev_histo, 0, 256 * sizeof(int));

    //因为数组大小是256，所以每个块中恰好有256个线程是最好的
    //且据验证，当启动的块数恰好是GPU多处理器数量的两倍时，性能达到最优
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount;

    //需要的三个参数：指向输入数组的指针，数据直方图的长度，以及指向输出数组的指针
    histo_kernel<<<blocks * 2, 256>>>(dev_buffer, SIZE, dev_histo);

    unsigned int histo[256];
    cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    //获取停止时间，并显示计时结果
    cudaEventRecord(stop, 0);
    cudaEventSynchronize( stop );

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Time to generate: %3.1f ms\n", elapsedTime);

    long histoCount = 0;
    for (int i=0; i<256; i++){
        histoCount += histo[i];
    }

    printf("Histogram Sum: %ld\n", histoCount);

    //在CPU上进行反向验证
    for (int i = 0; i < SIZE; i++){
        //buffer 是大随机数组，从头向尾遍历，出现的元素是几，在histo这个计数数组中对应的数字的次数就减一
        histo[buffer[i]] --;
    }//最后计算之后如果histo数组变为全0就证明CPU&GPU计算的结果是一样的

    for (int i = 0; i < 256; i++){
        if (histo[i] != 0){
            printf("Failure at %d!\n", i);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree( dev_histo );
    cudaFree( dev_buffer );
    free(buffer);

    return 0;
}
```

