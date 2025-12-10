---
title: cuda dot product 代码理解
date: 2025-12-10
category: 学习
tags: [CUDA，c++]
---



# CUDA 中的向量点积(常规)

***

## 1. 关键代码分步理解

```c++
#define imin(a,b) (a<b?a:b)
```

简单的三元表达式，取二者间较小者

```c++
const int threadsPerBlock = 256;
```

规定block大小，一个block里面跑256个线程，256 = 332 * 8，是最常用的block大小之一。效率高，硬件友好，通用性强。

```c++
__global__ void dot (float *a, float *b, float *c){

    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N){
        temp += a[tid] *b[tid];
        tid += blockDim.x * gridDim.x;
    }

    ...
}
```

`shared`为共享内存，所有的线程都可以将临时结果存储进来。编译器会为每个代码块创建共享变量的副本。对于块与块之间，这个共享内存是互不影响的，每个块中的线程都只会操作自己块对应的缓冲区（见下例）。所以要为共享内存分配足够的内存大小，让代码块中的每个线程都有一个条目。

例：

假设 launch kernel 时：

```C++
<<<4 blocks, 256 threads>>>
```

然后 kernel 里写了：

```c++
__shared__ float temp[256];
```

那么 GPU 实际上会分配 4 份 `temp`：

- block 0 有自己的 temp[256]
- block 1 有自己的 temp[256]
- block 2 有自己的 temp[256]
- block 3 有自己的 temp[256]

不同 block 的 temp 之间完全隔离，互不干扰。

`while` 循环中，每经过一次循环，这个线程就要跳到下一个它该处理的位置。这里是让每个线程每隔`blockDim.x * gridDim.x`取一个任务，这样能保证工作均摊，不重复。因为数据量可以很庞大，但是GPU的线程数是有限的，所以每个线程要跳过所有线程一次迭代的工作量才能遍历完整个数组。此为网格跨步循环grid-stride loop。

```c++
    //set cache values
    cache[cacheIndex] = temp;
```

使用共享内存的对应区域存放线程计算的点积对应部分，点积就是向量对应元素相乘的累加，这里放的是一个块的结果。

```c++
{
    ...
	//synchronize threads 使线程在此同步
    __syncthreads();
	...
}
```

`__syncthreads()`确保在硬件执行下一个指令之前，代码块中的每个线程都已经完成所有指令的执行。确保所有对共享数组的写入操作在任何线程尝试从该缓冲区读取数据之前都已经完成。

```c++
{
    ...
        //因为以下代码，每块线程数必须是2的某次幂，对于规约来说
    int i = blockDim.x/2;
    while (i != 0){
        if (cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }
    ...
}
```

求和简化，使用规约操作。每个线程都会获取cache[]中与其索引对应的条目，将其与偏移i处的条目相加，并将结果存储回cache[]。

```c++
{
    ...
    if (cacheIndex == 0)
    c[blockIdx.x] = cache[0];
    ...
}
```

在`while`循环结束后，每个块中的所有线程的运算结果会被规约进当前块的共享内存中的第0个元素中。这里借用`cacheIndex==0`的所有线程执行全局存储，实际上可以是任意一个，将所在块中规约出的结果存放进输出数组c的对应位置上。

当然，向量点积还差最后一步，那就是把输出数组中的元素求和。这个求和运算对于GPU来说有些浪费资源，因为数组规模太小，继续放在GPU上容易造成资源浪费，所以我们把这一步放回CPU中进行操作。

## 2. 完整代码

完整代码如下：

```c++
#include "../../common/book.h"

#define imin(a, b) (a<b?a:b)

const int N = 33 * 1024;//要处理的数据规模，即需要的线程数
const int threadsPerBlock = 256; //这是CUDA最常用的block大小之一
//256 = 32 * 8， 效率高，硬件友好，通用性强
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1) / threadsPerBlock);
//这里其实和之前一样在计算分配多少个块，但之前是老GPU，内存比较小

__global__ void dot (float *a, float *b, float *c){

    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N){
        temp += a[tid] *b[tid];
        tid += blockDim.x * gridDim.x;
    }

    //set cache values
    cache[cacheIndex] = temp;

    //synchronize threads 使线程在此同步
    __syncthreads();

    //因为以下代码，每块线程数必须是2的某次幂，对于规约来说
    int i = blockDim.x/2;
    while (i != 0){
        if (cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

int main(void){
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    //分配内存(CPU)
    a = (float*)malloc( N*sizeof(float) );
    b = (float*)malloc( N*sizeof(float) );

    //c用来存放各个块中规约出的最后一个数
    partial_c = (float*)malloc( blocksPerGrid*sizeof(float) );
    
    //分配内存(GPU)
    cudaMalloc( (void**)&dev_a, N * sizeof(float) );
    cudaMalloc( (void**)&dev_b, N * sizeof(float) );
    cudaMalloc( (void**)&dev_partial_c, N * sizeof(float) );

    //在主机上填入数据
    for (int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    cudaMemcpy( dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice );

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    cudaMemcpy( partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost );

    c = 0;
    for (int i = 0; i < blocksPerGrid; i++){
        c += partial_c[i];
    }

    #define sum_squares(x) (x*(x+1)*(2*x+1)/6)
    printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares( (float)(N - 1) ));

    cudaFree( dev_a );
    cudaFree( dev_b );
    cudaFree( dev_partial_c ); 

    free( a );
    free( b );
    free( partial_c );

    return 0;
}
```

