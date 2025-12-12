---
title: CUDA#2 shared memory bitmap
date: 2025-12-11
category: 学习
tags: [CUDA, c++, bitmap, 线程同步]
---

# CUDA shared memory bitmap

***

## 1. 关键代码步骤分解

图像维度，1024*1024；

PI的值，用来计算两条波形(三角函数)

```c++
#define DIM 1024 //图像维度
#define PI 3.1415926535897932f
```

**先来看main函数：**

```C++
    CPUBitmap bitmap (DIM, DIM);
    unsigned char *dev_bitmap;
```

首先根据define好的维度初始化一个bitmap对象，是什么对象不要紧，要做什么也不要紧，本篇笔记重要的是记录多维的线程、块并行，以及处理数据的概念。

在这里这个指针指向了这个bitmap，里面的元素类型式unsigned char，用来存放像素。一个像素单位在内存中占4个字节，后面索引会再次提到内存中像素的索引方式。

```c++
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
```

首先按照目标的图像大小为这个bitmap在GPU内存中分配空间。不用管`HANDLE_ERROR`,这个是书中为了帮助初学者快速定位错误原因的辅助函数，正常用的时候没有。

```c++
    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16, 16);
```

在本例中，网格的维度是64 * 64，块的维度是16 * 16，这样刚好实现1024 * 1024个线程，覆盖图像中的每一个像素。

```c++
    kernel<<<grids, threads>>>(dev_bitmap);
```

然后调用kernel函数并依照变量规定本次执行的规模；kernel函数在后面分析，这个kernel函数的结果是一个位图，也就是bitmap，来看后续操作。

```c++
    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
```

将生成的位图拷贝到CPU中，host中的bitmap对象获取一个指针用来装拷贝回来的地址。随后就是简单的显示图像和释放内存。

```c++
    bitmap.display_and_exit();
    cudaFree( dev_bitmap );
```

***

**下面来看kernel函数`__global__ void kernel(unsigned char *ptr)`**：

```c++
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
```

这里是初始化当先线程的横纵坐标，因为执行完当前的，下一次再执行就是最大维度执行完之后的一个序号。

```c++
    int offset = x + y * blockDim.x * gridDim.x;
```

这里CUDA将二位线程索引映射程一维索引`offset`。这里采用的是行优先展开，`y * blockDim.x * gridDim.x`是已经执行的行数*每行中的线程数，也就是已经执行的行数 * 总宽度。

接下来是缓冲区的设置，缓冲区作为所有当前块中的线程都能触及到的内存区，应当与块中线程总数的维度是一致的，在这里设置成16*16。

```c++
    __shared__ float shared[16][16];
```

决定周期的变量`period`以及控制计算的偏移量。

```c++
    //本个像素（线程）的值,图像尺寸是1024*1024， 选用128正好能把画面切成8份周期
    const float period = 128.0f; //使用f作后缀防止隐式转换，不加f是double
```

```c++
//+1.0和把范围从[-1,1]映射到[0,2],因为亮度不能为负数，最后除以4.0把扩张后的范围移回来
    //最后存放到该线程对应的缓冲区
    shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x*2.0f*PI/ period) + 1.0f) *(sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;
```

最重要的是线程同步确认函数，必须要所有线程都执行完毕后再开始进行内存读取。

```c++
    //不能删掉，删掉了会直接坏掉、花屏，一部分线程已经开始执行ptr，一部分还在计算亮度值
    __syncthreads();
```

这里翻转存储图像，就是为了体现出同步的作用，跨线程依赖必须要有同步点。在内存中每个像素占4个字节，所以要索引到对应像素在内存中的位置就要用原索引值*4。这4个字节对应的是RGBA，红绿蓝透明度。

```c++
    ptr[offset*4 + 0] = 0; //红色
    ptr[offset*4 + 1] = shared[15-threadIdx.x][15-threadIdx.y]; //绿色
    ptr[offset*4 + 2] = 0; //蓝色
    ptr[offset*4 + 3] = 255; //透明度为全不透明
```

## 2. 完整代码

```c++
#include "cuda.h"
#include "../../common/book.h"
#include "../../common/cpu_bitmap.h"

#define DIM 1024 //图像维度
#define PI 3.1415926535897932f


__global__ void kernel(unsigned char *ptr){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    //这里是cuda里把二维线程索引映射成一维索引offset
    int offset = x + y * blockDim.x * gridDim.x;
    
    //缓冲区的大小应该与每个块中的线程总数一致，所以在这里是16*16
    __shared__ float shared[16][16];

    //本个像素（线程）的值,图像尺寸是1024*1024， 选用128正好能把画面切成8份周期
    const float period = 128.0f; //使用f作后缀防止隐式转换，不加f是double

    //+1.0和把范围从[-1,1]映射到[0,2],因为亮度不能为负数，最后除以4.0把扩张后的范围移回来
    //最后存放到该线程对应的缓冲区
    shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x*2.0f*PI/ period) + 1.0f) *(sinf(y*2.0f*PI/ period) + 1.0f) / 4.0f;

    //不能删掉，删掉了会直接坏掉、花屏，一部分线程已经开始执行ptr，一部分还在计算亮度值
    __syncthreads();

    //这里翻转存储图像，就是为了体现出同步的作用，跨线程依赖必须要有同步点
    //在内存中每个像素占4个字节，所以要索引到对应像素在内存中的位置就要用原索引值*4
    //这4个字节对应的是RGBA，红绿蓝透明度
    ptr[offset*4 + 0] = 0; //红色
    ptr[offset*4 + 1] = shared[15-threadIdx.x][15-threadIdx.y]; //绿色
    ptr[offset*4 + 2] = 0; //蓝色
    ptr[offset*4 + 3] = 255; //透明度为全不透明

}


int main(void){
    CPUBitmap bitmap (DIM, DIM);
    unsigned char *dev_bitmap;

    //GPU开辟显存放位图
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

    //grid结构是64*64个block， block结构是16*16个线程
    //所以一共是1024*1024个线程，刚好覆盖图像的每一个像素
    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(dev_bitmap);

    //把结果拷回CPU
    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
    
    //显示图像
    bitmap.display_and_exit();

    //释放内存
    cudaFree( dev_bitmap );

}
```

output:

![output](https://cdn.jsdelivr.net/gh/kiu795/pic@main/img/output.png)
