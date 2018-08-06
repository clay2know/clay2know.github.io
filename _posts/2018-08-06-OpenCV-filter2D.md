---
layout:     post
title:      OpenCV 源码
subtitle:   OpenCV-filter2D（一）
date:       2018-08-05
author:     DJ
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - OpenCV
    - filter2D
    - 源码阅读
---
# 手撕OpenCV源码之filter2D(一)
在上篇的GaussianBlur中提到,gaussianBlur使用的是filter2D的实现,因此上篇仅仅描述了高斯滤波器的
生成细节,并没有针对滤波的计算细节及代码实现进行分析.本篇将详细介绍OpenCV中滤波的实现细节.
# filter2D函数的整体结构分析    
上篇文章中没有提到,本系列分析源码所使用的opencv版本是opencv-3.4.0.       
首先从OpenCV的主函数入手.     

```
void cv::filter2D( InputArray _src, OutputArray _dst, int ddepth,
                   InputArray _kernel, Point anchor0,
                   double delta, int borderType )         

```

* 路径
`opencv-3.4.0/modules/imgproc/filter.cpp`         

* 函数原型      

```          
void cv::filter2D( InputArray _src, OutputArray _dst, int ddepth,
                   InputArray _kernel, Point anchor0,
                   double delta, int borderType )
{
  //opencv的profiling,可以在内部追踪函数执行状况,默认情况下是关闭的,不会产生性能开销.
    CV_INSTRUMENT_REGION()
//若平台支持OpenCL则使用OpenCL执行
    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
               ocl_filter2D(_src, _dst, ddepth, _kernel, anchor0, delta, borderType))
//将传入的参数转换为Mat结构
    Mat src = _src.getMat(), kernel = _kernel.getMat();
//在api说明中知道调用cv::filter2D是允许将ddepth设置小于0的
    if( ddepth < 0 )
        ddepth = src.depth();
//为函数输出创建Mat结构
    _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
    Mat dst = _dst.getMat();
    Point anchor = normalizeAnchor(anchor0, kernel.size());
//计算src在进行了边界填充后的数组中的位置
    Point ofs;
    Size wsz(src.cols, src.rows);
    if( (borderType & BORDER_ISOLATED) == 0 )
        src.locateROI( wsz, ofs );

//调用hal命名空间下的filter2D函数
    hal::filter2D(src.type(), dst.type(), kernel.type(),
                  src.data, src.step, dst.data, dst.step,
                  dst.cols, dst.rows, wsz.width, wsz.height, ofs.x, ofs.y,
                  kernel.data, kernel.step,  kernel.cols, kernel.rows,
                  anchor.x, anchor.y,
                  delta, borderType, src.isSubmatrix());
}  
```     
接下来对代码注释中涉及的一些内容进行解释:     
* 关于OpenCL               
在OpenCV的早起版本中其实就已经有OpenCL的优化了(比如2.4.8);当时版本中需要用户自己指定。
在当前的opencv-3.4.0中则把OpenCL版本的实现做了改进;并且会优先选择使用OpenCL。`CV_OCL_RUN`就是对OpenCL的检测，若OpenCL可用则执行OpenCL版本的filter2D。
可以看一下`CV_OCL_RUN`这个宏           
路径：`opencv/core/include/opencv2/core/opencl/ocl_defs.hpp`       

```
#define CV_OCL_RUN_(condition, func, ...)                                   \
    {                                                                       \
        if (cv::ocl::isOpenCLActivated() && (condition) && func)            \
        {                                                                   \
            printf("%s: OpenCL implementation is running\n", CV_Func);      \
            fflush(stdout);                                                 \
            CV_IMPL_ADD(CV_IMPL_OCL);                                       \
            return __VA_ARGS__;                                             \
        }                                                                   \
        else                                                                \
        {                                                                   \
            printf("%s: Plain implementation is running\n", CV_Func);       \
            fflush(stdout);                                                 \
        }                                                                   \
    }

```         
可以看出，首先检测了opencl平台的可用性；之后添加OCL实现（`CV_IMPL_ADD(CV_IMPL_OCL)`）;接下来简单了解一下opencv目前的一些优化平台。

路径：`opencv/modules/modules/core/include/opencv2/core/utility.hpp`

```            
   #define CV_IMPL_PLAIN  0x01 // native CPU OpenCV implementation
   #define CV_IMPL_OCL    0x02 // OpenCL implementation
   #define CV_IMPL_IPP    0x04 // IPP implementation
   #define CV_IMPL_MT     0x10 // multithreaded implementation

   #define CV_IMPL_ADD(impl)                                                   \
       if(cv::useCollection())                                                 \
       {                                                                       \
           cv::addImpl(impl, CV_Func);                                         \
       }
   #else
   #define CV_IMPL_ADD(impl)
   #endif
```           
从这里可以看出，目前opencv主要的优化是，cpu，opencl，IPP库，以及多线程优化。另外opencv是至此cuda的，只是cuda是单独的模块。其他几种优化是作为通用模块实现的。
从cv::filter2D代码的组织形式看，opencv是优先使用opencl的，可以推测，opencv中opencl的实现性能是比较优秀的。         

* 接下来的处理再注释中写的比较明确了，不再赘述。           
* hal::filter2D             
先上源码：              

```         
void filter2D(int stype, int dtype, int kernel_type,
              uchar * src_data, size_t src_step,
              uchar * dst_data, size_t dst_step,
              int width, int height,
              int full_width, int full_height,
              int offset_x, int offset_y,
              uchar * kernel_data, size_t kernel_step,
              int kernel_width, int kernel_height,
              int anchor_x, int anchor_y,
              double delta, int borderType,
              bool isSubmatrix)
{
    bool res;
    res = replacementFilter2D(stype, dtype, kernel_type,
                              src_data, src_step,
                              dst_data, dst_step,
                              width, height,
                              full_width, full_height,
                              offset_x, offset_y,
                              kernel_data, kernel_step,
                              kernel_width, kernel_height,
                              anchor_x, anchor_y,
                              delta, borderType, isSubmatrix);
    if (res)
        return;
        CV_IPP_RUN_FAST(ippFilter2D(stype, dtype, kernel_type,
                                     src_data, src_step,
                                     dst_data, dst_step,
                                     width, height,
                                     full_width, full_height,
                                     offset_x, offset_y,
                                     kernel_data, kernel_step,
                                     kernel_width, kernel_height,
                                     anchor_x, anchor_y,
                                     delta, borderType, isSubmatrix))

           res = dftFilter2D(stype, dtype, kernel_type,
                             src_data, src_step,
                             dst_data, dst_step,
                             full_width, full_height,
                             offset_x, offset_y,
                             kernel_data, kernel_step,
                             kernel_width, kernel_height,
                             anchor_x, anchor_y,
                             delta, borderType);
           if (res)
               return;
               ocvFilter2D(stype, dtype, kernel_type,
                              src_data, src_step,
                              dst_data, dst_step,
                              width, height,
                              full_width, full_height,
                              offset_x, offset_y,
                              kernel_data, kernel_step,
                              kernel_width, kernel_height,
                              anchor_x, anchor_y,
                              delta, borderType);
}             
```            
这段代码很简单，主要是各个平台计算法的执行顺序；首先是replacementFilter2D，这其实是可分离滤波器，从名字也可以看出来，是替代filter2D；上篇文章中提到的GaussianBlur就是可分离滤波器。满足这种性质的滤波器可以实现更好的计算性能，所以先检测，是否满足可分离特性。关于可分离滤波器会再后续的文章中详细说明，暂且不表。第二是调用IPP库，这是Inter的一个计算库；第三是使用dft进行滤波，再opencv中滤波器尺寸大于11的，会使用dft进行滤波，对于大尺寸滤波器，dft会获得更好的性能。在dftFilter2D函数内部会判断滤波器尺寸；最后是ocvFilter2D，这是使用c++实现的二维滤波。
dftFilter2D函数内部判断：

```
int dft_filter_size = 50;//不使用SSE的情况下       
if (kernel_width * kernel_height < dft_filter_size)
           return false;
```
到此为止，filter2D API中的函数结构介绍结束，目前本系列文章中着重介绍c++实现，也就是ocvFilter2D；在后续的文章中也会介绍dft的c++实现；对于各中优化版本的实现，不做深究。以后的别的系列可能会介绍opencl版本的优化。           
下一篇文章中将详细介绍ocvFilter2D的实现。           
