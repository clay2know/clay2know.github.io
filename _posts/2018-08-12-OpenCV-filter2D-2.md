---
layout:     post
title:      OpenCV 源码
subtitle:   OpenCV-filter2D（二）
date:       2018-08-05
author:     DJ
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - OpenCV源码
    - filter2D
---
# 手撕OpenCV源码之filter2D(二)
## cv::filter2D                 
前文对这个函数的分析是为了了解filter的实现结构，所以比较粗略，本文将更细致的分析opencv中filter2D的c++实现的细节，不涉及各种加速的实现方式。
首先还是看函数原型：              
```                
4894 void cv::filter2D( InputArray _src, OutputArray _dst, int ddepth,
4895                    InputArray _kernel, Point anchor0,
4896                    double delta, int borderType )
4897 {
4898     CV_INSTRUMENT_REGION()
4899
4900     CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2,
4901                ocl_filter2D(_src, _dst, ddepth, _kernel, anchor0, delta, borderType))
4902
4903     Mat src = _src.getMat(), kernel = _kernel.getMat();
4904
4905     if( ddepth < 0 )
4906         ddepth = src.depth();
4907
4908     _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
4909     Mat dst = _dst.getMat();
4910     Point anchor = normalizeAnchor(anchor0, kernel.size());
4911
4912     Point ofs;
4913     Size wsz(src.cols, src.rows);
4914     if( (borderType & BORDER_ISOLATED) == 0 )
4915         src.locateROI( wsz, ofs );
4916
4917     hal::filter2D(src.type(), dst.type(), kernel.type(),
4918                   src.data, src.step, dst.data, dst.step,
4919                   dst.cols, dst.rows, wsz.width, wsz.height, ofs.x, ofs.y,
4920                   kernel.data, kernel.step,  kernel.cols, kernel.rows,
4921                   anchor.x, anchor.y,
4922                   delta, borderType, src.isSubmatrix());
4923 }

```          
### 输入参数介绍：             
- src/dst/kernel    
这三个参数的含义很容易理解，分别是输入，滤波器和输出；他们的数据类型分别为InputArray和OutputArray;在opencv中有很多函数的输入和输出是这样的数据类型。他支持Mat，Vector， UMat，以及CUDA的GPU_MAT,HOST_MEM等等，非常强大；而OutputArray是继承InputArray的，InputArray作为输入参数，是有const限定的，防止参数被修改；而OutputArray中是没有的。
- anchor0     
这个参数是指滤波器的锚点位置，不理解的同学可以看后文的详细介绍。             
- delta    
这个参数很简单，就是在滤波结果上加上这个值。一般都是0     
- borderType         
这个参数是边界填充的类型，在滤波的过程中，会根据滤波器的尺寸在图像的边界填充一定的数量的像素值，以保证输入与输出具有相同的尺寸，这个参数指定边界填充的规则;目前支持一下几种规则：`opencv/modules/core/include/opencv2/core/base.hpp`              
```           
268 enum BorderTypes {
269     BORDER_CONSTANT    = 0, //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
270     BORDER_REPLICATE   = 1, //!< `aaaaaa|abcdefgh|hhhhhhh`
271     BORDER_REFLECT     = 2, //!< `fedcba|abcdefgh|hgfedcb`
272     BORDER_WRAP        = 3, //!< `cdefgh|abcdefgh|abcdefg`
273     BORDER_REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`
274     BORDER_TRANSPARENT = 5, //!< `uvwxyz|abcdefgh|ijklmno`
275
276     BORDER_REFLECT101  = BORDER_REFLECT_101, //!< same as BORDER_REFLECT_101
277     BORDER_DEFAULT     = BORDER_REFLECT_101, //!< same as BORDER_REFLECT_101
278     BORDER_ISOLATED    = 16 //!< do not look outside of ROI
279 };
```                
注释描述的很形象，这里就不赘述了。        
- ddepth        
输入图像的深度，也就是输入图像的类型，目前主要支持一下几种：                
```          
src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
src.depth() = CV_64F, ddepth = -1/CV_64F
```           
### 函数剖析             
关于函数的整体结构在之前的博文中都有介绍，这里主要是从一些函数语句入手，细致分析函数的实现。        
#### 获取输入数据         
获取输入数据主要涉及一下几行代码：             
```       
4903 Mat src = _src.getMat(), kernel = _kernel.getMat();
4908 _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
4909 Mat dst = _dst.getMat();
```          
这三个参数的数据类型都是InputArray/OutputArray是代理数据类型；因此需要转换为Mat或者vector等具体数据类型；接下来看getMat函数。
##### getMat函数实现       
* 函数定义(`opencv/modules/core/include/opencv2/core/mat.hpp`)：               
```
Mat getMat(int idx=-1) const;
```       
* 函数实现(`opencv/modules/core/include/opencv2/core
/mat.inl.hpp`)：          
```            
 145 inline Mat _InputArray::getMat(int i) const
 146 {
 147     if( kind() == MAT && i < 0 )
 148         return *(const Mat*)obj;
 149     return getMat_(i);
 150 }
```                  

很明显了，filter2D在调用的时候，使用了默认参数`i=-1`，也就是说默认输入是Mat的类型.所以直接返回const Mat类型；ofs是InputArray类中定义的`void*`类型。 但是从判断条件看，还有一个`kind() == MAT`。                 
我们首先看MAT的定义(这是InputArray类中的一个枚举类型)：                      

```               
155     enum {
 156         KIND_SHIFT = 16,
 157         FIXED_TYPE = 0x8000 << KIND_SHIFT,
 158         FIXED_SIZE = 0x4000 << KIND_SHIFT,
 159         KIND_MASK = 31 << KIND_SHIFT,
 160
 161         NONE              = 0 << KIND_SHIFT,
 162         MAT               = 1 << KIND_SHIFT,
 163         MATX              = 2 << KIND_SHIFT,
 164         STD_VECTOR        = 3 << KIND_SHIFT,
 165         STD_VECTOR_VECTOR = 4 << KIND_SHIFT,
 166         STD_VECTOR_MAT    = 5 << KIND_SHIFT,
 167         EXPR              = 6 << KIND_SHIFT,
 168         OPENGL_BUFFER     = 7 << KIND_SHIFT,
 169         CUDA_HOST_MEM     = 8 << KIND_SHIFT,
 170         CUDA_GPU_MAT      = 9 << KIND_SHIFT,
 171         UMAT              =10 << KIND_SHIFT,
 172         STD_VECTOR_UMAT   =11 << KIND_SHIFT,
 173         STD_BOOL_VECTOR   =12 << KIND_SHIFT,
 174         STD_VECTOR_CUDA_GPU_MAT = 13 << KIND_SHIFT,
 175         STD_ARRAY         =14 << KIND_SHIFT,
```             

可以看出，这里定义了所有InputArray支持的类型；
接着看kind()函数，该函数也定义在InputArray类中，官方的注释是这样的：                
```   
kind() can be used to distinguish Mat from
       `vector<>` etc., but normally it is not needed.
```               

通过注释能知道kind()就是用来判断InputArray的具体类型的；              
它的实现是这样的(`opencv/modules/core/src/matrix_wrap.cpp`)：                     

```              
 379 int _InputArray::kind() const
 380 {
 381     return flags & KIND_MASK;
 382 }
 ```
这是一个按位与，KIND_MASK在前文的枚举类型中定义的（31 << 16）; 而flags则是在InputArray类中定义的一个参数，在InputArray类中有一个构造函数：            
```           
inline _InputArray::_InputArray(const Mat& m) { init(MAT+ACCESS_READ, &m); }
```               
init函数也是在InputArray类中的：              
```          
inline void _InputArray::init(int _flags, const void* _obj)
   { flags = _flags; obj = (void*)_obj; }
```              
到此可以知道，`flags = MAT+ACCESS_READ`;从变量名可以知道，flags标识的意义是Mat数据类型，读写特性为只读；
其中，ACCESS_READ是namespace cv中的一个全局的枚举类型，定义如下：              
```
enum { ACCESS_READ=1<<24, ACCESS_WRITE=1<<25,
       ACCESS_RW=3<<24, ACCESS_MASK=ACCESS_RW, ACCESS_FAST=1<<26 };
```             
我们在这里做一个简单的计算：           
```        
         MAT : 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  ACCESS_READ: 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
ADD(READ,MAT): 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
KIND_MASK:     0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
MASK & flags:  0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```            
从计算可以知道，kind()函数中相与的结果是MAT；此时再看两个枚举类型的定义，可以知道，再OpenCV中，使用低20位标识数据类型(Mat,UMat,vertor等)，24，25，26位标识数据的读写特性。            
到这里为止相信大家已经明白了，在InputArray类中，关于getMat的整个实现流程。在这里给大家做一个总结，首先在我们需要采用具体的数据类型来初始化InputArray对象，这个时候，在InputArray对象的内部，初始化了两个参数，flags和obj；flags被初始化为枚举类型中的类型标识，而obj则被初始化化为具体类型，存储数据。此时当我们下次调用getMat获得具体类型的时候，直接调用getMat()函数，使用默认的函数，可以直接返回Mat类型。同样，如果以其他类型初始化InputArray对象，也会以相同的流程执行。            
那么，getMat的输入参数i又具有什么实质性的意义呢？
继续深究，当`i>=0`的时候，函数会调用getMat_()函数，这个函数的实现如下：                
##### getMat_的实现 (`opencv/modules/core/src/matrix_wrap.cpp`)：                 

```
  15 Mat _InputArray::getMat_(int i) const
  16 {
  17     int k = kind();
  18     int accessFlags = flags & ACCESS_MASK;
  19
  20     if( k == MAT )
  21     {   
  22         const Mat* m = (const Mat*)obj;
  23         if( i < 0 )
  24             return *m;
  25         return m->row(i);
  26     }
  27
  28     if( k == UMAT )
  29     {
  30         const UMat* m = (const UMat*)obj;
  31         if( i < 0 )
  32             return m->getMat(accessFlags);
  33         return m->getMat(accessFlags).row(i);
  34     }
         ...
      }
```         
这个函数可以返回指定的行数据，也就是行的起始位置。          
到此为止，关于输入数据的类型解析结束，关于InputArray/OutputArray/Mat等，都有相应的类来实现，也相当强大，以后会写博客专门介绍。             

#### 滤波器锚点介绍(normalizeAnchor)      
关于锚点的含义很容易理解，通俗讲就是滤波计算得到的结果，替换原图像中什么位置的点，大多是情况下，我们替换的是中心点，例如3×3的kernel，我们替换的是(1,1)位置的点(左上角为原点)。这里主要介绍`normalizeAnchor`这个函数，函数实现如下(`opencv/modules/imgproc/src/filterengine.hpp`)：                 
```      
357 static inline Point normalizeAnchor( Point anchor, Size ksize )
358 {
359    if( anchor.x == -1 )
360        anchor.x = ksize.width/2;
361    if( anchor.y == -1 )
362        anchor.y = ksize.height/2;
363    CV_Assert( anchor.inside(Rect(0, 0, ksize.width, ksize.height)) );
364    return anchor;
365 }
```
可以看到，当输入为(-1,-1)的时候会默认为中心点。inside函数的定义如下( `opencv/modules/core/include/opencv2/core/types.hpp`)：     

```
//! checks whether the point is inside the specified rectangle
 183     bool inside(const Rect_<_Tp>& r) const;

```                         
 该函数是用于判断锚点是否超出越界。                    
 
#### 边界设定                
 
 这一部分主要涉及这两条语句：               
 ```  
 if( (borderType & BORDER_ISOLATED) == 0 )
 4915         src.locateROI( wsz, ofs );
 ```              
 关于边界填充类型的枚举前文已经介绍了，大家可以翻到前面看，容易知道，不需要边界填充的时候，相与结果是1，否则相与结果都是0，也就是说，大多数情况下，需要执行locateROI函数；                     
 
## ocvFilter2D                      

 接下来函数调用了hal::filter2D函数，关于这个函数在前一篇博文中介绍过：   
 ```
 4917     hal::filter2D(src.type(), dst.type(), kernel.type(),
 4918                   src.data, src.step, dst.data, dst.step,
 4919                   dst.cols, dst.rows, wsz.width, wsz.height, ofs.x, ofs.y,
 4920                   kernel.data, kernel.step,  kernel.cols, kernel.rows,
 4921                   anchor.x, anchor.y,
 4922                   delta, borderType, src.isSubmatrix());
 ```                
 接下来直接看不进行硬件加速的版本，
 ```
 4714 static void ocvFilter2D(int stype, int dtype, int kernel_type,
4715                         uchar * src_data, size_t src_step,
4716                         uchar * dst_data, size_t dst_step,
4717                         int width, int height,
4718                         int full_width, int full_height,
4719                         int offset_x, int offset_y,
4720                         uchar * kernel_data, size_t kernel_step,
4721                         int kernel_width, int kernel_height,
4722                         int anchor_x, int anchor_y,
4723                         double delta, int borderType)
4724 {
4725     int borderTypeValue = borderType & ~BORDER_ISOLATED;
4726     Mat kernel = Mat(Size(kernel_width, kernel_height), kernel_type, kernel_data, kernel_step);
4727     Ptr<FilterEngine> f = createLinearFilter(stype, dtype, kernel, Point(anchor_x, anchor_y), delt     a,
4728                                              borderTypeValue);
4729     Mat src(Size(width, height), stype, src_data, src_step);
4730     Mat dst(Size(width, height), dtype, dst_data, dst_step);
4731     f->apply(src, dst, Size(full_width, full_height), Point(offset_x, offset_y));
4732 }
 ```                   
 这个函数很简单，做了3件事，第一是申请Mat，为kernel，src和dst分别申请了Mat；第二是初始化了一个线性滤波器，接触过OpenCV的同学可能知道，OpenCV中关于滤波，有一个FilterEngine类。
 本文就介绍到此，接下来关于createLinearFilter都属于FilterEngine中的内容，在下篇博文中继续剖析。
# 欢迎指正
# 作者：2know
# 个人网站：2know.top

