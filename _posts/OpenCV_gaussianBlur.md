# 手撕OpenCV源码之gaussiablur
## GaussianBlur API解析           
首先看源码:              
```       
void cv::GaussianBlur( InputArray _src, OutputArray _dst, Size ksize,
                   double sigma1, double sigma2,
                   int borderType )
{
  //初始化及边界类型等的判断
    CV_INSTRUMENT_REGION()

    int type = _src.type();
    Size size = _src.size();
    _dst.create( size, type );

    if( borderType != BORDER_CONSTANT && (borderType & BORDER_ISOLATED) != 0 )
    {
        if( size.height == 1 )
            ksize.height = 1;
        if( size.width == 1 )
            ksize.width = 1;
    }
//容易理解,高斯滤波器如果尺寸为1,根据高斯函数可以知道该系数为1,所以就是将输入复制到输出
    if( ksize.width == 1 && ksize.height == 1 )
    {
        _src.copyTo(_dst);
        return;
    }
//OpenCV中针对一些ksize = 3和5的情况做了OpenCL优化,所以初始化OpenCL相关函数
    bool useOpenCL = (ocl::isOpenCLActivated() && _dst.isUMat() && _src.dims() <= 2 &&
               ((ksize.width == 3 && ksize.height == 3) ||
               (ksize.width == 5 && ksize.height == 5)) &&
               _src.rows() > ksize.height && _src.cols() > ksize.width);
    (void)useOpenCL;

    int sdepth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
//获取gaussianKernels
    Mat kx, ky;
    createGaussianKernels(kx, ky, type, ksize, sigma1, sigma2);
//调用opencl进行计算
    CV_OCL_RUN(useOpenCL, ocl_GaussianBlur_8UC1(_src, _dst, ksize, CV_MAT_DEPTH(type), kx, ky, borderType));
//如果不是ksize=3或者5的情况,考虑使用filter2D的opencl优化程序计算
    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2 && (size_t)_src.rows() > kx.total() && (size_t)_src.cols() > kx.total(),
               ocl_sepFilter2D(_src, _dst, sdepth, kx, ky, Point(-1, -1), 0, borderType))
//如果OpenCL版的filter2D依然不能计算,则选择cpu版本的gaussianBlur
    Mat src = _src.getMat();
    Mat dst = _dst.getMat();

    Point ofs;
    Size wsz(src.cols, src.rows);
    if(!(borderType & BORDER_ISOLATED))
        src.locateROI( wsz, ofs );

    CALL_HAL(gaussianBlur, cv_hal_gaussianBlur, src.ptr(), src.step, dst.ptr(), dst.step, src.cols, src.rows, sdepth, cn,
             ofs.x, ofs.y, wsz.width - src.cols - ofs.x, wsz.height - src.rows - ofs.y, ksize.width, ksize.height,
             sigma1, sigma2, borderType&~BORDER_ISOLATED);

    CV_OVX_RUN(true,
               openvx_gaussianBlur(src, dst, ksize, sigma1, sigma2, borderType))

    CV_IPP_RUN_FAST(ipp_GaussianBlur(src, dst, ksize, sigma1, sigma2, borderType));
//若CPU版本的gaussianBlur仍然不能计算,则选择CPU版本的filter2D
    sepFilter2D(src, dst, sdepth, kx, ky, Point(-1, -1), 0, borderType);
}       
```              
从上述代码的大致分析中可以知道,OpenCV的GaussianBlur本质上依然是filter2D,只是针对一些特殊情况进行了GPU和CPU版本的优化,如果输入的维度等信息不满足这些特殊情况,则选择使用filter2D进行计算.关于优化不是本文的重点,filter2D会在后续的博文中进行详细分析,所以这里只对获取GaussianKernel的部分进行介绍.                  
## 获取GaussianKernel        
还是先上源码:             
```       
static void createGaussianKernels( Mat & kx, Mat & ky, int type, Size ksize,
                                   double sigma1, double sigma2 )
{
  //初始化
    int depth = CV_MAT_DEPTH(type);
    if( sigma2 <= 0 )
        sigma2 = sigma1;
//如果用户没有设置ksize,则需要根据sigma设定ksize
    // automatic detection of kernel size from sigma
    if( ksize.width <= 0 && sigma1 > 0 )
        ksize.width = cvRound(sigma1*(depth == CV_8U ? 3 : 4)*2 + 1)|1;
    if( ksize.height <= 0 && sigma2 > 0 )
        ksize.height = cvRound(sigma2*(depth == CV_8U ? 3 : 4)*2 + 1)|1;
//判断ksize是否合法
    CV_Assert( ksize.width > 0 && ksize.width % 2 == 1 &&
        ksize.height > 0 && ksize.height % 2 == 1 );
//保证sigma合法
    sigma1 = std::max( sigma1, 0. );
    sigma2 = std::max( sigma2, 0. );
//获取GaussianKernels,其数据类型为float或者double
    kx = getGaussianKernel( ksize.width, sigma1, std::max(depth, CV_32F) );
    if( ksize.height == ksize.width && std::abs(sigma1 - sigma2) < DBL_EPSILON )
        ky = kx;
    else
        ky = getGaussianKernel( ksize.height, sigma2, std::max(depth, CV_32F) );
}

}            
```
上述代码的逻辑也非常简单,这里主要解释一下,sigma和ksize的关系.根据高斯函数的分布特性,可以知道,函数分布在区间[u - 3 * sigma, u + 3 * sigma]范围内的概率大于99%.因此模板大小的选取往往与sigma有关.         
看代码中的公式,ksize = round(2 * 3 * sigma + 1) | 1;注意与1按位或,是保证结果为奇数.另外需要注意,OpenCV认为当图像类型为CV_8U的时候能量集中区域为3 * sigma,其他类型图像的能量集中区域为4*sigma.           
接着往下看,会发现,OpenCV中获取了两个方向的GaussianKernels,kx和ky.当两个方向的sigma相同,尺寸相同的时候,两个方向上的kernels是相同的.这是因为gaussianBlur是一种可分离滤波器,为了减少计算量,OpenCV采用先对行滤波,再对列滤波的方式进行滤波,这是一种优化方式.      
细心的读者可能发现在第一部分中OpenCV调用的filter2D其实是sepFilter2D,这是一种可分离的二维滤波器,同样是出于优化考虑的.           
继续看代码:                  
```             
cv::Mat cv::getGaussianKernel( int n, double sigma, int ktype )
{
    const int SMALL_GAUSSIAN_SIZE = 7;
    //定义了固定的filter即Kernels.
    static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
    {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };
//对滤波器的类型进行判断,1,尺寸为奇数;2,尺寸小于等于7;3.sigma小于等于0(注)
    const float* fixed_kernel = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ?
        small_gaussian_tab[n>>1] : 0;
//前文注释中介绍了,Kernels的数据类型为float,double也是ok的.
    CV_Assert( ktype == CV_32F || ktype == CV_64F );
    Mat kernel(n, 1, ktype);
    float* cf = kernel.ptr<float>();
    double* cd = kernel.ptr<double>();
//确定sigma,如果sigma > 0,ok,不用修改;否则按照公式计算(注)
    double sigmaX = sigma > 0 ? sigma : ((n-1)*0.5 - 1)*0.3 + 0.8;
    double scale2X = -0.5/(sigmaX*sigmaX);//高斯公式
    double sum = 0;

    int i;
    for( i = 0; i < n; i++ )
    {
        double x = i - (n-1)*0.5;
        //如果fixed_kernel为真,也就是符合上文中的3个条件,则区固定的系数;否则按照高斯公式计算
        double t = fixed_kernel ? (double)fixed_kernel[i] : std::exp(scale2X*x*x);
        //对kernels进行归一化
        if( ktype == CV_32F )
        {
            cf[i] = (float)t;
            sum += cf[i];
        }
        else
        {
            cd[i] = t;
            sum += cd[i];
        }
    }

    sum = 1./sum;
    for( i = 0; i < n; i++ )
    {
        if( ktype == CV_32F )
            cf[i] = (float)(cf[i]*sum);
        else
            cd[i] *= sum;
    }

    return kernel;
}              
```           
这个函数最终确定了gaussianKernels的计算规则.分为两种情况;1.取固定系数;2.是按照高斯公式计算.            
* 取固定系数     
当kernels的尺寸为1,3,5,7 并且用户没有设置sigma的时候(sigma <= 0),就会取固定的系数.这是一种默认的值是高斯函数的近似.            
* 按照高斯公式计算             
当kernels尺寸超过7的时候,如果sigma设置合法(用户设置了sigma),则按照高斯公式计算.当sigma不合法(用户没有设置sigma),则按照((n-1)*0.5 - 1)*0.3 + 0.8计算.n为kernels的尺寸.            
以上是OpenCV中关于高斯滤波器系数以及高斯滤波的计算规则,欢迎指正.
