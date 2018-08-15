---
layout:		post
title:        	bit operate
subtitle:     	about pow of 2
date:         	2018-08-15
author:       	DJ
header-image: 	image/post-bg-ios-web.jpg
catalog:      	true
tags:
	- c++
	- bit operate
---  
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

# 关于2的幂                  

## 判断该数是否是2的整数次幂                      

```
static inline int is_pow_of_2(int val){
	return !(val & (val-1));
}
```         
## 求能大于等于当前数字的最小2的整数次幂                             

```
static inline int next_pow_of_2(int val){
	if(is_pow_of_2(val))
		return val;

	if(sizeof(val) == 4){
		val |= val >> 1;
		val |= val >> 2;
		val |= val >> 4;
		val |= val >> 8;
		val |= val >> 16;
	}

	if(sizeof(val) == 8){
		val |= val >> 1;
		val |= val >> 2;
		val |= val >> 4;
		val |= val >> 8;
		val |= val >> 16;
		val |= val >> 32;
	}

	return val + 1;
}
```              

