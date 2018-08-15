---
layout:     post
title:      bit opearte
subtitle:   pow of 2
date:       2018-08-16
author:     DJ
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - C++
    - bit Opearting
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

