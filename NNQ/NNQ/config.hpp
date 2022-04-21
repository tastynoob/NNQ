#pragma once


#if __cplusplus > 201103L
#include <initializer_list>
#define cpp11 1
#else

//#warning "你当前所用的编译器C++版本不支持部分功能，这可能会影响部分使用，比如<initializer_list>头文件的使用"

#endif

#include "porting.hpp"


//设置为1 关闭网络训练功能
#define JUST_TO_RUN 0


//nnq框架所使用的基础整数类型
//常用于for循环或者其它地方
typedef unsigned int quint;
typedef int qint;
//qtype用于数学计算
typedef float qtype;

//开辟内存,开辟的内存大小为 sizeof(type) * len
#define qalloc(type,len) (new type[len])
#define qfree(ptr) (delete[] ptr)
#define qerror(msg) (throw msg)


