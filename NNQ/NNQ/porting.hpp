#pragma once

#include "config.hpp"

//针对不同平台的运算机制的移植接口



//向量点对点运算
#define vector_cal(Y,A,B,dims,cal) \
do {\
for(qTp i=0;i<dims;i++){\
(Y)[i] = (A)[i] cal (B)[i];\
};\
}while(0)


//向量内积
#define vector_inner(Y,A,B,dims) \
do {\
(Y)=0;\
for(qTp i=0;i<dims;i++){\
(Y) += (A)[i] * (B)[i];\
};\
}while(0)














