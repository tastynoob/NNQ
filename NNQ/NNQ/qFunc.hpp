

#include "config.hpp"
#include "qMatrix.hpp"
#include "NNL/layer.hpp"

namespace nnq {

class Func {
public:
    //生成服从正态分布的随机数
    static qtype random_normal(qtype u, qtype sigma);

    static qtype sigmoid(qtype x);
    static qtype relu(qtype x);
    static qtype leaky_relu(qtype x, qtype ln);
    static qtype tanh(qtype x);

    //绝对损失函数 Y=|real-ideal|,grad = Y'(real)
    static void absolute_loss(qMatrix& Y,qMatrix& real, qMatrix& ideal, qMatrix& grad);
    //平方损失函数 Y=（real-ideal)^2,grad = 2*(real-ideal)
    static void square_loss(qMatrix& Y, qMatrix& real, qMatrix& ideal, qMatrix& grad);
    //交叉熵损失函数 Y=-ideal*log(real),grad = -ideal/real，必须配合softmax使用
    static void cross_entropy_loss(qMatrix& Y, qMatrix& real, qMatrix& ideal, qMatrix& grad);


    static inline nnl::Layer* Linear(quint rows,quint cols,qtype ln,qMatrix* _ws=nullptr,qMatrix* _bs=nullptr) {
        nnl::Layer* l = new nnl::Linear(rows,cols,ln,_ws,_bs);
        return l;
    }

    static inline nnl::Layer* Sigmoid(quint inlens) {
        nnl::Layer* l = new nnl::Sigmoid(inlens);
        return l;
    }
    static inline nnl::Layer* Tanh(quint inlens) {
        nnl::Layer* l = new nnl::Tanh(inlens);
        return l;
    }

    static inline nnl::Layer* Relu(quint inlens) {
        nnl::Layer* l = new nnl::Relu(inlens);
        return l;
    }
    static inline nnl::Layer* LeakyRelu(quint inlens,qtype ln=0.001) {
        nnl::Layer* l = new nnl::LeakyRelu(inlens,ln);
        return l;
    }
    static inline nnl::Layer* Softmax(quint inlens) {
        nnl::Layer* l = new nnl::Softmax(inlens);
        return l;
    }

};



class Datald {

    Datald(char* path);
    //读取指定通道数量的3维矩阵
    //如果读取的通道数量大于最大通道，只返回最大通道大小的3维矩阵
    qMat3& ReadMat3(quint channs);
};

}