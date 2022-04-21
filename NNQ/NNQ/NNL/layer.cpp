#include "layer.hpp"
#include "../qFunc.hpp"
#include <random>
#include <stdio.h>

//0代表正常，1代表训练模式
quint mode = 0;

namespace nnq {
    namespace nnl {


        void Layer::Saveto(quint nl) {}


        Linear::Linear(quint inlens, quint outlens, qtype ln,const qMatrix* _ws,const qMatrix* _bs) {
            this->ln = ln;

            if (_ws == nullptr) {
                qtype* data_ws = qalloc(qtype, inlens * outlens);
                ws.Init(data_ws, outlens, inlens);
            }
            else {
                ws >= *_ws;
            }

            if (_bs == nullptr) {
                qtype* data_bs = qalloc(qtype, outlens);
                bs.Init(data_bs, outlens, 1);
            }
            else {
                bs >= *_bs;
            }



            qtype* data_os = qalloc(qtype, outlens);
            os.Init(data_os, outlens, 1);
            qtype* data_g = qalloc(qtype, inlens);
            grad_l.Init(data_g, inlens, 1);

            //初始化ws
            for (quint i = 0; i < ws.rows * ws.cols; i++) {
                ws.eles[i] = Func::random_normal(0, 0.5);
            }
            //初始化bs
            for (quint i = 0; i < bs.rows * bs.cols; i++) {
                bs.eles[i] = Func::random_normal(0, 0.5);
            }
        }

        qMatrix& Linear::forward(qMatrix& input) {
            this->input = &input;
            ws.Mul(input, os);//os = W*X
            os.ADD(bs, os);//O = O + B
            return os;
        }

        //grad_为当前层的输出梯度
        qMatrix& Linear::backward(qMatrix& grad_) {
            //首先反向更新n-1层的梯度
            //grad_l = W.T * grad_
            for (quint i = 0; i < grad_l.rows; i++) {
                grad_l[i][0] = 0;
                for (quint j = 0; j < grad_.rows; j++) {
                    grad_l[i][0] += ws[j][i] * grad_[j][0];
                }
            }

            //再更新当前层的参数
            // W减去grad与input的转置矩阵的积
            // ln为学习率
            // W = W - ln * grad_ Mul input.T
            // B = B - ln * grad_
            for (quint i = 0; i < ws.rows; i++) {
                for (quint j = 0; j < ws.cols; j++) {
                    ws[i][j] -= ln * grad_[i][0] * (*input)[j][0];
                }
                bs[i][0] -= ln * grad_[i][0];
            }

            return grad_l;
        }
#if JUST_TO_RUN == 0
        void Linear::Saveto(quint nl) {
            char buff[] = "00linear.bin";
            char a = (nl / 10) % 10;
            char b = nl % 10;
            buff[0] += a;
            buff[1] += b;
            FILE* fp = fopen(buff, "w");
            //保存ws,注意后面有2个大S
            fprintf(fp, "SS%dSS%dSS", (uint32_t)ws.rows, (uint32_t)ws.cols);
            fwrite(ws.eles, sizeof(qtype), ws.rows * ws.cols, fp);
            //保存bs
            fprintf(fp, "SS%dSS%dSS", (uint32_t)bs.rows, (uint32_t)bs.cols);
            fwrite(ws.eles, sizeof(qtype), bs.rows * bs.cols, fp);
            fclose(fp);
        }
#endif




        Sigmoid::Sigmoid(quint inlens) {
            qtype* data_os = qalloc(qtype, inlens);
            os.Init(data_os, inlens, 1);
        }
        qMatrix& Sigmoid::forward(qMatrix& input) {
            for (quint i = 0; i < input.rows; i++) {
                os[i][0] = Func::sigmoid(input[i][0]);
            }
            return os;
        }
        qMatrix& Sigmoid::backward(qMatrix& grad_) {
            //sigmodd的导数为sigmodd(x) * (1-sigmodd(x))
            for (quint i = 0; i < grad_.rows; i++) {
                grad_[i][0] = os[i][0] * (1 - os[i][0]) * grad_[i][0];
            }
            return grad_;
        }

        Relu::Relu(quint inlens) {
            qtype* data_os = qalloc(qtype, inlens);
            os.Init(data_os, inlens, 1);
        }
        qMatrix& Relu::forward(qMatrix& input) {
            this->input = &input;
            for (quint i = 0; i < input.rows; i++) {
                os[i][0] = Func::relu(input[i][0]);
            }
            return os;
        }
        qMatrix& Relu::backward(qMatrix& grad_) {
            for (quint i = 0; i < grad_.rows; i++) {
                grad_[i][0] = (*input[i][0] > 0) ? grad_[i][0] : 0;
            }
            return grad_;
        }
        /*leakyRelu*/
        LeakyRelu::LeakyRelu(quint inlens, qtype ln) {
            this->ln = ln;
            qtype* data_os = qalloc(qtype, inlens);
            os.Init(data_os, inlens, 1);
        }
        qMatrix& LeakyRelu::forward(qMatrix& input) {
            this->input = &input;
            for (quint i = 0; i < input.rows; i++) {
                os[i][0] = Func::leaky_relu(input[i][0], ln);
            }
            return os;
        }
        qMatrix& LeakyRelu::backward(qMatrix& grad_) {
            for (quint i = 0; i < grad_.rows; i++) {
                grad_[i][0] = (*input[i][0] > 0) ? grad_[i][0] : 0.01 * grad_[i][0];
            }
            return grad_;
        }


        Tanh::Tanh(quint inlens) {
            qtype* data_os = qalloc(qtype, inlens);
            os.Init(data_os, inlens, 1);

        }

        qMatrix& Tanh::forward(qMatrix& input) {
            for (quint i = 0; i < input.rows; i++) {
                os[i][0] = Func::tanh(input[i][0]);
            }
            return os;
        }

        qMatrix& Tanh::backward(qMatrix& grad_) {
            for (quint i = 0; i < grad_.rows; i++) {
                grad_[i][0] = (1 - os[i][0] * os[i][0]) * grad_[i][0];
            }
            return grad_;
        }

        Softmax::Softmax(quint inlens) {
            qtype* data_os = qalloc(qtype, inlens);
            os.Init(data_os, inlens, 1);
        }
        qMatrix& Softmax::forward(qMatrix& input) {
            qtype sum = 0;
            for (quint i = 0; i < input.rows; i++) {
                sum += exp(input[i][0]);
            }
            for (quint i = 0; i < input.rows; i++) {
                os[i][0] = exp(input[i][0]) / sum;
            }
            return os;
        }
        qMatrix& Softmax::backward(qMatrix& grad_) {
            for (quint i = 0; i < grad_.rows; i++) {
                qtype sum = 0;
                for (quint j = 0; j < grad_.rows; j++) {
                    if (i == j) {
                        sum += os[i][0] * (1 - os[i][0]);
                    }
                    else {
                        sum += -os[i][0] * os[j][0];
                    }
                }
                grad_[i][0] = sum * grad_[i][0];
            }
            return grad_;
        }
    }
} // namespace nnq
