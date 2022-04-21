
#include "qFunc.hpp"
#include <random>


namespace nnq {


    //生成服从正态分布的随机数
    qtype Func::random_normal(qtype u, qtype sigma) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(u, sigma);
        return d(gen);
    }

    qtype Func::sigmoid(qtype x) {
        return 1.0 / (1.0 + exp(-x));
    }

    qtype Func::relu(qtype x) {
        return x > 0 ? x : 0;
    }
    qtype Func::leaky_relu(qtype x, qtype ln) {
        return x > 0 ? x : ln * x;
    }
    qtype Func::tanh(qtype x) {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }



    void Func::absolute_loss(qMatrix& Y, qMatrix& real, qMatrix& ideal, qMatrix& grad)
    {
        if (real.rows != ideal.rows || real.cols != ideal.cols)
        {
            qerror("real and ideal must have the same size");
            return;
        }
        for (quint i = 0; i < real.rows; i++)
        {
            for (quint j = 0; j < real.cols; j++)
            {
                Y[i][j] = (real[i][j] - ideal[i][j]) > 0 ? (real[i][j] - ideal[i][j]) : (ideal[i][j] - real[i][j]);
                grad[i][j] = real[i][j] - ideal[i][j];
            }
        }
    }

    void Func::square_loss(qMatrix& Y, qMatrix& real, qMatrix& ideal, qMatrix& grad)
    {
        if (real.rows != ideal.rows || real.cols != ideal.cols)
        {
            qerror("real and ideal must have the same size");
            return;
        }
        for (quint i = 0; i < real.rows; i++)
        {
            for (quint j = 0; j < real.cols; j++)
            {
                Y[i][j] = (real[i][j] - ideal[i][j]) * (real[i][j] - ideal[i][j]);
                grad[i][j] = 2 * (real[i][j] - ideal[i][j]);
            }
        }
    }

    void Func::cross_entropy_loss(qMatrix& Y, qMatrix& real, qMatrix& ideal, qMatrix& grad)
    {
        if (real.rows != ideal.rows || real.cols != ideal.cols)
        {
            qerror("real and ideal must have the same size");
            return;
        }
        for (quint i = 0; i < real.rows; i++)
        {
            for (quint j = 0; j < real.cols; j++)
            {
                if (real[i][j] <= 0) {
                    qerror("cross entropy data error!");
                }
                Y[i][j] = -ideal[i][j] * log(real[i][j]);
                grad[i][j] = -ideal[i][j] / real[i][j];
            }
        }
    }


    //Datald::Datald(char* path) {

    //}

    //qMat3& Datald::ReadMat3(quint channs)
    //{

    //}

}