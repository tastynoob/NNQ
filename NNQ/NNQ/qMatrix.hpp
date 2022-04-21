#pragma once

#include "config.hpp"

namespace nnq {


class qMatrix {
public:
    //矩阵的行
    quint rows = 0;
    //矩阵的列
    quint cols = 0;
    qtype* eles = nullptr;

#ifdef cpp11
    constexpr qMatrix(const std::initializer_list<std::initializer_list<qtype>> li) {
        rows = li.size();
        if (rows == 0) {
            cols = 0;
            eles = nullptr;
            return;
        }
        cols = li.begin()->size();
        eles = qalloc(qtype, rows * cols);
        quint i = 0;
        for (quint row = 0; row < rows; row++) {
            for (auto& col : li.begin()[row]) {
                eles[i++] = col;
            }
            for (quint col = li.begin()[row].size(); col < cols; col++) {
                eles[i++] = 0;
            }
        }
    }
#endif
    constexpr qMatrix():eles(nullptr), rows(0), cols(0) {}
    //矩阵复制
    qMatrix(const qMatrix& qmat);
    //初始化矩阵大小，不初始化元素
    qMatrix(quint rows, quint cols);
    //datas数组可释放
    qMatrix(quint rows, quint cols, qtype* datas);
    //datas数组不可施放
    qMatrix(qtype* datas, quint rows, quint cols);
    ~qMatrix();
    //要求datas数组不可释放
    void Init(qtype* datas, quint rows, quint cols);
    //矩阵复制
    void Copy(const qMatrix& qmat);
    constexpr qtype* operator[](quint index) {
        qtype* k = &eles[index * cols];
        return &eles[index * cols];
    }
    void ADD(qMatrix& B, qMatrix& Y);
    // Y = A x B
    // 该函数是不安全的
    void Mul(qMatrix& B, qMatrix& Y);
    //哈达玛矩阵乘法
    void HadamardMul(qMatrix& B, qMatrix& Y);
    //克罗内克矩阵乘法
    void KroneckerMul(qMatrix& B, qMatrix& Y);
    ////数据复制
    //qMatrix& operator=(qMatrix& m);
    //数据拷贝,必须保证2个矩阵大小相同
    qMatrix& operator<=(qMatrix& m);
    //矩阵替换，将自身数据替换为矩阵m
    qMatrix& operator>=(const qMatrix& m);
};


//静态矩阵
//需要提前初始化矩阵大小和元素
template<quint R, quint C>
class qSMatrix : public qMatrix {
public:
    qtype s_eles[R * C];

    constexpr qSMatrix() {
        this->rows = R;
        this->cols = C;
        this->eles = s_eles;
    }
};

//列向量
class qVec : public qMatrix {
public:
    qVec(const std::initializer_list<qtype>& li);
    qVec(quint len);
    constexpr qtype& operator[](quint index) {
        return eles[index];
    }
};
//静态列向量
template<quint R>
class qSVec :public qVec {
    public:
    qtype s_eles[R];
    constexpr qSVec() {
        this->rows = R;
        this->cols = 1;
        this->eles = s_eles;
    }
    constexpr qSVec(std::initializer_list<qtype> li) {
        this->rows = R;
        this->cols = 1;
        this->eles = s_eles;
        quint i = 0;
        for (auto& e : li) {
            s_eles[i++] = e;
        }
        for (;i < R;i++) {
            s_eles[i] = 0;
        }
    }
};

//3维矩阵
class qMat3:public qMatrix {
public:
    quint chans;
    inline qMat3(quint rows, quint cols, quint chans=1) {
        this->rows = rows;
        this->cols = cols;
        this->chans = chans;
        this->eles = qalloc(qtype, rows * cols * chans);
    }
    inline qtype* GetChann(quint n) {
        return &eles[n * rows * cols];
    }
};

}