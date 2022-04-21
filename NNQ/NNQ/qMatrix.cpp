
#include "qMatrix.hpp"

namespace nnq {
//矩阵复制
qMatrix::qMatrix(const qMatrix& qmat) {
	this->Copy(qmat);
}

//初始化矩阵大小，不初始化元素
qMatrix::qMatrix(quint rows, quint cols) {
	this->Init(qalloc(qtype, rows * cols), rows, cols);
}
//datas可释放
qMatrix::qMatrix(quint rows, quint cols, qtype* datas) {
	qtype* eles_tp = qalloc(qtype, rows * cols);
	for (quint i = 0; i < rows * cols; i++) {
		eles_tp[i] = datas[i];
	}
	this->Init(eles_tp, rows, cols);
}
//datas不可释放
qMatrix::qMatrix(qtype* datas, quint rows, quint cols)
{
	Init(datas, rows, cols);
}

//利用数组初始化矩阵
//要求datas数组不可释放
void qMatrix::Init(qtype* datas, quint rows, quint cols) {
	this->rows = rows;
	this->cols = cols;
	this->eles = datas;
}

qMatrix::~qMatrix() 	{
	qfree(eles);
}

//矩阵复制
void qMatrix::Copy(const qMatrix& qmat) {
	if(this->eles != nullptr) {
		qfree(this->eles);
	}
	rows = qmat.rows;
	cols = qmat.cols;
	eles = qalloc(qtype, rows * cols);
	for (quint i = 0; i < rows * cols; i++) {
		eles[i] = qmat.eles[i];
	}
}

// constexpr qtype* qMatrix::operator[](quint index) {
// 	qtype* k = &eles[index * cols];
// 	return &eles[index * cols];
// }

void qMatrix::ADD(qMatrix& B, qMatrix& Y) {
	if (this->rows != B.rows || this->cols != B.cols) {
		qerror("qMatrix::ADD: matrix size not match");
		return;
	}
	for (quint i = 0; i < rows * cols; i++) {
		Y.eles[i] = eles[i] + B.eles[i];
	}
}

// Y = A x B
// 该函数是不安全的
void qMatrix::Mul(qMatrix& B, qMatrix& Y) {
	if (cols != B.rows) {
		qerror("qMatrix::Mul: matrix size not match");
		return;
	}
	for (quint i = 0; i < rows; i++) {
		for (quint j = 0; j < B.cols; j++) {
			Y.eles[i * B.cols + j] = 0;
			for (quint k = 0; k < cols; k++) {
				Y.eles[i * B.cols + j] += eles[i * cols + k] * B.eles[k * B.cols + j];
			}
		}
	}
}

void qMatrix::HadamardMul(qMatrix& B, qMatrix& Y) {
	if (rows != B.rows || cols != B.cols) {
		qerror("qMatrix::HadamardMul: matrix size not match");
		return;
	}
	for (quint i = 0; i < rows * cols; i++) {
		Y.eles[i] = eles[i] * B.eles[i];
	}
}
void qMatrix::KroneckerMul(qMatrix& B, qMatrix& Y) 	{
	if (cols != B.rows) 		{
		qerror( "qMatrix::KroneckerMul: matrix size not match");
		return;
	}
	quint rows_y = rows * B.rows;
	quint cols_y = cols * B.cols;
	for (quint i = 0; i < rows; i++) {
		for (quint j = 0; j < cols; j++) {
			for (quint k = 0; k < B.rows; k++) {
				for (quint l = 0; l < B.cols; l++) {
					Y[i + k][j + l] = *this[i][j] * B[k][l];
				}
			}
		}
	}
}
//qMatrix& qMatrix::operator=(qMatrix& m) 	{
//	if (this == &m) return *this;
//	if(this->rows != m.rows || this->cols != m.cols) {
//		qerror("qMatrix::operator=: matrix size not match");
//		return *this;
//	}
//	this->Copy(m);
//	return *this;
//}

//数据拷贝,必须保证2个矩阵大小相同
qMatrix& qMatrix::operator<=(qMatrix& m) {
	if (this->rows != m.rows || this->cols != m.cols) 		{
		qerror("qMatrix::operator<=: matrix size not match");
		return *this;
	}
	for (quint i = 0; i < rows * cols; i++) {
		eles[i] = m.eles[i];
	}
	return *this;
}

qMatrix& qMatrix::operator>=(const qMatrix& m){
	this->rows = m.rows;
	this->cols = m.cols;
	this->eles = m.eles;
	return *this;
}


qVec::qVec(const std::initializer_list<qtype>& list) {
	this->Init(qalloc(qtype, list.size()), list.size(),1);
	quint i = 0;
	for (auto& e : list) {
		eles[i++] = e;
	}
}
qVec::qVec(quint len) {
	this->Init(qalloc(qtype, len), len, 1);
}



}