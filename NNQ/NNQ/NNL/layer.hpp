#pragma once


#include "../config.hpp"
#include "../qMatrix.hpp"


namespace nnq {
	namespace nnl {
		//所有层的基类
		class Layer {
		public:
			virtual qMatrix& forward(qMatrix& input) = 0;
			//输入梯度
			//grad_: 当前层的梯度
			virtual qMatrix& backward(qMatrix& grad_) = 0;
			virtual void Saveto(quint nl);
			inline qMatrix& operator()(qMatrix& input) {
				return forward(input);
			}
			inline qMatrix& operator[](qMatrix& grad_) {
				return backward(grad_);
			}
		};

		//线性层，WX+B
		class Linear : public Layer {
		public:
			qtype ln;
			//保存前向传输的输入
			qMatrix* input = nullptr;
			qMatrix grad_l;//n-1层的梯度
			qMatrix ws;
			qMatrix bs;
			qMatrix os;
			//pr_setzero:设梯度为0的概率[0-100]
			Linear(quint inlens, quint outlens, qtype ln,const qMatrix* _ws = nullptr,const qMatrix* _bs = nullptr);
			qMatrix& forward(qMatrix& input);
			qMatrix& backward(qMatrix& grad_);
			void Saveto(quint nl);
		};
		//Sigmoid
		class Sigmoid : public Layer {
		public:
			qMatrix os;
			Sigmoid(quint inlens);
			qMatrix& forward(qMatrix& input);
			qMatrix& backward(qMatrix& grad_);
		};
		//Relu
		class Relu : public Layer {
		public:
			//保存前向传输的输入
			qMatrix* input = nullptr;
			qMatrix os;
			Relu(quint inlens);
			qMatrix& forward(qMatrix& input);
			qMatrix& backward(qMatrix& grad_);
		};
		//LeakyRelu
		class LeakyRelu : public Layer {
		public:
			//保存前向传输的输入
			qMatrix* input = nullptr;
			qMatrix os;
			qtype ln;
			LeakyRelu(quint inlens, qtype ln);
			qMatrix& forward(qMatrix& input);
			qMatrix& backward(qMatrix& grad_);
		};

		//Tanh
		class Tanh : public Layer {
		public:
			qMatrix os;
			Tanh(quint inlens);
			qMatrix& forward(qMatrix& input);
			qMatrix& backward(qMatrix& grad_);
		};

		//Softmax
		class Softmax : public Layer {
		public:
			qMatrix os;
			Softmax(quint inlens);
			qMatrix& forward(qMatrix& input);
			qMatrix& backward(qMatrix& grad_);
		};




	}
}
