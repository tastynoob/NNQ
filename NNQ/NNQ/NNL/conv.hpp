#pragma once

#include "layer.hpp"

namespace nnq {
	namespace nnl {

		class Conv2d : public Layer {
			qMat3 os;
		public:
			Conv2d(quint inShape[3], quint kerShape[3], quint step[2]);
			qMatrix& forward(qMatrix& input);
			qMatrix& backward(qMatrix& grad_);
		};
	}
}