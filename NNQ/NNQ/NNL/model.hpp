#pragma once

#include "layer.hpp"

namespace nnq {
	namespace nnl {
		class Model : public Layer {

			quint depth = 0;
			Layer** layers = nullptr;
		public:
			Model(std::initializer_list<Layer*> li);
			qMatrix& forward(qMatrix& input);
			qMatrix& backward(qMatrix& grad_);
			void Saveto(quint nl);
		};
	}
}