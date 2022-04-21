#include "model.hpp"


namespace nnq {
	namespace nnl {

        Model::Model(std::initializer_list<Layer*> li) {
            depth = li.size();
            if (depth == 0) {
                return;
            }
            layers = qalloc(Layer*, depth);
            for (quint i = 0; i < depth; i++) {
                layers[i] = (Layer*)li.begin()[i];
            }
        }
        qMatrix& Model::forward(qMatrix& input) {
            qMatrix* m = &input;
            for (quint i = 0; i < depth; i++) {
                m = &(layers[i]->forward(*m));
            }
            return *m;
        }
        qMatrix& Model::backward(qMatrix& grad_) {
            qMatrix* m = &grad_;
            for (qint i = depth - 1; i >= 0; i--) {
                m = &(layers[i]->backward(*m));
            }
            return *m;
        }
        void Model::Saveto(quint nl)
        {
            for (int i = 0; i < depth; i++) {
                layers[i]->Saveto(i + nl);
            }
        }
	}
}