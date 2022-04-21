#include <iostream>
#include <time.h>
#include <math.h>
#include <queue>
#include <math.h>
#include <string>
#include <vector>



using namespace std;
#include "NNQ/nnq.hpp"
using namespace nnq;

nnl::Model m = {
        Func::Linear(1,20,0.001),
        Func::Sigmoid(20),
        Func::Linear(20,20,0.001),
        Func::Sigmoid(20),
        Func::Linear(20,1,0.0001),
};

int main() {
    qVec in(1);
    qVec out(1);
    qVec ideal(1);
    qVec loss(1);
    qVec grad(1);
    
    srand((unsigned)time(NULL));
    mode = 1;
    for(int i = 0; i < 20000; i++) {
        
        for (int j = 0; j < 100; j++) {
            float a = sin(j / 10);
            in[0] = j/10;
            ideal[0] = a;
            out <= m(in);
            Func::square_loss(loss,out, ideal, grad);
            m[grad];
        }
        
        qtype ls =
            loss[0];
        cout << ls << endl;
    }
    mode = 0;
    for (int j = 0; j < 100; j++) {
        int k = abs(rand()) % 100;
        float a = sin(k / 10);
        in[0] = k / 10;

        out <= m(in);
        cout << "real:" << out[0] << " ideal:" << a << endl;
    }

    
}





