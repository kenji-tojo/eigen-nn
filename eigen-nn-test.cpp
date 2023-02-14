#include <iostream>

#include "eigen-nn/module.hpp"

int main() {
    using namespace std;
    using namespace Eigen;

    int in_dim = 2;
    int out_dim = 3;
    int hidden_dim = 32;
    int hidden_depth = 1;

    eignn::module::MLP m{in_dim, out_dim, hidden_dim, hidden_depth};

    int batch_size = 8;
    MatrixXf x;
    x.setZero(in_dim, batch_size);

    m.forward(x,x);
    cout << "output shape: " << x.cols() << "x" << x.rows() << endl;
}