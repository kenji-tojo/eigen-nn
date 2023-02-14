#include <iostream>

#include "eigen-nn/module.hpp"
#include "eigen-nn/sampler.hpp"
#include "eigen-nn/loss.hpp"


namespace {

void random_matrix(Eigen::MatrixXf &m, eignn::Sampler<float> &sampler) {
    for (int ii = 0; ii < m.rows(); ++ii) {
        for (int jj = 0; jj < m.cols(); ++jj) {
            m(ii,jj) = sampler.sample();
        }
    }
}

} // namespace


int main() {
    using namespace std;
    using namespace Eigen;

    int in_dim = 2;
    int out_dim = 2;
    int hidden_dim = 32;
    int hidden_depth = 3;

    eignn::module::MLP mlp{in_dim, out_dim, hidden_dim, hidden_depth};

    int batch_size = 128;
    MatrixXf x, y, d_x, d_loss;
    x.resize(in_dim, batch_size);

    const float step_size = 1e-1f;
    const int epochs = 50;
    eignn::MSELoss loss;
    eignn::Sampler<float> sampler;

    for (int ii = 0; ii < epochs; ++ii) {
        random_matrix(x, sampler);
        mlp.forward(x, y);
        float loss_val;
        loss.eval(y, x, loss_val, d_loss);
        mlp.reverse(-1.f*d_loss*step_size, d_x);
        assert(d_x.rows() == x.rows() && d_x.cols() == x.cols());
        cout << "epoch no." << ii << ": loss = " << loss_val << endl;
    }
}