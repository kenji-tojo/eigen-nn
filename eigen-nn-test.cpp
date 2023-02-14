#include <iostream>

#include "eignn/module.hpp"
#include "eignn/sampler.hpp"
#include "eignn/loss.hpp"
#include "eignn/encoder.hpp"


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

    std::vector<int> freq{1, 2, 3, 4, 5, 6};

    const int coords = 2;
    const int in_dim = coords*(1+2*freq.size());
    const int out_dim = coords;
    const int hidden_dim = 32;
    const int hidden_depth = 3;

    eignn::module::MLP mlp{in_dim, out_dim, hidden_dim, hidden_depth};

    const int batch_size = 128;
    MatrixXf x, x_enc, y;
    MatrixXf d_x, d_loss;
    x.resize(coords, batch_size);

    const float step_size = 1e-1f;
    const int epochs = 50;
    eignn::MSELoss loss;
    eignn::Sampler<float> sampler;

    for (int ii = 0; ii < epochs; ++ii) {
        random_matrix(x, sampler);
        eignn::encoder::fourier_feature(x, x_enc, freq);
        mlp.forward(x_enc, y);
        float loss_val;
        loss.eval(y, x, loss_val, d_loss);
        mlp.reverse(-1.f*step_size*d_loss, d_x);
        assert(d_x.rows() == in_dim && d_x.cols() == batch_size);
        cout << "epoch no." << ii << ": loss = " << loss_val << endl;
    }
}