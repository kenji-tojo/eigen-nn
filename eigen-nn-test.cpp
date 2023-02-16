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
    const int hidden_depth = 1;

    eignn::module::MLP mlp{in_dim, out_dim, hidden_dim, hidden_depth};

    const int batch_size = 32;
    MatrixXf x, x_enc;
    MatrixXf d_loss;
    x.resize(coords, batch_size);

    const float step_size = 1.f;
    const int epochs = 30;
    eignn::MSELoss loss;
    eignn::Sampler<float> sampler;

    for (int ii = 0; ii < epochs; ++ii) {
        random_matrix(x, sampler);
        eignn::encoder::fourier_feature(x, x_enc, freq);
        mlp.forward(x_enc);
        float loss_val;
        loss.eval(mlp.y(), x, loss_val, d_loss);
        mlp.reverse(step_size*d_loss);
        assert(mlp.x_bar().rows() == in_dim && mlp.x_bar().cols() == batch_size);
        cout << "epoch no." << ii << ": loss = " << loss_val << endl;
    }
}