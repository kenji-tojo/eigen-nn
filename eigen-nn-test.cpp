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

    eignn::module::FourierFeature ff;
    ff.freq = {1,2,3,4,5,6};

    const int coords = 2;
    const int in_dim = coords*(1+2*ff.freq.size());
    const int out_dim = coords;
    const int hidden_dim = 32;
    const int hidden_depth = 1;

    eignn::module::MLP mlp{in_dim, out_dim, hidden_dim, hidden_depth};

    const int batch_size = 32;
    MatrixXf x, loss_bar;
    x.resize(coords, batch_size);

    const float step_size = 1.f;
    const int epochs = 30;
    eignn::MSELoss loss;
    eignn::Sampler<float> sampler;

    for (int ii = 0; ii < epochs; ++ii) {
        random_matrix(x, sampler);
        ff.forward(x);
        mlp.forward(ff.y());
        float loss_val;
        loss.eval(mlp.y(), x, loss_val, loss_bar);
        mlp.reverse(step_size*loss_bar);
        ff.reverse(mlp.x_bar());
        assert(ff.x_bar().rows() == x.rows() && ff.x_bar().cols() == x.cols());
        cout << "epoch no." << ii << ": loss = " << loss_val << endl;
    }
}