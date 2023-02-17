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

void create_pixel_dataset(
        int width, int height,
        Eigen::MatrixXf &coords,
        Eigen::MatrixXf &rgb,
        const std::vector<int> &idx
) {
    coords.resize(2, width*height);
    rgb.resize(3, width*height);

    for (int iw = 0; iw < width; ++iw) {
        for (int ih = 0; ih < height; ++ih) {
            int pix_id = idx[height*iw+ih];
            coords(0, pix_id) = (float(iw)+.5f)/float(width);
            coords(1, pix_id) = (float(ih)+.5f)/float(height);
            rgb(0, pix_id) = coords(0, pix_id);
            rgb(1, pix_id) = coords(1, pix_id);
            rgb(2, pix_id) = .5f;
        }
    }
}

} // namespace


int main() {
    using namespace std;
    using namespace Eigen;

    const int width = 16;
    const int height = 16;
    const int pixels = width * height;


    eignn::module::FourierFeature ff;
    ff.freq = {1,2,3};
    const int freqs = ff.freq.size();


    const int grid_res = 4;
    const int feature_dim = 2;
    const int table_size_log2 = 2;

    ArrayXi grid_shape;
    grid_shape.resize(2);
    grid_shape[0] = grid_shape[1] = grid_res;
    eignn::module::FeatureGrid<2> grid{grid_shape, feature_dim, table_size_log2};

    const int in_dim = (grid.dim()+2)*(1+2*freqs);
    const int out_dim = 3;

    const int hidden_dim = 4;
    const int hidden_depth = 1;
    eignn::module::MLP mlp{in_dim, out_dim, hidden_dim, hidden_depth};


    const int batch_size = 32;
    const int batches = pixels/batch_size;

    MatrixXf x, y_tar, loss_bar;
    x.resize(2, batch_size);

    const float step_size = 1e-1f;
    const int epochs = 3;

    eignn::MSELoss loss;

    eignn::Shuffler shuffler;
    MatrixXf coords, rgb;
    std::vector<int> idx(pixels);
    for (int ii = 0; ii < idx.size(); ++ii)
        idx[ii] = ii;

    for (int epoch_id = 0; epoch_id < epochs; ++epoch_id) {
        shuffler.shuffle(idx);
        ::create_pixel_dataset(width, height, coords, rgb, idx);

        for (int batch_id = 0; batch_id < batches; ++batch_id) {
            x = coords.block(0,batch_size*batch_id,2,batch_size);
            y_tar = rgb.block(0,batch_size*batch_id,3,batch_size);

            grid.forward(x);
            ff.forward(grid.y());
            mlp.forward(ff.y());
            assert(!std::isnan(mlp.y().sum()));

            float loss_val;
            loss.eval(mlp.y(),y_tar,loss_val,loss_bar);
            assert(!std::isnan(loss_val));

            mlp.reverse(step_size*loss_bar);
            ff.reverse(mlp.x_bar());
            grid.reverse(ff.x_bar());
            assert(grid.x_bar().rows() == x.rows() && grid.x_bar().cols() == x.cols());

            cout << "epoch no." << epoch_id+1 << ": loss = " << loss_val << endl;
        }
    }
}