#include <iostream>
#include <vector>
#include <array>
#include <random>

#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include "eignn/module.hpp"
#include "eignn/loss.hpp"
#include "eignn/encoder.hpp"
#include "eignn/optimizer.hpp"


namespace nb = nanobind;
using namespace nb::literals;


namespace {

void create_pixel_dataset(
        nb::tensor<float, nb::shape<nb::any, nb::any, nb::any>> &img,
        Eigen::MatrixXf &coords,
        Eigen::MatrixXf &rgb,
        const std::vector<int> &idx,
        int padding
) {
    const auto width = img.shape(0);
    const auto height = img.shape(1);
    coords.resize(2,width*height+padding);
    rgb.resize(3,width*height+padding);
    for (int iw = 0; iw < width; ++iw) {
        for (int ih = 0; ih < height; ++ih) {
            coords(0, idx[height*iw+ih]) = (float(iw)+.5f)/float(width);
            coords(1, idx[height*iw+ih]) = (float(ih)+.5f)/float(height);
            rgb(0,idx[height*iw+ih]) = img(iw,ih,0);
            rgb(1,idx[height*iw+ih]) = img(iw,ih,1);
            rgb(2,idx[height*iw+ih]) = img(iw,ih,2);
        }
    }
}

} // namespace


NB_MODULE(eignn, m) {

    m.def("fit_nn", [](
            nb::tensor<float, nb::shape<nb::any, nb::any, nb::any>> &img,
            const int hidden_dim,
            const int hidden_depth,
            const float step_size,
            const int batch_size,
            const int epochs,
            const int min_res,
            const int levels,
            const int feature_dim,
            const int table_size_log2,
            nb::tensor<float, nb::shape<nb::any>> &_freq
    ) {
        using namespace Eigen;

        const auto width = img.shape(0);
        const auto height = img.shape(1);
        const auto channels = img.shape(2);
        const auto pixels = width*height;

        std::cout << "img size: "
                  << width << "x" << height << "x" << channels
                  << std::endl;


        eignn::module::FeatureGrid<2> grid{
            min_res, levels, feature_dim, table_size_log2
        };


        eignn::module::FourierFeature ff;
        const int freqs = _freq.shape(0);
        ff.freq.resize(freqs);
        for (int ii = 0; ii < freqs; ++ii)
            ff.freq[ii] = _freq(ii);


        const int coords_ndim = 2;
        const int in_dim = grid.dim()*grid.levels()+coords_ndim*(1+2*freqs);
        const int out_dim = 3;
        eignn::module::MLP mlp{in_dim, out_dim, hidden_dim, hidden_depth};


        eignn::MSELoss loss;
        eignn::Optimizer optimizer(mlp.parameters());


        eignn::Shuffler shuffler;
        MatrixXf coords, rgb;
        std::vector<int> idx(pixels);
        for (int ii = 0; ii < idx.size(); ++ii)
            idx[ii] = ii;


        const int batches = 1 + pixels/batch_size;
        MatrixXf x, y_tar, loss_adj;
        x.resize(2, batch_size);


        for (int epoch_id = 0; epoch_id < epochs; ++epoch_id) {
            shuffler.shuffle(idx);
            ::create_pixel_dataset(img, coords, rgb, idx, batch_size);

            for (int batch_id = 0; batch_id < batches; ++batch_id) {
                const int start_col = batch_size * batch_id;
                const int end_col = start_col + batch_size;
                x = coords.block(0,start_col,2,batch_size);
                y_tar = rgb.block(0,start_col,3,batch_size);

                ff.forward(x);
                grid.forward(ff.y);
                mlp.forward(grid.y);
                assert(!std::isnan(mlp.y.sum()));

                float loss_val;
                loss.eval(mlp.y,y_tar,loss_val,loss_adj);
                assert(!std::isnan(loss_val));
                if (end_col > pixels)
                    loss_adj.block(0, pixels-start_col,loss_adj.rows(), end_col-pixels).setZero();

                mlp.adjoint(step_size*loss_adj);
                grid.adjoint(mlp.x_adj);
                ff.adjoint(grid.x_adj);
                assert(ff.x_adj.rows() == x.rows() && ff.x_adj.cols() == x.cols());

                optimizer.descent();

                if (batch_id == 0 || batch_id % (batches/10) != 0)
                    continue;

                std::cout << "epoch: " << epoch_id+1 << "/" << epochs
                          << " batch: " << batch_id+1 << "/" << batches
                          << ": loss = " << loss_val << std::endl;
            }
        }


        for (int ii = 0; ii < idx.size(); ++ii)
            idx[ii] = ii;
        ::create_pixel_dataset(img, coords, rgb, idx, 0);


        for (int iw = 0; iw < width; ++iw) {
            x = coords.block(0,height*iw,2,height);

            ff.forward(x);
            grid.forward(ff.y);
            mlp.forward(grid.y);

            for (int ih = 0; ih < height; ++ih)
                for (int ic = 0; ic < channels; ++ic)
                    img(iw,ih,ic) = mlp.y(ic,ih);
        }
    });

}