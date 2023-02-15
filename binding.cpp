#include <iostream>
#include <vector>
#include <array>
#include <random>

#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include "eignn/module.hpp"
#include "eignn/loss.hpp"
#include "eignn/encoder.hpp"


namespace nb = nanobind;
using namespace nb::literals;


namespace {

void enumerate_coords_2d(
        nb::tensor<float, nb::shape<nb::any, nb::any, nb::any>> &img,
        Eigen::MatrixXf &coords,
        Eigen::MatrixXf &rgb,
        const std::vector<int> &idx
) {
    const auto width = img.shape(0);
    const auto height = img.shape(1);
    coords.resize(2,width*height);
    rgb.resize(3,width*height);
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
            const int epochs
    ) {
        using namespace Eigen;

        const auto width = img.shape(0);
        const auto height = img.shape(1);
        const auto channels = img.shape(2);
        const auto pixels = width*height;

        std::cout << "img size: "
                  << width << "x" << height << "x" << channels
                  << std::endl;


        std::vector<int> idx(pixels);
        for (int ii = 0; ii < idx.size(); ++ii)
            idx[ii] = ii;
        std::random_device rd;
        std::mt19937 g(rd());

        MatrixXf coords, rgb;
        ::enumerate_coords_2d(img, coords, rgb, idx);


        std::vector<int> freq{1, 2, 4, 8};
        const int in_dim = 2*(1+2*freq.size());
        const int out_dim = 3;

        eignn::module::MLP mlp{in_dim, out_dim, hidden_dim, hidden_depth};
//        eignn::module::Sequential mlp;
//        mlp.modules.push_back(std::make_unique<eignn::module::Linear<true>>(in_dim,out_dim));
//        mlp.modules.push_back(std::make_unique<eignn::module::Sigmoid>());

#if defined(NDEBUG)
        const int batches = pixels/batch_size;
#else
        const int batches = 10;
#endif

        MatrixXf x, x_enc, y, y_tar;
        MatrixXf d_x, d_loss;
        x.resize(2, batch_size);

        eignn::MSELoss loss;

        for (int epoch_id = 0; epoch_id < epochs; ++epoch_id) {
            std::shuffle(idx.begin(), idx.end(), g);
            ::enumerate_coords_2d(img, coords, rgb, idx);
            for (int batch_id = 0; batch_id < batches; ++batch_id) {
                x = coords.block(0,batch_size*batch_id,2,batch_size);
                y_tar = rgb.block(0,batch_size*batch_id,3,batch_size);
                eignn::encoder::fourier_feature(x,x_enc,freq);
                mlp.forward(x_enc,y);
                assert(!std::isnan(y.sum()));
                float loss_val;
                loss.eval(y,y_tar,loss_val,d_loss);
                assert(!std::isnan(loss_val));
                d_loss *= -1.f * step_size;
                mlp.reverse(d_loss, d_x);
                assert(d_x.rows() == in_dim && d_x.cols() == batch_size);

#if defined(NDEBUG)
                if (batch_id == 0 || batch_id % (batches/10) != 0)
                    continue;
#endif

                std::cout << "epoch: " << epoch_id+1 << "/" << epochs
                          << " batch: " << batch_id+1 << "/" << batches
                          << ": loss = " << loss_val << std::endl;
            }
        }


        for (int ii = 0; ii < idx.size(); ++ii)
            idx[ii] = ii;
        ::enumerate_coords_2d(img, coords, rgb, idx);


        auto img_out = new float[pixels*channels];
        for (int iw = 0; iw < width; ++iw) {
            x = coords.block(0,height*iw,2,height);
            eignn::encoder::fourier_feature(x,x_enc,freq);
            mlp.forward(x_enc,y);

            for (int ih = 0; ih < height; ++ih) {
                for (int ic = 0; ic < channels; ++ic) {
                    img_out[channels*height*iw+channels*ih+ic] = y(ic,ih);
                }
            }
        }

        size_t shape[3]{width,height,channels};
        return nb::tensor<nb::numpy, float>{img_out,3,shape};
    });

}