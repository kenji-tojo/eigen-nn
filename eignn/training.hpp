#pragma once

#include "eignn/module.hpp"
#include "eignn/loss.hpp"
#include "eignn/encoder.hpp"
#include "eignn/optimizer.hpp"


namespace eignn {

template<typename Tensor_, typename Index_>
void create_pixel_dataset(
        Tensor_ &img,
        Eigen::MatrixXf &coords,
        Eigen::MatrixXf &rgb,
        const std::vector<Index_> &idx_map
) {
    static_assert(std::is_integral_v<Index_>);

    assert(img.ndim() == 3);
    const unsigned int width = img.shape(0);
    const unsigned int height = img.shape(1);
    const unsigned int channels = img.shape(2);

    coords.resize(2,width*height);
    rgb.resize(channels,width*height);
    for (int iw = 0; iw < width; ++iw) {
        for (int ih = 0; ih < height; ++ih) {
            coords(0, idx_map[height*iw+ih]) = (float(iw)+.5f)/float(width);
            coords(1, idx_map[height*iw+ih]) = (float(ih)+.5f)/float(height);
            for (int ic = 0; ic < channels; ++ic)
                rgb(ic,idx_map[height*iw+ih]) = img(iw,ih,ic);
        }
    }
}


template<typename Tensor_, typename Encoder_>
std::unique_ptr<eignn::module::MLP> fit_field(
        Tensor_ &img,
        const int epochs,
        const int batch_size,
        const float learning_rate,
        const int hidden_dim,
        const int hidden_depth,
        Encoder_ &enc
) {
    using namespace Eigen;

    assert(img.ndim() == 3);
    const unsigned int width = img.shape(0);
    const unsigned int height = img.shape(1);
    const unsigned int pixels = width*height;

    const int in_dim = enc.out_dim();
    const int out_dim = 3;
    auto mlp = std::make_unique<module::MLP>(
            in_dim, out_dim, hidden_dim, hidden_depth
    );

    MSELoss loss;
    Adam optimizer;
    optimizer.learning_rate = learning_rate;
    optimizer.add_parameters(mlp->parameters());
    optimizer.add_parameters(enc.parameters());

    Shuffler shuffler;
    MatrixXf coords, rgb;
    std::vector<unsigned int> idx_map;
    idx_map.resize(pixels);
    for (int ii = 0; ii < idx_map.size(); ++ii) idx_map[ii] = ii;

    MatrixXf x, y_tar, loss_adj;

    for (int epoch_id = 0; epoch_id < epochs; ++epoch_id) {
        shuffler.shuffle(idx_map);
        create_pixel_dataset(img, coords, rgb, idx_map);

        const unsigned int batches = pixels/batch_size;
        for (int batch_id = 0; batch_id < batches; ++batch_id) {
            const unsigned int start = batch_size * batch_id;
            const unsigned int end   = batch_id < batches-1 ? start+batch_size : pixels;

            x = coords.block(0,start,2,end-start);
            y_tar = rgb.block(0,start,3,end-start);

            enc.forward(x);
            mlp->forward(enc.y);
            assert(!std::isnan(mlp->y.sum()));

            float loss_val;
            loss.eval(mlp->y, y_tar, loss_val, loss_adj);
            assert(!std::isnan(loss_val));

            mlp->adjoint(loss_adj);
            enc.adjoint(mlp->x_adj);
            assert(enc.x_adj.rows()==x.rows() && enc.x_adj.cols()==x.cols());

            optimizer.descent();

            if (batches/10 > 0 && (batch_id+1) % (batches/10) != 0) continue;

            std::cout << "epoch: " << epoch_id+1 << "/" << epochs << "; "
                      << "batch: " << batch_id+1 << "/" << batches << "; "
                      << "loss = " << loss_val << std::endl;
        }
    }

    return mlp;
}


template<typename Tensor_, typename Encoder_>
void render_field(
        Tensor_ &img,
        eignn::module::MLP &mlp,
        Encoder_ &enc
) {
    using namespace Eigen;

    assert(img.ndim() == 3);
    const unsigned int width = img.shape(0);
    const unsigned int height = img.shape(1);
    const unsigned int channels = img.shape(2);
    const unsigned int pixels = width*height;

    std::vector<unsigned int> idx_map;
    idx_map.resize(pixels);
    for (int ii = 0; ii < idx_map.size(); ++ii) idx_map[ii] = ii;

    MatrixXf coords, rgb;
    create_pixel_dataset(img, coords, rgb, idx_map);

    for (int iw = 0; iw < width; ++iw) {
        const MatrixXf &x = coords.block(0,height*iw,2,height);
        enc.forward(x);
        mlp.forward(enc.y);
        for (int ih = 0; ih < height; ++ih)
            for (int ic = 0; ic < channels; ++ic)
                img(iw,ih,ic) = mlp.y(ic,ih);
    }

}


} // namespace eignn