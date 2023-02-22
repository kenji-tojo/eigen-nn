#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include "eignn/training.hpp"


namespace nb = nanobind;
using namespace nb::literals;


NB_MODULE(eignn, m) {

    m.def("fit_field_hash", [](
            nb::tensor<float, nb::shape<nb::any, nb::any, nb::any>> &img,
            const int epochs,
            const int batch_size,
            const float learning_rate,
            const int hidden_dim,
            const int hidden_depth,
            const int min_res,
            const int feature_dim,
            const int levels,
            const int table_size_log2
    ) {
        assert(img.ndim() == 3);
        const unsigned int width = img.shape(0);
        const unsigned int height = img.shape(1);
        const unsigned int channels = img.shape(2);

        std::cout << "img size: "
                  << width << "x" << height << "x" << channels
                  << std::endl;

        eignn::module::FeatureGrid</*ndim_=*/2> grid{
            min_res, feature_dim, levels, table_size_log2
        };

        auto mlp = eignn::fit_field(
                img, epochs, batch_size, learning_rate,
                hidden_dim, hidden_depth, grid
        );
        assert(mlp);

        eignn::render_field(img, *mlp, grid);
    });


    m.def("fit_field_ff", [](
            nb::tensor<float, nb::shape<nb::any, nb::any, nb::any>> &img,
            const int epochs,
            const int batch_size,
            const float learning_rate,
            const int hidden_dim,
            const int hidden_depth,
            const int freqs
    ) {
        assert(img.ndim() == 3);
        const unsigned int width = img.shape(0);
        const unsigned int height = img.shape(1);
        const unsigned int channels = img.shape(2);

        std::cout << "img size: "
                  << width << "x" << height << "x" << channels
                  << std::endl;

        eignn::module::FourierFeature</*ndim_=*/2> ff;
        ff.freqs = freqs;

        auto mlp = eignn::fit_field(
                img, epochs, batch_size, learning_rate,
                hidden_dim, hidden_depth, ff
        );
        assert(mlp);

        eignn::render_field(img, *mlp, ff);
    });


    m.def("fit_field_vanilla", [](
            nb::tensor<float, nb::shape<nb::any, nb::any, nb::any>> &img,
            const int epochs,
            const int batch_size,
            const float learning_rate,
            const int hidden_dim,
            const int hidden_depth
    ) {
        assert(img.ndim() == 3);
        const unsigned int width = img.shape(0);
        const unsigned int height = img.shape(1);
        const unsigned int channels = img.shape(2);

        std::cout << "img size: "
                  << width << "x" << height << "x" << channels
                  << std::endl;

        eignn::module::FourierFeature</*ndim_=*/2> ff;
        ff.freqs = 0;

        auto mlp = eignn::fit_field(
                img, epochs, batch_size, learning_rate,
                hidden_dim, hidden_depth, ff
        );
        assert(mlp);

        eignn::render_field(img, *mlp, ff);
    });

}