#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include "eignn/training.hpp"


namespace nb = nanobind;
using namespace nb::literals;


namespace eignn::binding {

class NeuralField2D {
public:
    NeuralField2D() = default;

    void set_network(int hidden_dim, int hidden_depth, int out_dim) {
        auto ff = std::make_unique<module::FourierFeature</*ndim_=*/2>>();
        ff->freqs = 0;
        enc = std::move(ff);

        int in_dim = enc->out_dim();
        mlp = std::make_unique<module::MLP>(
                in_dim, out_dim, hidden_dim, hidden_depth
        );

        register_parameters();
    }

    void set_network_ff(int hidden_dim, int hidden_depth, int out_dim, int freqs) {
        auto ff = std::make_unique<module::FourierFeature</*ndim_=*/2>>();
        ff->freqs = freqs;
        enc = std::move(ff);

        int in_dim = enc->out_dim();
        mlp = std::make_unique<module::MLP>(
                in_dim, out_dim, hidden_dim, hidden_depth
        );

        register_parameters();
    }

    void set_network_hash(
            int hidden_dim, int hidden_depth, int out_dim,
            int min_res, int feature_dim, int levels, int table_size_log2

    ) {
        enc = std::make_unique<module::FeatureGrid</*ndim_=*/2>>(
                min_res, feature_dim, levels, table_size_log2
        );

        int in_dim = enc->out_dim();
        mlp = std::make_unique<module::MLP>(
                in_dim, out_dim, hidden_dim, hidden_depth
        );

        register_parameters();
    }

    void fit(
            nb::tensor<float, nb::shape<nb::any, nb::any, nb::any>> &img,
            const int epoch_start, const int epoch_end, const int epochs, const int batch_size, const float learning_rate
    ) {
        if (!mlp || !enc) {
            std::cerr << "error: no network is set" << std::endl;
            return;
        }

        assert(img.ndim() == 3);
        const unsigned int width = img.shape(0);
        const unsigned int height = img.shape(1);
        const unsigned int channels = img.shape(2);

        std::cout << "img size: "
                  << width << "x" << height << "x" << channels
                  << std::endl;

        optimizer->learning_rate = learning_rate;
        fit_field(img, epoch_start, epoch_end, epochs, batch_size, *optimizer, *mlp, *enc);
    }

    void render(nb::tensor<float, nb::shape<nb::any, nb::any, nb::any>> &img) {
        if (!mlp || !enc) {
            std::cerr << "error: no network is set" << std::endl;
            return;
        }

        render_field(img, *mlp, *enc);
    }

private:
    std::unique_ptr<eignn::module::MLP> mlp;
    std::unique_ptr<eignn::module::Encoder> enc;

    std::unique_ptr<Optimizer> optimizer = std::make_unique<Adam>();

    void register_parameters() {
        optimizer->add_parameters(mlp->parameters());
        optimizer->add_parameters(enc->parameters());
    }
};

} // namespace eignn::binding


using namespace eignn::binding;

NB_MODULE(eignn, m) {
    nb::class_<NeuralField2D>(m, "NeuralField2D")
            .def(nb::init())
            .def("set_network", &NeuralField2D::set_network)
            .def("set_network_ff", &NeuralField2D::set_network_ff)
            .def("set_network_hash", &NeuralField2D::set_network_hash)
            .def("fit", &NeuralField2D::fit)
            .def("render", &NeuralField2D::render);
}