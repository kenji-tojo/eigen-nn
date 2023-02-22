#include <iostream>
#include <array>

#include "eignn/training.hpp"


namespace {

struct Image {
public:
    void resize(int width, int height, int channels) {
        m_shape[0] = width;
        m_shape[1] = height;
        m_shape[2] = channels;
        m_data.resize(width*height*channels);
    }

    [[nodiscard]] size_t ndim() const { return m_shape.size(); }
    unsigned int &shape(size_t index) { return m_shape[index]; }

    float &operator()(size_t iw, size_t ih, size_t ic) {
        assert(iw<m_shape[0] && ih<m_shape[1] && ic<m_shape[2]);
        unsigned int index = m_shape[1]*m_shape[2]*iw + m_shape[2]*ih + ic;
        return m_data[index];
    }

private:
    std::array<unsigned int, 3> m_shape{0,0,1};
    std::vector<float> m_data;
};

} // namespace


int main() {
    using namespace std;
    using namespace Eigen;

    const int width = 16;
    const int height = 16;

    ::Image img;
    img.resize(width, height, /*channels=*/3);
    for (int iw = 0; iw < width; ++iw) {
        for (int ih = 0; ih < height; ++ih) {
            img(iw, ih, 0) = (float(iw)+.5f)/float(width);
            img(iw, ih, 1) = (float(ih)+.5f)/float(height);
            img(iw, ih, 2) = .5f;
        }
    }


    const int epochs = 3;
    const int batch_size = 32;
    const float learning_rate = 1e-3f;
    const int hidden_dim = 4;
    const int hidden_depth = 1;

    {
        std::cout << "testing vanilla mlp" << std::endl;

        eignn::module::FourierFeature</*ndim_=*/2> ff;
        ff.freqs = 0;

        auto mlp = eignn::fit_field(
                img, epochs, batch_size, learning_rate,
                hidden_dim, hidden_depth, ff
        );
        assert(mlp);


        std::cout << "testing fourier feature" << std::endl;
        ff.freqs = 3;
        mlp = eignn::fit_field(
                img, epochs, batch_size, learning_rate,
                hidden_dim, hidden_depth, ff
        );
        assert(mlp);

        eignn::render_field(img, *mlp, ff);
        std::cout << "img shape: "
                  << img.shape(0) << "x"
                  << img.shape(1) << "x"
                  << img.shape(2) << std::endl;
    }

    {
        std::cout << "testing hash encoding" << std::endl;

        const int min_res = 16;
        const int levels = 2;
        const int feature_dim = 2;
        const int table_size_log2 = 5;

        eignn::module::FeatureGrid</*ndim_=*/2> grid{
                min_res, feature_dim, levels, table_size_log2
        };

        auto mlp = eignn::fit_field(
                img, epochs, batch_size, learning_rate,
                hidden_dim, hidden_depth, grid
        );
        assert(mlp);

        eignn::render_field(img, *mlp, grid);
        std::cout << "img shape: "
                  << img.shape(0) << "x"
                  << img.shape(1) << "x"
                  << img.shape(2) << std::endl;
    }
}