#include <iostream>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include "eignn/module.hpp"


namespace nb = nanobind;
using namespace nb::literals;


NB_MODULE(eignn, m) {

    m.def("fit_nn", [](
            nb::tensor<float, nb::shape<nb::any, nb::any, nb::any>> &img
    ) {
        using namespace std;

        const auto width = img.shape(0);
        const auto height = img.shape(1);
        const auto channels = img.shape(2);

        cout << "img size: "
             << width << "x" << height << "x" << channels
             << endl;

        std::vector<float> img_out(width*height*channels);
        for (int iw = 0; iw < width; ++iw) {
            for (int ih = 0; ih < height; ++ih) {
                for (int ic = 0; ic < channels; ++ic) {
                    img_out[channels*height*iw+channels*ih+ic] = img(iw,ih,ic);
                }
            }
        }

        size_t shape[3]{width,height,channels};
        return nb::tensor<nb::numpy, float>{img_out.data(),3,shape};
    });

}